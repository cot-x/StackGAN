#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#from comet_ml import Experiment
#experiment = Experiment()


# In[ ]:


import os
import math
import random
import cv2
import time
import argparse
import itertools
import string

from tqdm import tqdm
from PIL import Image, ImageFile
from pickle import load, dump
from pathlib import Path

import numpy as np
import pandas as pd
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image

#import torchtext
#if 'legacy' in dir(torchtext):
#    import torchtext.legacy as torchtext

from torch import einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[ ]:


class Mish(nn.Module):
    @staticmethod
    def mish(x):
        return x * torch.tanh(F.softplus(x))
    
    def forward(self, x):
        return Mish.mish(x)


# In[ ]:


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        
        self.dropout = nn.Dropout(dropout)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if not (heads == 1 and dim_head == dim) else nn.Identity()
        
    def forward(self, x, return_attention=False):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        query, key, value = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        
        attention_score = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        attention_prob = F.softmax(attention_score, dim=-1)
        attention_prob = self.dropout(attention_prob)
        
        context = torch.matmul(attention_prob, value)
        context = rearrange(context, 'b h n d -> b n (h d)')
        
        if return_attention:
            return self.to_out(context), attention_prob
        else:
            return self.to_out(context)


# In[ ]:


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            Mish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


# In[ ]:


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# In[ ]:


class Transformer(nn.Module):
    def __init__(self, dim, mlp_dim, depth=4, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
            
    def forward(self, x):
        for attention, feedforward in self.layers:
            x = attention(x) + x
            x = feedforward(x) + x
        return x


# In[ ]:


class PoolFormer(nn.Module):
    def __init__(self, dim, mlp_dim, pool_size=3, depth=4, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, nn.AvgPool2d(pool_size, stride=1, padding=pool_size//2, count_include_pad=False)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
            
    def forward(self, x):
        for pooling, feedforward in self.layers:
            x = pooling(x) + x
            x = feedforward(x) + x
        return x


# In[ ]:


class PositionalEncoder(nn.Module):
    def __init__(self, vocab_size, sentence_size):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.pe = torch.Tensor(sentence_size, vocab_size)
        for pos in range(sentence_size):
            for i in range(0, vocab_size, 2):
                self.pe[pos, i] = math.sin(pos / (10000**((2*i)/vocab_size)))
                self.pe[pos, i+1] = math.cos(pos / (10000**((2*(i+1))/vocab_size)))
        self.pe = self.pe.unsqueeze(0)
        self.pe.requires_grad = False
    
    def to(self, device):
        self.pe = self.pe.to(device)
        return super().to(device)
    
    def forward(self, x):
        return math.sqrt(self.vocab_size) * x + self.pe


# In[ ]:


class Embedding(nn.Module):
    def __init__(self, vocab_size, sentence_size, dim, dropout=0.):
        super().__init__()
        
        self.word_embedding = nn.Embedding(vocab_size, dim)
        #self.position_embedding = nn.Embedding(sentence_size, dim)
        self.position_embedding = PositionalEncoder(dim, sentence_size)
        self.LayerNorm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids):
        word_embedding = self.word_embedding(input_ids)
        
        #position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        #position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        #position_embedding = self.position_embedding(position_ids)
        
        #embedding = word_embedding + position_embedding
        embedding = self.position_embedding(word_embedding)
        embedding = self.LayerNorm(embedding)
        embedding = self.dropout(embedding)
        
        return embedding


# In[ ]:


class TextTransformer(nn.Module):
    def __init__(self, vocab_size, sentence_size, dim_out=128, dim=512, mlp_dim=1024,
                 pool='cls', channels=3, dropout=0., emb_dropout=0.):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool
        
        self.embedding = Embedding(vocab_size, sentence_size, dim, emb_dropout)
        self.transformer = Transformer(dim, mlp_dim, dropout=dropout)
        
        self.f = nn.Identity()

        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_out)
        )
    
    def to(self, *args, **kwargs):
        self.embedding.position_embedding.to(args[0])
        return super().to(*args, **kwargs)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.f(x)
        return self.mlp(x)


# In[ ]:


class VisionTransformer(nn.Module):
    def __init__(self, max_patches, patch_size, num_classes, dim=512, mlp_dim=1024,
                 pool='cls', channels=3, dropout=0., emb_dropout=0.):
        super().__init__()
        
        def pair(t):
            return t if isinstance(t, tuple) else (t, t)
        
        self.patch_height, self.patch_width = pair(patch_size)
        patch_dim = channels * self.patch_height * self.patch_width
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        self.patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.patch_height, p2 = self.patch_width),
            nn.Linear(patch_dim, dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, max_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        #self.transformer = Transformer(dim, mlp_dim, dropout=dropout)
        self.transformer = PoolFormer(dim, mlp_dim, dropout=dropout)
        
        self.f = nn.Identity()

        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        h, w = img.size(2), img.size(3)
        resize = transforms.Resize((h - h % self.patch_height, w - w % self.patch_width))
        img = resize(img)
        
        x = self.patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x += self.pos_embedding[:, :(n + 1)]
        
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.f(x)
        
        return self.mlp(x)


# In[ ]:


class CLIP(nn.Module):
    def __init__(self, max_patches, patch_size, vocab_size, sentence_size, img_classes=128, text_classes=128, dim_embed=64):
        super().__init__()
        
        self.image_encoder = VisionTransformer(max_patches, patch_size, img_classes)
        self.text_encoder = TextTransformer(vocab_size, sentence_size, text_classes)
        self.weight_image = nn.Parameter(torch.randn(img_classes, dim_embed))
        self.weight_text = nn.Parameter(torch.randn(text_classes, dim_embed))
        self.logit_scale = nn.Parameter(torch.randn(1))
    
    def to(self, *args, **kwargs):
        self.text_encoder.to(args[0])
        return super().to(*args, **kwargs)
    
    def forward(self, image, text):
        img = self.image_encoder(image)
        text = self.text_encoder(text)
        
        image_norm = torch.mm(img, self.weight_image).norm(dim=1, keepdim=True)
        text_norm = torch.mm(text, self.weight_text).norm(dim=1, keepdim=True)
        
        logits = torch.mm(image_norm, text_norm.T) * self.logit_scale.exp()
        
        n = logits.size(0)
        labels = torch.arange(n).to(logits.device)
        loss_image = F.cross_entropy(logits, labels)
        loss_text = F.cross_entropy(logits.T, labels)
        loss = (loss_image + loss_text) / 2
        
        return loss


# In[ ]:


class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.paths = self.get_paths(img_dir)
        self.transform = transform
    
    def get_paths(self, img_dir):
        paths = []
        print('Make image-path of directories.')
        for root, dirs, files in tqdm(os.walk(img_dir)):
            for file in files:
                paths += [Path(os.path.join(root, file))]
        return paths
    
    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path)
        if self.transform is not None:
            image = self.transform(image)
        return image
    
    def __len__(self):
        return len(self.paths)


# In[ ]:


class TextDataset:
    def __init__(self, text_dir):
        self.filenames = []
        self.texts = []
        print('Make text-data of directories.')
        for root, dirs, files in tqdm(os.walk(text_dir)):
            for file in files:
                filename = os.path.splitext(file)
                if filename[1] == '.txt':
                    with open(os.path.join(root, file), 'r') as f:
                        self.filenames += [filename[0] + '.jpg']
                        self.texts += [TextDataset.tokenizer(f.read())]
    
    @staticmethod
    def tokenizer(text):
        for s in string.punctuation:
            if not(s == '.' or s == ','):
                text = text.replace(s, ' ')
            else:
                text = text.replace(s, ' ' + s + ' ')
        text = text.strip().split()
        return text
    
    def __getitem__(self, index):
        return self.filenames[index], self.texts[index]
        
    def __len__(self):
        return len(self.filenames)
    
    def tolist(self):
        return sum(self.texts, [])


# In[ ]:


class TextAndImageDataset(ImageDataset):
    @staticmethod
    def make_vocab(textdata: TextDataset, vocab_size=None):
        print('Generate word-ids.')
        word2id = {}
        word2id['<pad>'] = 0
        word2id['<unk>'] = 1
        
        #wc = collections.Counter(textdata.tolist())
        #for i, (w, _) in enumerate(wc.most_common(vocab_size), 2):
        #    word2id[w] = i
        
        id2word = {v: k for k, v in word2id.items()}

        for texts in tqdm(textdata):
            for words in texts:
                for word in words:
                    if word not in word2id:
                        id = len(word2id)
                        word2id[word] = id
                        id2word[id] = word
        
        return word2id, id2word
    
    def __init__(self, text_dir, img_dir, sentence_size, vocab_size=None, transform=None):
        super().__init__(img_dir, transform)
        
        self.sentence_size = sentence_size
        
        if os.path.exists('textdata.dat'):
            with open(os.path.join('.', 'textdata.dat'), 'rb') as f:
                self.textdata = load(f)
                print('Loaded textdata.dat.')
        else:
            self.textdata = TextDataset(text_dir)
            with open(os.path.join('.', 'textdata.dat'), 'wb') as f:
                dump(self.textdata, f)
                print('Saved textdata.dat.')
        
        if os.path.exists('word2id.dat') and os.path.exists('id2word.dat'):
            with open(os.path.join('.', 'word2id.dat'), 'rb') as f:
                self.word2id = load(f)
                print('Loaded word2id.dat.')
            with open(os.path.join('.', 'id2word.dat'), 'rb') as f:
                self.id2word = load(f)
                print('Loaded id2word.dat.')
        else:
            self.word2id, self.id2word = TextAndImageDataset.make_vocab(self.textdata, vocab_size)
            with open(os.path.join('.', 'word2id.dat'), 'wb') as f:
                dump(self.word2id, f)
                print('Saved word2id.dat.')
            with open(os.path.join('.', 'id2word.dat'), 'wb') as f:
                dump(self.id2word, f)
                print('Saved id2word.dat.')
        
    
    def to_string(self, tensor):
        text = ''
        for d in tensor.tolist():
            text += self.id2word[d] + ' '
        return text
    
    def to_tokens(self, data):
        tokens = []
        for d in data:
            tokens += [self.word2id[d] if d in self.word2id else self.word2id['<unk>']]
        return tokens
    
    def __getitem__(self, index):
        filename, text = self.textdata[index]
        
        path = [path for path in self.paths if filename == path.name][0]
        image = Image.open(path)
        if self.transform is not None:
            image = self.transform(image)
        else:
            toTensor = transforms.ToTensor()
            image = toTensor(image)
        
        text = self.to_tokens(text)
        text.extend([self.word2id['<pad>'] for _ in range(self.sentence_size - len(text))])
        text = torch.LongTensor(text)
        
        return image, text
    
    def __len__(self):
        return len(self.textdata)


# In[ ]:


class Solver:
    def __init__(self, args):
        has_cuda = torch.cuda.is_available() if not args.cpu else False
        self.device = torch.device("cuda" if has_cuda else "cpu")
        
        self.args = args
        
        self.load_dataset()
        
        self.CLIP = CLIP(self.args.CLIP_max_patches, self.args.CLIP_patch_size,
                         len(self.dataset.word2id), self.args.CLIP_sentence_size).to(self.device)
        self.CLIP.apply(self.weights_init)
        self.optimizer_CLIP = optim.Adam(self.CLIP.parameters(), lr=self.args.lr, betas=(0, 0.9))
        
        self.epoch = 0
    
    def weights_init(self, module):
        if type(module) == nn.Linear or type(module) == nn.Conv2d or type(module) == nn.ConvTranspose2d:
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0)
            
    def load_dataset(self, img_size=256):
        self.dataset = TextAndImageDataset(self.args.text_dir, self.args.image_dir, self.args.CLIP_sentence_size,
                                           transform=transforms.Compose([
                                               transforms.Resize(int(img_size)),
                                               transforms.RandomCrop(img_size),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor()
                                           ]))
        self.dataloader = DataLoader(self.dataset, batch_size=self.args.batch_size,
                                     shuffle=True, drop_last=True, num_workers=os.cpu_count())
        self.max_iters = len(iter(self.dataloader))
            
    def save_state(self, epoch):
        self.CLIP.cpu()
        torch.save(self.CLIP.state_dict(), os.path.join(self.args.weight_dir, f'weight_CLIP.{epoch}.pth'))
        self.CLIP.to(self.device)
        
    def load_state(self):
        if os.path.exists('weight_CLIP.pth'):
            self.CLIP.load_state_dict(torch.load('weight_CLIP.pth', map_location=self.device))
            print('Loaded CLIP network state.')
    
    def save_resume(self):
        with open(os.path.join('.', f'resume.pkl'), 'wb') as f:
            dump(self, f)
    
    @staticmethod
    def load(args, resume=False):
        if resume and os.path.exists('resume.pkl'):
            with open(os.path.join('.', 'resume.pkl'), 'rb') as f:
                print('Load resume.')
                solver = load(f)
                solver.args = args
                return solver
        else:
            return Solver(args)
    
    def trainCLIP(self, epoch, iters, max_iters, images, texts):
        loss = self.CLIP(images, texts)
        
        self.optimizer_CLIP.zero_grad()
        loss.backward()
        self.optimizer_CLIP.step()
        
        return loss.item()
    
    def train(self):
        print(f'Use Device: {self.device}')
        torch.backends.cudnn.benchmark = True
        
        self.CLIP.train()
        
        hyper_params = {}
        hyper_params['Text Dir'] = self.args.text_dir
        hyper_params['Image Dir'] = self.args.image_dir
        hyper_params['Result Dir'] = self.args.result_dir
        hyper_params['Weight Dir'] = self.args.weight_dir
        hyper_params['CLIP_max_patches'] = self.args.CLIP_max_patches
        hyper_params['CLIP_patch_size'] = self.args.CLIP_patch_size
        hyper_params['CLIP_sentence_size'] = self.args.CLIP_sentence_size
        hyper_params['Learning Rate'] = self.args.lr
        hyper_params['Batch Size'] = self.args.batch_size
        hyper_params['Num Train'] = self.args.num_train
        
        for key in hyper_params.keys():
            print(f'{key}: {hyper_params[key]}')
        #experiment.log_parameters(hyper_params)
        
        while self.args.num_train > self.epoch:
            self.epoch += 1
            epoch_loss_CLIP = 0.0
            
            for iters, (images, texts) in enumerate(tqdm(self.dataloader)):
                iters += 1
                loss = {}
                
                images = images.to(self.device, non_blocking=True)
                texts = texts.to(self.device, non_blocking=True)
                
                loss_CLIP = self.trainCLIP(self.epoch, iters, self.max_iters, images, texts)
                loss['CLIP/loss'] = loss_CLIP
                
                epoch_loss_CLIP += loss['CLIP/loss']
                #experiment.log_metrics(loss)
            
            epoch_loss = epoch_loss_CLIP
            
            print(f'Epoch[{self.epoch}]'
                  + f' + Loss[CLIP({epoch_loss_CLIP}) = {epoch_loss}]')
            
            self.save_state('last')
                    
            if not self.args.noresume:
                self.save_resume()


# In[ ]:


def main(args):
    solver = Solver.load(args, resume=not args.noresume)
    solver.load_state()
    
    solver.train()
    #experiment.end()


# In[ ]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_dir', type=str, default='/usr/share/datasets/cub2002011/cvpr2016_cub/text_c10/')
    parser.add_argument('--image_dir', type=str, default='/usr/share/datasets/cub2002011/CUB_200_2011/images/')
    parser.add_argument('--result_dir', type=str, default='results')
    parser.add_argument('--weight_dir', type=str, default='weights')
    parser.add_argument('--CLIP_max_patches', type=int, default=128)
    parser.add_argument('--CLIP_patch_size', type=int, default=32)
    parser.add_argument('--CLIP_sentence_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_train', type=int, default=10)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--noresume', action='store_true')
    
    args, unknown = parser.parse_known_args()
    
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    if not os.path.exists(args.weight_dir):
        os.mkdir(args.weight_dir)
    
    main(args)

