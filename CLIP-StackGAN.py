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


class FReLU(nn.Module):
    def __init__(self, n_channel, kernel=3, stride=1, padding=1):
        super().__init__()
        self.funnel_condition = nn.Conv2d(n_channel, n_channel, kernel_size=kernel,stride=stride, padding=padding, groups=n_channel)
        self.normalize = nn.BatchNorm2d(n_channel)

    def forward(self, x):
        tx = self.normalize(self.funnel_condition(x))
        out = torch.max(x, tx)
        return out


# In[ ]:


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.shortcut = nn.Sequential()
        self.residual = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
            FReLU(in_features),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features)
        )
        self.activate = FReLU(in_features)

    def forward(self, x):
        shortcut = self.shortcut(x)
        return self.activate(self.residual(x) + shortcut)


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
            nn.Mish(inplace=True),
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


class STAGE1_G(nn.Module):
    def __init__(self, dim_c_code=128, dim_noise=128, dim_ideal=1024):
        super().__init__()
        
        self.dim_noise = dim_noise
        
        def upBlock(dim_in, dim_out):
            block = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(dim_out),
                FReLU(dim_out)
            )
            return block
        
        self.encoder = nn.Sequential(
            nn.Linear(dim_c_code + dim_noise, dim_ideal * 4 * 4),
            nn.Mish(inplace=True)
        )

        self.upsample1 = upBlock(dim_ideal, dim_ideal // 2)       # 8x8
        self.upsample2 = upBlock(dim_ideal // 2, dim_ideal // 4)  # 16x16
        self.upsample3 = upBlock(dim_ideal // 4, dim_ideal // 8)  # 32x32
        self.upsample4 = upBlock(dim_ideal // 8, dim_ideal // 16) # 64x64
        
        self.toRGB = nn.Sequential(
            nn.Conv2d(dim_ideal // 16, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, text):
        noise = torch.randn(text.size(0), self.dim_noise).to(text.device)
        c_code = torch.cat((noise, text), 1)
        
        estimate = self.encoder(c_code)
        
        h_code = estimate.view(estimate.size(0), -1, 4, 4)
        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)
        h_code = self.upsample4(h_code)
        fake_img = self.toRGB(h_code)
        
        _noise = torch.randn(c_code.shape).to(text.device)
        true_pdf = self.encoder(_noise).softmax(dim=1)
        
        # Variational Conditional GAN's Loss
        kl_loss = nn.KLDivLoss(reduction='sum')
        vc_loss = - kl_loss(estimate.softmax(dim=1), true_pdf)
        
        return fake_img, vc_loss, noise


# In[ ]:


class STAGE1_D(nn.Module):
    def __init__(self, dim_c_code=128, dim_ideal=64):
        super().__init__()
        
        def downBlock(dim_in, dim_out):
            block = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(dim_out),
                FReLU(dim_out)
            )
            return block
        
        self.downconv = nn.Sequential(
            downBlock(3, dim_ideal),                 # 32x32
            downBlock(dim_ideal, dim_ideal * 2),     # 16x16
            downBlock(dim_ideal * 2, dim_ideal * 4), # 8x8
            downBlock(dim_ideal * 4, dim_ideal * 8)  # 4x4
        )
        
        self.conv_patch = nn.Conv2d(dim_ideal * 8 + dim_c_code, 1, kernel_size=3, stride=1, padding=1)
        
    def forward(self, image, text):
        cond = self.downconv(image)
        
        text = text.view(text.size(0), -1, 1, 1)
        c_code = text.repeat(1, 1, 4, 4)
        
        c_code = torch.cat((cond, c_code), 1)
        
        patch = self.conv_patch(c_code)
        return patch


# In[ ]:


class STAGE2_G(nn.Module):
    def __init__(self, dim_c_code=128, dim_ideal=128, n_residual=4):
        super().__init__()
        
        def upBlock(dim_in, dim_out):
            block = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(dim_out),
                FReLU(dim_out)
            )
            return block
        
        self.downsample = nn.Sequential(
            nn.Conv2d(3, dim_ideal, kernel_size=3, stride=1, padding=1),
            FReLU(dim_ideal),
            nn.Conv2d(dim_ideal, dim_ideal * 2, kernel_size=4, stride=2, padding=1),     # 32x32
            nn.BatchNorm2d(dim_ideal * 2),
            FReLU(dim_ideal * 2),
            nn.Conv2d(dim_ideal * 2, dim_ideal * 4, kernel_size=4, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(dim_ideal * 4),
            FReLU(dim_ideal * 4)
        )
        
        layers = [
            nn.Conv2d(dim_c_code + dim_ideal * 4, dim_ideal * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_ideal * 4),
            FReLU(dim_ideal * 4)
        ]
        for _ in range(n_residual):
            layers += [ResidualBlock(dim_ideal * 4)]
        self.encoder = nn.Sequential(*layers)
        
        self.upsample1 = upBlock(dim_ideal * 4, dim_ideal * 2)   # 32x32
        self.upsample2 = upBlock(dim_ideal * 2, dim_ideal)       # 64x64
        self.upsample3 = upBlock(dim_ideal, dim_ideal // 2)      # 128x128
        self.upsample4 = upBlock(dim_ideal // 2, dim_ideal // 4) # 256x256
        
        self.toRGB = nn.Sequential(
            nn.Conv2d(dim_ideal // 4, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, image, text):
        cond = self.downsample(image)
        text = text.view(text.size(0), -1, 1, 1)
        c_code = text.repeat(1, 1, 16, 16)
        c_code = torch.cat([cond, c_code], 1)
        
        estimate = self.encoder(c_code)

        h_code = self.upsample1(estimate)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)
        h_code = self.upsample4(h_code)
        fake_img = self.toRGB(h_code)
        
        noise = torch.randn(c_code.shape).to(c_code.device)
        true_pdf = self.encoder(noise).softmax(dim=1)
        
        # Variational Conditional GAN's Loss
        kl_loss = nn.KLDivLoss(reduction='sum')
        vc_loss = - kl_loss(estimate.softmax(dim=1), true_pdf)
        
        return fake_img, vc_loss


# In[ ]:


class STAGE2_D(nn.Module):
    def __init__(self, dim_c_code=128, dim_ideal=64):
        super().__init__()
        
        def downBlock(dim_in, dim_out):
            block = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(dim_out),
                FReLU(dim_out)
            )
            return block
        
        self.downconv = nn.Sequential(
            downBlock(3, dim_ideal),                   # 128x128
            downBlock(dim_ideal, dim_ideal * 2),       # 64x64
            downBlock(dim_ideal * 2, dim_ideal * 4),   # 32x32
            downBlock(dim_ideal * 4, dim_ideal * 8),   # 16x16
            downBlock(dim_ideal * 8, dim_ideal * 16),  # 8x8
            downBlock(dim_ideal * 16, dim_ideal * 32), # 4x4
            nn.Conv2d(dim_ideal * 32, dim_ideal * 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_ideal * 16),
            FReLU(dim_ideal * 16),
            nn.Conv2d(dim_ideal * 16, dim_ideal * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_ideal * 8),
            FReLU(dim_ideal * 8)
        )
        
        self.conv_patch = nn.Conv2d(dim_ideal * 8 + dim_c_code, 1, kernel_size=3, stride=1, padding=1)
    
    def forward(self, image, text):
        cond = self.downconv(image)
        
        text = text.view(text.size(0), -1, 1, 1)
        c_code = text.repeat(1, 1, 4, 4)
        
        c_code = torch.cat((cond, c_code), 1)
        
        patch = self.conv_patch(c_code)
        return patch


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
        self.pseudo_aug = 0.0
        self.epoch = 0
        
        self.load_dataset()
        
        self.CLIP = CLIP(self.args.CLIP_max_patches, self.args.CLIP_patch_size,
                         len(self.dataset.word2id), self.args.CLIP_sentence_size).to(self.device)
        self.CLIP.apply(self.weights_init)
        
        self.stage1_g = STAGE1_G().to(self.device)
        self.stage1_d = STAGE1_D().to(self.device)
        self.stage2_g = STAGE2_G().to(self.device)
        self.stage2_d = STAGE2_D().to(self.device)

        self.stage1_g.apply(self.weights_init)
        self.stage1_d.apply(self.weights_init)
        self.stage2_g.apply(self.weights_init)
        self.stage2_d.apply(self.weights_init)
        
        self.optimizer_CLIP = optim.Adam(self.CLIP.parameters(), lr=self.args.lr, betas=(0, 0.9))
        self.optimizer_G = optim.Adam(itertools.chain(self.stage1_g.parameters(),
                                                      self.stage2_g.parameters()),
                                      lr=self.args.lr, betas=(0, 0.9))
        self.optimizer_D = optim.Adam(itertools.chain(self.stage1_d.parameters(),
                                                      self.stage2_d.parameters()),
                                      lr=self.args.lr * self.args.mul_lr_dis, betas=(0, 0.9))
    
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
        self.CLIP.cpu(), self.stage1_g.cpu(), self.stage1_d.cpu(), self.stage2_g.cpu(), self.stage2_d.cpu()
        torch.save(self.CLIP.state_dict(), os.path.join(self.args.weight_dir, f'weight_CLIP.{epoch}.pth'))
        torch.save(self.stage1_g.state_dict(), os.path.join(self.args.weight_dir, f'weight_G1.{epoch}.pth'))
        torch.save(self.stage1_d.state_dict(), os.path.join(self.args.weight_dir, f'weight_D1.{epoch}.pth'))
        torch.save(self.stage2_g.state_dict(), os.path.join(self.args.weight_dir, f'weight_G2.{epoch}.pth'))
        torch.save(self.stage2_d.state_dict(), os.path.join(self.args.weight_dir, f'weight_D2.{epoch}.pth'))
        self.CLIP.to(self.device), self.stage1_g.to(self.device), self.stage1_d.to(self.device), self.stage2_g.to(self.device), self.stage2_d.to(self.device)
        
    def load_state(self):
        if os.path.exists('weight_CLIP.pth'):
            self.CLIP.load_state_dict(torch.load('weight_CLIP.pth', map_location=self.device))
            print('Loaded CLIP network state.')
        if os.path.exists('weight_G1.pth'):
            self.stage1_g.load_state_dict(torch.load('weight_G1.pth', map_location=self.device))
            print('Loaded Stage1_G network state.')
        if os.path.exists('weight_D1.pth'):
            self.stage1_d.load_state_dict(torch.load('weight_D1.pth', map_location=self.device))
            print('Loaded Stage1_D network state.')
        if os.path.exists('weight_G2.pth'):
            self.stage2_g.load_state_dict(torch.load('weight_G2.pth', map_location=self.device))
            print('Loaded Stage2_G network state.')
        if os.path.exists('weight_D2.pth'):
            self.stage2_d.load_state_dict(torch.load('weight_D2.pth', map_location=self.device))
            print('Loaded Stage2_D network state.')
    
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
    
    def trainCLIP(self, images, texts):
        clip_loss = self.CLIP(images, texts)
        
        # Backward and optimize.
        self.optimizer_CLIP.zero_grad()
        clip_loss.backward()
        self.optimizer_CLIP.step()
        
        # Logging.
        loss = {}
        loss['CLIP/loss'] = clip_loss.item()
        
        return loss
    
    def trainGAN(self, epoch, iters, max_iters, real_img, texts, a=0, b=1, c=1):
        ### Train with LSGAN.
        ### for example, (a, b, c) = 0, 1, 1 or (a, b, c) = -1, 1, 0
        
        resize_64 = transforms.Resize(64)
        resize_256 = transforms.Resize(256)
        real_img_64 = resize_64(real_img)
        real_img_256 = resize_256(real_img)
        loss = {}
        
        # ================================================================================ #
        #                             Train the CLIP                                       #
        # ================================================================================ #
        
        self.CLIP.train()
        loss = self.trainCLIP(real_img_256, texts)
        self.CLIP.eval()
        
        # ================================================================================ #
        #                             Train the discriminator                              #
        # ================================================================================ #
        
        text = self.CLIP.text_encoder(texts)
        text = text.detach()
        
        fake_img_1, vc_loss_1, noise = self.stage1_g(text)
        fake_score_1 = self.stage1_d(fake_img_1, text)
        
        fake_img_2, vc_loss_2 = self.stage2_g(fake_img_1, text)
        fake_score_2 = self.stage2_d(fake_img_2, text)
        
        # for Mode-Seeking
        _fake_img_2 = Variable(fake_img_2.data)
        _noise = Variable(noise.data)
        
        real_score_1 = self.stage1_d(real_img_64, text)
        real_score_2 = self.stage2_d(real_img_256, text)
        
        # Compute loss with real images.
        real_src_loss = torch.sum((real_score_1 + real_score_2 - b) ** 2)
        
        # Compute loss with fake images.
        p = random.uniform(0, 1)
        if 1 - self.pseudo_aug < p:
            fake_src_loss = torch.sum((fake_score_1 + fake_score_2 - b) ** 2) # Pseudo: fake is real.
        else:
            fake_src_loss = torch.sum((fake_score_1 + fake_score_2 - a) ** 2)
        
        vc_loss = (vc_loss_1 + vc_loss_2) * 1e-5
        
        # Update Probability Augmentation.
        lz = (torch.sign(torch.logit(real_score_1 + real_score_2)).mean()
              - torch.sign(torch.logit(fake_score_1 + fake_score_2)).mean()) / 2
        if lz > self.args.aug_threshold:
            self.pseudo_aug += 0.01
        else:
            self.pseudo_aug -= 0.01
        self.pseudo_aug = min(1, max(0, self.pseudo_aug))
        
        # Backward and optimize.
        d_loss = 0.5 * (real_src_loss + fake_src_loss) / self.args.batch_size + vc_loss
        self.optimizer_D.zero_grad()
        d_loss.backward()
        self.optimizer_D.step()
        
        # Logging.
        loss['D/loss'] = d_loss.item()
        loss['D/vc_loss'] = vc_loss.item()
        loss['D/pseudo_aug'] = self.pseudo_aug
        
        # ================================================================================ #
        #                               Train the generator                                #
        # ================================================================================ #
        
        text = self.CLIP.text_encoder(texts)
        text = text.detach()
        
        fake_img_1, vc_loss_1, noise = self.stage1_g(text)
        fake_score_1 = self.stage1_d(fake_img_1, text)
        
        fake_img_2, vc_loss_2 = self.stage2_g(fake_img_1, text)
        fake_score_2 = self.stage2_d(fake_img_2, text)
        
        # Compute loss with fake images.
        fake_src_loss = torch.sum((fake_score_1 + fake_score_2 - c) ** 2)
        
        # Mode Seeking Loss
        lz = torch.mean(torch.abs(fake_img_2 - _fake_img_2)) / torch.mean(torch.abs(noise - _noise))
        eps = 1 * 1e-5
        ms_loss = 1 / (lz + eps)
        
        # Backward and optimize.
        g_loss = 0.5 * fake_src_loss / self.args.batch_size + self.args.lambda_ms * ms_loss
        self.optimizer_G.zero_grad()
        g_loss.backward()
        self.optimizer_G.step()

        # Logging.
        loss['G/loss'] = g_loss.item()
        loss['G/ms_loss'] = ms_loss.item()
        
        # Save
        if iters == max_iters:
            self.save_state('last')
            img_name = str(epoch) + '_' + str(iters) + '_1.png'
            img_path = os.path.join(self.args.result_dir, img_name)
            save_image(fake_img_1, img_path)
            img_name = str(epoch) + '_' + str(iters) + '_2.png'
            img_path = os.path.join(self.args.result_dir, img_name)
            save_image(fake_img_2, img_path)
        
        return loss
    
    def train(self):
        print(f'Use Device: {self.device}')
        torch.backends.cudnn.benchmark = True
        
        self.CLIP.eval()
        self.stage1_g.train()
        self.stage1_d.train()
        self.stage2_g.train()
        self.stage2_d.train()
        
        hyper_params = {}
        hyper_params['Text Dir'] = self.args.text_dir
        hyper_params['Image Dir'] = self.args.image_dir
        hyper_params['Result Dir'] = self.args.result_dir
        hyper_params['Weight Dir'] = self.args.weight_dir
        hyper_params['CLIP_max_patches'] = self.args.CLIP_max_patches
        hyper_params['CLIP_patch_size'] = self.args.CLIP_patch_size
        hyper_params['CLIP_sentence_size'] = self.args.CLIP_sentence_size
        hyper_params['Prob-Aug-Threshold'] = self.args.aug_threshold
        hyper_params['Learning Rate'] = self.args.lr
        hyper_params["Mul Discriminator's LR"] = self.args.mul_lr_dis
        hyper_params['Batch Size'] = self.args.batch_size
        hyper_params['Num Train'] = self.args.num_train
        hyper_params['Lambda Mode-Seeking'] = self.args.lambda_ms

        for key in hyper_params.keys():
            print(f'{key}: {hyper_params[key]}')
        #experiment.log_parameters(hyper_params)
        
        while self.args.num_train > self.epoch:
            self.epoch += 1
            epoch_loss_CLIP = 0.0
            epoch_loss_G = 0.0
            epoch_loss_D = 0.0
            
            for iters, (images, texts) in enumerate(tqdm(self.dataloader)):
                iters += 1
                
                images = images.to(self.device, non_blocking=True)
                texts = texts.to(self.device, non_blocking=True)
                
                loss = self.trainGAN(self.epoch, iters, self.max_iters, images, texts)
                
                epoch_loss_CLIP += loss['CLIP/loss']
                epoch_loss_G += loss['G/loss']
                epoch_loss_D += loss['D/loss']
                #experiment.log_metrics(loss)
            
            epoch_loss = epoch_loss_CLIP + epoch_loss_G + epoch_loss_D
            
            print(f'Epoch[{self.epoch}] CLIP({epoch_loss_CLIP}) + G({epoch_loss_G}) + D({epoch_loss_D}) = {epoch_loss}')
                    
            if not self.args.noresume:
                self.save_resume()
    
    def generate(self, text):
        self.CLIP.eval()
        self.stage1_g.eval()
        self.stage2_g.eval()
        
        text = self.dataset.to_tokens(TextData.tokenizer(text))
        text.extend([self.dataset.word2id['<pad>'] for _ in range(self.dataset.sentence_size - len(text))])
        text = torch.LongTensor(text).unsqueeze(0).to(self.device)
        
        print(self.dataset.to_string(text[0]))
        
        text = self.CLIP.text_encoder(text)
        fake_img_1, _, _ = self.stage1_g(text)
        fake_img_2, _ = self.stage2_g(fake_img_1, text)

        save_image(fake_img_1[0], os.path.join(self.args.result_dir, f'generated_1_{time.time()}.png'))
        save_image(fake_img_2[0], os.path.join(self.args.result_dir, f'generated_2_{time.time()}.png'))
        print('New picture was generated.')


# In[ ]:


def main(args):
    solver = Solver.load(args, resume=not args.noresume)
    solver.load_state()
    
    if args.generate != '':
        solver.generate(args.generate)
        return
    
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
    parser.add_argument('--aug_threshold', type=float, default=0.6)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--mul_lr_dis', type=float, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_train', type=int, default=100)
    parser.add_argument('--lambda_ms', type=float, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--noresume', action='store_true')
    parser.add_argument('--generate', type=str, default='')
    
    args, unknown = parser.parse_known_args()
    
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    if not os.path.exists(args.weight_dir):
        os.mkdir(args.weight_dir)
    
    main(args)

