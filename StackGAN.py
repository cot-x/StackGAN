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

from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding

ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[ ]:


class Bert(nn.Module):
    #model_name = 'tohoku-nlp/bert-base-japanese-v3'
    model_name = 'google-bert/bert-base-cased'
    
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(Bert.model_name)
        
    def to(self, device):
        self.model.to(device)
        return self

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()
    
    def forward(self, **x):
        x = self.model(**x)['pooler_output']
        return x


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


class STAGE1_G(nn.Module):
    def __init__(self, dim_c_code=768, dim_noise=128, dim_ideal=1024):
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
    def __init__(self, dim_c_code=768, dim_ideal=64):
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
    def __init__(self, dim_c_code=768, dim_ideal=128, n_residual=4):
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
    def __init__(self, dim_c_code=768, dim_ideal=64):
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
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp']
    
    def __init__(self, img_dir, transform=None):
        self.paths = self.get_paths(img_dir)
        self.transform = transform
    
    def get_paths(self, img_dir):
        img_dir = Path(img_dir)
        paths = [p for p in img_dir.iterdir() if p.suffix in ImageDataset.IMG_EXTENSIONS]
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


class TextData:
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path, sep='|')
        self.filenames = df.iloc[:,0].tolist()
        self.texts = [str(text) for text in df.iloc[:,2].tolist()]
    
    def __getitem__(self, index):
        return self.filenames[index], self.texts[index]
        
    def __len__(self):
        return len(self.texts)
    
    def tolist(self):
        return sum(self.texts, [])


# In[ ]:


class TextAndImageDataset(ImageDataset):
    def __init__(self, csv_path, img_dir, transform=None):
        super().__init__(img_dir, transform)
        self.text_data = TextData(csv_path)
    
    def __getitem__(self, index):
        filename, text = self.text_data[index]
        
        path = [path for path in self.paths if filename == path.name][0]
        image = Image.open(path)
        if self.transform is not None:
            image = self.transform(image)
        else:
            toTensor = transforms.ToTensor()
            image = toTensor(image)
        
        return image, text


# In[ ]:


class RandomErasing:
    def __init__(self, p=0.5, erase_low=0.02, erase_high=0.33, aspect_rl=0.3, aspect_rh=3.3):
        self.p = p
        self.erase_low = erase_low
        self.erase_high = erase_high
        self.aspect_rl = aspect_rl
        self.aspect_rh = aspect_rh

    def __call__(self, image):
        if np.random.rand() <= self.p:
            c, h, w = image.shape

            mask_area = np.random.uniform(self.erase_low, self.erase_high) * (h * w)
            mask_aspect_ratio = np.random.uniform(self.aspect_rl, self.aspect_rh)
            mask_w = int(np.sqrt(mask_area / mask_aspect_ratio))
            mask_h = int(np.sqrt(mask_area * mask_aspect_ratio))

            mask = torch.Tensor(np.random.rand(c, mask_h, mask_w) * 255)

            left = np.random.randint(0, w)
            top = np.random.randint(0, h)
            right = left + mask_w
            bottom = top + mask_h

            if right <= w and bottom <= h:
                image[:, top:bottom, left:right] = mask
        
        return image


# In[ ]:


class Util:
    @staticmethod
    def loadImages(batch_size, folder_path, size):
        imgs = ImageFolder(folder_path, transform=transforms.Compose([
            transforms.Resize(int(size)),
            transforms.RandomCrop(size),
            transforms.ToTensor()
        ]))
        return DataLoader(imgs, batch_size=batch_size, shuffle=True, drop_last=True)


# In[ ]:


class Solver:
    def __init__(self, args):
        has_cuda = torch.cuda.is_available() if not args.cpu else False
        self.device = torch.device("cuda" if has_cuda else "cpu")
        
        self.args = args
        
        self.load_dataset()
        
        self.text_encoder = Bert().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(Bert.model_name)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        self.stage1_g = STAGE1_G().to(self.device)
        self.stage1_d = STAGE1_D().to(self.device)
        self.stage2_g = STAGE2_G().to(self.device)
        self.stage2_d = STAGE2_D().to(self.device)

        self.stage1_g.apply(self.weights_init)
        self.stage1_d.apply(self.weights_init)
        self.stage2_g.apply(self.weights_init)
        self.stage2_d.apply(self.weights_init)
        
        self.optimizer_G = optim.Adam(itertools.chain(self.stage1_g.parameters(),
                                                      self.stage2_g.parameters()),
                                      lr=self.args.lr, betas=(0, 0.9))
        self.optimizer_D = optim.Adam(itertools.chain(self.stage1_d.parameters(),
                                                      self.stage2_d.parameters()),
                                      lr=self.args.lr * self.args.mul_lr_dis, betas=(0, 0.9))
        
        self.scheduler_G = CosineAnnealingLR(self.optimizer_G, T_max=4, eta_min=self.args.lr/4)
        self.scheduler_D = CosineAnnealingLR(self.optimizer_D, T_max=4, eta_min=(self.args.lr * self.args.mul_lr_dis)/4)
        
        self.pseudo_aug = 0.0
        self.epoch = 0
    
    def weights_init(self, module):
        if type(module) == nn.Linear or type(module) == nn.Conv2d or type(module) == nn.ConvTranspose2d:
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0)
            
    def load_dataset(self, img_size=256):
        self.dataset = TextAndImageDataset(self.args.csv_path, self.args.image_dir,
                                           transform=transforms.Compose([
                                               transforms.Resize(int(img_size)),
                                               transforms.RandomCrop(img_size),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor()
                                           ]))
        self.dataloader = DataLoader(self.dataset, batch_size=self.args.batch_size,
                                     shuffle=True, drop_last=True, num_workers=os.cpu_count())
        self.max_iters = len(iter(self.dataloader))

    def tokenize(self, texts):
        texts = self.tokenizer(texts)
        texts = self.data_collator(texts)
        return texts
            
    def save_state(self, epoch):
        self.stage1_g.cpu(), self.stage1_d.cpu(), self.stage2_g.cpu(), self.stage2_d.cpu()
        torch.save(self.stage1_g.state_dict(), os.path.join(self.args.weight_dir, f'weight_G1.{epoch}.pth'))
        torch.save(self.stage1_d.state_dict(), os.path.join(self.args.weight_dir, f'weight_D1.{epoch}.pth'))
        torch.save(self.stage2_g.state_dict(), os.path.join(self.args.weight_dir, f'weight_G2.{epoch}.pth'))
        torch.save(self.stage2_d.state_dict(), os.path.join(self.args.weight_dir, f'weight_D2.{epoch}.pth'))
        self.stage1_g.to(self.device), self.stage1_d.to(self.device), self.stage2_g.to(self.device), self.stage2_d.to(self.device)
        
    def load_state(self):
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
    def load(args, resume=True):
        if resume and os.path.exists('resume.pkl'):
            with open(os.path.join('.', 'resume.pkl'), 'rb') as f:
                print('Load resume.')
                solver = load(f)
                solver.args = args
                return solver
        else:
            return Solver(args)
    
    def trainGAN(self, epoch, iters, max_iters, real_img, texts, a=0, b=1, c=1):
        ### Train with LSGAN.
        ### for example, (a, b, c) = 0, 1, 1 or (a, b, c) = -1, 1, 0
        
        resize_64 = transforms.Resize(64)
        resize_256 = transforms.Resize(256)
        real_img_64 = resize_64(real_img)
        real_img_256 = resize_256(real_img)
        
        # ================================================================================ #
        #                             Train the discriminator                              #
        # ================================================================================ #
        
        text = self.text_encoder(**texts)
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
        
        # Update Pseudo Augmentation.
        lz = (torch.sign(torch.logit(real_score_1 + real_score_2)).mean()
              - torch.sign(torch.logit(fake_score_1 + fake_score_2)).mean()) / 2
        if lz > 0.6:
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
        loss = {}
        loss['D/loss'] = d_loss.item()
        loss['D/vc_loss'] = vc_loss.item()
        loss['D/pseudo_aug'] = self.pseudo_aug
        
        # ================================================================================ #
        #                               Train the generator                                #
        # ================================================================================ #
        
        text = self.text_encoder(**texts)
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
            self.save_state(epoch)
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

        self.text_encoder.eval()
        self.stage1_g.train()
        self.stage1_d.train()
        self.stage2_g.train()
        self.stage2_d.train()
        
        hyper_params = {}
        hyper_params['CSV Path'] = self.args.csv_path
        hyper_params['Image Dir'] = self.args.image_dir
        hyper_params['Result Dir'] = self.args.result_dir
        hyper_params['Weight Dir'] = self.args.weight_dir
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
            epoch_loss_G = 0.0
            epoch_loss_D = 0.0
            
            for iters, (images, texts) in enumerate(tqdm(self.dataloader)):
                iters += 1
                
                images = images.to(self.device, non_blocking=True)
                texts = self.tokenize(texts).to(self.device)
                
                loss = self.trainGAN(self.epoch, iters, self.max_iters, images, texts)
                
                epoch_loss_D += loss['D/loss']
                epoch_loss_G += loss['G/loss']
                #experiment.log_metrics(loss)
            
            self.scheduler_G.step()
            self.scheduler_D.step()
            
            epoch_loss = epoch_loss_G + epoch_loss_D
            
            print(f'Epoch[{self.epoch}]'
                  + f' LR[G({self.scheduler_G.get_last_lr()[0]:.5f}) D({self.scheduler_D.get_last_lr()[0]:.5f})]'
                  + f' G({epoch_loss_G}) + D({epoch_loss_D}) = {epoch_loss}]')
                    
            if not self.args.noresume:
                self.save_resume()
    
    def generate(self, text):
        self.text_encoder.eval()
        self.stage1_g.eval()
        self.stage2_g.eval()
        
        texts = self.tokenize([text]).to(self.device)
        print(self.tokenizer.convert_ids_to_tokens(texts['input_ids'][0].tolist()))
        
        texts = self.text_encoder(**texts)
        fake_img_1, _, _ = self.stage1_g(texts)
        fake_img_2, _ = self.stage2_g(fake_img_1, texts)
        
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
    parser.add_argument('--csv_path', type=str, default='/mnt/c/Datasets/flickr-images-dataset/results.csv')
    parser.add_argument('--image_dir', type=str, default='/mnt/c/Datasets/flickr-images-dataset/flickr30k_images/')
    parser.add_argument('--result_dir', type=str, default='results')
    parser.add_argument('--weight_dir', type=str, default='weights')
    parser.add_argument('--lr', type=float, default=0.0001)
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

