import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import transforms

from model import SRresnet
from data_loader import Dataset, RandomCrop, Resize, Normalization
from utils import save, load, init_weights

parser = argparse.ArgumentParser(description='Train the CycleGAN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str, dest='mode')
parser.add_argument('--train_continue', default='off', choices=['on', 'off'], type=str, dest='train_continue')

parser.add_argument('--lr', default=1e-3, type=float, dest='lr')
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_epochs', default=200, type=int)

parser.add_argument('--data_dir', default='./datasets', type=str, dest='data_dir')
parser.add_argument('--ckpt_dir', default='./epochs', type=str, dest='ckpt_dir')
parser.add_argument('--log_dir', default='./log', type=str, dest='log_dir')
parser.add_argument('--result_dir', default='./training_results/', type=str, dest='result_dir')

parser.add_argument('--task', default='denoising', choices=['denoising', 'inpainting', 'super_resolution'], type=str, dest='task')
parser.add_argument('--opts', nargs='+', default=['random', 30.0], dest='opts')

parser.add_argument('--ny', default=320, type=int, dest='ny')
parser.add_argument('--nx', default=480, type=int, dest='nx')
parser.add_argument('--nch', default=3, type=int, dest='nch')
parser.add_argument('--nker', default=64, type=int, dest='nker')
parser.add_argument('--learning_type', default='plain', choices=['plain', 'residual'], type=str, dest='learning_type')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# training parameters
mode = args.mode
train_continue = args.train_continue

lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir

task = args.task
opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]

ny = args.ny
nx = args.nx
nch = args.nch
nker = args.nker

network = args.network
learning_type = args.learning_type

transform_train = transforms.Compose([
    Resize(shape=(286, 286, 3)),
    RandomCrop((256, 256)),
    Normalization(mean=0.5, std=0.5)
])
transform_val = transforms.Compose([
    Resize(shape=(286, 286, 3)),
    RandomCrop((256, 256)),
    Normalization(mean=0.5, std=0.5)
])

# data loading
train_set = Dataset(os.path.join(data_dir, 'train'), transform=transform_train, task=task, opts=opts)
train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True)
val_set = Dataset(os.path.join(data_dir, 'val'), transform=transform_val, task=task, opts=opts)
val_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=batch_size, shuffle=True)

num_batch_train = int((train_set.__len__() / batch_size) + ((train_set.__len__() / batch_size) != 0))
num_batch_val = int((val_set.__len__() / batch_size) + ((val_set.__len__() / batch_size) != 0))

if not os.path.exists(result_dir):
    os.makedirs(result_dir)
    os.makedirs(os.path.join(result_dir, 'train'))
    os.makedirs(os.path.join(result_dir, 'val'))

train_result_dir = os.path.join(result_dir, 'train')
val_result_dir = os.path.join(result_dir, 'val')

net = SRresnet(in_channels=)

optim = torch.optim.Adam(net.parameters(), lr=lr)

# fn_loss = nn.BCELoss().to(device) # binary cross entropy
fn_loss = nn.MSELoss().to(device)

fn_tonumpy = lambda x: x.to('cpu').datach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

if mode == 'train':
    if train_continue == 'on':
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    for epoch in range(1, num_epoch + 1):
        net.train()
        loss_mse = []
        for batch, data in enumerate(train_loader, 1):
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            optim.zero_grad()

            loss = fn_loss(output, label)
            loss.backward()

            optim.step()

            # 손실함수
            loss_mse += [loss.item()]

            print('train: epoch %04d / %04d | batch %04d / %04d | loss %.4f' %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_mse)))

            if epoch % 10 == 0:

                label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5))
                input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

                input = np.clip(input, a_min=0, a_max=1)
                output = np.clip(output, a_min=0, a_max=1)

                plt.imsave(os.path.join(train_result_dir, 'png', '%04d_label.png' % epoch), label[0].squeeze(), cmap=None)
                plt.imsave(os.path.join(train_result_dir, 'png', '%04d_input.png' % epoch), input[0].squeeze(), cmap=None)
                plt.imsave(os.path.join(train_result_dir, 'png', '%04d_output.png' % epoch), output[0].squeeze(), cmap=None)

        with torch.no_grad():
            net.eval()
            loss_mse = []

            for batch, data in enumerate(val_loader, 1):
                # forward
                label = data['label'].to(device)
                input = data['input'].to(device)

                output = net(output)

                # loss
                loss = fn_loss(output, label)

                loss_mse += [loss.item()]

                print('valid: epoch %04d / %04d | batch %04d / %04d | loss %.4f' %
                      (epoch, num_epoch, batch, num_batch_val, np.mean(loss_mse)))

                if epoch % 10 == 0:

                    label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5))
                    input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                    output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

                    input = np.clip(input, a_min=0, a_max=1)
                    output = np.clip(output, a_min=0, a_max=1)

                    plt.imsave(os.path.join(val_result_dir, 'png', '%04d_label.png' % epoch), label[0].squeeze(), cmap=None)
                    plt.imsave(os.path.join(val_result_dir, 'png', '%04d_input.png' % epoch), input[0].squeeze(), cmap=None)
                    plt.imsave(os.path.join(val_result_dir, 'png', '%04d_output.png' % epoch), output[0].squeeze(), cmap=None)

        if epoch % 10 == 0:
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)








