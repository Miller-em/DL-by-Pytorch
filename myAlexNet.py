import os
import numpy as np
import math
import argparse
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import cv2
from PIL import Image

# a modified model of AlexNet, include 5 convlotional layers and 3 full-connect layers
# I modified some parameters of AlexNet
# size of input images is 28 x 28, in grayscale (1 channel)
# output is a vector with a length of 10 (0-9)
class myAlexNet(nn.Module):
  def __init__(self, imgChannel):
    super(myAlexNet, self).__init__()
    # conv1
    self.conv1 = nn.Sequential(
                                nn.Conv2d(in_channels=imgChannel, out_channels=32, kernel_size=3, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2),
                                nn.LocalResponseNorm(size = 5)
                                )
    # conv2
    self.conv2 = nn.Sequential(
                                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1,padding=2),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2),
                                nn.LocalResponseNorm(size = 5)
                                )
    # conv3
    self.conv3 = nn.Sequential(
                                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride =1, padding=1),
                                nn.ReLU()
                                )
    # conv4
    self.conv4 = nn.Sequential(
                                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride =1, padding=1),
                                nn.ReLU()
                                )
    # conv5
    self.conv5 = nn.Sequential(
                                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride =1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2)
                                )
    self.fc1 = nn.Linear(256 * 4 * 4, 1024)
    self.fc2 = nn.Linear(1024, 512)
    self.fc3 = nn.Linear(512, 10)
  
  def forward(self, input):
    #print(input.size())
    out = self.conv1(input)
    #print(out.size())
    out = self.conv2(out)
    #print(out.size())
    out = self.conv3(out)
    #print(out.size())
    out = self.conv4(out)
    #print(out.size())
    out = self.conv5(out)
    #print(out.size())
    out = out.view(-1, 256 * 4 * 4)
    #print(out.size())
    out = self.fc1(out)
    #print(out.size())
    out = self.fc2(out)
    #print(out.size())
    out = self.fc3(out)
    #print(out.size())
    return out

# train function
def train(epochs, trainLoader, model, device,Lr,momen):
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=Lr, momentum=momen)
  model.to(device)
  for e in range(epochs):
    for i, (imgs, labels) in enumerate(trainLoader):
      imgs = imgs.to(device)
      labels = labels.to(device)
      out = model(imgs)
      loss = criterion(out, labels)
      optimizer.zero_grad() # if don't call zero_grad, the grad of each batch will be accumulated
      loss.backward()
      optimizer.step()
      if i%20==0:
        print('epoch: {}, batch: {}, loss: {}'.format(e + 1, i + 1, loss.data))
  torch.save(model, 'myAlexMnistDemo.pth') # save net model and parameters

# test function
def test(testLoader, model, device):
  model.to(device)
  with torch.no_grad(): # when in test stage, no grad
    correct = 0
    total = 0
    for (imgs, labels) in testLoader:
      imgs = imgs.to(device)
      labels = labels.to(device)
      out = model(imgs)
      _, pre = torch.max(out.data, 1)
      total += labels.size(0)
      correct += (pre == labels).sum().item()
    print('Accuracy: {}'.format(correct / total))

# predict function
def predict(input, model, device):
  model.to(device)
  with torch.no_grad():
    input=input.to(device)
    out = model(input)
    _, pre = torch.max(out.data, 1)
    return pre.item()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--stage", type=str, default='train', help="is train or test")
  parser.add_argument("--epochs", type=int, default=30, help="number of epochs of training")
  parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
  parser.add_argument("--lr", type=float, default=0.001, help="SGD: learning rate")
  parser.add_argument("--momentum", type=float, default=0.9, help="SGD: momentum")
  parser.add_argument("--img_size", type=tuple, default=(28,28), help="size of each image dimension")
  parser.add_argument("--channels", type=int, default=1, help="number of image channels")
  parser.add_argument("--predictImg", type=str, default='', help="image need to be predicted")
  opt = parser.parse_args()
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  if opt.stage=='train': # in train stage
    dataloader = torch.utils.data.DataLoader(
      datasets.MNIST(
        "data",
        train=True,
        download=False,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
      ),
      batch_size=opt.batch_size,
      shuffle=True,
      num_workers=8,
    )
    model = myAlexNet(opt.channels)
    train(opt.epochs, dataloader, model, device, opt.lr, opt.momentum)

  elif opt.stage == 'test':
    testLoader = dataloader = torch.utils.data.DataLoader(
      datasets.MNIST(
        "your dataset path",
        train=False,
        download=False,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
      ),
      batch_size=opt.batch_size,
      shuffle=True,
      num_workers=8,
    )
    model = torch.load('myAlexMnistDemo.pth')
    test(testLoader, model, device)
    
  elif opt.stage == 'predict':
    model = torch.load('myAlexMnistDemo.pth')
    transform=transforms.Compose(
            [transforms.Grayscale(),
            transforms.Resize(opt.img_size),
            #transforms.Normalize([0.5], [0.5]),
            transforms.ToTensor(),]
        )
    img = Image.open(opt.predictImg).convert('RGB')
    print(type(img))
    img = transform(img)
    img = img.unsqueeze(0)
    ans = predict(img, model, device)
    print('prediction of this image is a hand writing number: {}'.format(ans))



if __name__ == '__main__':
  main()


