from model import Net
import dataset
from skimage.io import imread

# Pytorch Libraries and modules
import torch
from torch.optim import RMSprop
from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d, Dropout, CrossEntropyLoss