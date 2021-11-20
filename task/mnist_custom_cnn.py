from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

class myConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        #
    
    @staticmethod
    def backward(ctx, grad_output):
        #

class myConv2d(nn.Module):
    def __init__(self, input_features, output_features):
        super(myConv2d, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        # self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        # self.weight.data.uniform_(-0.1,0.1)
    
    def forward(self, input):
        return myConv2dFunction.apply(input, self.weight)