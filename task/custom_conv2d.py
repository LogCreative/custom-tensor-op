import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class myConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        # Since we didn't use other parameters,
        # we ignore that in the customized model.
        #
        # by default:
        # stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None
        super(myConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kenerl_size = kernel_size
        # weight and bias are sampled from (-\sqrt{k}, \sqrt{k})
        # where k = groups/(in_channels*kernel_size[0]*kernel_size[1])
        sqrtk = math.sqrt(1/(in_channels * kernel_size[0] * kernel_size[1]))
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels,kernel_size[0],kernel_size[1]))
        self.weight.data.uniform_(-sqrtk, sqrtk)
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.bias.data.uniform_(-sqrtk, sqrtk)
    
    def forward(self, input):
        return myConv2dFunction.apply(input, self.weight, self.bias)

class myConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        # forward
        ctx.save_for_backward(input, weight, bias)
        return F.conv2d(input,weight,bias)
        
    @staticmethod
    def backward(ctx, grad_output):
        # backward
        input, weight, bias = ctx.saved_tensors
        out_channels, in_channels, kernel_height, kernel_width = list(weight.size())
        grad_input = F.conv2d(grad_output, torch.Tensor.rot90(weight,2,[2,3]).transpose(0,1), padding=(kernel_width-1,kernel_height-1))
        grad_weight = F.conv2d(input.transpose(0,1), grad_output.transpose(0,1)).transpose(0,1)
        grad_bias = grad_output.sum([0,2,3])
        return grad_input, grad_weight, grad_bias