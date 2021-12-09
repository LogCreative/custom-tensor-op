import math
import torch
from torch.autograd.function import InplaceFunction
import torch.nn as nn
import torch.nn.functional as F
import custom_conv2d_cpu

# TODO: replace to F.conv2d
def conv2dbasis(input, kernal, padding=0):
    h,w = list(input.size())
    kh,kw = list(kernal.size())
    oh,ow = h-kh+2*padding+1,w-kw+2*padding+1
    output = torch.Tensor(oh,ow)
    input_ = F.pad(input, (padding,padding,padding,padding), "constant", 0)
    for i in range(oh):
        for j in range(ow):
            output[i,j] = input_[i:i+kh,j:j+kw].mul(kernal).sum()
    return output # imm

# TODO: to make it cuda available
device = "cpu"
# API Reference:
# https://pytorch.org/docs/master/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
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

device = "cpu"
# learning reference:
# https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html#:~:text=%EE%80%80PyTorch%EE%80%81%3A%20Defining%20New%20autograd%20%EE%80%80Functions%EE%80%81.%20A%20fully-connected%20ReLU,Variables%2C%20and%20uses%20%EE%80%80PyTorch%EE%80%81%20autograd%20to%20compute%20gradients.
class myConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        # forward
        global device
        ctx.save_for_backward(input, weight, bias)
        batch_size, in_channels, in_height, in_width = list(input.size())
        out_channels, in_channels, kernel_height, kernel_width = list(weight.size())
        return F.conv2d(input,weight,bias)
        
    @staticmethod
    def backward(ctx, grad_output):
        # backward
        input, weight, bias = ctx.saved_tensors
        batch_size, in_channels, in_height, in_width = list(input.size())
        out_channels, in_channels, kernel_height, kernel_width = list(weight.size())
        grad_input = F.conv2d(grad_output.repeat(1,in_channels,1,1), torch.Tensor.rot90(weight,2,[2,3]).repeat(in_channels,1,1,1), padding=(kernel_width-1,kernel_height-1))
        grad_weight = F.conv2d(input, grad_output.repeat(1,in_channels,1,1)).sum([0]).unsqueeze(0)
        grad_bias = grad_output.sum([0,2,3])
        return grad_input, grad_weight, grad_bias

if __name__=="__main__":
    testInput = (torch.randn(3,3,5,5,dtype=torch.double,requires_grad=True),torch.randn(1,3,3,3,dtype=torch.double,requires_grad=True),torch.randn(1,dtype=torch.double,requires_grad=True))
    print(torch.autograd.gradcheck(myConv2dFunction.apply, testInput, eps=1e-3, rtol=1e-4))