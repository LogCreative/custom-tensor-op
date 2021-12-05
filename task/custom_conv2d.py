import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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

# learning reference:
# https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html#:~:text=%EE%80%80PyTorch%EE%80%81%3A%20Defining%20New%20autograd%20%EE%80%80Functions%EE%80%81.%20A%20fully-connected%20ReLU,Variables%2C%20and%20uses%20%EE%80%80PyTorch%EE%80%81%20autograd%20to%20compute%20gradients.
class myConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        # forward
        ctx.save_for_backward(input, weight, bias)
        batch_size, in_channels, in_height, in_width = list(input.size())
        out_channels, in_channels, kernel_height, kernel_width = list(weight.size())
        output = torch.zeros(batch_size, out_channels, in_height-kernel_height+1, in_width-kernel_width+1)
        for i in range(batch_size):
            for j in range(out_channels):
                for k in range(in_channels):
                    output[i,j] += conv2dbasis(input[i,k],weight[j,k])
                output[i,j] += bias[j]
        return output
        
    @staticmethod
    def backward(ctx, grad_output):
        # backward
        input, weight, bias = ctx.saved_tensors
        batch_size, in_channels, in_height, in_width = list(input.size())
        out_channels, in_channels, kernel_height, kernel_width = list(weight.size())
        grad_input = torch.zeros(input.size())
        grad_weight = torch.zeros(weight.size())
        grad_bias = torch.zeros(bias.size())
        # Eq (17.3.10) grad_input = \sum grad_output * W^rot180
        for i in range(batch_size):
            for j in range(out_channels):
                for k in range(in_channels):
                    weight_ = torch.Tensor.rot90(weight[j,k], 2)
                    grad_input[i,k] += conv2dbasis(grad_output[i,j], weight_, padding=kernel_height-1)
        # Eq (17.3.19) grad_weight = input * grad_output
                    grad_weight[j,k] += conv2dbasis(input[i,k], grad_output[i,j])
        # Eq (17.3.21) grad_bias = grad_output
                grad_bias[j] = grad_output[i,j].sum()
        return grad_input, grad_weight, grad_bias