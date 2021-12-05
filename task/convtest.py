import unittest
import torch
from torch import Tensor
import torch.nn.functional as F
from custom_conv2d import conv2dbasis, myConv2dFunction

class ConvTest(unittest.TestCase):
    def testBasicConv(self):
        res = conv2dbasis(
            Tensor([[2,5,0,1],[6,7,1,2],[3,0,4,0],[3,9,7,4]]),
            Tensor([[1,0,1],[-1,1,0],[0,1,-1]]),
            0)
        self.assertEqual(Tensor.equal(res, Tensor([[-1,4],[6,16]])), True)
    
    def testPaddingConv(self):
        res = conv2dbasis(
            Tensor([[2,5,0,1],[6,7,1,2],[3,0,4,0],[3,9,7,4]]),
            Tensor([[1,0,1],[-1,1,0],[0,1,-1]]),
            1
        )
        self.assertEqual(Tensor.equal(res, Tensor([[1,9,-6,3],[14,-1,4,1],[4,6,16,1],[3,13,-2,1]])), True)

    @unittest.expectedFailure
    def testMyConv2dFunction(self):
        testInput = (torch.randn(3,3,5,5,dtype=torch.double,requires_grad=True),torch.randn(1,3,3,3,dtype=torch.double,requires_grad=True),torch.randn(1,dtype=torch.double,requires_grad=True))
        print(torch.autograd.gradcheck(myConv2dFunction.apply, testInput, eps=1e-3, rtol=1e-4,nondet_tol=1e-3))

if __name__=="__main__":
    unittest.main()