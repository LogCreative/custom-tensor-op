import unittest
from torch import Tensor
from mnist_custom_conv2d import conv2dbasis

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

if __name__=="__main__":
    unittest.main()