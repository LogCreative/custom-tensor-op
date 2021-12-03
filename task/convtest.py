import unittest
from torch import Tensor
from mnist_custom_conv2d import conv2dbasis

class ConvTest(unittest.TestCase):
    def testBasicConv(self):
        res = conv2dbasis(
            Tensor([[2,5,0,1],[6,7,1,2],[3,0,4,0],[3,9,7,4]]),
            Tensor([[1,0,1],[-1,1,0],[0,1,-1]]),
            Tensor(2,2))
        self.assertEqual(Tensor.equal(res, Tensor([[-1,4],[6,16]])), True)

if __name__=="__main__":
    unittest.main()