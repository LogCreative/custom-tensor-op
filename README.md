# custom-tensor-op


## 运行情况

### Linear 层

Linear 层已经被实现，所以在此仅给出运行情况：

||||
|--------|--------------|--------------------------|
|硬件环境|CPU（vCPU数目）|4(8)|
||GPU(型号，数目)| GTX 1050 Ti x1|
|软件环境|OS版本|Windows 11|
||深度学习框架<br>python包名称及版本|torch==1.10.0|
||CUDA版本|11.5|
||||

||||
|---------------|---------------------------|-----|
| 实现方式（Linear层）| 性能评测 | 单次时间 |
|<br/> <br/>PyTorch原有张量运算<br/> <br/>&nbsp;| 9905/10000 221s | 34.5s |
|<br/> <br/>基于Python API的定制化张量运算<br/> <br/>&nbsp;|9914/10000  218s| 33.2s |
|<br/> <br/>基于C++的定制化张量运算<br/> <br/>&nbsp;|9909/10000 221s| 32.6s |
||||

线性层的自定义可以在一定程度上加快速度。

### Conv2d 层

Conv2d 的运行情况

||||
|---------------|---------------------------|---|
| 实现方式（Conv2d层）| 性能评测 | 单次时间* |
|<br/> <br/>PyTorch原有张量运算<br/> <br/>&nbsp;|9905/10000 221s| 35.3s |
|<br/> <br/>基于Python API的定制化张量运算<br/> <br/>&nbsp;|9907/10000  235s| 39.1s |
|<br/> <br/>基于C++的定制化张量运算<br/> <br/>&nbsp;|9907/10000 234s| 39.1s |
|||||

<div style="font-size: small">*单次时间没有取平均，更加细致的时间分析将会采用下面的方法。</div>

使用 [convperf](task/convperf.py) 对卷积层的测试结果如下（5000次取平均）：

|CPU 测试结果 |              Forward     |    Forward+Backward|
|---|---|---|
|native  | 0.006395 |    0.014766 |
|pyver   | 0.006491 |    0.019503 |
|cppver  | 0.006585 |    0.019705 |


|CUDA 测试结果 |               Forward   |      Forward+Backward|
|---|---|---|
|native |  0.001082  | 0.001365 |
|pyver  |  0.001043  | 0.003303 |
|cppver |  0.001108  | 0.003151 |

不论是实际地在 MNIST 环境中训练得到的结果，还是纯净测试得到的结果，都可以看到自定义实现的 Python 版本和 C++ 版本在含反向传播上差别不大（最终结果），比原生的版本都慢一些。

[Profiler](task/mnist_conv_benchmark.py) 的分析（CUDA）显示是卷积调用次数过多导致，自定义实现的函数仍有可优化空间。在上表的结果可以发现，Forward层面基本没有区别（毕竟都只是调用 F.conv），但是 Forward+Backward 在 CPU 上慢了 33%，在 CUDA 上慢了 141%。这也就意味着自动图运算生成的反向传播算法可以减少卷积顶层调用的次数，类似于编译优化，虽然手动编写反向传播是可行的，但是对于专用运算器使用专用的方法进行优化会更好，下表 Profiler 的结果显示在 Backward 上并没有使用 Conv2d。

||# of Calls (aten::conv2d)|
|--|--|
|native|1896|
|pyver|5648|

|Native Run over CUDA| # of Calls |
|--|--|
| aten::cudnn_convolution_backward | 1876 |

而是使用了内置的 cudnn_convolution_backward 实现，调用一次即可得到结果，猜测在数据并行性上有了较好的优化，直接输入一次数据，就执行两次等价时间的卷积运算，更好地提高硬件利用率。

对于 CPU 优化而言，可以使用 TVM 对循环结构进行调整，减少数据依赖性，提高硬件利用率，提高 60% 左右的效率。细节详见 [LogCreative/CompilerPrinciple](https://github.com/LogCreative/CompilerPrinciple) 。

## 卷积层原理

[Pytorch的API](https://pytorch.org/docs/master/generated/torch.nn.Conv2d.html#torch.nn.Conv2d) 告诉我们
```python
torch.nn.Conv2d(in_channels, out_channels, kernel_size, 
stride=1, padding=0, dilation=1, groups=1, 
bias=True, padding_mode='zeros', 
device=None, dtype=None)
```
检视 MNIST 代码后发现我们可以直接使用后面所有的默认值，不需要实现全部的参数。所以本项目的函数原型为
```python
myConv2d(in_channels, out_channels, kernel_size)
```

然后仿照线性层，实现`myConv2dFunction`调用函数的`forward`和`backward`函数。

见[卷积推导](https://logcreative.github.io/custom-tensor-op/img/conv.pdf)（PDF）。对应的细节的下标实现见 [CPU 版本](task/custom_conv2d_cpu.py)，但请注意该模块不能在 GPU 版本中运行成功，有数据传输问题，为避免此将直接通过下一节的 Pytorch API 实现。该代码仅供展示原理，并且在 Python 层直接运算矩阵会有精度问题。代码一定程度参考[配套代码](https://github.com/microsoft/ai-edu/blob/master/%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/%E7%AC%AC8%E6%AD%A5%20-%20%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/src/ch17-CNNBasic/MiniFramework/ConvLayer.py)。

## Python 版本实现

运行方法

```cmd
cd task
python mnist_custom_conv2d.py
```

为了转换为 GPU 可以处理的[代码](task/custom_conv2d.py)，需要统一下标的处理方式，以直接使用 `F.conv2d` 函数，会使用 `transpose` 进行处理。

## C++ 版本实现

运行方法

```cmd
cd task/myconv2d_cpp
python setup.py install --user
cd ..
python mnist_custom_conv2d_cpp.py
```

[代码](task/myconv2d_cpp/myconv2d.cpp)结构与 Python 版本类似，但是需要注意`conv2d`在c++中的调用方式可选参数需要使用`Conv2dFuncOptions`包裹。

本项目含有[单元测试](task/convtest.py)。