# custom-tensor


## 运行情况

Linear 层已经被实现，所以在此仅给出运行情况：

||||
|--------|--------------|--------------------------|
|硬件环境|CPU（vCPU数目）|4(8)|
||GPU(型号，数目)| GTX 1050 Ti x1|
|软件环境|OS版本|Windows 11|
||深度学习框架<br>python包名称及版本|torch==1.10.0|
||CUDA版本|11.5|
||||

|||
|---------------|---------------------------|
| 实现方式（Linear层）| &nbsp; &nbsp; &nbsp; &nbsp; 性能评测 |
|<br/> <br/>PyTorch原有张量运算<br/> <br/>&nbsp;| 9905/10000 221s |
|<br/> <br/>基于Python API的定制化张量运算<br/> <br/>&nbsp;|9914/10000  218s|
|<br/> <br/>基于C++的定制化张量运算<br/> <br/>&nbsp;|9909/10000 221s|
||||

线性层的自定义不会显著影响性能。

Conv2d 的运行情况

|||
|---------------|---------------------------|
| 实现方式（Conv2d层）| &nbsp; &nbsp; &nbsp; &nbsp; 性能评测 |
|<br/> <br/>PyTorch原有张量运算<br/> <br/>&nbsp;|9905/10000 221s|
|<br/> <br/>基于Python API的定制化张量运算<br/> <br/>&nbsp;|9907/10000  235s|
|<br/> <br/>基于C++的定制化张量运算<br/> <br/>&nbsp;|9907/10000 234s|
||||

卷积层的自定义会稍微增加一些时间，但是主要应该认为是大量矩阵在 CPU 和 GPU 之间传输所致。

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

见 [卷积推导](https://logcreative.github.io/custom-tensor/img/conv.pdf)。对应的细节的下标实现见 [CPU 版本](task/custom_conv2d_cpu.py)，但请注意该模块不能在 GPU 版本中运行成功，有数据传输问题，为避免此将直接通过下一节的 Pytorch API 实现。该代码仅供展示原理，并且在 Python 层直接运算矩阵会有精度问题。代码一定程度参考[配套代码](https://github.com/microsoft/ai-edu/blob/master/%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/%E7%AC%AC8%E6%AD%A5%20-%20%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/src/ch17-CNNBasic/MiniFramework/ConvLayer.py)。

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