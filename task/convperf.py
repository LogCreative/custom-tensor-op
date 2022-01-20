import time
import torch
import torch.nn as nn
from custom_conv2d import myConv2d
from custom_conv2d_cpp import myConv2dCpp

NUMTEST = 500

# Test hyper-parameters: C_in = 1, C_out = 32, kernel = (3,3)
# batch_size = 64, width = 28, height = 28

C_IN = 1
C_OUT = 32
KERNEL = (3,3)

BATCH_SIZE = 64
HEIGHT = 28
WIDTH = 28

device = "cpu"

def run(net: nn.Module, forward_only=False):
    testInput = torch.randn(BATCH_SIZE,C_IN,HEIGHT,WIDTH).to(device)
    # Input: batch_size, C_in, height, width

    start_time = time.time()
    result = net(testInput)
    loss = torch.sum(result)
    if not forward_only:
        loss.backward()
    elapsed_time = time.time() - start_time
    
    if device == "cuda":
        torch.cuda.empty_cache()
    return elapsed_time

def perf(net, forward_only=False):
    with torch.no_grad() if forward_only else torch.enable_grad():
        time_sum = 0
        for i in range(NUMTEST*2):
            elapsed = run(net, forward_only)
            if i >= NUMTEST:
                time_sum += elapsed
        return time_sum / NUMTEST

if __name__=='__main__':
    native = nn.Conv2d(C_IN, C_OUT, KERNEL).to(device)
    pyver = myConv2d(C_IN, C_OUT, KERNEL).to(device)
    cppver = myConv2dCpp(C_IN, C_OUT, KERNEL).to(device)
    print(device, "test, Average with", NUMTEST, "times")
    print("\t\tForward\t\tForward+Backward")
    print("cppver\t",perf(cppver, True),"\t",perf(cppver))
    print("pyver\t",perf(pyver, True),"\t",perf(pyver))
    print("native\t",perf(native, True),"\t",perf(native))