# Hello DPC++

直接使用icpx或者icx 编译你的 .cpp或.h文件是很直接的方式，就像使用gcc/g++那样。但是随着项目变得复杂，处理文件的依赖关系变得困难，因此我会使用Cmake作为项目构建工具，就像其他的C/C++开源项目那样。
Compiling your .cpp or .h files directly with icpx or icx is a straightforward way to do it, just like with gcc/g++. But as the project gets more complex, it becomes difficult to deal with file dependencies, so I would use Cmake as a project builder, just like any other C/C++ open source project.


你需要安装Cmake，在Linux上它的版本要求大于3.22.1，在Windows上它的版本要求大于3.22.3，具体的细节请查看这个文件: **/opt/intel/oneapi/compiler/latest/lib/cmake/IntelDPCPP/ReadMeDPCPP.txt**

You need to install Cmake, on Linux it requires a version greater than 3.22.1, on Windows it requires a version greater than 3.22.3, for details check out this file: .
**/opt/intel/oneapi/compiler/latest/lib/cmake/IntelDPCPP/ReadMeDPCPP.txt**


DPC++ 是一个异构编程语言，你可以决定一段代码的运行设备。这分为主机代码和内核代码(设备代码)，一直以来我们编写的都是主机代码(除非你擅长编写Shader)。
DPC++ is a heterogeneous programming language where you can determine the device a piece of code runs on. This is divided into host code and kernel code (device code), and it has always been host code that we have written (unless you are good at writing Shaders).

作为GettingStarted，我不打算涉及太多内容，能学会如何设置代码运行设备就可以了。
Being GettingStarted, I'm not going to cover too much ground, it's nice to learn how to set up the code to run the device.
## sycl::queue

队列是连接命令与设备的桥梁。在sycl::queue()的构造函数中设置内核代码的运行设备。
The queue is the bridge between commands and devices. The device on which the kernel code runs is set in the constructor of sycl::queue().

```C++
sycl::queue queue(sycl::gpu_selector_v); // for gpu

sycl::queue queue(sycl::cpu_selector_v); // for cpu

sycl::queue queue(sycl::default_selector_v); // decided by dpc++ runtime
```

你应该停止使用 "sycl::gpu_selector"，"sycl::cpu_selector"，"sycl::default_selector"
You should stop using "sycl::gpu_selector", "sycl::cpu_selector", "sycl::default_ selector"

## 编译demo代码 Compile demo code

```sh
mkdir build
cd build
cmake .. && make && ../bin/./main
```

终端输出：

```text
*************************device info************************
设备类型: GPU
设备名称：NVIDIA GeForce RTX 4070 Ti SUPER
最大工作小组大小:1024
全局内存大小: 15.5875 Gbytes
最大计算单元数量: 66
*************************************************************
```
