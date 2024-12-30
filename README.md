# oneAPI-Basic-Tutorial
 本仓库包含了一系列关于oneAPI或DP C++的教程，旨在记录自身的学习，同时也给帮助你快速使用英特尔提供的跨架构编程工具和Data Parallel C++。适合萌新，逐步掌握异构计算开发技巧。
 This repository contains a series of tutorials on oneAPI or DP C++, designed to document your own learning, but also given to help you quickly use the cross-architecture programming tools and Data Parallel C++ provided by Intel. It is suitable for budding developers to gradually master heterogeneous computing development skills.

# 参考书籍 Reference books
 [《Data Parallel C++》](https://link.springer.com/book/10.1007/978-1-4842-5574-2)
 [《Data Parallel C++ zh》](https://github.com/xiaoweiChen/Data-Paralle-Cpp/releases/tag/0.0.1)
[ 《oneapi_programming-guide》](./pdf/oneapi_programming-guide_2024.2-771723-824015.pdf)
# 安装 Install oneAPI

## 标准安装 Standard Installation 

[《Official Standard Installation Manual for Linux》](./pdf/oneapi_installation-guide-linux_2024.2-766279-824598.pdf)
[《Official Standard Installation Manual for Windows》](./pdf/oneapi_installation-guide-windows_2024.2-766284-824599.pdf)

 ## For Arch Linux user
 
 ### Install:
> yay -S intel-oneapi-dpcpp-cpp

then：
>yay -Q|grep intel-oneapi

output:
>  intel-oneapi-common
>  
>  intel-oneapi-compiler-dpcpp-cpp-runtime
>  
>  intel-oneapi-compiler-dpcpp-cpp-runtime-libs
>  
>  intel-oneapi-compiler-shared
>  
>  intel-oneapi-compiler-shared-runtime
>  
>  intel-oneapi-compiler-shared-runtime-libs
>  
>  intel-oneapi-dev-utilities
>  
>  intel-oneapi-dpcpp-cpp
>  
>  intel-oneapi-dpcpp-debugger
>  
>  intel-oneapi-mkl 
>  
>  intel-oneapi-mkl-sycl
>  
>  intel-oneapi-openmp
>  
>  intel-oneapi-tbb 
>  
>  intel-oneapi-tcm

### 配置oneAPI环境变量 Configuring oneAPI Environment Variables 

添加编译器，调试器目录到 PATH 环境变量中 ~/.bashrc 或者 .zshrc 文件。

Add compiler and debugger directories to the PATH environment variable in ~/.bashrc or .zshrc file.

> export PATH=/opt/intel/oneapi/debugger/latest/bin/:\$PATH
> export PATH=/opt/intel/oneapi/compiler/latest/bin/:\$PATH

添加动态库/静态库目录到 LD_LIBRARY_PAT环境变量中。

>export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/latest/lib:$LD_LIBRARY_PATH

加载和生效 load and active Env 

> sudo chmod +x /opt/intel/oneapi/setvars.sh
> source /opt/intel/oneapi/setvars.sh &&  source ~/.zshrc   
### 检查 Check

运行 run
>  icpx --version

获得输出 output
>  Intel(R) oneAPI DPC++/C++ Compiler 2024.1.0 (2024.1.0.20240308)
>  Target: x86_64-unknown-linux-gnu
>  Thread model: posix
>  InstalledDir: /opt/intel/oneapi/compiler/2024.1/bin/compiler
>  Configuration file: /opt/intel/oneapi/compiler/2024.1/bin/compiler/../icpx.cfg


### * Install Nvidia driver and CUDA(12.x)
对Nvida用户你需要安装最新驱动和CUDA 12.x
For Nvidia user,need instal latest driver and CUDA 12.x
> yay -S nvidia nvidia-utils opencl-nvidia cuda

配置CUDA环境变量  Configuring CUDA environment variables 

> export PATH=/opt/cuda/bin:\$PATH
> export LD_LIBRARY_PATH=/opt/cuda/lib:\$LD_LIBRARY_PATH

下载oneAPI的CUDA配置脚本 Download oneAPI config script for CUDA
 https://developer.codeplay.com/products/oneapi/nvidia/download/

+ Choose a Version ：Your Intel(R) oneAPI DPC++/C++ compiler version
+ Choose a CUDA® Version ： 12
+ Choose a Platform：Linux

```sh
chmod +x oneapi-for-nvidia-gpus-2024.1.0-cuda-12.0-linux.sh
sudo ./oneapi-for-nvidia-gpus-2024.1.0-cuda-12.0-linux.sh
```

运行 run 
> sycl-ls 

输出 output
>  \[opencl:cpu:0\] Intel(R) OpenCL, Intel(R) Core(TM) i5-14600KF OpenCL 3.0 (Build 0)  \[2024.17.3.0.08_160000\]
>  
>  \[ext_oneapi_cuda:gpu:0\] NVIDIA CUDA BACKEND, NVIDIA GeForce RTX 4070 Ti SUPER 8.9  \[CUDA 12.5\]

如果一切正常，那么安装完成。if every thing ok, done！