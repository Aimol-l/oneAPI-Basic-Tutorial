# 创建和访问数据CreatAndAccessDate

## 没有数据，计算就没有意义。

在上一及节中，实现了能够在不同设备上使用sycl::queue,并且能获取对应设备的基本信息。为了使用这些设备，首先就需要你有数据给这些设备计算，比如：向量数据，矩阵/图像数据，张量数据等等....
无论如何，第一步都是在host上先获取数据，然后再传输到device中。比如说读取硬盘上的图片，表格数据或是临时创建的数据。为了方便，我们使用临时创建的std::vector 数据。

```c++
#include<vector>
static constexpr size_t SIZE = 50; 
std::vector<int> data(SIZE,100);
```

如何管理这些数据是DPC++的核心模型，它提供了数据管理的方式3种方式：
+ 统一共享内存 (USM)
+ 缓冲区
+ 图像(一种特殊的缓冲区,，不做讨论)
##  统一共享内存 (USM)
### USM的隐式数据传输
隐式数据传输意味着，我们编写代码的时候不需要关心data是如何从host端传输到device端或是如何回传的，只需要在需要的地方直接使用，这背后的工作的是dpcpp-cpp-runtime帮我们做的。支持USM 的设备支持统一虚拟地址空间,拥有统一虚拟地址空间意味着主机上的 USM 返回的指针都是设备上的有效指针。

```c++
int* usm_data = sycl::malloc_shared<int>(data.size(),queue);
std::copy(data.begin(), data.end(), usm_data);
```

sycl::malloc_shared函数分配的内存是在 host 和 device中共享的，可直接在host或device中直接访问，这与sycl::malloc_device和sycl::malloc_host不同。


| malloc_shared    |  malloc_device   |  malloc_host |
| --- | --- | --|
| host,device共享|  device独享   |  host独享 |

### USM的显式数据传输
显式数据传输意味着，完全控制数据移动。某些应用程序中,控制复制数据的数量和
复制数据的时间,对于获得最佳性能非常重要。理想情况下,可以将计算与数据移动重叠,确保硬件高效率运行。但是代价就是可能会造成错误: 可能会忽略复制,不正确的数据量可能被复制,或者源或目标指针可能不正确(你必须知道你在做什么)。

比如说你在host上想要读取device上的内存，那么就必须进行显式的内存复制，将device内存数据复制到host内存上。

```c++
int* device_data = sycl::malloc_device<int>(SIZE,queue);
queue.memset(device_data, 100, SIZE * sizeof(int));

int* host_data = sycl::malloc_host<int>(SIZE,queue);
queue.memcpy(host_data, device_data, SIZE * sizeof(int));
```

## 编译demo代码 Compile demo code

demo代码实现了向量的加法运算。试着比较explicit_usm函数和implicit_usm函数的区别。

```sh
cmake -B build 
cd build
cmake .. -DTARGET_DEVICE=GPU && make && ../bin/./main
```