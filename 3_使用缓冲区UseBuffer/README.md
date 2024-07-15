# 使用缓冲区 UseBuffer

## 缓冲区和访问器
在上一节中，我们使用 sycl::malloc_shared,sycl::malloc_device,sycl::malloc_host 去管理数据，它是基于指针的操作。

当然在dpc++中还有另外一种管理数据的方法：缓冲区和访问器。

缓冲区是比USM更高级别的抽象，而USM还保留了指针。缓冲区的抽象级别允许在应用程序的任何设备上可用,包含在运行时数据管理。
利用缓冲区存储的数据不能像传统C++那样使用data[index] 指针的方式访问数据，需要一个帮手：访问器。

```c++
std::vector<int> vec(100,200);

//USM
int* data = sycl::malloc_shared<int>(vec.size(),queue);
queue.memcpy(data, vec.data(), vec.size() * sizeof(int)).wait();

//Buffer
sycl::buffer buf(vec);
//sycl::buffer buf{vec.begin(), vec.end()};
//sycl::buffer buf(vec, sycl::range{vec.size()});

```
## 访问器 accessor 

在这之前还有一个概念没有解释，我们在上一节中已经使用过：queue.submit();
queue.submit()函数用于提交一个或多个 DPC++ 任务到队列中。这个函数通常用于在设备上执行并行计算。它接受一个lambda表达式作为参数，这个lambda表达式定义了并行计算的任务，也被称为命令组。

```c++
queue.submit([&](sycl::handler &h) {
    /*
    定义并行计算的任务
    */ 
    h.parallel_for(sycl::range<1>(vec.size()), [=](sycl::id<1> index) {
        /*
        编写设备(内核)代码
        */ 
    });
});
queue.wait(); // 等待设备代码运算完成。
```
由于缓冲区表示的数据不能直接访问,必须创建访问器对象进行访问。访问器告知运行时希望在
何处以及如何访问数据,从而允许运行时确保正确的数据在正确的时间出现在正确的位置。

```c++

queue.submit([&](sycl::handler &h){
    sycl::accessor acc{buf,h,sycl::read_write};
    h.parallel_for(sycl::range<1>(SIZE), [=](sycl::id<1> index) {
        acc[index] += 100;
    });
}).wait();

```
> sycl::accessor acc{buf,h,sycl::read_write};
是在命令组中定义的访问器
sycl::read_write 被称为访问模式，类似的还有：sycl::read_only,sycl::write_only

+ sycl::read_write(默认)
+ sycl::read_only(访问者只能读取缓冲区中的元素，不能写入) 同时告诉运行时，buf只要从host读取device,而不用写回。
+ sycl::write_only(访问者只能写入缓冲区中的元素，不能读取)


## sycl::host_accessor vs sycl::accessor

sycl::host_accessor是sycl::read_only的，只允许在主机上读取缓冲区中的数据。
sycl::accessor可以是三者。

sycl::host_accessor 通常用于在主机上读取设备缓冲区中的数据，或者在主机上初始化设备缓冲区中的数据。
sycl::accessor 通常用于在设备上执行计算时访问设备缓冲区中的数据