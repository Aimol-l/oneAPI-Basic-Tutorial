#include <iostream>
#include<vector>
#include "utils.h"
static constexpr size_t SIZE = 50;  

template<typename T>
void implicit_usm(std::vector<T>& vec1,std::vector<T>& vec2){
    if(vec1.size() != vec2.size() || vec1.data() == vec2.data()) return;

    auto queue = init_queue();
    T* usm_data1 = sycl::malloc_shared<T>(vec1.size(),queue);
    T* usm_data2 = sycl::malloc_shared<T>(vec2.size(),queue);
    
    std::copy(vec1.begin(), vec1.end(), usm_data1);
    std::copy(vec2.begin(), vec2.end(), usm_data2);

    // vector add
    queue.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(SIZE), [=](sycl::id<1> index) {
            usm_data1[index]  = usm_data1[index] + usm_data2[index];
        });
    }).wait();

    for(size_t i =0;i<SIZE;++i) std::cout<<usm_data1[i]<<",";

    // free shared memory
    sycl::free(usm_data1,queue);
    sycl::free(usm_data2,queue);
}

template<typename T>
void explicit_usm(std::vector<T>& vec1,std::vector<T>& vec2){
    if(vec1.size() != vec2.size() || vec1.data() == vec2.data()) return;
    auto queue = init_queue();

    T* usm_data1 = sycl::malloc_device<T>(vec1.size(), queue); //设备device上的内存，不能在host使用。如访问usm_data1[0]
    T* usm_data2 = sycl::malloc_device<T>(vec2.size(), queue); //设备device上的内存，不能在host使用。如访问usm_data1[0]

    queue.memcpy(usm_data1, vec1.data(), vec1.size() * sizeof(T)).wait();// 将数据从host复制到device
    queue.memcpy(usm_data2, vec2.data(), vec2.size() * sizeof(T)).wait();

    // vector add
    queue.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(vec1.size()), [=](sycl::id<1> index) {
            usm_data1[index] += usm_data2[index];
        });
    }).wait();

    std::vector<T> sum_(SIZE);
    queue.memcpy(sum_.data(), usm_data1, sum_.size() * sizeof(T)).wait();// 将结果从device复制到host

    for(size_t i =0;i<SIZE;++i) std::cout<<sum_[i]<<",";
    
    // ffree device memory
    sycl::free(usm_data1, queue);
    sycl::free(usm_data2, queue);
}
int main() {
    std::vector vec1(SIZE,100);
    std::vector vec2(SIZE,200);

    implicit_usm<int>(vec1,vec2);
    explicit_usm<int>(vec1,vec2);

    return 0;
}
