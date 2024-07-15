#include <iostream>
#include<vector>
#include "utils.h"
static constexpr size_t SIZE = 500;  

template<typename T>
void buffer_add(std::vector<T>& vec1,std::vector<T>& vec2){
    if(vec1.size() != vec2.size() || vec1.data() == vec2.data()) return;
    auto queue = init_queue();
    
    sycl::buffer buf_1(vec1);
    sycl::buffer buf_2(vec2);

    queue.submit([&](sycl::handler &h){
        sycl::accessor acc_1{buf_1,h,sycl::read_write};
        sycl::accessor acc_2{buf_2,h,sycl::read_only};
        h.parallel_for(sycl::range<1>(SIZE), [=](sycl::id<1> index) {
            acc_1[index]  = acc_1[index] + acc_2[index];
        });
    }).wait();

    sycl::host_accessor result{buf_1};
    for(size_t i =0;i<SIZE;++i) std::cout<<result[i]<<",";
}

int main() {
    std::vector vec1(SIZE,100);
    std::vector vec2(SIZE,200);

    buffer_add<int>(vec1,vec2);
    return 0;
}
