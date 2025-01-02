#include "utils.hpp"
#include "memory.hpp"
#include <format>
#include <iostream>


int main(int argc, char const *argv[]){
    std::cout<<"hello sycl/dpc++ !!!"<<std::endl;


    // 标准c/c++中的申请内存
    standard_c<float>(1000);
    standard_cpp<float>(1000);


    // 统一共享内存(USM) && sycl::buffer
    
    // dpc++ 中的申请内存
    dpcpp_host<float>(1000);
    dpcpp_device<float>(1000);
    dpcpp_shared<float>(1000);

    return 0;
}
