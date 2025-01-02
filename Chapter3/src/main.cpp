#include "utils.hpp"
#include <format>
#include <iostream>
#include "Tensor.hpp"


int main(int argc, char const *argv[]){
    std::cout<<"hello sycl/dpc++ !!!"<<std::endl;

    auto t1 = Tensor<float>::Fill({4,3},3.14f);
    auto t2 = Tensor<float>::Fill({3,3},2.178f);
    // auto t3 = Tensor<float>::Zeros({3,3});
    // auto t4 = Tensor<float>::Ones({3,3});

    // 张量运算
    // auto t3 = t1+t2;
    // auto t3 = t1-t2;
    // auto t3 = t1*t2;
    // auto t3 = t1/t2;

    // 标量运算
    //t1 += 1;
    // t1 -= 2;
    //t1 *= 3;
    //  t1 /= 2;
    t1.print();
    t1.t();
    t1.print();
    return 0;
}
