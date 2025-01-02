#pragma once
#include <iostream>
#include "utils.hpp"

template<typename T>
bool standard_c(size_t size){
    std::cout<<"standard_c: malloc & free"<<std::endl;
    T* data = (T*)std::malloc(size * sizeof(T));
    if (data == nullptr) {
        std::cerr << "内存分配失败"<< std::endl;
        return false;
    }
    //**********************************
    for(size_t i=0;i<size;i++) data[i] = i;
    
    //**********************************
    std::free(data);
    return true;
}

template<typename T>
bool standard_cpp(size_t size){
    std::cout<<"standard_cpp: new & delete"<<std::endl;
    T* data = nullptr;
    try {
        data = new T[size];
    } catch (std::bad_alloc& e) {
        std::cerr << "内存分配失败: " << e.what() << std::endl;
        return false;
    }
    //**********************************
    for(size_t i=0;i<size;i++) data[i] = i;
    //**********************************
    delete[] data;
    return true;
}

template<typename T>
bool dpcpp_host(size_t size){
    std::cout<<"dpcpp_host: malloc_host & free"<<std::endl;
    T* data = sycl::malloc_host<T>(size,queue);
    //**********************************
    for(size_t i=0;i<size;i++) data[i] = i;
    //**********************************
    sycl::free(data,queue);
    return true;
}

template<typename T>
bool dpcpp_device(size_t size){
    std::cout<<"dpcpp_device: malloc_device & free"<<std::endl;
    T* data = sycl::malloc_device<T>(size,queue);
    //**********************************
    queue.memset(data,1,size);
    //**********************************
    sycl::free(data,queue);
    return true;
}
template<typename T>
bool dpcpp_shared(size_t size){
    std::cout<<"dpcpp_shared: malloc_shared & free"<<std::endl;
    T* data = sycl::malloc_shared<T>(size,queue);
    //**********************************
    for(size_t i=0;i<size;i++) data[i] = i;
    //**********************************
    sycl::free(data,queue);
    return true;
}