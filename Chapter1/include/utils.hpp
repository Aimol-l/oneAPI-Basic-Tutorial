#pragma once
#include <sycl/sycl.hpp>


// 选择合适硬件用来进行异构计算
sycl::queue InitQueue(){
    // sycl::queue queue(sycl::cpu_selector_v);
    // sycl::queue queue(sycl::gpu_selector_v);
    sycl::queue queue(sycl::default_selector_v);
    auto sycl_device = queue.get_device();
    auto sycl_context = queue.get_context();
    std::cout<<"*************************device info************************"<<std::endl;
    sycl::info::device_type info = queue.get_device().get_info<sycl::info::device::device_type>();
    if(sycl::info::device_type::cpu == info) std::cout << "设备类型: CPU" << std::endl;
    if(sycl::info::device_type::gpu == info) std::cout << "设备类型: GPU" << std::endl;
    if(sycl::info::device_type::accelerator == info) std::cout << "设备类型: ???" << std::endl;
    std::cout<<"设备名称："<< queue.get_device().get_info<sycl::info::device::name>()<<std::endl;
    std::cout<<"最大工作小组大小:"<<queue.get_device().get_info<sycl::info::device::max_work_group_size>()<<std::endl;
    std::cout<<"全局内存大小: "<< queue.get_device().get_info<sycl::info::device::global_mem_size>()/float(1073741824) << " Gbytes"<<std::endl;
    std::cout<<"最大计算单元数量: "<< queue.get_device().get_info<sycl::info::device::max_compute_units>()<<std::endl;
    std::cout<<"*************************************************************"<<std::endl;
    return queue;
}

auto queue = InitQueue();
