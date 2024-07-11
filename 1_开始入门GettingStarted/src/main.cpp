#include <iostream>
#include <CL/sycl.hpp>

using std::cout;
using std::endl;
using sycl::queue;

int main() {

#ifdef ACC_DEVIDE_GPU
    sycl::queue queue(sycl::gpu_selector_v); //手动指定GPU设备
#endif
#ifdef ACC_DEVIDE_CPU
    sycl::queue queue(sycl::cpu_selector_v); //手动指定CPU设备
#endif
    // sycl::queue queue(sycl::default_selector_v); //selected by dpc++ runtime
    //****************************************************************
    auto sycl_device = queue.get_device();
    auto sycl_context = queue.get_context();
    cout<<"*************************device info************************"<<endl;
    sycl::info::device_type info = queue.get_device().get_info<sycl::info::device::device_type>();
    if(sycl::info::device_type::cpu == info) std::cout << "设备类型: CPU" << std::endl;
    if(sycl::info::device_type::gpu == info) std::cout << "设备类型: GPU" << std::endl;
    if(sycl::info::device_type::accelerator == info) std::cout << "设备类型: ???" << std::endl;
    cout<<"设备名称："<< queue.get_device().get_info<sycl::info::device::name>()<<endl;
    cout<<"最大工作小组大小:"<<queue.get_device().get_info<sycl::info::device::max_work_group_size>()<<endl;
    cout<<"全局内存大小: "<< queue.get_device().get_info<sycl::info::device::global_mem_size>()/float(1073741824) << " Gbytes"<<endl;
    cout<<"最大计算单元数量: "<< queue.get_device().get_info<sycl::info::device::max_compute_units>()<<endl;
    cout<<"*************************************************************"<<endl;
    return 0;
}
