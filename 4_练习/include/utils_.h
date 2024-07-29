#pragma once 
  
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <CL/sycl.hpp>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <array>

template<typename ...U>
void println(U... u){
    int i =0;
    auto printer = [&i]<typename Arg>(Arg arg){
        if(sizeof...(U) == ++i) std::cout<<arg<<std::endl;
        else std::cout<<arg<<" ";
    };
    (printer(u),...);
}

auto init_queue(){

#ifdef ACC_DEVIDE_GPU
    sycl::queue queue(sycl::gpu_selector_v); //手动指定GPU设备
#endif
#ifdef ACC_DEVIDE_CPU
    sycl::queue queue(sycl::cpu_selector_v); //手动指定CPU设备
#endif
    //sycl::queue queue(sycl::default_selector_v); //selected by dpc++ runtime
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

static constexpr size_t M = 50;
static constexpr size_t N = 60;
static constexpr size_t P = 70;

void rgbrbg_to_rrggbb(unsigned char *image,int width, int height,int channels){
    size_t length = width*height*channels;
    std::vector<unsigned char> new_image(length);
    for(int h = 0;h<height;h++){
        for(int w = 0;w<width;w++){
            auto r = image[h*width*channels+w*channels +0];
            auto g = image[h*width*channels+w*channels +1];
            auto b = image[h*width*channels+w*channels +2];
            new_image[h*width + w] = r;
            new_image[width*height + h*width + w] = g;
            new_image[2*width*height + h*width + w] = b;
        }
    }
    for(size_t i =0;i<width*height*channels;i++) image[i] = new_image[i];
}
void rrggbb_to_rgbrgb(unsigned char *image,int width, int height,int channels){
    size_t length = width*height*channels;
    std::vector<unsigned char> new_image(length);
    for(int c =0;c<channels;c++){
        for(int h=0;h<height;h++){
            for(int w=0;w<width;w++){
                auto color = image[c*width*height+h*width+w];
                new_image[h*width*channels + w*channels +c] = color;
            }
        }
    }
    for(size_t i =0;i<width*height*channels;i++) image[i] = new_image[i];
}
void image_conv(std::string &filePath){
    auto queue = init_queue();
    int Laplacian[9] = {0,1,0,1,-4,1,0,1,0};
    int Sobel_dx[9] = {-1,0,1,-2,0,2,-1,0,1};
    int Sobel_dy[9] = {1,2,1,0,0,0,-1,-2,-1};
    int LoG[9] = {1,1,1,1,-8,1,1,1,1};
    int Hessian[9] = {1,0,0,0,1,0,0,0,1};
    sycl::buffer<int,2> kernel(Hessian,sycl::range(3, 3)); // 模板参数是 数据类型和维度
    //加载图图片
    int width, height, channels;
    unsigned char *input_image = stbi_load(filePath.c_str(), &width, &height, &channels, 0);
    println("channels=",channels,"width=",width,"height=",height);
    if(input_image == nullptr) return;
    rgbrbg_to_rrggbb(input_image,width,height,channels);

    std::vector<unsigned char> output_image(channels*height*width);
    sycl::buffer<unsigned char, 3> input_buffer(input_image, sycl::range<3>(channels, height, width));
    sycl::buffer<unsigned char, 3> output_buffer(output_image.data(),sycl::range<3>(channels, height, width));
    //提交卷积
    queue.submit([&](auto &h) {
        sycl::accessor input_img(input_buffer, h, sycl::read_only);
        sycl::accessor out_img(output_buffer, h, sycl::write_only);
        sycl::accessor kern(kernel, h, sycl::read_only);
        h.parallel_for(sycl::range<3>(channels, height, width), [=](auto index) {
            int channel = index[0];
            int i = index[1];
            int j = index[2]; // i,j是当前像素的位置
            int sum = 0;
            for (int m = 0; m < 3; ++m) {
                for (int n = 0; n < 3; ++n) {
                    int ii = i + m - 1; // 计算卷积核中心点的位置
                    int jj = j + n - 1;
                    // 边界处理
                    if (ii >= 0 && ii < height && jj >= 0 && jj < width) {
                        sum += input_img[channel][ii][jj] * kern[m][n];
                    }
                }
            }
            out_img[index] = std::min(std::max(sum, 0), 255);
        });
    }).wait();
    // 保存图片
    sycl::host_accessor result{output_buffer};
    rrggbb_to_rgbrgb(output_image.data(),width,height,channels);
    std::string output_file_path = "../assets/out.jpg"; 
    stbi_write_jpg(output_file_path.c_str(), width, height, channels, output_image.data(), 100);
    stbi_image_free(input_image);
}
void matrix_mul(){
    auto queue = init_queue();
    sycl::buffer<float, 2> A(sycl::range(M, N));
    sycl::buffer<float, 2> B(sycl::range(N, P));

    queue.submit([&](auto &h) {
        sycl::accessor a(A, h, sycl::write_only);
        h.parallel_for(sycl::range(M, N), [=](auto index) {
            a[index] = 3.0f;
        });
    });
    queue.submit([&](auto &h) {
        sycl::accessor b(B, h, sycl::write_only);
        h.parallel_for(sycl::range(N, P), [=](auto index) {
            b[index] = 6.0f;
        });
    });
    queue.wait();

    sycl::buffer<float, 2> C(sycl::range(M, P));
    queue.submit([&](auto &h) {
        sycl::accessor a(A, h, sycl::read_only);
        sycl::accessor b(B, h, sycl::read_only);
        sycl::accessor c(C, h, sycl::write_only);

        int width_a = a.get_range()[1];

        h.parallel_for(sycl::range(M, P), [=](auto index) {
            int row = index[0];
            int col = index[1];
            float sum = 0.0f;
            for (int i = 0; i < width_a; i++) 
                sum += a[row][i] * b[i][col];
            c[index] = sum;
        });
    }).wait();

    // 输出结果
    sycl::host_accessor result{C};
    for(size_t i =0;i<M;++i){
        for(size_t j=0;j<P;j++)
            std::cout<<result[i][j]<<" ";
        std::cout<<"\n";
    }
}
void matrix_add(){
    auto queue = init_queue();
    sycl::buffer<float, 2> A(sycl::range(M, M));
    sycl::buffer<float, 2> B(sycl::range(M, M));

    queue.submit([&](auto &h) {
        sycl::accessor a(A, h, sycl::write_only);
        sycl::accessor b(B, h, sycl::write_only);
        h.parallel_for(sycl::range(M, M), [=](auto index) {
            a[index] = 3.14f;
            b[index] = 6.18f;
        });

    }).wait();

    sycl::buffer<float, 2> C(sycl::range(M, M));
    queue.submit([&](auto &h) {
        sycl::accessor a(A, h, sycl::read_only);
        sycl::accessor b(B, h, sycl::read_only);
        sycl::accessor c(C, h, sycl::write_only);
        h.parallel_for(sycl::range(M, M), [=](auto index) {
            c[index] = a[index] + b[index];
        });
    }).wait();

    // 输出结果
    sycl::host_accessor result{C};
    for(size_t i =0;i<M;++i){
        for(size_t j=0;j<M;j++)
            std::cout<<result[i][j]<<" ";
        std::cout<<"\n";
    }
}