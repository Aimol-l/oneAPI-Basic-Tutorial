#pragma once
#include <vector>
#include <numeric>
#include <iostream>
#include <algorithm>
#include <concepts>
#include <cstddef> // for size_t
#include <initializer_list>
#include "utils.hpp"


template<typename T>
concept Arithmetic = std::is_arithmetic_v<T>; // c++20约束，T只能是基本数据类型

template<Arithmetic T>
struct Matrix {
    T* data;            // 指向数据的指针
    size_t* shape;      // 形状数组，例如 [2, 3, 4]
    size_t dims;        // 张量的维度数量，例如 3 表示三维张量
    Matrix() : data(nullptr), shape(nullptr), dims(0) {}
    ~Matrix() {
        delete[] data;
        delete[] shape;
    }
    // 深拷贝构造函数
    Matrix(const Matrix& other) : dims(other.dims) {
        if (other.shape) {
            shape = new size_t[dims];
            std::copy(other.shape, other.shape + dims, shape);
        } else {
            shape = nullptr;
        }
        if (other.data) {
            size_t total_size = 1;
            for (size_t i = 0; i < dims; ++i) total_size *= shape[i];
            data = new T[total_size];
            std::copy(other.data, other.data + total_size, data);
        } else {
            data = nullptr;
        }
    }
    // 深拷贝赋值运算符
    Matrix& operator=(const Matrix& other) {
        if (this == &other) return *this; // 避免自赋值
        // 释放现有资源
        delete[] data;
        delete[] shape;
        // 分配并复制数据
        dims = other.dims;
        if (other.shape) {
            shape = new size_t[dims];
            std::copy(other.shape, other.shape + dims, shape);
        } else {
            shape = nullptr;
        }
        if (other.data) {
            size_t total_size = 1;
            for (size_t i = 0; i < dims; ++i) total_size *= shape[i];
            data = new T[total_size];
            std::copy(other.data, other.data + total_size, data);
        } else {
            data = nullptr;
        }
        return *this;
    }
    // 计算总元素数量
    size_t total_size() const {
        size_t total_size = 1;
        for (size_t i = 0; i < dims; ++i) total_size *= shape[i];
        return total_size;
    }
};

template<Arithmetic T>
class Tensor{
private:
    Matrix<T> mat;
    void add(const Matrix<T>& A,T scalar){
        size_t total_size = this->mat.total_size();
        T* device_A = sycl::malloc_device<T>(total_size,queue);
        queue.memcpy(device_A,A.data,total_size*sizeof(T));
        // 计算
        queue.submit([&](sycl::handler& h){
            h.parallel_for(sycl::range(total_size), [=](auto idx) {
                device_A[idx] += scalar;
            });
        }).wait();
        queue.memcpy(A.data,device_A,total_size*sizeof(T)).wait();
        sycl::free(device_A,queue);
    }
    void sub(const Matrix<T>& A,T scalar){
        size_t total_size = this->mat.total_size();
        T* device_A = sycl::malloc_device<T>(total_size,queue);
        queue.memcpy(device_A,A.data,total_size*sizeof(T));
        // 计算
        queue.submit([&](sycl::handler& h){
            h.parallel_for(sycl::range(total_size), [=](auto idx) {
                device_A[idx] -= scalar;
            });
        }).wait();
        queue.memcpy(A.data,device_A,total_size*sizeof(T)).wait();
        sycl::free(device_A,queue);
    }
    void dot(const Matrix<T>& A,T scalar){
        size_t total_size = this->mat.total_size();
        T* device_A = sycl::malloc_device<T>(total_size,queue);
        queue.memcpy(device_A,A.data,total_size*sizeof(T));
        // 计算
        queue.submit([&](sycl::handler& h){
            h.parallel_for(sycl::range(total_size), [=](auto idx) {
                device_A[idx] *= scalar;
            });
        }).wait();
        queue.memcpy(A.data,device_A,total_size*sizeof(T)).wait();
        sycl::free(device_A,queue);
    }
    void divide(const Matrix<T>& A,T scalar){
        size_t total_size = this->mat.total_size();
        T* device_A = sycl::malloc_device<T>(total_size,queue);
        queue.memcpy(device_A,A.data,total_size*sizeof(T));
        // 计算
        queue.submit([&](sycl::handler& h){
            h.parallel_for(sycl::range(total_size), [=](auto idx) {
                device_A[idx] /= scalar;
            });
        }).wait();
        queue.memcpy(A.data,device_A,total_size*sizeof(T)).wait();
        sycl::free(device_A,queue);
    }
    void add(const Matrix<T>& A,const Matrix<T>& B,const Matrix<T>& dst)const{
        size_t total_size = this->mat.total_size(); //9
        // 在设备上分配内存
        T* device_A = sycl::malloc_device<T>(total_size,queue);
        T* device_B = sycl::malloc_device<T>(total_size,queue);
        T* device_dst = sycl::malloc_device<T>(total_size,queue);
        // 将A.data 和 B.data 复制到设备内存中
        queue.memcpy(device_A,A.data,total_size*sizeof(T));
        queue.memcpy(device_B,B.data,total_size*sizeof(T));
        // 计算
        queue.submit([&](sycl::handler& h){
            h.parallel_for(sycl::range<1>(total_size), [=](auto idx) {
                device_dst[idx] = device_A[idx] + device_B[idx];
            });
        }).wait();
        // 将device_dst复制回host
        queue.memcpy(dst.data,device_dst,total_size*sizeof(T)).wait();
        // 回收设备内存
        sycl::free(device_A,queue);
        sycl::free(device_B,queue);
        sycl::free(device_dst,queue);
    }
    void sub(const Matrix<T>& A,const Matrix<T>& B,const Matrix<T>& dst)const{
        size_t total_size = this->mat.total_size();
        // 在设备上分配内存
        T* device_A = sycl::malloc_device<T>(total_size,queue);
        T* device_B = sycl::malloc_device<T>(total_size,queue);
        T* device_dst = sycl::malloc_device<T>(total_size,queue);
        // 将A.data 和 B.data 复制到设备内存中
        queue.memcpy(device_A,A.data,total_size*sizeof(T));
        queue.memcpy(device_B,B.data,total_size*sizeof(T));
        // 计算
        queue.submit([&](sycl::handler& h){
            h.parallel_for(sycl::range(total_size), [=](auto idx) {
                device_dst[idx] = device_A[idx] - device_B[idx];
            });
        }).wait();
        // 将device_dst复制回host
        queue.memcpy(dst.data,device_dst,total_size*sizeof(T)).wait();
        // 回收设备内存
        sycl::free(device_A,queue);
        sycl::free(device_B,queue);
        sycl::free(device_dst,queue);
    }
    void dot(const Matrix<T>& A,const Matrix<T>& B,const Matrix<T>& dst)const{
        size_t total_size = this->mat.total_size();
        // 在设备上分配内存
        T* device_A = sycl::malloc_device<T>(total_size,queue);
        T* device_B = sycl::malloc_device<T>(total_size,queue);
        T* device_dst = sycl::malloc_device<T>(total_size,queue);
        // 将A.data 和 B.data 复制到设备内存中
        queue.memcpy(device_A,A.data,total_size*sizeof(T));
        queue.memcpy(device_B,B.data,total_size*sizeof(T));
        // 计算
        queue.submit([&](sycl::handler& h){
            h.parallel_for(sycl::range(total_size), [=](auto idx) {
                device_dst[idx] = device_A[idx] * device_B[idx];
            });
        }).wait();
        // 将device_dst复制回host
        queue.memcpy(dst.data,device_dst,total_size*sizeof(T)).wait();
        // 回收设备内存
        sycl::free(device_A,queue);
        sycl::free(device_B,queue);
        sycl::free(device_dst,queue);
    }
    void mul(const Matrix<T>& A,const Matrix<T>& B,const Matrix<T>& dst)const{
        size_t rows = A.shape[0];
        size_t common_dim = A.shape[1];
        size_t cols = B.shape[1];
        T* device_A = sycl::malloc_device<T>(rows*common_dim,queue);
        T* device_B = sycl::malloc_device<T>(common_dim*cols,queue);
        T* device_dst = sycl::malloc_device<T>(rows*cols,queue);
        // 从host端复制到device端
        queue.memcpy(device_A,A.data,rows*common_dim*sizeof(T));
        queue.memcpy(device_B,B.data,common_dim*cols*sizeof(T));
        // 计算
        queue.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<2>(rows, cols), [=](sycl::id<2> idx){
                size_t row = idx[0];
                size_t col = idx[1];
                T sum = 0;
                for (size_t k = 0; k < common_dim; ++k)
                    sum += device_A[row * common_dim + k] * device_B[k * cols + col];
                device_dst[row * cols + col] = sum;
            });
        }).wait();
        // 将device_dst复制回host
        queue.memcpy(dst.data,device_dst,rows*cols*sizeof(T)).wait();
        // 回收设备内存
        sycl::free(device_A,queue);
        sycl::free(device_B,queue);
        sycl::free(device_dst,queue);
    }
    void divide(const Matrix<T>& A,const Matrix<T>& B,const Matrix<T>& dst)const{
        size_t total_size = this->mat.total_size();
        // 在设备上分配内存
        T* device_A = sycl::malloc_device<T>(total_size,queue);
        T* device_B = sycl::malloc_device<T>(total_size,queue);
        T* device_dst = sycl::malloc_device<T>(total_size,queue);
        // 将A.data 和 B.data 复制到设备内存中
        queue.memcpy(device_A,A.data,total_size*sizeof(T));
        queue.memcpy(device_B,B.data,total_size*sizeof(T));
        // 计算
        queue.submit([&](sycl::handler& h){
            h.parallel_for(sycl::range(total_size), [=](auto idx) {
                if(sycl::fabs(device_B[idx]) < 0.00001){
                    device_dst[idx] = std::numeric_limits<T>::quiet_NaN(); // 或者其他默认值
                }else{
                    device_dst[idx] = device_A[idx] / device_B[idx];
                }
            });
        }).wait();
        // 将device_dst复制回host
        queue.memcpy(dst.data,device_dst,total_size*sizeof(T)).wait();
        // 回收设备内存
        sycl::free(device_A,queue);
        sycl::free(device_B,queue);
        sycl::free(device_dst,queue);
    }
    
    void transpose(const Matrix<T>& A){
        // 获取大小
        size_t rows = A.shape[0];
        size_t cols = A.shape[1];
        // 分配设备内存
        T* A_data = sycl::malloc_device<T>(rows*cols,queue);
        T* T_data = sycl::malloc_device<T>(rows*cols,queue);
        // 将A.data 复制到设备内存
        queue.memcpy(A_data,A.data,rows*cols*sizeof(T));
        // 计算
        queue.submit([&](sycl::handler& h){
            h.parallel_for(sycl::range<2>(rows, cols), [=](sycl::id<2> idx) {
                size_t i = idx[0];
                size_t j = idx[1];
                T_data[j * rows + i] = A_data[i * cols + j];
            });
        }).wait();
        //取回数据
        queue.memcpy(A.data,T_data,rows*cols*sizeof(T));
        sycl::free(T_data,queue);
        sycl::free(A_data,queue);
        std::swap(A.shape[0], A.shape[1]);
    }
    void check_shapes(const Tensor<T>& other)const{
        if (this->mat.dims != other.mat.dims) {
            std::string info = std::format("Tensors must have the same number of dimensions: {} vs {}",this->mat.dims,other.mat.dims);
            throw std::invalid_argument(info);
        }
        for (size_t i = 0; i < this->mat.dims; ++i) {
            if (this->mat.shape[i] != other.mat.shape[i]) {
                std::string info = std::format("Tensors must have the same shape in each dimension at index {}:{} vs {}",i,this->mat.shape[i],other.mat.shape[i]);
               throw std::invalid_argument(info);
            }
        }
    }
    size_t compute_index(const size_t* indices, const size_t* shape, size_t dims) {
        size_t index = 0;
        size_t stride = 1;
        for (int i = static_cast<int>(dims) - 1; i >= 0; --i) {
            index += indices[i] * stride;
            stride *= shape[i];
        }
        return index;
    }
    void print_matrix(const Matrix<T>& mat, size_t depth, size_t* indices) {
        if (depth == mat.dims) {
            std::cout<<mat.data[this->compute_index(indices, mat.shape, mat.dims)]<<" ";
            return;
        }
        std::cout << "[";
        for (size_t i = 0; i < mat.shape[depth]; ++i) {
            indices[depth] = i;
            print_matrix(mat, depth + 1, indices);
        }
        std::cout << "]";
    }
public:
    Tensor() = default;
    ~Tensor() = default;
    Tensor(const Tensor<T>& other) {
        this->mat.dims = other.mat.dims;
        if (other.mat.shape != nullptr) {
            this->mat.shape = new size_t[this->mat.dims];
            std::copy(other.mat.shape, other.mat.shape + this->mat.dims, this->mat.shape);
        } else {
            this->mat.shape = nullptr;
        }
        size_t total_size = this->mat.total_size();
        if (other.mat.data != nullptr && total_size > 0) {
            this->mat.data = new T[total_size];
            std::copy(other.mat.data, other.mat.data + total_size, this->mat.data);
        } else {
            this->mat.data = nullptr;
        }
    }
    Tensor(std::initializer_list<T> val,std::initializer_list<size_t> shape){
        // 计算总元素数量
        size_t total_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
        // 验证 val 和 shape 是否匹配
        if (val.size() != total_elements) {
            throw std::runtime_error("The size of values does not match the specified shape.");
        }
        // 初始化形状
        mat.dims = shape.size();
        mat.shape = new size_t[mat.dims];
        std::copy(shape.begin(), shape.end(), mat.shape);
        // 初始化数据
        mat.data = new T[total_elements];
        std::copy(val.begin(), val.end(), mat.data);
    }
    static Tensor Zeros(std::initializer_list<size_t> shape){
        Tensor<T> tensor;
        // 计算总元素数量
        size_t total_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
        // 分配内存
        tensor.mat.dims = shape.size();
        tensor.mat.shape = new size_t[tensor.mat.dims];
        std::copy(shape.begin(), shape.end(), tensor.mat.shape);
        tensor.mat.data = new T[total_elements];
        std::fill(tensor.mat.data, tensor.mat.data + total_elements, 0);
        return tensor;
    }
    static Tensor Ones(std::initializer_list<size_t> shape){
        Tensor<T> tensor;
        // 计算总元素数量
        size_t total_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
        // 分配内存
        tensor.mat.dims = shape.size();
        tensor.mat.shape = new size_t[tensor.mat.dims];
        std::copy(shape.begin(), shape.end(), tensor.mat.shape);
        tensor.mat.data = new T[total_elements];
        std::fill(tensor.mat.data, tensor.mat.data + total_elements, 1);
        return tensor;
    }
    static Tensor Fill(std::initializer_list<size_t> shape,T val){
        Tensor<T> tensor;
        // 计算总元素数量
        size_t total_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
        // 分配内存
        tensor.mat.dims = shape.size();
        tensor.mat.shape = new size_t[tensor.mat.dims];
        std::copy(shape.begin(), shape.end(), tensor.mat.shape);
        tensor.mat.data = new T[total_elements];
        std::fill(tensor.mat.data, tensor.mat.data + total_elements, val);
        return tensor;
    }

    Tensor<T>& operator=(const Tensor<T>& other) {
        if (this == &other) return *this; // 避免自赋值
        this->mat = other.mat;
        return *this;
    }

    // 运算符重载 for tensor
    Tensor<T> operator+(const Tensor<T>& other) const{
        this->check_shapes(other);
        Tensor<T> result = Tensor<T>(other);
        this->add(this->mat,other.mat,result.mat);
        return result;
    }
    Tensor<T> operator-(const Tensor<T>& other) const{
        this->check_shapes(other);
        Tensor<T> result = Tensor<T>(other);
        this->sub(this->mat,other.mat,result.mat);
        return result;
    }
    Tensor<T> operator*(const Tensor<T>& other) const{
        this->check_shapes(other);
        // 必须要求 this->mat  和 other.mat是二维
        if(this->mat.dims != 2 || other.mat.dims != 2){
            std::string info = std::format("Both Tensor must be 2 dims");
            throw std::invalid_argument(info);
        }
        // 检查矩阵尺寸是否匹配
        size_t rows = this->mat.shape[0];
        size_t common_dim = this->mat.shape[1];
        size_t cols = other.mat.shape[1];
        if(common_dim != other.mat.shape[0]){
            std::string info = std::format("this cols {} != other rows {}",common_dim,other.mat.shape[0]);
            throw std::invalid_argument(info);
        }
        Tensor<T> result = this->Zeros({rows,cols});
        this->mul(this->mat,other.mat,result.mat);
        return result;
    }
    Tensor<T> operator/(const Tensor<T>& other) const{
        this->check_shapes(other);
        Tensor<T> result = Tensor<T>(other);
        this->divide(this->mat,other.mat,result.mat);
        return result;
    }
   
    // 运算符重载 for scalar
    Tensor<T>& operator+=(T scalar) {
        if(this->mat.dims == 0){
            throw std::invalid_argument("0 dims");
        }
        this->add(this->mat,scalar);
        return *this;
    }
    Tensor<T>& operator-=(T scalar) {
        this->sub(this->mat,scalar);
        return *this;
    }
    Tensor<T>& operator*=(T scalar) {
        this->dot(this->mat,scalar);
        return *this;
    }
    Tensor<T>& operator/=(T scalar) {
        if(std::fabs(scalar) < 0.00001){
            std::string info = std::format("scalar is 0");
            throw std::invalid_argument(info);
        }
        this->divide(this->mat,scalar);
        return *this;
    }
    
    void t(){
        // 必须要求 this->mat是二维
        if(this->mat.dims != 2){
            std::string info = std::format("this Tensor must be 2 dims");
            throw std::invalid_argument(info);
        }
        this->transpose(this->mat);
    }
    void print(){
        // 打印矩阵
        size_t* indices = new size_t[mat.dims]();
        this->print_matrix(mat, 0, indices);
        std::cout << std::endl;
        //打印维度 
        std::cout << "Shape: [";
        for (size_t i = 0; i < mat.dims; ++i)
            std::cout << mat.shape[i] << (i < mat.dims - 1 ? ", " : "]\n");
        delete[] indices;
    }
    std::vector<size_t> shape(){
        std::vector<size_t> shapes;
        for(size_t i=0;i<this->mat.dims;i++){
            shapes.push_back(this->mat.shape[i]);
        }
        return shapes;
    }
};
