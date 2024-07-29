#include <iostream>
#include "utils_.h"
#include <CL/sycl.hpp>


int main(int argc, char const *argv[]){

    // matrix_add();
    //matrix_mul();

    std::string input_img = {"../assets/input.jpg"};
    image_conv(input_img);
    return 0;
}
