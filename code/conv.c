#include "tensor.h"
#include "math_structs.h"

double corr(matrixt input, matrixt kernel){
    if(input->rows != kernel->cols || input->cols != kernel->cols){
        printf("Correlation: shape of input and mask not matching");
        exit(1);
    }
    double res = 0;
    for(int i = 0; i < input->rows; i++){
        for(int j = 0; j < input-> cols; j++){
            res+=input->contents[i][j]*kernel->contents[i][j];
        }
    }
    return res;
}

double conv(matrixt input, matrixt kernel){
    return corr(input, matrix_transposeOf(kernel));
}

matrixt corr2d(matrixt input, matrixt kernel){
    //if input shape is (a,b) and kernel is (c,d) output shape is (a-c+1)*(b-d+1)
    matrixt res = matrix_make(input->rows+kernel->rows-1, input->cols+kernel->cols-1);
    matrix_zeros(res);
    for(int i = 0; i < res->rows; i++){
        for(int j = 0; j < res->cols; j++){
            res->contents[i][j]=corr(submatrix(i,j,i+kernel->rows,j+kernel->cols,input), kernel);//change this for padding 
        }
    }
    return res;
}

tensor Conv2D(int in_channels, int out_channels, int * kernel_size, int conv_stride, int padding){
    
}

//TODO: Flatten and dropout layer