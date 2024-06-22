#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "tensor.h"

double MSE(tensor t1, tensor t2){
    for (int i = 0; i < t1->n_dims; i++) {
        if (t1->shape[i] != t2->shape[i]) {
            fprintf(stderr, "MSE: Shape %d not equal to target shape %d at %d\n", t1->shape[i], t2->shape[i], i);
            exit(1);
        }
    }
    double result = 0;
    for(int i = 0; i < t1->n_elems; i++){
        result += (t1->data[i] - t2->data[i])*(t1->data[i] - t2->data[i]);
    } // and yes may overflow here ik don't panic yet
    result /= t1->n_elems;
    return result;
}

tensor MSE_der(tensor t1, tensor t2){
    if(t1->n_dims != t2->n_dims){
        fprintf(stderr,"MSE_der: Dimension of tensors different");
        exit(1);
    }
    for (int i = 0; i < t1->n_dims; i++) {
        if (t1->shape[i] != t2->shape[i]) {
            fprintf(stderr, "MSE: Shape %d not equal to target shape %d at %d\n", t1->shape[i], t2->shape[i], i);
            exit(1);
        }
    }
    tensor res = tensor_copy(t1);
    for(int i = 0; i < res->n_elems; i++){
        res -> data[i] = 2.0*(t1 ->data[i]-t2->data[i]);
    }
    return res;
}

double binary_cross_entropy(tensor t1, tensor t2){
    if(t1->n_dims != 2){
        fprintf("BINARY cross entropy cannot be called on tensor with dimension %d", t1->n_dims);
        exit(1);
    }
    if(t2->n_dims != 2){
        fprintf("BINARY cross entropy cannot be called on tensor with dimension %d", t2->n_dims);
        exit(1);
    }
    for (int i = 0; i < 2; i++) {
        if (t1->shape[i] != t2->shape[i]) {
            fprintf(stderr, "BCE: Shape %d not equal to target shape %d at %d\n", t1->shape[i], t2->shape[i], i);
            exit(1);
        }
    }
    double res = 0;
    for(int i = 0; i < t1->n_elems; i++){ 
        if(t1->data[i] == 0 || t1->data[i] == 1){
            fprintf(stderr, "BCE: error calculating the logarithm of 0");
            exit(1);
        }
        res += log(t1->data[i])*t2->data[i] + (1-t2->data[i])*log(1-t1->data[i]); 
    }
    res /= t1->n_elems;
    res *= -1;
    return res;
}

double binary_cross_entropy_der(){

}

tensor sigmoid(tensor t){
    tensor res = tensor_copy(t);
    for(int i = 0; i < res->n_elems; i++){
        res->data[i] = 1.0 /(1.0 +exp(-1*t->data[i]));
    }
    return t;
}

tensor sigmoid_der(tensor t){
    tensor res = tensor_copy(t);
    for(int i = 0; i < res->n_elems; i++){
        res->data[i] = t->data[i]*(1-t->data[i]);
    }
    return res;
}

tensor relu(tensor t) {
    tensor res = tensor_copy(t);
    for(int i = 0; i < t->n_elems; i++){
        res->data[i] = MAX(0,res->data[i]);
    }
    return res;
}

tensor relu_der(tensor t) {
    tensor res = tensor_copy(t);
    for(int i = 0; i < t->n_elems; i++){
        res->data[i] = (t->data[i]>=0)?1:0;
    }
    return res;
}

tensor tanh_reg(tensor t){
    tensor res = tensor_copy(t);
    for(int i = 0; i < t->n_elems; i++){
        res->data[i] = tanh(res->data[i]);
    }
    return res;
}

tensor tanh_der(tensor t){
    tensor res = tensor_copy(t);
    for(int i = 0; i < t->n_elems; i++){
        res->data[i] = 1.0 - tanh(res->data[i])*tanh(res->data[i]);
    }
    return res;
}