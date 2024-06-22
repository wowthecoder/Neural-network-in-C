#include <stdlib.h>
#include <stdio.h>

#include "tensor.h"

tensor tensor_make(double* data, int n_elems, int* shape, int n_dims){
    tensor t = malloc(sizeof(struct tensor));
    if(t == NULL){
        fprintf(stderr, "Error allocating memory\n");
        exit(1);
    }
    t->n_elems = n_elems;
    for (int i = 0; i < n_elems; i++) {
        t->data[i] = data[i];
    }
    t->strides = malloc(n_dims * sizeof(int));
    t->strides[n_dims-1] = 1;
    for (int i = n_dims - 2; i >= 0; i--) {
        t->strides[i] = shape[i + 1] * t->strides[i + 1];
    }
    t->shape = malloc(n_dims * sizeof(int));
    int test_shape_valid = 1;
    for (int i = 0; i < n_dims; i++) {
        t->shape[i] = shape[i];
        test_shape_valid *= shape[i];
    } 
    if(test_shape_valid != n_elems){
        fprintf(stderr,"Shape not matching with the total number of elements");
        exit(1);
    }
    return t;
}

static int tensor_idx(tensor t, int *idxs) {
    int idx = 0;
    for (int i = 0; i < t->n_dims; i++) {
        idx += idxs[i] * t->strides[i];
    }
    return idx;
}

// idxs contains i,j,k, etc. so we have t[i][j][k]
double tensor_get(tensor t, int *idxs) {
    return t->data[tensor_idx(t, idxs)];
}

// set the value of t[i][j][k]
void tensor_set(tensor t, int *idxs, double val) {
    t->data[tensor_idx(t, idxs)] = val;
}

// same as numpy reshape
// only change the shape and strides, not the actual data
void tensor_reshape(tensor t, int n_dims, int *shape) {
    t->n_dims = n_dims;
    for (int i = 0; i < n_dims; i++) {
        t->shape[i] = shape[i];
    }
    t->strides[n_dims-1] = 1;
    for (int i = n_dims - 2; i >= 0; i--) {
        t->strides[i] = shape[i + 1] * t->strides[i + 1];
    }
}

//returns a copy of the input tensor
tensor tensor_copy(tensor t) {
    return tensor_make(t->data, t->n_elems, t->shape, t->n_dims);
}

//frees the 
void tensor_free(tensor t) {
    free(t->data);
    free(t->shape);
    free(t->strides);
    free(t);
}

tensor tensor_add(tensor t1, tensor t2){
    if(t1->n_dims != t2->n_dims){
        fprintf(stderr, "Addition: dimension %d not equal to dimension %d.\n",t1->n_dims, t2->n_dims);
        exit(1);
    }
    tensor res = tensor_copy(t1);
    for (int i = 0; i < t1->n_elems; i++) {
        res->data[i] += t2->data[i];
    }
    for (int i = 0; i < t1->n_dims; i++) {
        if (t1->shape[i] != t2->shape[i]) {
            fprintf(stderr, "Addition: shape %d not equal to shape %d at %d\n", t1->shape[i], t2->shape[i], i);
            exit(1);
        }
        res->shape[i] = t1->shape[i];
    }
    return res;
}

tensor tensor_sub(){

}

tensor tensor_dot(){

}

tensor tensor_scale(){

}

tensor tensor_div(){

}
