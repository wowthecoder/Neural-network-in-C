typedef struct tensor {
    double *data;
    int n_elems;
    int *shape;
    int n_dims; // length of shape
    int *strides;
};
typedef struct tensor *tensor;

extern tensor tensor_make(double* data, int n_elems, int* shape, int n_dims);
extern double tensor_get(tensor t, int *idxs);
extern void tensor_set(tensor t, int *idxs, double val);
extern void tensor_reshape(tensor t, int n_dims, int *shape);
extern tensor tensor_copy(tensor t);
extern void tensor_free(tensor t);
