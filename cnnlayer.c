#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "math_structs.h"
#include "cnnlayer.h"

cnn_layer *cnn_layer_create() {
    return (cnn_layer *) calloc(1, sizeof(cnn_layer));
}

bool cnn_layer_init(cnn_layer *layer, int batch_size, int num_outputs, int ksize, activation_func f, activation_func fprime, cnn_layer *prev) {
    layer->prev = prev;
    layer->irows = prev->outputs->contents[0][0]->rows;
    layer->icols = prev->outputs->contents[0][0]->cols;
    layer->num_outputs = num_outputs;
    layer->outputs = r4t_make(num_outputs, batch_size, layer->irows-ksize+1, layer->icols-ksize+1);
    if (layer->outputs == NULL) {
        return true;
    }
    if (prev != NULL) {
        layer->num_inputs = prev->num_outputs;
        int num_inputs = layer->num_inputs;
        // Allocate row pointers, each input is 1 row
        layer->kernals = r4t_make(num_outputs, num_inputs, ksize, ksize);
        // Assign random values
        r4t_rands(layer->kernals);
        layer->biases = r4t_make(num_outputs, 1, layer->irows-ksize+1, layer->icols-ksize+1);
        r4t_rands(layer->biases);
        layer->deltas = r4t_make(num_outputs, batch_size, layer->irows-ksize+1, layer->icols-ksize+1);
        if (layer->biases == NULL || layer->deltas == NULL) {
            return true;
        }
        layer->activate = f;
        layer->activate_der = fprime;
    } 
    return false;
}

void cnn_layer_free(cnn_layer *layer) {
    matrix_free(layer->outputs);
    if (layer->prev != NULL) {
        matrix_free(layer->kernals);
        matrix_free(layer->biases);
        matrix_free(layer->deltas);
    }
    free ( layer );
}

void cnn_layer_compute_outputs(cnn_layer *layer) {
    r4t_MultDotCorr(layer->outputs, layer->kernals, layer->prev->outputs);
    r4t_add(layer->outputs, layer->biases);
    r4t old_outputs = layer->outputs; // save ptr to old output matrix
    layer->outputs = (layer->activate)(layer->outputs);
    matrix_free(old_outputs);
}