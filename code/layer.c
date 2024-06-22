#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "math_structs.h"
#include "layer.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))

dense_layer *layer_create() {
    return (dense_layer *) calloc(1, sizeof(dense_layer));
}

// input layer just pipes the input to the output, no weights, biases or anything
bool layer_init(dense_layer *layer, int num_outputs, int batch_size, activation_func f, activation_func fprime, double dropout_prob, dense_layer *prev) {
    layer->prev = prev;
    layer->num_outputs = num_outputs;
    layer->outputs = matrix_make(num_outputs, batch_size);
    layer->dropout_mask = matrix_make(num_outputs, 1);
    for (int i = 0; i < num_outputs; i++) {
        layer->dropout_mask->contents[i][0] = 1;
    }
    layer->dropout_p = dropout_prob;
    if (layer->outputs == NULL || layer->dropout_mask == NULL) {
        return true;
    }
    if (prev != NULL) {
        layer->num_inputs = prev->num_outputs;
        int num_inputs = layer->num_inputs;
        layer->zs = matrix_make(num_outputs, batch_size);
        layer->weights = matrix_make(num_outputs, num_inputs);
        // Assign random values
        matrix_rands(layer->weights);
        layer->biases = matrix_make(num_outputs, 1);
        layer->deltas = matrix_make(num_outputs, batch_size);
        // layer->adam = adam_create(num_outputs, batch_size, 0.9, 0.999, pow(10, -7));
        if (layer->biases == NULL || layer->deltas == NULL) {
            return true;
        }
        layer->activate = f;
        layer->activate_der = fprime;
    } 
    return false;
}

void layer_free(dense_layer *layer) {
    matrix_free(layer->outputs);
    matrix_free(layer->dropout_mask);
    if (layer->prev != NULL) {
        matrix_free(layer->zs);
        matrix_free(layer->weights);
        matrix_free(layer->biases);
        matrix_free(layer->deltas);
        // adam_free(layer->adam);
    }
    free ( layer );
}

// y = relu(Xw + b) [or tanh for output layer]
void layer_compute_outputs(dense_layer *layer, bool training) {
    matrix_matMult(layer->zs, layer->weights, layer->prev->outputs); // Xw
    matrix_add_vector(layer->zs, layer->biases); // Xw + b
    layer->outputs = (layer->activate)(layer->zs);
    if (training) {
        dropout(layer->dropout_mask, layer->dropout_p);
        matrixt scaled_mask = matrix_scalarMult(layer->dropout_mask, 1.0 / (1 - layer->dropout_p));
        matrix_dot_vector(layer->outputs, scaled_mask);
        matrix_free(scaled_mask);
    }
}

// layer->deltas[i] = activate_der(layer->outputs[i]) * sum_gradients;
void layer_compute_deltas(dense_layer const *layer, int epoch) {
    matrixt act_zs = (layer->activate_der)(layer->zs);
    matrixt nextW_T = matrix_transposeOf(layer->next->weights);
    matrix_matMult(layer->deltas, nextW_T, layer->next->deltas);
    matrix_elemMult(layer->deltas, act_zs, layer->deltas);
    // dropout
    matrix_dot_vector(layer->deltas, layer->dropout_mask);

    matrix_free(act_zs);
    matrix_free(nextW_T);
}

// layer->weights[i][j] += l_rate * layer->prev->outputs[j] * layer->deltas[i];
// layer->biases[i] += l_rate * layer->deltas[i];
// Commented parts are for adam optimizers but it doesn't work
void layer_update(dense_layer const *layer, double l_rate, int batch_size) {
    double factor = l_rate / batch_size;
    // matrixt inputs_T = matrix_transposeOf(layer->prev->outputs);
    // matrixt lr_deltas = matrix_scalarMult(layer->deltas, factor);
    // matrixt sum_deltas = matrix_sum_rows(lr_deltas);
    // matrixt weight_changes = matrix_make(layer->weights->rows, layer->weights->cols);
    // matrix_matMult(weight_changes, lr_deltas, inputs_T);
    // matrix_subtract(layer->weights, weight_changes);
    // matrix_subtract(layer->biases, sum_deltas);

    adam_optimize(layer->adam, layer->deltas, inputs_T, epoch);


    matrix_free(inputs_T);
    matrix_free(lr_deltas);
    matrix_free(weight_changes);
    matrix_free(sum_deltas);
}

/*
SOME ROUGH WORK WITH FCN FUNCTIONS
*/

// this is commented as we are yet to wrap matrix contents and dimensions together
// ignore above - done 14/06/2024!

// matrixt FCN_fp(matrixt input, fcn fcn){
//     // caching input for backprop
//     fcn->input = matrix_copyOf(input);
//     matrixt z = matrix_make(fcn->outSize, 1);
//     matrix_matMult(z, fcn->weights, input);
//     matrix_add(z, fcn->biases);
//     fcn->output = matrix_copyOf(z);
//     return z;
// }
