#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "math_structs.h"
#include "math_r4t.h"
#include "layer.h"
#include "flattenlayer.h"

flatten_layer *flatten_layer_create() {
    return (flatten_layer *) calloc(1, sizeof(flatten_layer));
}


bool flatten_layer_init(flatten_layer *layer, int num_outputs, int irows, int icols, int erows, int ecols, dense_layer *prev) {
    layer->prev = prev;
    layer->num_outputs = num_outputs;
    layer->outputs = matrix_make(irows*icols*erows*ecols, 1);
    if (layer->outputs == NULL) {
        return true;
    }
    r4t prev_r4t = prev->num_outputs;
    layer->deltas = r4t_make(prev_r4t->rows, prev_r4t->cols, prev_r4t->elemrows, prev_r4t->elemcols);
    if (layer->deltas == NULL) {
        return true;
    }
    return false;
}


void flatten_layer_compute_outputs(flatten_layer *layer) {
    // output of previous layer is an nx1 matrix of pxq matrices
    int pos = 0;
    r4t prev_out = layer->prev->outputs;
    for (int i = 0; i < prev_out->rows; i++) {
        for (int j = 0; j < prev_out->cols; i++) {
            matrixt m = prev_out->contents[i][j];
            for (int p = 0; p < prev_out->elemrows; p++) {
                for (int q = 0; q < prev_out->elemcols; q++) {
                    layer->outputs->contents[pos][0] = m->contents[p][q];
                    pos++;
                }
            }
        }
    }
}

// matrix deltas are just restructured into r4t deltas
void flatten_layer_compute_deltas(flatten_layer const *layer) {
    int pos = 0;
    r4t del = layer->deltas;
    for (int i = 0; i < del->rows; i++) {
        for (int j = 0; j < del->cols; i++) {
            for (int p = 0; p < del->elemrows; p++) {
                for (int q = 0; q < del->elemcols; q++) {
                    layer->deltas->contents[i][j]->contents[p][q] = layer->next->deltas->contents[pos][0];
                    pos++;
                }
            }
        }
    }
}

// no need to update flatten layer - no trainabke parameters