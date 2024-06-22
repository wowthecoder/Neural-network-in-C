#ifndef __ANN_H__
#define __ANN_H__

#include <stdbool.h>

#include "layer.h"
#include "math_structs.h"
#include "math_funcs.h"

/* Represents a N layer artificial neural network. */
typedef struct ann {
    int num_inputs, num_outputs;
    /* The head and tail of layers doubly linked list. */
    dense_layer *input_layer;
    dense_layer *output_layer;
    /* The derivative of the loss function */
    loss_der loss_prime;
    /* max number of samples it can take in at once */
    int batch_size;
} ann_t;

/* Creates and returns a new ann. */
ann_t *ann_create(int num_layers, int batch_size, loss_der f, activation_func *acts, activation_func *act_ders, double *dropout_probs, int *layer_outputs);
/* Frees the space allocated to ann. */
void ann_free(ann_t *ann);
/* Forward run of given ann with batch of inputs. */
void ann_predict_batch(ann_t const *ann, matrixt inputs, bool training);
/* During testing/deployment, predict the value of 1 sample. */
void ann_predict_one(ann_t const *ann, matrixt inputs);
/* Trains the ann with single backprop update on a batch at once. */
void ann_train(ann_t const *ann, matrixt inputs, matrixt targets, double l_rate, int epoch);

#endif