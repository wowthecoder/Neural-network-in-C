#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "ann.h"

/*
THIS IS THE SCARY FILE - WILL EXPLAIN MORE IN NEXT MEETING - Sam


effectively a network is an array of layer structs as will be defined in net_components

we need to things - process and train

process:

- takes an input matrix, returns an output matrix
- achieves output by calling the forward prop function of each layer sequentially, having the output of one be the input to the next

train:

- we will just use regular gradient descent, and our training cycle will just be based on regression using SSR (or MSE?)
- our training cycle will 'process' each input to generate a matrix of observations
- observation matrix used to find dE/dO then this is fed back through every layer in the network array to update params
- this repeats for the specified ammount of cycles

- find some way to print/plot loss with each cyle
- then were done!

*/

// activation func * is an array of activation functions, one for each layer excluding input layer
// acts[0] and act_ders[0] should be NULL, both has size num_layers
ann_t *ann_create(int num_layers, int batch_size, loss_der f, activation_func *acts, activation_func *act_ders, double *dropout_probs, int *layer_outputs)
{
    ann_t *new_ann = malloc(sizeof(ann_t));
    if (new_ann == NULL) {
        return NULL;
    }
    for (int i = 0; i < num_layers; i++) {
        dense_layer *curr_layer = layer_create();
        if (curr_layer == NULL) {
            return NULL;
        }
        if (i == 0) {
            new_ann->input_layer = curr_layer;
            new_ann->output_layer = curr_layer;
        }
        else {
            dense_layer *last = new_ann->output_layer; // find the original last of the list
            curr_layer->prev = last; // backlink current layer to last
            last->next = curr_layer; // link last to current layer
            new_ann->output_layer = curr_layer; // current layer is now the last of the list
        }
        layer_init(curr_layer, layer_outputs[i], batch_size, acts[i], act_ders[i], dropout_probs[i], curr_layer->prev);
    }
    new_ann->num_inputs = layer_outputs[0];
    new_ann->num_outputs = layer_outputs[num_layers-1];
    new_ann->loss_prime = f;
    new_ann->batch_size = batch_size;
    return new_ann;
}

/* Frees the space allocated to ann. */
void ann_free(ann_t *ann)
{
    for (dense_layer *curr = ann->input_layer; curr != NULL; curr = curr->next) {
        layer_free( curr );
    }
    free( ann );
}

/* Forward run of given ann with inputs. */
void ann_predict_batch(ann_t const *ann, matrixt inputs, bool training)
{
    // Cannot just assign this pointer for some reason
    // ann->input_layer->outputs = inputs;
    dense_layer *in = ann->input_layer;
    for (int i = 0; i < in->num_outputs; i++) {
        for (int j = 0; j < ann->batch_size; j++) {
            in->outputs->contents[i][j] = inputs->contents[i][j];
        }
    }
    for (dense_layer *curr = in->next; curr != NULL; curr = curr->next) {
        layer_compute_outputs(curr, training);
    }
}

/*
  * Trains the ann with single backprop update. 
  * targets is a 2D array of length num_samples, each sample has 1 or more outputs (1 for regression, multiple for classification)
  * inputs is a 2D array of length num_samples, each samples has some features.
  * num_samples is to indicate whether we have the full batch (num_samples = batch_size) or not
*/
void ann_train(ann_t const *ann, matrixt inputs, matrixt targets, double l_rate, int epoch)
{
    /* Sanity checks. */
    assert(ann != NULL);
    assert(inputs != NULL);
    assert(targets != NULL);
    assert(l_rate > 0);

    /* Run forward pass. */
    ann_predict_batch(ann, inputs, true);

    /* Compute gradients of output layer */
    dense_layer *out_layer = ann->output_layer;
    matrixt act_zs = (out_layer->activate_der)(out_layer->zs);
    matrixt diff = (ann->loss_prime)(targets, out_layer->outputs);
    // both act_output and diff has shape (num_outputs * num_samples) 
    // element-wise multiply both matrices to get the delta matrix
    matrix_elemMult(out_layer->deltas, act_zs, diff);
    

    /* Backpropagation (exclude input layer) */
    for (dense_layer *curr = ann->output_layer->prev; curr != ann->input_layer; curr = curr->prev) {
        layer_compute_deltas(curr);
    }

    /* Gradient descent step (update to do backwards) */
    for (dense_layer *curr = ann->input_layer->next; curr != NULL; curr = curr->next) {
        layer_update(curr, l_rate, ann->batch_size, epoch);
    }

    matrix_free(act_zs);
    matrix_free(diff);
}
