#include <stdbool.h>

#include "math_structs.h"
#include "math_funcs.h"
#include "adam.h"

/* Represents a single neural layer. */
typedef struct dlayer {
  /* Number of inputs and outputs (neurons).*/
  int num_inputs, num_outputs;
  /* Output of EACH neuron, before activation. */
  matrixt zs;
  /* Output of EACH neuron, after activation. */
  matrixt outputs;
  /* Pointers to previous and next layer if any. */
  struct dlayer *prev;
  struct dlayer *next;
  /* Incoming weights of EACH neuron. */
  matrixt weights;
  /* Biases of EACH neuron. */
  matrixt biases;
  /* Delta errors of EACH neuron. */
  matrixt deltas;
  /* To indicate whether this neuron will be dropped out or not */
  matrixt dropout_mask;
  /* Activation function for this layer */
  activation_func activate;
  /* Derivative of activation function (needed for backpropagation) */
  activation_func activate_der;
  /* Dropout probability p (Probability that this layer gets dropped) */
  double dropout_p;
  /* Adam Optimizer */
  // adam_optimizer adam;
} dense_layer;

/* Creates a single layer. */
dense_layer *layer_create();
/* Initialises the given layer. */
bool layer_init(dense_layer *layer, int num_outputs, int batch_size, activation_func f, activation_func fprime, double dropout_prob, dense_layer *prev);
/* Frees a given layer. */
void layer_free(dense_layer *layer);
/* Computes the outputs of the current and all subsequent layers given inputs. */
void layer_compute_outputs(dense_layer *layer, bool training);
/* Computes the delta errors for this layer and all previous layers (backpropagate). */
void layer_compute_deltas(dense_layer const *layer, int epoch);
/* Updates weights and biases according to the delta errors given learning rate. */
void layer_update(dense_layer const *layer, double l_rate, int batch_size);