#include <stdbool.h>

#include "math_structs.h"
#include "math_funcs.h"
#include "math_r4t.h"


/* Represents a single neural layer. */
typedef struct clayer {
  /* Number of inputs and outputs (neurons).*/
  int num_inputs, num_outputs;
  int irows, icols, ksize;
  /* Output of EACH neuron. */
  r4t outputs;
  /* Pointers to previous and next layer if any. */
  struct clayer *prev;
  struct clayer *next;
  /* Incoming kernals of EACH neuron. */
  r4t kernals;
  /* Biases of EACH neuron. */
  r4t biases;
  /* Delta errors of EACH neuron. */
  r4t deltas;
  /* Activation function for this layer */
  activation_func activate;
  /* Derivative of activation function (needed for backpropagation) */
  activation_func activate_der;
} cnn_layer;

extern cnn_layer *cnn_layer_create();
extern bool cnn_layer_init(cnn_layer *layer, int batch_size, int num_outputs, int ksize, activation_func f, activation_func fprime, cnn_layer *prev);
extern void cnn_layer_free(cnn_layer *layer);
extern void cnn_layer_compute_outputs(cnn_layer *layer);