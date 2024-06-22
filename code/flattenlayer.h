#include <stdbool.h>

#include "math_structs.h"
#include "math_funcs.h"
#include "math_r4t.h"
#include "layer.h"

/* Represents a single flatten layer. */
typedef struct flayer {
  /* dimensions of input */
  int irows, icols, erows, ecols;
  /* Number outputs (neurons).*/
  int num_outputs;
  /* Output of EACH neuron. */
  matrixt outputs;
  /* Pointers to previous and next layer if any. */
  struct clayer *prev;
  struct dlayer *next;

  r4t deltas;
} flatten_layer;


extern flatten_layer *flatten_layer_create();
extern bool flatten_layer_init(flatten_layer *layer, int num_outputs, int irows, int icols, int erows, int ecols, dense_layer *prev);
extern void flatten_layer_compute_outputs(flatten_layer *layer);
extern void flatten_layer_compute_deltas(flatten_layer const *layer);