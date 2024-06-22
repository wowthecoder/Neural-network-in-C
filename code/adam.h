#include <stdbool.h>

#include "math_structs.h"

typedef struct adam_optimizer {
  matrixt m_weights;
  matrixt m_bias;
  matrixt s_weights;
  matrixt s_bias;
  matrixt mhat_weights;
  matrixt mhat_bias;
  matrixt shat_weights;
  matrixt shat_bias;
  double beta1, beta2;
  /* used in gradient descent to avoid division by zero */
  double epsilon;
};
typedef struct adam_optimizer *adam_optimizer;

adam_optimizer adam_create(int num_neurons, int num_inputs, int batch_size, double beta1, double beta2, double epsilon);
void adam_optimize(adam_optimizer adam, matrixt deltas, int epoch);
matrixt sqrt_epsilon(matrixt s_hat, double epsilon);
void adam_update(adam_optimizer adam, matrixt m, bool weights, double l_rate);
void adam_free(adam_optimizer adam);