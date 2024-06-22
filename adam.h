#include "math_structs.h"

typedef struct adam_optimizer {
  matrixt momentum;
  matrixt s;
  matrixt m_hat;
  matrixt s_hat;
  double beta1, beta2;
  /* used in gradient descent to avoid division by zero */
  double epsilon;
};
typedef struct adam_optimizer *adam_optimizer;

adam_optimizer adam_create(int num_neurons, int batch_size, double beta1, double beta2, double epsilon);
void adam_optimize(adam_optimizer adam, matrixt deltas, int epoch);
matrixt sqrt_epsilon(matrixt s_hat, double epsilon);
void adam_free(adam_optimizer adam);