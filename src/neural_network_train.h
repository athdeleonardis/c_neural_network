#ifndef NEURAL_NETWORK_TRAIN
#define NEURAL_NETWORK_TRAIN

#include "neural_network.h"

//
// 'neural_network_train.h' definitions
//

/**
 * Train the neural network on a single case.
 * @param nn The neural network to train.
 * @param input The input matrix to evaluate the neural network on.
 * @param output The matrix representing the expected output of the neural network.
 * @param p The training parameter. Weights will be adjusted proportional to this parameter.
*/
void neural_network_train_case(neural_network_t *nn, matrix_t *input, matrix_t *output, double p);

#endif
