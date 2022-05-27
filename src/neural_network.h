#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include "matrix.h"

//
// 'neural_network.h' Definitions
//

typedef struct {
    matrix_t *weights;
    matrix_t *biases;
} layer_t;

typedef struct {
    int input_size;
    int output_size;
    int hidden_layer_count;
    int *hidden_layer_sizes;
    layer_t *layers;
} neural_network_t;

neural_network_t *neural_network_create(int input_size, int output_size, int hidden_layer_count, int *hidden_layer_sizes);
void neural_network_delete(neural_network_t *);
void neural_network_print(neural_network_t *);
void neural_network_layers_randomize(neural_network_t *);
matrix_t **neural_network_evaluate(neural_network_t *, int n_cases, matrix_t **inputs);

#endif
