#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include "activation_function.h"
#include "matrix.h"

//
// 'neural_network.h' Definitions
//

/**
 * A hidden layer between output layers of the neural network.
*/
typedef struct {
    matrix_t *weights;
    matrix_t *biases;
    activation_function_t activation_function;
} layer_t;

/**
 * A struct representing a simple feed-forward neural network.
*/
typedef struct {
    int input_size;
    int output_size;
    int hidden_layer_count;
    int *hidden_layer_sizes;
    layer_t *layers;
} neural_network_t;

/**
 * Create a feed-forward neural network with the given input parameters.
 * @param input_size The number of rows of the input matrix.
 * @param output_size The number of rows of the output matrix.
 * @param hidden_layer_count The number of hidden layers.
 * @param hidden_layer_sizes The number of rows of each hidden layer output. The length of this array should equal 'hidden_layer_count'.
 * @param activation_functions The name of activation functions of each hidden layer. The length of this array should equal 'hidden_layer_count+1'.
 * @return A neural network with undefined weights and biases matching the input parameters.
*/
neural_network_t *neural_network_create(int input_size, int output_size, int hidden_layer_count, int *hidden_layer_sizes, char **activation_functions);

/**
 * Delete the inputted neural network to prevent memory leaks.
 * @param nn The neural network to be deleted.
*/
void neural_network_delete(neural_network_t *nn);

/**
 * Print the inputted neural network to the console.
 * @param nn The neural network to be printed to the console.
*/
void neural_network_print(neural_network_t *nn);

/**
 * Randomize all weights and biases of the inputted neural network to be between (-1,1).
 * @param nn The neural network to be randomized.
*/
void neural_network_layers_randomize(neural_network_t *nn);

/**
 * Evaluate the inputted neural network against an array of inputs, returning an array of outputs.
 * @param nn The neural network to compute the inputs against.
 * @param n_cases The number of input cases to compute.
 * @param inputs The inputs to be passed to the neural network. The length of this array should equal 'n_cases'.
 * @return The outputs of the neural network from each input, respectively. The length of this array equals 'n_cases'.
*/
matrix_t **neural_network_evaluate(neural_network_t *nn, int n_cases, matrix_t **inputs);

#endif
