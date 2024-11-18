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
    matrix_t weights;
    matrix_t biases;
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
 * Initialize the neural network's layers' weight and bias matrices from a single array.
 * @param nn The neural network with layers to be initialized.
 * @param data The array which will be partitioned for the neural network's weight and bias matrices.
 * @param activation_function_names The activation function names of each of the neural network's layers.
 */
void neural_network_layers_from_array(neural_network_t *nn, double *data, char **activation_function_names);

/**
 * Delete the inputted neural network to prevent memory leaks. Only intended to delete neural networks allocated by 'neural_network_create'.
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
 * Evaluate the inputted neural network against an array of inputs, placing the respective outputs into the outputs array.
 * @param nn The neural network to compute the inputs against.
 * @param n_cases The number of input cases to compute.
 * @param inputs The inputs to be passed to the neural network. The length of this array should equal 'n_cases'.
 * @param outputs The array of matrices in which the outputs will be placed. The length of this array should equal 'n_cases'.
*/
void neural_network_evaluate(neural_network_t *nn, int n_cases, matrix_t *inputs, matrix_t *outputs);

#endif
