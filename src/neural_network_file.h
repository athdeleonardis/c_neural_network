#ifndef NEURAL_NETWORK_FILE
#define NEURAL_NETWORK_FILE

#include "neural_network.h"

/**
 * Save the input, output and hidden layer sizes of the inputted neural network.
 * @param nn The neural network struct data to be saved.
 * @param filename The file location where the data is to be saved.
*/
void neural_network_save_static(neural_network_t *nn, const char *filename);

/**
 * Save the input, output and hidden layer sizes of the inputted neural network, as well as all the weights and biases.
 * @param nn The neural network struct data to be saved.
 * @param filename The file location where the data is to be saved.
*/
void neural_network_save_dynamic(neural_network_t *nn, const char *filename);

/**
 * Load the input, output and hidden layer sizes of a neural network.
 * @param filename The file location from which the data is loaded.
*/
neural_network_t *neural_network_load_static(const char *filename);

/**
 * Load the input, output and hidden layer sizes of a neural network, as well as all the weights and biases.
 * @param filename The file location from which the data is loaded.
*/
neural_network_t *neural_network_load_dynamic(const char *filename);

#endif
