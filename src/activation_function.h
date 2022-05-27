#ifndef NEURAL_NETWORK_ACTIVATION_FUNCTIONS
#define NEURAL_NETWORK_ACTIVATION_FUNCTIONS

#include "matrix.h"

//
// 'neural_network_activation_functions.h' Definitions
//

#define ACTIVATION_FUNCTION_NAME_SIZE 16

typedef struct {
    char name[ACTIVATION_FUNCTION_NAME_SIZE];
    matrix_map_t function;
    matrix_map_t derivative;
} activation_function_t;

activation_function_t activation_function_get(const char *name);
void activation_function_copy(activation_function_t src, activation_function_t *dest);

#endif
