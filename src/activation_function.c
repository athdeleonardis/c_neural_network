#include "activation_function.h"

#include "error.h"

#include <math.h>
#include <string.h>

//
// Auxilliary Definitions
//

double sigmoid(double);
double sigmoid_derivative(double);

//
// 'neural_network_activation_functions.h' Implementations
//

// TODO: Parsing tree
activation_function_t activation_function_get(const char *name) {
    // Chain of if-elses hooray
    if (strcmp(name, "sigmoid") == 0) {
        activation_function_t af = { "sigmoid", sigmoid, sigmoid_derivative };
        return af;
    }
    
    make_error("Activation function does not exist");
}

void activation_function_copy(activation_function_t src, activation_function_t *dest) {
    strcpy(dest->name, src.name);
    dest->function = src.function;
    dest->derivative = src.derivative;
}

//
// Auxilliary Implementations
//

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double sigmoid_derivative(double x) {
    return sigmoid(x) * (1 - sigmoid(x));
}