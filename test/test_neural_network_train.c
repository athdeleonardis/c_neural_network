#include "../src/neural_network.h"
#include "../src/neural_network_train.h"
#include "../src/random.h"

#include <stdio.h>
#include <stdlib.h>

/**
 * This file trains a neural network against the simple case of the XOR gate, i.e.
 * (0, 0) -> 0
 * (1, 0) -> 1
 * (0, 1) -> 1
 * (1, 1) -> 0
 * 
 * This highlights the non-linear behaviour of the neural network, as XOR is a non-linear function of two inputs.
*/

#define N_TRAINING_CASES 1000000

int main(int argc, char *argv[]) {
    random_init();
    
    printf("Step 1: Create the neural network\n");
    int hidden_layer_sizes[1] = { 4 };
    char *activation_functions[2] = { "sigmoid", "sigmoid" };
    neural_network_t *nn = neural_network_create(2, 1, 1, hidden_layer_sizes, activation_functions);
    neural_network_layers_randomize(nn);
    neural_network_print(nn);
    
    printf("\nStep 2: Create the input and output data\n");
    matrix_t **inputs = (matrix_t **)malloc(4 * sizeof(matrix_t));
    matrix_t **outputs = (matrix_t **)malloc(4 * sizeof(matrix_t));
    for (int i = 0; i < 4; i++) {
        inputs[i] = matrix_create(1, 2);
        inputs[i]->data[0] = i % 2;
        inputs[i]->data[1] = (i / 2) % 2;

        outputs[i] = matrix_create(1, 1);
        outputs[i]->data[0] = ((int)inputs[i]->data[0] + (int)inputs[i]->data[1]) % 2; // XOR
    }

    matrix_t **outputs_before = neural_network_evaluate(nn, 4, inputs);
    for (int i = 0; i < 4; i++) {
        printf("Case %d:\nInput: ", i);
        matrix_print(inputs[i]);
        printf("Output expected: ");
        matrix_print(outputs[i]);
        printf("Output: ");
        matrix_print(outputs_before[i]);
    }

    printf("\nStep 3: Train the neural network on the data\n");
    for (int i = 0; i < N_TRAINING_CASES; i++) {
        int num_case = random_int_between(0, 4);
        neural_network_train_case(nn, inputs[num_case], outputs[num_case], 0.01);
    }

    neural_network_print(nn);

    matrix_t **outputs_calculated = neural_network_evaluate(nn, 4, inputs);
    neural_network_delete(nn);
    for (int i = 0; i < 4; i++) {
        printf("Case %d:\nInput: ", i);
        matrix_print(inputs[i]);
        printf("Output expected: ");
        matrix_print(outputs[i]);
        printf("Output: ");
        matrix_print(outputs_calculated[i]);
    }

    printf("Errors: [");
    for (int i = 0; i < 4; i++) {
        double error_percentage = (matrix_get(outputs_calculated[i], 1, 0) - matrix_get(outputs[i], 1, 0)) * 100;
        if (error_percentage < 0)
            error_percentage = -error_percentage;
        if (i)
            printf(", ");
        printf("%.01f%%", error_percentage);
    }
    printf("]\n");

    for (int i = 0; i < 4; i++) {
        matrix_delete(inputs[i]);
        matrix_delete(outputs[i]);
        matrix_delete(outputs_before[i]);
        matrix_delete(outputs_calculated[i]);
    }
    free(inputs);
    free(outputs);
    free(outputs_before);
    free(outputs_calculated);
}
