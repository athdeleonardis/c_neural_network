#include "../src/neural_network.h"
#include "../src/neural_network_train.h"
#include "../src/random.h"

#include <stdio.h>
#include <stdlib.h>

#define N_TRAINING_CASES 1000000

int main(int argc, char *argv[]) {
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

    matrix_t **outputs_after = neural_network_evaluate(nn, 4, inputs);
    neural_network_delete(nn);
    for (int i = 0; i < 4; i++) {
        printf("Case %d:\nInput: ", i);
        matrix_print(inputs[i]);
        printf("Output expected: ");
        matrix_print(outputs[i]);
        printf("Output: ");
        matrix_print(outputs_after[i]);
    }

    for (int i = 0; i < 4; i++) {
        matrix_delete(inputs[i]);
        matrix_delete(outputs[i]);
        matrix_delete(outputs_before[i]);
        matrix_delete(outputs_after[i]);
    }
    free(inputs);
    free(outputs);
    free(outputs_before);
    free(outputs_after);
}