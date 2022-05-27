#define N_CASES 6

#include "../src/neural_network.h"
#include "../src/random.h"
#include "../src/matrix.h"

#include <stdlib.h>
#include <stdio.h>

int main(int argc, char *argv) {
    random_init();

    printf("Step 1: Create the neural network\n");
    int hidden_layer_sizes[2] = {4, 3};
    neural_network_t *nn = neural_network_create(2, 2, 2, hidden_layer_sizes);
    neural_network_layers_randomize(nn);
    neural_network_print(nn);
    
    printf("\nStep 2: Create the input data\n");
    matrix_t **inputs = (matrix_t **)malloc(N_CASES * sizeof(matrix_t *));
    for (int i = 0; i < N_CASES; i++) {
        inputs[i] = matrix_create(1, 2);
        inputs[i]->data[0] = random_double_between(-1, 1);
        inputs[i]->data[1] = random_double_between(-1, 1);
        matrix_print(inputs[i]);
    }

    printf("\nStep 3: Evaluate the inputs with the neural network\n");
    matrix_t **outputs = neural_network_evaluate(nn, N_CASES, inputs);
    neural_network_delete(nn);
    for (int i = 0; i < N_CASES; i++) {
        matrix_print(outputs[i]);

        free(inputs[i]);
        free(outputs[i]);
    }

    free(inputs);
    free(outputs);
}
