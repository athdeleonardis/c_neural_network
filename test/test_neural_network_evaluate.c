#define N_CASES 6

#include "../src/neural_network.h"
#include "../src/random.h"
#include "../src/matrix.h"

#include <stdlib.h>
#include <stdio.h>

#define INPUT_SIZE 2
#define HIDDEN_LAYER_COUNT 2
#define HIDDEN_LAYER_SIZE_1 4
#define HIDDEN_LAYER_SIZE_2 3
#define OUTPUT_SIZE 2

int main(int argc, char *argv) {
    random_init();

    printf("Step 1: Create the neural network\n");
    int hidden_layer_sizes[HIDDEN_LAYER_COUNT] = { HIDDEN_LAYER_SIZE_1, HIDDEN_LAYER_SIZE_2 };
    char *activation_functions[3] = { "sigmoid", "sigmoid", "sigmoid" };
    neural_network_t *nn = neural_network_create(
        INPUT_SIZE, OUTPUT_SIZE,
        HIDDEN_LAYER_COUNT,
        hidden_layer_sizes,
        activation_functions
    );
    neural_network_layers_randomize(nn);
    neural_network_print(nn);
    
    printf("\nStep 2: Create the input data\n");
    double input_data[N_CASES * INPUT_SIZE];
    matrix_t inputs[N_CASES];
    int input_data_offset = 0;
    for (int i = 0; i < N_CASES; i++) {
        matrix_initialize_from_array(inputs+i, 1, INPUT_SIZE, input_data, &input_data_offset);
        for (int j = 0; j < INPUT_SIZE; j++) {
            inputs[i].data[j] = random_double_between(-1, 1);
        }
        matrix_print(inputs+i);
    }

    printf("\nStep 3: Evaluate the inputs with the neural network\n");
    double output_data[N_CASES * OUTPUT_SIZE];
    matrix_t outputs[N_CASES];
    int output_data_offset = 0;
    for (int i = 0; i < N_CASES; i++) {
        matrix_initialize_from_array(outputs+i, 1, OUTPUT_SIZE, output_data, &output_data_offset);
    }
    neural_network_evaluate(nn, N_CASES, inputs, outputs);
    neural_network_delete(nn);
    for (int i = 0; i < N_CASES; i++) {
        matrix_print(outputs+i);
    }
}
