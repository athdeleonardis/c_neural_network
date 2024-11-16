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

#define INPUT_SIZE 2
#define HIDDEN_LAYER_COUNT 1
#define HIDDEN_LAYER_SIZES_1 4
#define OUTPUT_SIZE 1

#define N_INPUT_CASES 4

#define N_TRAINING_CASES 1000000

int main(int argc, char *argv[]) {
    random_init();
    
    printf("Step 1: Create the neural network\n");
    int hidden_layer_sizes[HIDDEN_LAYER_COUNT] = { HIDDEN_LAYER_SIZES_1 };
    char *activation_functions[2] = { "sigmoid", "sigmoid" };
    neural_network_t *nn = neural_network_create(
        INPUT_SIZE,
        OUTPUT_SIZE,
        HIDDEN_LAYER_COUNT,
        hidden_layer_sizes,
        activation_functions
    );
    neural_network_layers_randomize(nn);
    neural_network_print(nn);
    
    printf("\nStep 2: Create the input and output data\n");
    // Space for all input cases, output cases, outputs calculated before training, outputs calculated after training.
    double case_data[(INPUT_SIZE + OUTPUT_SIZE * 3) * N_INPUT_CASES];
    matrix_t inputs[N_INPUT_CASES];
    matrix_t outputs[N_INPUT_CASES];
    matrix_t outputs_calculated_before[N_INPUT_CASES];
    matrix_t outputs_calculated_after[N_INPUT_CASES];
    int case_data_offset = 0;
    for (int i = 0; i < N_INPUT_CASES; i++) {
        case_data[case_data_offset] = i % 2;
        case_data[case_data_offset + 1] = (i / 2) % 2;
        matrix_initialize_from_array(inputs+i, 1, INPUT_SIZE, case_data, &case_data_offset);
        // i = 0 -> 0, i = 3 -> 0
        // i = 1 -> 1, i = 2 -> 1
        // (2*i+1) % 2
        case_data[case_data_offset] = (2 * i + 1) % 2;
        matrix_initialize_from_array(outputs+i, 1, OUTPUT_SIZE, case_data, &case_data_offset);
    }
    for (int i = 0; i < N_INPUT_CASES; i++) {
        matrix_initialize_from_array(outputs_calculated_before+i, 1, OUTPUT_SIZE, case_data, &case_data_offset);
    }
    for (int i = 0; i < N_INPUT_CASES; i++) {
        matrix_initialize_from_array(outputs_calculated_after+i, 1, OUTPUT_SIZE, case_data, &case_data_offset);
    }

    neural_network_evaluate(nn, 4, inputs, outputs_calculated_before);
    for (int i = 0; i < 4; i++) {
        printf("Case %d:\nInput: ", i);
        matrix_print(inputs+i);
        printf("Output expected: ");
        matrix_print(outputs+i);
        printf("Output calculated: ");
        matrix_print(outputs_calculated_before+i);
    }

    printf("\nStep 3: Train the neural network on the data\n");
    for (int i = 0; i < N_TRAINING_CASES; i++) {
        int num_case = random_int_between(0, 4);
        neural_network_train_case(nn, &inputs[num_case], &outputs[num_case], 0.01);
    }

    neural_network_print(nn);

    neural_network_evaluate(nn, N_INPUT_CASES, inputs, outputs_calculated_after);
    neural_network_delete(nn);
    for (int i = 0; i < 4; i++) {
        printf("Case %d:\nInput: ", i);
        matrix_print(inputs+i);
        printf("Output expected: ");
        matrix_print(outputs+i);
        printf("Output calculated: ");
        matrix_print(outputs_calculated_after+i);
    }

    printf("Errors: [");
    for (int i = 0; i < 4; i++) {
        double error_percentage = (matrix_get(&outputs_calculated_after[i], 1, 0) - matrix_get(&outputs[i], 1, 0)) * 100;
        if (error_percentage < 0)
            error_percentage = -error_percentage;
        if (i)
            printf(", ");
        printf("%.01f%%", error_percentage);
    }
    printf("]\n");
}
