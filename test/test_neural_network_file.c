#include "../src/neural_network.h"
#include "../src/neural_network_file.h"
#include "../src/random.h"
#include "../src/error.h"

#include <string.h>
#include <stdio.h>

void cnd_print(int check, const char *message) {
    if (check)
        printf("%s", message);
}

int main(int argc, char *argv[]) {
    int do_log = 0;
    if (argc > 1)
        do_log = strcmp(argv[1], "do_log") == 0;
    
    random_init();

    cnd_print(do_log, "Step 1: Create neural network\n");
    int hidden_layer_sizes[] = {8, 4};
    char *activation_functions[3] = { "sigmoid", "sigmoid", "sigmoid" };
    neural_network_t *nn1 = neural_network_create(2, 2, 2, hidden_layer_sizes, activation_functions);
    neural_network_layers_randomize(nn1);
    if (do_log)
        neural_network_print(nn1);

    cnd_print(do_log, "\nStep 2: Save neural network\n");
    neural_network_save_dynamic(nn1, "models/test.model.dynamic");

    cnd_print(do_log, "\nStep 3: Load neural network\n");
    neural_network_t *nn2 = neural_network_load_dynamic("models/test.model.dynamic");
    if (do_log)
        neural_network_print(nn2);

    cnd_print(do_log, "\nStep 4: Compare saved and loaded neural networks\n");
    cnd_make_error(nn1->input_size != nn2->input_size, "Saved and loaded input sizes do not match.");
    cnd_make_error(nn1->output_size != nn2->output_size, "Saved and loaded output sizes do not match.");
    cnd_make_error(nn1->hidden_layer_count != nn2->hidden_layer_count, "Saved and loaded hidden layer counts do not match.");

    for (int i = 0; i < nn1->hidden_layer_count; i++) {
        cnd_make_error(nn1->hidden_layer_sizes[i] != nn2->hidden_layer_sizes[i], "Saved and loaded hidden layer sizes do not match.");
    }
    for (int i = 0; i < nn1->hidden_layer_count + 1; i++) {
        (strcmp(nn1->layers[i].activation_function.name, nn2->layers[i].activation_function.name) != 0, "Saved and loaded activation functions do not match.");
    }
    for (int i = 0; i < nn1->hidden_layer_count + 1; i++) {
        matrix_t *weights1 = &nn1->layers[i].weights;
        matrix_t *weights2 = &nn2->layers[i].weights;
        cnd_make_error(weights1->cols != weights2->cols || weights1->rows != weights2->rows, "Saved and loaded weight matrix dimensions do not match.");
        for (int j = 0; j < weights1->cols * weights1->rows; j++) {
            cnd_make_error(weights1->data[i] != weights2->data[i], "Saved and loaded weight data do not match.");
        }

        matrix_t *biases1 = &nn1->layers[i].biases;
        matrix_t *biases2 = &nn2->layers[i].biases;
        cnd_make_error(biases1->cols != biases2->cols || biases1->rows != biases2->rows, "Saved and loaded bias matrix dimensions do not match.");
        for (int j = 0; j < biases1->cols * biases1->rows; j++) {
            cnd_make_error(biases1->data[i] != biases2->data[i], "Saved and loaded weight data do not match.");
        }
    }

    neural_network_delete(nn1);
    neural_network_delete(nn2);
}
