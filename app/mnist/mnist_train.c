#include <time.h>
#include <string.h>

#include "mnist.h"
#include "../../src/neural_network.h"
#include "../../src/neural_network_train.h"
#include "../../src/neural_network_file.h"
#include "../../src/random.h"

//
// 'mnist_train.c' definitions
//

// Neural Network parameters
#define INPUT_LAYER_SIZE IMAGE_WIDTH * IMAGE_WIDTH
// #define HIDDEN_LAYER_COUNT 2
// #define HIDDEN_LAYER_SIZE_1 14*14
// #define HIDDEN_LAYER_SIZE_2 7*7
#define HIDDEN_LAYER_COUNT 1
#define HIDDEN_LAYER_SIZE_1 32
#define OUTPUT_LAYER_SIZE OUTPUT_SIZE

// Training parameters
#define TRAINING_DATA_COUNT 60000
#define TRAINING_PARAMETER 0.001

neural_network_t *initialize_neural_network();
void save_neural_network(neural_network_t *nn, time_t timer, int iteration, int do_overwrite);

//
// 'mnist_train.h' implementations
//

void mnist_train(const char *model_filename, int epochs, int do_overwrite) {
    double input_data[BATCH_COUNT * INPUT_SIZE];
    mnist_handle_t mnist_handle = mnist_handle_init(TRAINING_DATA_COUNT, input_data);
    mnist_images_load("datasets/mnist/train-images.idx3-ubyte", &mnist_handle);
    mnist_labels_load("datasets/mnist/train-labels.idx1-ubyte", &mnist_handle);

    double output_data[OUTPUT_DATA_SIZE];
    mnist_initialize_output_data(output_data);
    matrix_t outputs[10];
    mnist_initialize_outputs(outputs, output_data);

    neural_network_t *neural_network = NULL;
    if (model_filename) {
        neural_network = neural_network_load_dynamic(model_filename);
        printf("Loaded model.\n");
    }
    else {
        neural_network = initialize_neural_network();
    }

    time_t timer = time(NULL);
    printf("Training...\n");
    for (int i = 0; i < epochs; i++) {
        printf("Iteration %d\n", i);
        while (mnist_has_batch(&mnist_handle)) {
            int batch_size = mnist_load_batch(&mnist_handle);
            for (int j = 0; j < batch_size; j++) {
                unsigned char label = mnist_handle.output_data[j];
                matrix_t *label_matrix = &outputs[label];
                neural_network_train_case(neural_network, &mnist_handle.inputs[j], label_matrix, TRAINING_PARAMETER);
            }
            printf("%d ", mnist_handle.index);
            fflush(stdout);
        }
        printf("\n");
        save_neural_network(neural_network, timer, i, do_overwrite);
        mnist_reset(&mnist_handle);
    }
    printf("Done!\n");
    mnist_handle_close(&mnist_handle);
}

//
// 'mnist_train.c' implementations
//

neural_network_t *initialize_neural_network() {
    int hidden_layer_sizes[HIDDEN_LAYER_COUNT] = { HIDDEN_LAYER_SIZE_1 };
    char *activation_function_names[HIDDEN_LAYER_COUNT+1] = { "sigmoid", "sigmoid" };
    neural_network_t *neural_network = neural_network_create(
        INPUT_LAYER_SIZE,
        OUTPUT_LAYER_SIZE,
        HIDDEN_LAYER_COUNT,
        hidden_layer_sizes,
        activation_function_names
    );
    neural_network_layers_randomize(neural_network);
    return neural_network;
}

void save_neural_network(neural_network_t *nn, time_t timer, int epoch, int do_overwrite) {
    char filename[64];
    int offset = 13;
    strncpy(filename, "models/mnist-", 13);

    // timestamp
    struct tm* tm_info;
    tm_info = localtime(&timer);
    offset += strftime(filename+offset, 26, "%Y-%m-%d-%H:%M:%S", tm_info);

    // epoch, only labelled if not overwriting
    if (!do_overwrite)
        offset += sprintf(filename+offset, "-%d", epoch);

    strcpy(filename+offset, ".model.dynamic");
    neural_network_save_dynamic(nn, filename);
}
