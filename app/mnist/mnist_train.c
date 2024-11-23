#include <time.h>
#include <string.h>

#include "mnist.h"
#include "../../src/neural_network.h"
#include "../../src/neural_network_train.h"
#include "../../src/neural_network_file.h"

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
#define BATCH_SIZE 100
#define TRAINING_DATA_COUNT 60000
#define TRAINING_PARAMETER 0.001

neural_network_t *initialize_neural_network();
void save_neural_network(neural_network_t *nn, time_t timer, int iteration, int do_overwrite);

//
// 'mnist_train.h' implementations
//

void mnist_train(const char *model_filename, int epochs, int do_overwrite) {
    unsigned char input_data_buffer[BATCH_SIZE * INPUT_SIZE];
    mnist_handle_t mnist_handle = mnist_handle_init(TRAINING_DATA_COUNT, BATCH_SIZE, input_data_buffer);
    mnist_images_load("datasets/mnist/train-images.idx3-ubyte", &mnist_handle);
    mnist_labels_load("datasets/mnist/train-labels.idx1-ubyte", &mnist_handle);

    double inputs_data[BATCH_SIZE * INPUT_SIZE];
    matrix_t inputs[BATCH_SIZE];
    matrix_initialize_multiple_from_array(inputs, BATCH_SIZE, 1, INPUT_SIZE, inputs_data);

    double output_map_data[OUTPUT_DATA_SIZE];
    mnist_initialize_output_data(output_map_data);
    matrix_t output_map[10];
    mnist_initialize_outputs(output_map, output_map_data);

    unsigned char outputs[BATCH_SIZE];

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
        printf("Epoch %d\n", i+1);
        int batch_size;
        while (batch_size = mnist_load_batch(&mnist_handle, inputs_data, outputs)) {
            for (int j = 0; j < batch_size; j++) {
                unsigned char label = outputs[j];
                matrix_t *label_matrix = &output_map[label];
                neural_network_train_case(neural_network, &inputs[j], label_matrix, TRAINING_PARAMETER);
            }
            printf("%d\r", mnist_handle.index);
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
    char filename[100];
    int offset = 13;
    strncpy(filename, "models/mnist-", 13);

    // timestamp
    struct tm* tm_info;
    tm_info = localtime(&timer);
    offset += strftime(filename+offset, 36, "TimeStamp(%Y-%m-%d-%H-%M-%S)", tm_info);

    // epoch, only labelled if not overwriting
    if (!do_overwrite)
        offset += sprintf(filename+offset, "-Epoch(%d)", epoch+1);

    strcpy(filename+offset, ".model.dynamic");
    printf("Saving to file '%s'.\n", filename);
    neural_network_save_dynamic(nn, filename);
}
