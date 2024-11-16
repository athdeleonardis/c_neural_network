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
#define HIDDEN_LAYER_COUNT 2
#define HIDDEN_LAYER_SIZE_1 14*14
#define HIDDEN_LAYER_SIZE_2 7*7
#define OUTPUT_LAYER_SIZE OUTPUT_SIZE

#define TRAINING_DATA_COUNT 60000
#define TRAINING_PARAMETER 0.01

neural_network_t *initialize_neural_network();

//
// 'mnist_train.h' implementations
//

void mnist_train() {
    mnist_handle_t mnist_handle = mnist_handle_init(TRAINING_DATA_COUNT);
    mnist_images_load("datasets/mnist/train-images.idx3-ubyte", &mnist_handle);
    mnist_labels_load("datasets/mnist/train-labels.idx1-ubyte", &mnist_handle);

    double output_data[OUTPUT_DATA_SIZE];
    initialize_output_data(output_data);
    matrix_t outputs[10];
    initialize_outputs(outputs, output_data);

    neural_network_t *neural_network = initialize_neural_network();

    printf("Training...\n");
    while (mnist_has_batch(&mnist_handle)) {
        int batch_size = mnist_load_batch(&mnist_handle);
        for (int i = 0; i < batch_size; i++) {
            neural_network_train_case(neural_network, &mnist_handle.inputs[i], &outputs[mnist_handle.output_data[i]], TRAINING_PARAMETER);
        }
        printf("%d\n", mnist_handle.index);
    }
    printf("Done!\n");
    mnist_handle_close(&mnist_handle);

    neural_network_save_dynamic(neural_network, "models/mnist.model.dynamic");
}

//
// 'mnist_train.c' implementations
//

neural_network_t *initialize_neural_network() {
    int hidden_layer_sizes[HIDDEN_LAYER_COUNT] = { HIDDEN_LAYER_SIZE_1, HIDDEN_LAYER_SIZE_2 };
    char *activation_function_names[HIDDEN_LAYER_COUNT+1] = { "sigmoid", "sigmoid", "sigmoid" };
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
