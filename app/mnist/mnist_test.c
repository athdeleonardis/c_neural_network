#include "mnist.h"
#include "mnist_test.h"
#include "../../src/neural_network.h"
#include "../../src/neural_network_file.h"

//
// 'mnist_test.c' definitions
//

#define BATCH_SIZE 100
#define TESTING_DATA_COUNT 10000

//
// 'mnist_test.h' implementations
//

void mnist_test(const char *model_filename) {
    unsigned char input_data_buffer[BATCH_SIZE * INPUT_SIZE];
    mnist_handle_t mnist_handle = mnist_handle_init(TESTING_DATA_COUNT, BATCH_SIZE, input_data_buffer);
    mnist_images_load("datasets/mnist/t10k-images.idx3-ubyte", &mnist_handle);
    mnist_labels_load("datasets/mnist/t10k-labels.idx1-ubyte", &mnist_handle);

    neural_network_t *neural_network = neural_network_load_dynamic(model_filename);

    // Storage for inputs loaded from the MNIST handle.
    double inputs_data[BATCH_SIZE * INPUT_SIZE];
    matrix_t inputs[BATCH_SIZE];
    matrix_initialize_multiple_from_array(inputs, BATCH_SIZE, 1, INPUT_SIZE, inputs_data);

    // Storage for outputs calculated through 'neural_network_evaluate'.
    double outputs_calculated_batch_data[BATCH_SIZE * OUTPUT_SIZE];
    matrix_t outputs_calculated_batch[BATCH_SIZE];
    matrix_initialize_multiple_from_array(outputs_calculated_batch, BATCH_SIZE, 1, OUTPUT_SIZE, outputs_calculated_batch_data);

    // Storage of output labels loaded from the MNIST handle.
    unsigned char outputs[BATCH_SIZE];

    int num_correct = 0;
    printf("Testing:\n");
    int num_cases;
    while (num_cases = mnist_load_batch(&mnist_handle, inputs_data, outputs)) {
        neural_network_evaluate(neural_network, num_cases, inputs, outputs_calculated_batch);
        for (int i = 0; i < num_cases; i++) {
            unsigned char output_number_calculated = mnist_output_to_number(&outputs_calculated_batch[i]);
            unsigned char output_number = outputs[i];
            num_correct += (output_number_calculated == output_number);
        }
        printf("Cases tested: %d\n", mnist_handle.index);
        printf("Num correct: %d\n", num_correct);
    }
    mnist_handle_close(&mnist_handle);

    printf("Done!\n");
    printf("Perctentage correct: %.01f\n", ((double)num_correct * 100) / TESTING_DATA_COUNT);
}
