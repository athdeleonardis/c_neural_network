#include "mnist.h"
#include "mnist_test.h"
#include "../../src/neural_network.h"
#include "../../src/neural_network_file.h"

//
// 'mnist_test.c' definitions
//

#define TESTING_DATA_COUNT 10000

//
// 'mnist_test.h' implementations
//

void mnist_test(const char *model_filename) {
    double input_data[BATCH_COUNT * INPUT_SIZE];
    mnist_handle_t mnist_handle = mnist_handle_init(TESTING_DATA_COUNT, input_data);
    mnist_images_load("datasets/mnist/t10k-images.idx3-ubyte", &mnist_handle);
    mnist_labels_load("datasets/mnist/t10k-labels.idx1-ubyte", &mnist_handle);

    neural_network_t *neural_network = neural_network_load_dynamic(model_filename);

    double outputs_calculated_batch_data[BATCH_COUNT * OUTPUT_SIZE];
    matrix_t outputs_calculated_batch[BATCH_COUNT];
    int outputs_calculated_batch_data_offset = 0;
    for (int i = 0; i < BATCH_COUNT; i++) {
        matrix_initialize_from_array(&outputs_calculated_batch[i], 1, OUTPUT_SIZE, outputs_calculated_batch_data, &outputs_calculated_batch_data_offset);
    }

    int num_correct = 0;
    printf("Testing:\n");
    while (mnist_has_batch(&mnist_handle)) {
        int num_cases = mnist_load_batch(&mnist_handle);
        neural_network_evaluate(neural_network, num_cases, mnist_handle.inputs, outputs_calculated_batch);
        for (int i = 0; i < num_cases; i++) {
            unsigned char output_number_calculated = mnist_output_to_number(&outputs_calculated_batch[i]);
            unsigned char output_number = mnist_handle.output_data[i];
            num_correct += (output_number_calculated == output_number);
        }
        printf("Cases tested: %d\n", mnist_handle.index);
        printf("Num correct: %d\n", num_correct);
    }
    mnist_handle_close(&mnist_handle);

    printf("Done!\n");
    printf("Perctentage correct: %.01f\n", ((double)num_correct * 100) / TESTING_DATA_COUNT);
}
