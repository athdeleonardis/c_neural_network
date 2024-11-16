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

void mnist_test() {
    mnist_handle_t mnist_handle = mnist_handle_init(TESTING_DATA_COUNT);
    mnist_images_load("datasets/mnist/t10k-images.idx3-ubyte", &mnist_handle);
    mnist_labels_load("datasets/mnist/t10k-labels.idx1-ubyte", &mnist_handle);

    neural_network_t *neural_network = neural_network_load_dynamic("models/mnist.model.dynamic");

    int num_correct = 0;
    while (mnist_has_batch(&mnist_handle)) {
        int num_cases = mnist_load_batch(&mnist_handle);
    }
    mnist_handle_close(&mnist_handle);
}