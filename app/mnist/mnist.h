#include <stdio.h>
#include <stdint.h>
#include "../../src/matrix.h"

//
// 'mnist.h' definitions
//

#define IMAGE_WIDTH 28
#define INPUT_SIZE IMAGE_WIDTH * IMAGE_WIDTH
#define BATCH_COUNT 100
#define BATCH_INPUT_SIZE BATCH_COUNT * INPUT_SIZE
#define OUTPUT_SIZE 10
#define OUTPUT_DATA_SIZE OUTPUT_SIZE * OUTPUT_SIZE

typedef struct {
    FILE *inputs_file;
    FILE *outputs_file;
    int32_t num_cases;
    int32_t index;
    unsigned char input_data_unformatted[BATCH_INPUT_SIZE];
    double input_data[BATCH_INPUT_SIZE];
    matrix_t inputs[BATCH_COUNT];
    unsigned char output_data[BATCH_COUNT];
} mnist_handle_t;

mnist_handle_t mnist_handle_init(int32_t num_cases);
void mnist_handle_close(mnist_handle_t *handle);
void mnist_images_load(const char *filename, mnist_handle_t *handle);
void mnist_labels_load(const char *filename, mnist_handle_t *handle);
int mnist_has_batch(mnist_handle_t *handle);
int mnist_load_batch(mnist_handle_t *handle);
void initialize_output_data(double *data);
void initialize_outputs(matrix_t *outputs, double *data);
