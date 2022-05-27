#include "neural_network_train.h"

#include <stdlib.h>

//
// Auxilliary Definitions
//

typedef struct {
    matrix_t **pre_activation;
    matrix_t **outputs;
} neural_network_eval_t;

neural_network_eval_t new_eval(int hidden_layer_count);
void del_eval(neural_network_eval_t eval, int hidden_layer_count);
void get_eval(neural_network_t *, matrix_t *input, neural_network_eval_t);
void apply_eval(neural_network_t *, matrix_t *output, neural_network_eval_t, double p);

//
// Auxilliary Implementations
//

neural_network_eval_t new_eval(int hidden_layer_count) {
    neural_network_eval_t eval = {};
    eval.pre_activation = (matrix_t **)malloc((hidden_layer_count + 1)*sizeof(matrix_t));
    eval.outputs = (matrix_t **)malloc((hidden_layer_count + 1)*sizeof(matrix_t));
    return eval;
}

void del_eval(neural_network_eval_t eval, int hidden_layer_count) {
    for (int i = 0; i < hidden_layer_count + 1; i++) {
        matrix_delete(eval.pre_activation[i]);
        matrix_delete(eval.outputs[i]);
    }
    free(eval.pre_activation);
    free(eval.outputs);
}

void do_eval(neural_network_t *nn, matrix_t *input, neural_network_eval_t eval) {
    matrix_t *prev_outputs;
    for (int i = 0; i < nn->hidden_layer_count + 1; i++) {
        if (i)
            prev_outputs = eval.outputs[i-1];
        else
            prev_outputs = input;
        eval.pre_activation[i] = matrix_multiply_add(nn->layers[i].weights, prev_outputs, nn->layers[i].biases);
        eval.outputs[i] = matrix_copy(eval.pre_activation[i]);
        matrix_apply_function(eval.outputs[i], nn->layers[i].activation_function.function);
    }
}

void apply_eval(neural_network_t *nn, matrix_t *output, neural_network_eval_t eval, double p) {
    
}

//
// 'neural_network_train.h' Implementations
//

void neural_network_train_case(neural_network_t *nn, matrix_t *input, matrix_t *output, double p) {
    neural_network_eval_t eval = new_eval(nn->hidden_layer_count);
    do_eval(nn, input, eval);
    apply_eval(nn, output, eval, p);
    del_eval(eval, nn->hidden_layer_count);
}
