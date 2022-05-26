#include "matrix.h"
#include "error.h"

#include <stdio.h>
#include <stdlib.h>

//
// Auxilliary Definitions
//

int matrix_cell_to_index(matrix_t *, int col, int row);
void matrix_data_delete(matrix_t *);

//
// Auxilliary Implementations
//

int matrix_cell_to_index(matrix_t *mat, int col, int row) {
    col %= mat->cols;
    row %= mat->rows;
    return col + row * mat->cols;
}

void matrix_data_delete(matrix_t *mat) {
    for (int i = 0; i < mat->cols * mat->rows; i++) {
        mat->data_deleter(mat->data[i]);
    }
}

//
// 'matrix.h' Implementations
//

matrix_t *matrix_create(int cols, int rows) {
    cnd_make_error(cols < 1, "Matrix cols must be >= 1");
    cnd_make_error(rows < 1, "Matrix rows must be >= 1");
    
    matrix_t *mat = (matrix_t *)malloc(sizeof(matrix_t));
    mat->cols = cols;
    mat->rows = rows;
    mat->data = (data_t *)malloc(cols * rows * sizeof(data_t));
    return mat;
}

void matrix_set_algebra(matrix_t *mat, data_adder_t data_adder, data_multiplier_t data_multiplier, data_t data_zero) {
    mat->data_adder = data_adder;
    mat->data_multiplier = data_multiplier;
    mat->data_zero = data_zero;
}

void matrix_set_data_functions(matrix_t *mat, data_deleter_t data_deleter, data_printer_t data_printer) {
    mat->data_deleter = data_deleter;
    mat->data_printer = data_printer;
}

void matrix_delete(matrix_t *mat) {
    if (mat->data_deleter != NULL)
        matrix_data_delete(mat);
    free(mat->data);
    free(mat);
}

void matrix_print(matrix_t *mat) {
    printf("Matrix: %dx%d [", mat->cols, mat->rows);
    for (int i = 0; i < mat->cols * mat->rows; i++) {
        mat->data_printer(mat->data[i]);
    }
    printf("]\n");
}

void matrix_set(matrix_t *mat, int col, int row, data_t data) {
    mat->data[matrix_cell_to_index(mat, col, row)] = data;
}

data_t matrix_get(matrix_t *mat, int col, int row) {
    return mat->data[matrix_cell_to_index(mat, col, row)];
}

matrix_t *matrix_multiply(matrix_t *mat_A, matrix_t *mat_B) {
    cnd_make_error(mat_A->cols != mat_B->rows, "Attempting to multiply incompatible matrices");
    cnd_make_error(mat_A->data_adder == NULL || mat_A->data_multiplier == NULL,
        "Attempting to multiply matrices without algebra");
    int k_max = mat_A->cols;

    // mat_C takes mat_B's algebra functions
    matrix_t *mat_C = matrix_create(mat_B->cols, mat_A->rows);
    matrix_set_algebra(mat_C, mat_B->data_adder, mat_B->data_multiplier, mat_B->data_zero);
    matrix_set_data_functions(mat_C, mat_B->data_deleter, mat_B->data_printer);

    for (int j = 0; j < mat_C->rows; j++) {
        for (int i = 0; i < mat_C->cols; i++) {
            data_t sum = mat_A->data_zero;
            for (int k = 0; k < k_max; k++) {
                data_t data_A = matrix_get(mat_A, k, j);
                data_t data_B = matrix_get(mat_B, i, k);
                // mat_C uses mat_A's algebra functions to calculate it's values
                data_t to_add = mat_A->data_multiplier(data_A, data_B);
                sum = mat_A->data_adder(sum, to_add);
            }
            matrix_set(mat_C, i, j, sum);
        }
    }
    return mat_C;
}