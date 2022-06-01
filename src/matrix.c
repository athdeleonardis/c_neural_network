#include "matrix.h"
#include "error.h"

#include <stdio.h>
#include <stdlib.h>

//
// Auxilliary Definitions
//

int matrix_compare_size(matrix_t *, matrix_t *);
int matrix_cell_to_index(matrix_t *, int col, int row);
void matrix_data_delete(matrix_t *);

//
// Auxilliary Implementations
//

int matrix_compare_size(matrix_t *mat_A, matrix_t *mat_B) {
    return mat_A->cols != mat_B->cols || mat_A->rows != mat_B->rows;
}

int matrix_cell_to_index(matrix_t *mat, int col, int row) {
    col %= mat->cols;
    row %= mat->rows;
    return col + row * mat->cols;
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
    mat->data = (double *)malloc(cols * rows * sizeof(double));
    return mat;
}

matrix_t *matrix_copy(matrix_t *mat) {
    matrix_t *new_mat = matrix_create(mat->cols, mat->rows);
    for (int i = 0; i < new_mat->cols * new_mat->rows; i++)
        new_mat->data[i] = mat->data[i];
    return new_mat;
}

matrix_t *matrix_transpose(matrix_t *mat) {
    matrix_t *mat_new = matrix_create(mat->rows, mat->cols);
    for (int j = 0; j < mat_new->rows; j++) {
        for (int i = 0; i < mat_new->cols; i++) {
            matrix_set(mat_new, i, j, matrix_get(mat, j, i));
        }
    }
    return mat_new;
}

void matrix_delete(matrix_t *mat) {
    free(mat->data);
    free(mat);
}

void matrix_print(matrix_t *mat) {
    printf("Matrix: %dx%d [", mat->cols, mat->rows);
    for (int i = 0; i < mat->cols * mat->rows; i++) {
        printf("%lf ", mat->data[i]);
    }
    printf("]\n");
}

void matrix_set(matrix_t *mat, int col, int row, double data) {
    mat->data[matrix_cell_to_index(mat, col, row)] = data;
}

double matrix_get(matrix_t *mat, int col, int row) {
    return mat->data[matrix_cell_to_index(mat, col, row)];
}

matrix_t *matrix_multiply(matrix_t *mat_A, matrix_t *mat_B) {
    cnd_make_error(mat_A->cols != mat_B->rows, "Attempting to multiply incompatible matrices");
    int k_max = mat_A->cols;

    // mat_C takes mat_B's algebra functions
    matrix_t *mat_C = matrix_create(mat_B->cols, mat_A->rows);

    for (int j = 0; j < mat_C->rows; j++) {
        for (int i = 0; i < mat_C->cols; i++) {
            double sum = 0;
            for (int k = 0; k < k_max; k++) {
                sum += matrix_get(mat_A, k, j) * matrix_get(mat_B, i, k);
            }
            matrix_set(mat_C, i, j, sum);
        }
    }
    return mat_C;
}

matrix_t *matrix_multiply_scalar_i(matrix_t *mat_A, matrix_t *mat_B) {
    cnd_make_error(matrix_compare_size(mat_A, mat_B), "Attemping to scalar multiply icompatible matrices");
    for (int i = 0; i < mat_A->cols * mat_A->rows; i++)
        mat_A->data[i] *= mat_B->data[i];
}

matrix_t *matrix_multiply_add(matrix_t *mat_A, matrix_t *mat_B, matrix_t *mat_X) {
    matrix_t *mat_C = matrix_multiply(mat_A, mat_B);
    matrix_add_i(mat_C, mat_X);
    return mat_C;
}

void matrix_add_i(matrix_t *mat_A, matrix_t* mat_B) {
    cnd_make_error(matrix_compare_size(mat_A, mat_B), "Attemping to add different sized matrices");
    for (int i = 0; i < mat_A->cols * mat_A->rows; i++)
        mat_A->data[i] += mat_B->data[i];
}

void matrix_apply_function_i(matrix_t *mat, matrix_map_t map) {
    for (int i = 0; i < mat->cols * mat->rows; i++)
        mat->data[i] = map(mat->data[i]);
}

void matrix_apply_function(matrix_t *mat, matrix_map_t map) {
    matrix_t *new_mat = matrix_create(mat->cols, mat->rows);
    for (int i = 0; i < mat->cols * mat->rows; i++)
        new_mat->data[i] = map(mat->data[i]);
}
