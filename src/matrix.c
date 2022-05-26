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