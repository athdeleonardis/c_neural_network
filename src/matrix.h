#ifndef MATRIX
#define MATRIX

//
// 'matrix.h' Definitions
//

#include "data.h"

typedef struct {
    int cols;
    int rows;
    data_t *data;
    data_adder_t data_adder;
    data_multiplier_t data_multiplier;
    data_t data_zero;
    data_deleter_t data_deleter;
    data_printer_t data_printer;
} matrix_t;

matrix_t *matrix_create(int cols, int rows);
void matrix_set_algebra(matrix_t *, data_adder_t, data_multiplier_t, data_t data_zero);
void matrix_set_data_functions(matrix_t *, data_deleter_t, data_printer_t);
void matrix_delete(matrix_t *);
void matrix_print(matrix_t *);
void matrix_set(matrix_t *, int col, int row, data_t data);
data_t matrix_get(matrix_t *, int col, int row);
matrix_t *matrix_multiply(matrix_t *mat_A, matrix_t *mat_B);

#endif