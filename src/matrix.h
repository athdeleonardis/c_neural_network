#ifndef MATRIX
#define MATRIX

//
// 'matrix.h' Definitions
//

typedef struct {
    int cols;
    int rows;
    double *data;
} matrix_t;

matrix_t *matrix_create(int cols, int rows);
void matrix_delete(matrix_t *);
void matrix_print(matrix_t *);
void matrix_set(matrix_t *, int col, int row, double);
double matrix_get(matrix_t *, int col, int row);
matrix_t *matrix_multiply(matrix_t *mat_A, matrix_t *mat_B);
matrix_t *matrix_multiply_add(matrix_t *mat_A, matrix_t *mat_B, matrix_t *mat_X);

#endif