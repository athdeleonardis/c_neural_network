#ifndef MATRIX
#define MATRIX

//
// 'matrix.h' Definitions
//

typedef double (*matrix_map_t)(double);

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

matrix_t *matrix_copy(matrix_t *);
matrix_t *matrix_transpose(matrix_t *);
matrix_t *matrix_multiply(matrix_t *mat_A, matrix_t *mat_B);
matrix_t *matrix_multiply_scalar_i(matrix_t *mat_A, matrix_t *mat_B);
matrix_t *matrix_multiply_add(matrix_t *mat_A, matrix_t *mat_B, matrix_t *mat_X);
void matrix_add_i(matrix_t *mat_A, matrix_t *mat_B);
void matrix_apply_function_i(matrix_t *, matrix_map_t);
void matrix_apply_function(matrix_t *, matrix_map_t);

#endif
