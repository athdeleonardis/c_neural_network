#ifndef MATRIX
#define MATRIX

//
// 'matrix.h' Definitions
//

/**
 * A function which takes a matrix entry and returns another matrix entry.
*/
typedef double (*matrix_map_t)(double);

/**
 *  @brief A 2D matrix.
*/
typedef struct {
    int cols;
    int rows;
    double *data;
} matrix_t;

/**
 * Create a matrix with the inputted number of columns and rows.
 * @param cols The number of columns of the returned matrix.
 * @param rows The number of rows of the returned matrix.
 * @returns A matrix with the inputted number of columns and rows.
*/
matrix_t *matrix_create(int cols, int rows);

/**
 * Delete the matrix to prevent memory leaks.
*/
void matrix_delete(matrix_t *);

/**
 * Print the matrix to the console.
*/
void matrix_print(matrix_t *);

/**
 * Set the element of the matrix at (col, row) to be 'val'.
 * @param col The column of the value to be set.
 * @param row The row of the value to be set.
 * @param val The value to be placed at (col, row).
*/
void matrix_set(matrix_t *, int col, int row, double val);

/**
 * Get the element of the matrix at (col, row).
 * @param col The column of the matrix entry.
 * @param row The row of the matrix entry.
 * @return The value of the matrix at (col, row).
*/
double matrix_get(matrix_t *, int col, int row);

/**
 * Copy every element of the inputted matrix into the returned matrix.
 * @return An exact copy of the matrix inputted.
*/
matrix_t *matrix_copy(matrix_t *);

/**
 * Place every element (i,j) of the inputted matrix into the outputted matrix (j,i).
 * @return The matrix transpose of the inputted matrix.
*/
matrix_t *matrix_transpose(matrix_t *mat);

/**
 * Perform a matrix multiplication of matrices A and B and return the result in a new matrix.
 * The columns of A must equal the rows of B.
 * @param mat_A Matrix A.
 * @param mat_B Matrix B.
 * @return The matrix multiplication of matrices A and B.
*/
matrix_t *matrix_multiply(matrix_t *mat_A, matrix_t *mat_B);

/**
 * Scalar multiply every entry of matrix A and matrix B, storing the result in-place in matrix A.
 * @param mat_A Matrix A.
 * @param mat_B Matrix B.
*/
void matrix_multiply_scalar_i(matrix_t *mat_A, matrix_t *mat_B);

/**
 * Multiply matrix A onto matrix B, then add matrix X, returning the result.
 * @param mat_A Matrix A.
 * @param mat_B Matrix B.
 * @param mat_X Matrix X.
 * @return The matrix multiplication of matrices A and B, added to matrix X.
*/
matrix_t *matrix_multiply_add(matrix_t *mat_A, matrix_t *mat_B, matrix_t *mat_X);

/**
 * Add all entries of matrix B to matrix A, in-place in matrix A.
 * @param mat_A Matrix A.
 * @param mat_B Matrix B.
*/
void matrix_add_i(matrix_t *mat_A, matrix_t *mat_B);

/**
 * Apply a function to every entry of a matrix, in-place.
 * @param mat The target matrix.
 * @param map The function to apply to every entry of the matrix.
*/
void matrix_apply_function_i(matrix_t *mat, matrix_map_t map);

/**
 * Apply a function to every entry of a matrix, placing the result into a new matrix.
 * @param mat The target matrix.
 * @param map The function to apply to every entry of the matrix.
 * @return The new matrix containing all mapped entries of the inputted matrix.
*/
matrix_t *matrix_apply_function(matrix_t *mat, matrix_map_t map);

#endif
