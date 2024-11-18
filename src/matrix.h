#ifndef MATRIX
#define MATRIX

//
// 'matrix.h' definitions
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
 * Modify the inputted matrix to have the entered columns and rows, giving it a newly allocated data array.
 * @param mat The matrix to be modified.
 * @param cols The number of columns for the entered matrix.
 * @param rows The number of rows for the entered matrix.
 */
void matrix_create_i(matrix_t *mat, int cols, int rows);

/**
 * Initialize a matrix with the inputted columns and rows, with data from the inputted array and offset.
 * Increments the offset by the size of the array (cols * rows).
 * @param mat The matrix to be initialized.
 * @param cols The columns of the matrix.
 * @param rows The rows of the matrix.
 * @param array The array containing the data of the matrix.
 * @param offset The position in the array of the matrix's data. It gets incremented by (cols * rows).
*/
void matrix_initialize_from_array(matrix_t *mat, int cols, int rows, double *array, int *offset);

/**
 * Delete the matrix to prevent memory leaks.
 * @param mat The matrix to be deleted.
*/
void matrix_delete(matrix_t *mat);

/**
 * Print the matrix to the console.
 * @param mat The matrix to be printed.
*/
void matrix_print(matrix_t *mat);

/**
 * Print the matrix dimensions to the console.
 * @param mat The matrix to be printed.
*/
void matrix_print_short(matrix_t *mat);

/**
 * Set the element of the matrix at (col, row) to be 'val'.
 * @param mat The matrix to emplace a value into.
 * @param col The column of the value to be set.
 * @param row The row of the value to be set.
 * @param val The value to be placed at (col, row).
*/
void matrix_set(matrix_t *mat, int col, int row, double val);

/**
 * Get the element of the matrix at (col, row).
 * @param mat The matrix to retrieve an element from.
 * @param col The column of the matrix entry.
 * @param row The row of the matrix entry.
 * @return The value of the matrix at (col, row).
*/
double matrix_get(matrix_t *mat, int col, int row);

/**
 * Copy every element of the inputted matrix into a new matrix, which is returned.
 * @param mat The matrix to be copied.
 * @return An exact copy of the matrix inputted.
*/
matrix_t *matrix_copy_n(matrix_t *mat);

/**
 * Copy every element of matrix I into matrix O.
 * @param mat_I Matrix I.
 * @param mat_O Matrix O.
*/
void matrix_copy_o(matrix_t *mat_I, matrix_t *mat_O);

/**
 * Place every element (i,j) of the inputted matrix into a new matrix's (j,i), which is then returned.
 * @param mat The matrix to be transposed.
 * @return The matrix transpose of the inputted matrix.
*/
matrix_t *matrix_transpose_n(matrix_t *mat);

/**
 * Place the matrix transpose of matrix I into matrix O.
 * @param mat_I Matrix I.
 * @param mat_O Matrix O. The output matrix.
*/
void matrix_transpose_o(matrix_t *mat_I, matrix_t *mat_O);

/**
 * Perform a matrix multiplication of matrices A and B and return the result in a new matrix.
 * The columns of A must equal the rows of B.
 * @param mat_A Matrix A.
 * @param mat_B Matrix B.
 * @return The matrix multiplication of matrices A and B.
*/
matrix_t *matrix_multiply(matrix_t *mat_A, matrix_t *mat_B);

/**
 * Perform a multiplication of matrices A and B and return the result in matrix O.
 * The columns of A must equal the rows of B.
 * The dimensions of O must be (B cols, A rows).
 * @param mat_A Matrix A.
 * @param mat_B Matrix B.
 * @param mat_O Matrix O. The output matrix.
*/
void matrix_multiply_o(matrix_t *mat_A, matrix_t *mat_B, matrix_t *mat_O);

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
 * @param mat_A Matrix A. The output matrix.
 * @param mat_B Matrix B.
*/
void matrix_add_i(matrix_t *mat_A, matrix_t *mat_B);

/**
 * Subtract all entries of matrix B from matrix A, in-place in matrix A.
 * @param mat_A Matrix A. The output matrix.
 * @param mat_B Matrix B.
*/
void matrix_subtract_i(matrix_t *mat_A, matrix_t *mat_B);

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
