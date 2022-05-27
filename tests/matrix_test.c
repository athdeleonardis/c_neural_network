#include "../src/matrix.h"
#include <stdio.h>

int int_printer(int n) {
    printf("%d ", n);
}

matrix_t *matrix_from_cmd(const char *name) {
    printf("Enter cols and rows of %s: ", name);
    int cols;
    int rows;
    scanf("%d", &cols);
    scanf("%d", &rows);

    matrix_t *mat = matrix_create(cols, rows);

    printf("Enter data for %s: ", name);
    for (int j = 0; j < rows; j++) {
        for (int i = 0; i < cols; i++) {
            int n;
            scanf("%d", &n);
            matrix_set(mat, i, j, (double)n);
        }
    }

    return mat;
}

int main() {
    matrix_t *mat_A = matrix_from_cmd("A");
    matrix_t *mat_B = matrix_from_cmd("B");
    matrix_t *mat_C = matrix_multiply(mat_A, mat_B);
    matrix_print(mat_A);
    matrix_print(mat_B);
    matrix_print(mat_C);
}
