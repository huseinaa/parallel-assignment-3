#include <stdio.h>
#include <stdlib.h>

// Matrix dimensions
const int Rows = 3;  // Total rows
const int Columns = 4;  // Columns in first matrix and rows in second
const int Depth = 5;  // Columns in second matrix

#define BLOCK_DIM 16

// Function to perform matrix multiplication with tiling using OpenACC
void TiledMatrixMultiplication(int* Matrix1, int* Matrix2, int* Result, int Rows, int Columns, int Depth) {
    #pragma acc data copyin(Matrix1[0:Rows*Columns], Matrix2[0:Columns*Depth]) copyout(Result[0:Rows*Depth])
    {
        #pragma acc parallel loop tile(BLOCK_DIM, BLOCK_DIM) collapse(2)
        for (int r = 0; r < Rows; r++) {
            for (int c = 0; c < Depth; c++) {
                int accumulatedSum = 0;
                for (int block = 0; block < (Columns - 1) / BLOCK_DIM + 1; block++) {
                    #pragma acc loop reduction(+:accumulatedSum)
                    for (int i = 0; i < BLOCK_DIM; i++) {
                        int index1 = r * Columns + block * BLOCK_DIM + i;
                        int index2 = (block * BLOCK_DIM + i) * Depth + c;
                        if (block * BLOCK_DIM + i < Columns) {
                            accumulatedSum += Matrix1[index1] * Matrix2[index2];
                        }
                    }
                }
                Result[r * Depth + c] = accumulatedSum;
            }
        }
    }
}

int main() {
    int *inputA, *inputB, *outputC;
    inputA = (int*) malloc(Rows * Columns * sizeof(int));
    inputB = (int*) malloc(Columns * Depth * sizeof(int));
    outputC = (int*) malloc(Rows * Depth * sizeof(int));

    // Initialize the input matrices
    for (int i = 0; i < Rows * Columns; ++i) inputA[i] = i + 1;
    for (int i = 0; i < Columns * Depth; ++i) inputB[i] = i + 1;

    // Compute matrix multiplication using OpenACC
    TiledMatrixMultiplication(inputA, inputB, outputC, Rows, Columns, Depth);

    // Display the results
    printf("Computed Matrix C (OpenACC):\n");
    for (int i = 0; i < Rows; ++i) {
        for (int j = 0; j < Depth; ++j) {
            printf("%d ", outputC[i * Depth + j]);
        }
        printf("\n");
    }

    // Cleanup allocated memory
    free(inputA);
    free(inputB);
    free(outputC);

    return 0;
}