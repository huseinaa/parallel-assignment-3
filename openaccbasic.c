#include <stdio.h>
#include <stdlib.h>
#include <openacc.h>

// Constants for matrix sizes
const int RowSize = 3;  // Rows in the first matrix
const int ColumnSize = 4;  // Columns in the first matrix, Rows in the second
const int ResultColumns = 5;  // Columns in the result matrix

// Perform matrix multiplication using OpenACC
void PerformMatrixMultiplicationOpenACC(int* InputMatrix1, int* InputMatrix2, int* ResultMatrix, int RowSize, int ColumnSize, int ResultColumns) {
    #pragma acc data copyin(InputMatrix1[0:RowSize*ColumnSize], InputMatrix2[0:ColumnSize*ResultColumns]) copyout(ResultMatrix[0:RowSize*ResultColumns])
    {
        #pragma acc parallel num_gangs(RowSize) num_workers(1)
        {
            #pragma acc loop gang
            for (int r = 0; r < RowSize; ++r) {
                #pragma acc loop worker
                for (int c = 0; c < ResultColumns; ++c) {
                    int total = 0;
                    #pragma acc loop vector reduction(+:total)
                    for (int j = 0; j < ColumnSize; ++j) {
                        total += InputMatrix1[r * ColumnSize + j] * InputMatrix2[j * ResultColumns + c];
                    }
                    ResultMatrix[r * ResultColumns + c] = total;
                }
            }
        }
    }
}

int main() {
    int *matrix1, *matrix2, *matrixResult;
    matrix1 = (int*) malloc(RowSize * ColumnSize * sizeof(int));
    matrix2 = (int*) malloc(ColumnSize * ResultColumns * sizeof(int));
    matrixResult = (int*) malloc(RowSize * ResultColumns * sizeof(int));

    // Initialize matrices with incremental values
    for (int i = 0; i < RowSize * ColumnSize; ++i) matrix1[i] = i + 1;
    for (int i = 0; i < ColumnSize * ResultColumns; ++i) matrix2[i] = i + 1;

    // Execute the matrix multiplication
    PerformMatrixMultiplicationOpenACC(matrix1, matrix2, matrixResult, RowSize, ColumnSize, ResultColumns);

    // Display the resulting matrix
    printf("Resulting Matrix (OpenACC):\n");
    for (int i = 0; i < RowSize; ++i) {
        for (int c = 0; c < ResultColumns; ++c) {
            printf("%d ", matrixResult[i * ResultColumns + c]);
        }
        printf("\n");
    }

    // Clean up memory
    free(matrix1);
    free(matrix2);
    free(matrixResult);

    return 0;
}