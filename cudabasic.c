#include <iostream>

// Matrix size definitions
const int RowCount = 3;  // Rows in the first matrix
const int ColCount = 4;  // Columns in the first matrix (and rows in the second)
const int DepthCount = 5;  // Columns in the second matrix

// CUDA kernel to perform straightforward matrix multiplication
__global__ void SimpleMatrixProduct(int* Mat1, int* Mat2, int* ResultMatrix, int RowCount, int ColCount, int DepthCount) {
    // Determine the row and column from thread and block indices
    int rowIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int colIndex = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate matrix product for each element
    if (rowIndex < RowCount && colIndex < DepthCount) {
        int elementSum = 0;
        for (int k = 0; k < ColCount; ++k) {
            elementSum += Mat1[rowIndex * ColCount + k] * Mat2[k * DepthCount + colIndex];
        }
        ResultMatrix[rowIndex * DepthCount + colIndex] = elementSum;
    }
}

int main() {
    // Allocate host memory
    int *hostMat1, *hostMat2, *hostResult;
    hostMat1 = new int[RowCount * ColCount];
    hostMat2 = new int[ColCount * DepthCount];
    hostResult = new int[RowCount * DepthCount];

    // Initialize the matrices with sequential values
    for (int i = 0; i < RowCount * ColCount; ++i) hostMat1[i] = i + 1;
    for (int i = 0; i < ColCount * DepthCount; ++i) hostMat2[i] = i + 1;

    // Allocate memory on the device
    int *devMat1, *devMat2, *devResult;
    cudaMalloc((void**)&devMat1, RowCount * ColCount * sizeof(int));
    cudaMalloc((void**)&devMat2, ColCount * DepthCount * sizeof(int));
    cudaMalloc((void**)&devResult, RowCount * DepthCount * sizeof(int));

    // Copy matrices to device memory
    cudaMemcpy(devMat1, hostMat1, RowCount * ColCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(devMat2, hostMat2, ColCount * DepthCount * sizeof(int), cudaMemcpyHostToDevice);

    // Set up dimensions for grid and blocks
    dim3 gridDimensions((DepthCount - 1) / 16 + 1, (RowCount - 1) / 16 + 1);
    dim3 blockDimensions(16, 16);

    // Execute the matrix multiplication kernel
    SimpleMatrixProduct<<<gridDimensions, blockDimensions>>>(devMat1, devMat2, devResult, RowCount, ColCount, DepthCount);

    // Copy the resulting matrix back to host memory
    cudaMemcpy(hostResult, devResult, RowCount * DepthCount * sizeof(int), cudaMemcpyDeviceToHost);

    // Display the resulting matrix
    std::cout << "Resultant Matrix:\n";
    for (int i = 0; i < RowCount; ++i) {
        for (int j = 0; j < DepthCount; ++j) {
            std::cout << hostResult[i * DepthCount + j] << " ";
        }
        std::cout << "\n";
    }

    // Clean up allocated memory
    delete[] hostMat1;
    delete[] hostMat2;
    delete[] hostResult;
    cudaFree(devMat1);
    cudaFree(devMat2);
    cudaFree(devResult);

    return 0;
}