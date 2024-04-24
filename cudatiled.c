#include <iostream>

// Matrix dimensions constants
const int Rows = 3;  // Rows count
const int Columns = 4;  // Columns count
const int Depth = 5;  // Columns in second matrix

#define BLOCK_SIZE 16

// CUDA kernel to perform matrix multiplication using tiling
__global__ void TiledMatrixMultiply(int* MatA, int* MatB, int* MatC, int Rows, int Columns, int Depth) {
    // Indices for threads
    int idx_row = blockIdx.y * blockDim.y + threadIdx.y;
    int idx_col = blockIdx.x * blockDim.x + threadIdx.x;

    // Declare shared memory for matrix blocks
    __shared__ int BlockA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int BlockB[BLOCK_SIZE][BLOCK_SIZE];

    // Initialize the accumulation variable
    int productSum = 0;

    // Loop over the matrix tiles
    for (int block = 0; block < (Columns - 1) / BLOCK_SIZE + 1; ++block) {
        // Load matrix tiles into shared memory
        if (idx_row < Rows && block * BLOCK_SIZE + threadIdx.x < Columns)
            BlockA[threadIdx.y][threadIdx.x] = MatA[idx_row * Columns + block * BLOCK_SIZE + threadIdx.x];
        else
            BlockA[threadIdx.y][threadIdx.x] = 0;

        if (idx_col < Depth && block * BLOCK_SIZE + threadIdx.y < Columns)
            BlockB[threadIdx.y][threadIdx.x] = MatB[(block * BLOCK_SIZE + threadIdx.y) * Depth + idx_col];
        else
            BlockB[threadIdx.y][threadIdx.x] = 0;

        // Sync threads to ensure all data is loaded
        __syncthreads();

        // Multiply and accumulate the product
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            productSum += BlockA[threadIdx.y][k] * BlockB[k][threadIdx.x];
        }

        // Sync threads to ensure all calculations are done before next tile
        __syncthreads();
    }

    // Store the computed result in the output matrix
    if (idx_row < Rows && idx_col < Depth) {
        MatC[idx_row * Depth + idx_col] = productSum;
    }
}

int main() {
    // Host memory allocation
    int *hostA, *hostB, *hostC;
    hostA = new int[Rows * Columns];
    hostB = new int[Columns * Depth];
    hostC = new int[Rows * Depth];

    // Initialize matrices with values
    for (int i = 0; i < Rows * Columns; ++i) hostA[i] = i + 1;
    for (int i = 0; i < Columns * Depth; ++i) hostB[i] = i + 1;

    // Device memory allocation
    int *deviceA, *deviceB, *deviceC;
    cudaMalloc((void**)&deviceA, Rows * Columns * sizeof(int));
    cudaMalloc((void**)&deviceB, Columns * Depth * sizeof(int));
    cudaMalloc((void**)&deviceC, Rows * Depth * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(deviceA, hostA, Rows * Columns * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, Columns * Depth * sizeof(int), cudaMemcpyHostToDevice);

    // Set grid and block dimensions
    dim3 gridDim((Depth - 1) / BLOCK_SIZE + 1, (Rows - 1) / BLOCK_SIZE + 1);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    // Execute the kernel
    TiledMatrixMultiply<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, Rows, Columns, Depth);

    // Retrieve results from the device to host
    cudaMemcpy(hostC, deviceC, Rows * Depth * sizeof(int), cudaMemcpyDeviceToHost);

    // Output the resulting matrix
    std::cout << "Resultant Matrix C:\n";
    for (int i = 0; i < Rows; ++i) {
        for (int j = 0; j < Depth; ++j) {
            std::cout << hostC[i * Depth + j] << " ";
        }
        std::cout << "\n";
    }

    // Memory cleanup
    delete[] hostA;
    delete[] hostB;
    delete[] hostC;
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    return 0;
}