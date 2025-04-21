#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define MAX_TEXT_SIZE 2000000  // Increased for 1M file
#define MAX_DISPLAY 20         // Maximum number of matches to display
#define THREADS_PER_BLOCK 256  // Number of threads per block
#define MAX_PATTERN_LENGTH 100 // Maximum pattern length

// CUDA kernel for computing hash values and initial matching
__global__ void compute_initial_hashes(char *d_text, int text_length, char *d_pattern, 
                                      int pattern_length, int pattern_hash, int d, int q, int h, 
                                      int *d_matches, int *d_positions, int *d_match_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i <= text_length - pattern_length) {
        // Compute hash for this text window
        int text_hash = 0;
        for (int j = 0; j < pattern_length; j++) {
            text_hash = (d * text_hash + d_text[i + j]) % q;
        }
        
        // Check if hash matches the pattern hash
        if (text_hash == pattern_hash) {
            // Check for exact character match
            bool is_match = true;
            for (int j = 0; j < pattern_length; j++) {
                if (d_text[i + j] != d_pattern[j]) {
                    is_match = false;
                    break;
                }
            }
            
            if (is_match) {
                // Atomic operation to safely update shared match count
                int idx = atomicAdd(d_match_count, 1);
                if (idx < MAX_DISPLAY) {
                    d_positions[idx] = i;
                }
                atomicAdd(d_matches, 1);
            }
        }
    }
}

// Function to compute pattern hash on CPU
int compute_hash(char *str, int length, int d, int q) {
    int i = 0;
    int hash_value = 0;

    for (i = 0; i < length; ++i) {
        hash_value = (d * hash_value + str[i]) % q;
    }

    return hash_value;
}

// Function to read text file content
char* read_text_file(const char* filename, int* length) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file: %s\n", filename);
        exit(1);
    }
    
    // Get file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    rewind(file);
    
    // Allocate buffer
    char* buffer = (char*)malloc(file_size + 1);
    if (buffer == NULL) {
        printf("Memory allocation failed\n");
        fclose(file);
        exit(1);
    }
    
    // Read file content
    size_t read_size = fread(buffer, 1, file_size, file);
    buffer[read_size] = '\0';  // Null-terminate the string
    
    fclose(file);
    *length = read_size;
    return buffer;
}

// CUDA error checking helper function
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[]) {
    char pattern[MAX_PATTERN_LENGTH];
    char *d_pattern;  // Device copy of pattern
    char *d_text;     // Device copy of text
    int *d_matches;   // Device total match counter
    int *d_positions; // Device array to store match positions
    int *d_match_count; // Device counter for number of displayed matches
    
    int d = 256;      // Number of characters in the alphabet
    int q = 101;      // A prime number for hash calculation
    char* text;
    int text_length;
    int pattern_length;
    int matches = 0;
    int match_count = 0;
    int positions[MAX_DISPLAY];
    
    clock_t start_time, end_time;
    double execution_time;
    float gpu_execution_time;
    cudaEvent_t gpu_start, gpu_end;
    
    const char* filename = "human_1m_upper.txt";
    
    strcpy(pattern, "TCGTG");
    pattern_length = strlen(pattern);
    
    printf("Reading file: %s\n", filename);
    printf("Searching for pattern: %s\n", pattern);
    
    // Read the file content
    text = read_text_file(filename, &text_length);
    printf("Read %d characters from file\n", text_length);
    
    // Measure total execution time including memory transfers
    start_time = clock();
    
    // Create CUDA events for GPU timing
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_end);
    
    // Allocate device memory
    checkCudaError(cudaMalloc((void**)&d_text, text_length * sizeof(char)), 
                   "Failed to allocate device memory for text");
    checkCudaError(cudaMalloc((void**)&d_pattern, pattern_length * sizeof(char)), 
                   "Failed to allocate device memory for pattern");
    checkCudaError(cudaMalloc((void**)&d_matches, sizeof(int)), 
                   "Failed to allocate device memory for matches counter");
    checkCudaError(cudaMalloc((void**)&d_positions, MAX_DISPLAY * sizeof(int)), 
                   "Failed to allocate device memory for positions");
    checkCudaError(cudaMalloc((void**)&d_match_count, sizeof(int)), 
                   "Failed to allocate device memory for match count");
    
    // Copy data to device
    checkCudaError(cudaMemcpy(d_text, text, text_length * sizeof(char), cudaMemcpyHostToDevice), 
                   "Failed to copy text to device");
    checkCudaError(cudaMemcpy(d_pattern, pattern, pattern_length * sizeof(char), cudaMemcpyHostToDevice), 
                   "Failed to copy pattern to device");
    
    // Compute pattern hash on CPU
    int pattern_hash = compute_hash(pattern, pattern_length, d, q);
    
    // Initialize counters
    checkCudaError(cudaMemset(d_matches, 0, sizeof(int)), "Failed to initialize matches counter");
    checkCudaError(cudaMemset(d_match_count, 0, sizeof(int)), "Failed to initialize match count");
    
    // Calculate h = d^(pattern_length-1) % q
    int h = 1;
    for (int i = 0; i < pattern_length - 1; i++) {
        h = (h * d) % q;
    }
    
    // Calculate number of blocks needed
    int num_blocks = (text_length - pattern_length + 1 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // Start GPU timing
    cudaEventRecord(gpu_start);
    
    // Launch kernel
    compute_initial_hashes<<<num_blocks, THREADS_PER_BLOCK>>>(
        d_text, text_length, d_pattern, pattern_length, pattern_hash, d, q, h, 
        d_matches, d_positions, d_match_count);
    
    // Check for kernel launch errors
    checkCudaError(cudaGetLastError(), "Kernel launch failed");
    
    // Wait for kernel to complete
    checkCudaError(cudaDeviceSynchronize(), "Kernel synchronization failed");
    
    // End GPU timing
    cudaEventRecord(gpu_end);
    cudaEventSynchronize(gpu_end);
    cudaEventElapsedTime(&gpu_execution_time, gpu_start, gpu_end);
    
    // Copy results back to host
    checkCudaError(cudaMemcpy(&matches, d_matches, sizeof(int), cudaMemcpyDeviceToHost), 
                   "Failed to copy matches back to host");
    checkCudaError(cudaMemcpy(&match_count, d_match_count, sizeof(int), cudaMemcpyDeviceToHost), 
                   "Failed to copy match count back to host");
    checkCudaError(cudaMemcpy(positions, d_positions, MAX_DISPLAY * sizeof(int), cudaMemcpyDeviceToHost), 
                   "Failed to copy positions back to host");
    
    // Display matches
    int display_count = match_count > MAX_DISPLAY ? MAX_DISPLAY : match_count;
    for (int i = 0; i < display_count; i++) {
        printf("Pattern found at index %d\n", positions[i]);
    }
    
    end_time = clock();
    execution_time = ((double) (end_time - start_time)) / CLOCKS_PER_SEC * 1000.0; // milliseconds
    
    printf("Total matches found: %d\n", matches);
    printf("Execution time: %.2f ms\n", execution_time);
    
    // Clean up
    cudaFree(d_text);
    cudaFree(d_pattern);
    cudaFree(d_matches);
    cudaFree(d_positions);
    cudaFree(d_match_count);
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_end);
    free(text);
    
    return 0;
}