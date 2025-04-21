#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <time.h>

// Maximum buffer sizes
#define MAX_TEXT_SIZE 2000000  // Increased for 1M file
#define MAX_PATTERNS 100
#define MAX_PATTERN_LENGTH 100
#define THREADS_PER_BLOCK 256

// Structure to store pattern data
typedef struct {
    char pattern[MAX_PATTERN_LENGTH];
    int length;
    int hash;
} PatternData;

// Function to compute hash on CPU
__host__ int computeHash(const char* str, int length, int d, int q) {
    int hash = 0;
    for (int i = 0; i < length; i++) {
        hash = (d * hash + str[i]) % q;
    }
    return hash;
}

// CPU function for result verification (to measure accuracy)
void cpuRabinKarp(const char* text, int text_length, const PatternData* patterns, int pattern_count, 
                  int d, int q, int* results, int* result_count) {
    *result_count = 0;
    
    for (int i = 0; i < text_length; i++) {
        for (int j = 0; j < pattern_count; j++) {
            int pattern_length = patterns[j].length;
            
            if (i + pattern_length > text_length) continue;
            
            // Calculate hash for current text window
            int text_hash = 0;
            for (int k = 0; k < pattern_length; k++) {
                text_hash = (d * text_hash + text[i + k]) % q;
            }
            
            if (text_hash == patterns[j].hash) {
                bool match = true;
                for (int k = 0; k < pattern_length; k++) {
                    if (text[i + k] != patterns[j].pattern[k]) {
                        match = false;
                        break;
                    }
                }
                
                if (match) {
                    results[2 * (*result_count)] = i;
                    results[2 * (*result_count) + 1] = j;
                    (*result_count)++;
                }
            }
        }
    }
}

// CUDA kernel for Rabin-Karp pattern matching
__global__ void rabinKarpKernel(const char* text, int text_length, const PatternData* patterns, 
                               int pattern_count, int d, int q, int* results, int* result_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < text_length; i += stride) {
        // Each thread starts from a different position in the text
        for (int j = 0; j < pattern_count; j++) {
            int pattern_length = patterns[j].length;
            
            // Make sure we don't go beyond text bounds
            if (i + pattern_length > text_length) continue;
            
            // Calculate hash for current text window
            int text_hash = 0;
            for (int k = 0; k < pattern_length; k++) {
                text_hash = (d * text_hash + text[i + k]) % q;
            }
            
            // If hash matches, verify character by character
            if (text_hash == patterns[j].hash) {
                bool match = true;
                for (int k = 0; k < pattern_length; k++) {
                    if (text[i + k] != patterns[j].pattern[k]) {
                        match = false;
                        break;
                    }
                }
                
                if (match) {
                    // Use atomicAdd to avoid race conditions when adding results
                    int pos = atomicAdd(result_count, 1);
                    if (pos < MAX_TEXT_SIZE) { // Ensure we don't exceed results buffer capacity
                        results[2*pos] = i;     // Store match position
                        results[2*pos+1] = j;   // Store index of matching pattern
                    }
                }
            }
        }
    }
}

// Function to measure execution time
double getExecutionTime(clock_t start, clock_t end) {
    return ((double) (end - start)) / CLOCKS_PER_SEC * 1000.0; // Convert to milliseconds
}

// Function to check if CPU and GPU results match
bool compareResults(int* cpu_results, int cpu_count, int* gpu_results, int gpu_count) {
    if (cpu_count != gpu_count) {
        printf("Result count mismatch: CPU=%d, GPU=%d\n", cpu_count, gpu_count);
        return false;
    }
    
    // Sort results for comparison (simple implementation)
    for (int i = 0; i < cpu_count; i++) {
        for (int j = i + 1; j < cpu_count; j++) {
            if (cpu_results[2*i] > cpu_results[2*j] || 
                (cpu_results[2*i] == cpu_results[2*j] && cpu_results[2*i+1] > cpu_results[2*j+1])) {
                // Swap positions
                int temp1 = cpu_results[2*i];
                int temp2 = cpu_results[2*i+1];
                cpu_results[2*i] = cpu_results[2*j];
                cpu_results[2*i+1] = cpu_results[2*j+1];
                cpu_results[2*j] = temp1;
                cpu_results[2*j+1] = temp2;
            }
        }
    }
    
    for (int i = 0; i < gpu_count; i++) {
        for (int j = i + 1; j < gpu_count; j++) {
            if (gpu_results[2*i] > gpu_results[2*j] || 
                (gpu_results[2*i] == gpu_results[2*j] && gpu_results[2*i+1] > gpu_results[2*j+1])) {
                // Swap positions
                int temp1 = gpu_results[2*i];
                int temp2 = gpu_results[2*i+1];
                gpu_results[2*i] = gpu_results[2*j];
                gpu_results[2*i+1] = gpu_results[2*j+1];
                gpu_results[2*j] = temp1;
                gpu_results[2*j+1] = temp2;
            }
        }
    }
    
    // Compare sorted results
    for (int i = 0; i < cpu_count; i++) {
        if (cpu_results[2*i] != gpu_results[2*i] || cpu_results[2*i+1] != gpu_results[2*i+1]) {
            printf("Mismatch at index %d: CPU=(%d,%d), GPU=(%d,%d)\n", 
                   i, cpu_results[2*i], cpu_results[2*i+1], gpu_results[2*i], gpu_results[2*i+1]);
            return false;
        }
    }
    
    return true;
}

// Function to read text file content
char* readTextFile(const char* filename, int* length) {
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

int main() {
    // Path to the human_1m_upper.txt file
    const char* filename = "human_1m_upper.txt";
    
    // Patterns to search for
    const char* patterns[] = {"TGCGATA", "TGTG", "AAAG", "TCGCT", "AAGG"};
    int num_patterns = 5;
    
    // Read text file
    int text_length;
    char* text = readTextFile(filename, &text_length);
    printf("Read %d characters from %s\n", text_length, filename);
    
    int d = 256; // Alphabet size
    int q = 101; // Prime number for hash
    
    // Allocate memory for pattern data on host
    PatternData* h_patterns = (PatternData*)malloc(num_patterns * sizeof(PatternData));
    
    // Initialize pattern data
    for (int i = 0; i < num_patterns; i++) {
        strncpy(h_patterns[i].pattern, patterns[i], MAX_PATTERN_LENGTH);
        h_patterns[i].length = strlen(patterns[i]);
        h_patterns[i].hash = computeHash(patterns[i], h_patterns[i].length, d, q);
    }
    
    // Allocate memory on GPU
    char* d_text;
    PatternData* d_patterns;
    int* d_results;
    int* d_result_count;
    
    cudaMalloc((void**)&d_text, text_length * sizeof(char));
    cudaMalloc((void**)&d_patterns, num_patterns * sizeof(PatternData));
    cudaMalloc((void**)&d_results, 2 * MAX_TEXT_SIZE * sizeof(int)); // To store (position, pattern index)
    cudaMalloc((void**)&d_result_count, sizeof(int));
    
    // Copy data to GPU
    cudaMemcpy(d_text, text, text_length * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_patterns, h_patterns, num_patterns * sizeof(PatternData), cudaMemcpyHostToDevice);
    cudaMemset(d_result_count, 0, sizeof(int));
    
    // Kernel configuration
    int blocks = (text_length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // Limit the number of blocks to prevent too much resource allocation
    if (blocks > 1024) blocks = 1024;
    
    printf("Launching kernel with %d blocks and %d threads per block\n", blocks, THREADS_PER_BLOCK);
    
    // Measure GPU execution time
    clock_t gpu_start, gpu_end;
    gpu_start = clock();
    
    // Run kernel
    rabinKarpKernel<<<blocks, THREADS_PER_BLOCK>>>(d_text, text_length, d_patterns, num_patterns, d, q, d_results, d_result_count);
    
    // Wait for kernel to finish
    cudaDeviceSynchronize();
    
    gpu_end = clock();
    double gpu_time = getExecutionTime(gpu_start, gpu_end);
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Copy results back to host
    int h_gpu_result_count;
    int* h_gpu_results = (int*)malloc(2 * MAX_TEXT_SIZE * sizeof(int));
    
    cudaMemcpy(&h_gpu_result_count, d_result_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gpu_results, d_results, 2 * h_gpu_result_count * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Measure CPU execution time for comparison
    clock_t cpu_start, cpu_end;
    int h_cpu_result_count;
    int* h_cpu_results = (int*)malloc(2 * MAX_TEXT_SIZE * sizeof(int));
    
    printf("Running CPU implementation for comparison...\n");
    cpu_start = clock();
    cpuRabinKarp(text, text_length, h_patterns, num_patterns, d, q, h_cpu_results, &h_cpu_result_count);
    cpu_end = clock();
    double cpu_time = getExecutionTime(cpu_start, cpu_end);
    
    // Display results
    printf("========= SEARCH RESULTS =========\n");
    printf("Patterns found (showing first 20):\n");
    int display_count = h_gpu_result_count > 20 ? 20 : h_gpu_result_count;
    for (int i = 0; i < display_count; i++) {
        int position = h_gpu_results[2*i];
        int pattern_idx = h_gpu_results[2*i+1];
        printf("Pattern \"%s\" found at position %d\n", 
               patterns[pattern_idx], position);
    }
    
    // Compare CPU and GPU results to measure accuracy
    bool accurate = compareResults(h_cpu_results, h_cpu_result_count, h_gpu_results, h_gpu_result_count);
    
    // Display performance statistics
    printf("\n========= PERFORMANCE STATISTICS =========\n");
    printf("CPU execution time: %.3f ms\n", cpu_time);
    printf("GPU execution time: %.3f ms\n", gpu_time);
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    printf("Accuracy: %s\n", accurate ? "100% (All results match)" : "Inaccurate (Results differ)");
    printf("Total matches found: %d\n", h_gpu_result_count);
    
    // Clean up memory
    free(text);
    free(h_patterns);
    free(h_gpu_results);
    free(h_cpu_results);
    cudaFree(d_text);
    cudaFree(d_patterns);
    cudaFree(d_results);
    cudaFree(d_result_count);
    
    return 0;
}