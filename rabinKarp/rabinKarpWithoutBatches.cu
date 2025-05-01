#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define MAX_TEXT_SIZE 10000000
#define MAX_PATTERN_SIZE 512
#define MAX_PATTERNS 1024
#define MAX_DISPLAY 20

// Macro for checking CUDA errors
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

typedef struct {
    char pattern[MAX_PATTERN_SIZE];
    int length;
    int hash;
} PatternInfo;

__device__ int compute_hash_gpu(char *str, int length, int d, int q) {
    int hash_value = 0;
    for (int i = 0; i < length; ++i) {
        hash_value = (d * hash_value + str[i]) % q;
    }
    return hash_value;
}

__global__ void rk_kernel_with_stride(
    char *text, char *patterns, int *pattern_lengths, int *pattern_hashes,
    int *match_positions, int *match_pattern_ids, int *match_count,
    int *pattern_match_counts, int text_length, int pattern_count, 
    int d, int q) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Process multiple positions per thread using stride pattern
    for (int i = tid; i < text_length; i += stride) {
        // Each thread handles all patterns at its assigned positions
        for (int p = 0; p < pattern_count; p++) {
            int pattern_length = pattern_lengths[p];
            int pattern_hash = pattern_hashes[p];
            
            // Skip if we don't have enough characters left
            if (i > text_length - pattern_length) continue;

            // Compute the hash of the current window
            int hash_value = 0;
            for (int j = 0; j < pattern_length; j++) {
                hash_value = (d * hash_value + text[i + j]) % q;
            }

            if (hash_value == pattern_hash) {
                bool match = true;
                for (int j = 0; j < pattern_length; j++) {
                    if (text[i + j] != patterns[p * MAX_PATTERN_SIZE + j]) {
                        match = false;
                        break;
                    }
                }

                if (match) {
                    // Increment per-pattern match count
                    atomicAdd(&pattern_match_counts[p], 1);
                    
                    // Store position and pattern ID for display
                    int idx = atomicAdd(match_count, 1);
                    if (idx < MAX_DISPLAY) {
                        match_positions[idx] = i;
                        match_pattern_ids[idx] = p;
                    }
                }
            }
        }
    }
}

int main() {
    const char* text_filename = "human_10m_upper.txt";
    const char* pattern_filename = "pattern.txt";
    int d = 256, q = 101;
    
    // Query device properties
    int deviceCount;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("No CUDA devices found. Exiting.\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);
    printf("Global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    
    // Read text file
    FILE* file = fopen(text_filename, "r");
    if (!file) {
        printf("Error opening file %s\n", text_filename);
        return 1;
    }

    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    rewind(file);

    char* text = (char*)malloc(size + 1);
    if (!text) {
        printf("Memory allocation failed\n");
        fclose(file);
        return 1;
    }
    
    fread(text, 1, size, file);
    text[size] = '\0';
    fclose(file);
    int text_length = size;
    
    // Read patterns file
    PatternInfo patterns[MAX_PATTERNS];
    int pattern_count = 0;
    
    FILE* patternFile = fopen(pattern_filename, "r");
    if (!patternFile) {
        printf("Error opening patterns file %s\n", pattern_filename);
        free(text);
        return 1;
    }

    // Read the entire file into a buffer
    char* buffer = (char*)malloc(MAX_PATTERNS * MAX_PATTERN_SIZE);
    if (!buffer) {
        printf("Memory allocation failed\n");
        fclose(patternFile);
        free(text);
        return 1;
    }
    
    size_t bytes_read = fread(buffer, 1, MAX_PATTERNS * MAX_PATTERN_SIZE - 1, patternFile);
    buffer[bytes_read] = '\0';
    fclose(patternFile);

    // Parse comma-separated patterns
    char* token = strtok(buffer, ",");
    while (token != NULL && pattern_count < MAX_PATTERNS) {
        if (strlen(token) >= MAX_PATTERN_SIZE) {
            printf("Warning: Pattern %d is too long. Truncating.\n", pattern_count);
            token[MAX_PATTERN_SIZE-1] = '\0';
        }
        strcpy(patterns[pattern_count].pattern, token);
        patterns[pattern_count].length = strlen(token);
        
        // Compute hash on CPU
        int hash_value = 0;
        for (int i = 0; i < patterns[pattern_count].length; i++) {
            hash_value = (d * hash_value + token[i]) % q;
        }
        patterns[pattern_count].hash = hash_value;
        
        printf("Pattern %d: %s (length: %d, hash: %d)\n", 
               pattern_count, patterns[pattern_count].pattern, 
               patterns[pattern_count].length, patterns[pattern_count].hash);
               
        pattern_count++;
        token = strtok(NULL, ",");
    }
    free(buffer);
    
    printf("Loaded %d patterns from %s\n", pattern_count, pattern_filename);
    
    // Prepare pattern data for GPU
    char* patterns_flat = (char*)malloc(pattern_count * MAX_PATTERN_SIZE);
    int* pattern_lengths = (int*)malloc(pattern_count * sizeof(int));
    int* pattern_hashes = (int*)malloc(pattern_count * sizeof(int));
    
    for (int i = 0; i < pattern_count; i++) {
        strcpy(&patterns_flat[i * MAX_PATTERN_SIZE], patterns[i].pattern);
        pattern_lengths[i] = patterns[i].length;
        pattern_hashes[i] = patterns[i].hash;
    }
    
    // Allocate device memory
    char *d_text, *d_patterns;
    int *d_pattern_lengths, *d_pattern_hashes;
    int *d_match_positions, *d_match_pattern_ids, *d_match_count;
    int *d_pattern_match_counts;
    
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_text, text_length));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_patterns, pattern_count * MAX_PATTERN_SIZE));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_pattern_lengths, pattern_count * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_pattern_hashes, pattern_count * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_match_positions, MAX_DISPLAY * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_match_pattern_ids, MAX_DISPLAY * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_match_count, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_pattern_match_counts, pattern_count * sizeof(int)));
    
    // Initialize counters to zero
    int match_count = 0;
    CHECK_CUDA_ERROR(cudaMemcpy(d_match_count, &match_count, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemset(d_pattern_match_counts, 0, pattern_count * sizeof(int)));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_text, text, text_length, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_patterns, patterns_flat, pattern_count * MAX_PATTERN_SIZE, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pattern_lengths, pattern_lengths, pattern_count * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pattern_hashes, pattern_hashes, pattern_count * sizeof(int), cudaMemcpyHostToDevice));
    
    // Timer to measure execution time
    cudaEvent_t start, stop;
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    
    // Launch kernel with configurable grid size
    int blockSize = 256;
    int gridSize = min(64, (text_length + blockSize - 1) / blockSize);  // Configurable grid size
    
    printf("Launching kernel with grid size: %d, block size: %d\n", gridSize, blockSize);
    
    rk_kernel_with_stride<<<gridSize, blockSize>>>(
        d_text, d_patterns, d_pattern_lengths, d_pattern_hashes,
        d_match_positions, d_match_pattern_ids, d_match_count,
        d_pattern_match_counts, text_length, pattern_count, 
        d, q
    );
    
    // Check for kernel errors
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Record stop time
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    
    // Copy results back
    int host_match_count;
    int match_positions[MAX_DISPLAY];
    int match_pattern_ids[MAX_DISPLAY];
    int* pattern_match_counts = (int*)calloc(pattern_count, sizeof(int));
    
    CHECK_CUDA_ERROR(cudaMemcpy(&host_match_count, d_match_count, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(match_positions, d_match_positions, MAX_DISPLAY * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(match_pattern_ids, d_match_pattern_ids, MAX_DISPLAY * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(pattern_match_counts, d_pattern_match_counts, pattern_count * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Output results
    printf("\nSample matches found:\n");
    for (int i = 0; i < MAX_DISPLAY && i < host_match_count; i++) {
        printf("Pattern '%s' found at index %d\n", 
               patterns[match_pattern_ids[i]].pattern, match_positions[i]);
    }
    
    if (host_match_count > MAX_DISPLAY) {
        printf("... (showing only first %d matches out of %d)\n", MAX_DISPLAY, host_match_count);
    }
    
    // Print per-pattern match counts
    printf("\nMatch counts per pattern:\n");
    int total_matches = 0;
    for (int i = 0; i < pattern_count; i++) {
        printf("Pattern '%s': %d matches\n", patterns[i].pattern, pattern_match_counts[i]);
        total_matches += pattern_match_counts[i];
    }
    
    printf("Total matches found across all patterns: %d\n", total_matches);
    printf("Execution time: %.2f ms\n", milliseconds);
    
    // Free memory
    free(text);
    free(patterns_flat);
    free(pattern_lengths);
    free(pattern_hashes);
    free(pattern_match_counts);
    
    CHECK_CUDA_ERROR(cudaFree(d_text));
    CHECK_CUDA_ERROR(cudaFree(d_patterns));
    CHECK_CUDA_ERROR(cudaFree(d_pattern_lengths));
    CHECK_CUDA_ERROR(cudaFree(d_pattern_hashes));
    CHECK_CUDA_ERROR(cudaFree(d_match_positions));
    CHECK_CUDA_ERROR(cudaFree(d_match_pattern_ids));
    CHECK_CUDA_ERROR(cudaFree(d_match_count));
    CHECK_CUDA_ERROR(cudaFree(d_pattern_match_counts));
    
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    
    return 0;
}