#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define MAX_TEXT_SIZE 2000000
#define MAX_PATTERN_SIZE 100
#define MAX_PATTERNS 100
#define MAX_DISPLAY 20

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

__global__ void rk_kernel(char *text, char *patterns, int *pattern_lengths, int *pattern_hashes, 
                          int *match_positions, int *match_pattern_ids, int *match_count,
                          int *pattern_match_counts, int text_length, int pattern_count, int d, int q) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= text_length) return;

    // Each thread handles a position in the text
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
                // Increment total match count
                int idx = atomicAdd(match_count, 1);
                
                // Increment per-pattern match count
                atomicAdd(&pattern_match_counts[p], 1);
                
                // Store position and pattern ID for display
                if (idx < MAX_DISPLAY) {
                    match_positions[idx] = i;
                    match_pattern_ids[idx] = p;
                }
            }
        }
    }
}

char* read_text_file(const char* filename, int* length) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file %s\n", filename);
        exit(1);
    }

    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    rewind(file);

    char* buffer = (char*)malloc(size + 1);
    if (!buffer) {
        printf("Memory allocation failed\n");
        fclose(file);
        exit(1);
    }
    
    fread(buffer, 1, size, file);
    buffer[size] = '\0';
    fclose(file);

    *length = size;
    return buffer;
}

int read_patterns_file(const char* filename, PatternInfo* patterns) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error opening patterns file %s\n", filename);
        exit(1);
    }

    char line[1024];
    if (fgets(line, sizeof(line), file) == NULL) {
        printf("Error reading patterns file or file is empty\n");
        fclose(file);
        exit(1);
    }
    fclose(file);

    // Remove newline if present
    int len = strlen(line);
    if (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) {
        line[len-1] = '\0';
        len--;
    }
    if (len > 0 && line[len-1] == '\r') {
        line[len-1] = '\0';
    }

    // Parse the comma-separated patterns
    int pattern_count = 0;
    char* token = strtok(line, ",");
    while (token != NULL && pattern_count < MAX_PATTERNS) {
        strcpy(patterns[pattern_count].pattern, token);
        patterns[pattern_count].length = strlen(token);
        token = strtok(NULL, ",");
        pattern_count++;
    }

    return pattern_count;
}

int compute_hash_cpu(char *str, int length, int d, int q) {
    int hash_value = 0;
    for (int i = 0; i < length; ++i) {
        hash_value = (d * hash_value + str[i]) % q;
    }
    return hash_value;
}

int main() {
    const char* text_filename = "human_10m_upper.txt";
    const char* pattern_filename = "pattern.txt";
    int d = 256, q = 101;

    // Read text file
    int text_length;
    char* text = read_text_file(text_filename, &text_length);

    // Read patterns
    PatternInfo patterns[MAX_PATTERNS];
    int pattern_count = read_patterns_file(pattern_filename, patterns);
    
    printf("Loaded %d patterns from %s\n", pattern_count, pattern_filename);
    for (int i = 0; i < pattern_count; i++) {
        // Precompute hash for each pattern
        patterns[i].hash = compute_hash_cpu(patterns[i].pattern, patterns[i].length, d, q);
        printf("Pattern %d: %s (length: %d, hash: %d)\n", 
               i, patterns[i].pattern, patterns[i].length, patterns[i].hash);
    }

    // Prepare data for GPU
    char *d_text, *d_patterns;
    int *d_pattern_lengths, *d_pattern_hashes;
    int *d_match_positions, *d_match_pattern_ids, *d_match_count;
    int *d_pattern_match_counts;
    
    // Host memory for results
    int *match_positions = (int*)malloc(MAX_DISPLAY * sizeof(int));
    int *match_pattern_ids = (int*)malloc(MAX_DISPLAY * sizeof(int));
    int match_count = 0;
    int *pattern_match_counts = (int*)calloc(pattern_count, sizeof(int));

    // Prepare pattern data
    char* patterns_flat = (char*)malloc(pattern_count * MAX_PATTERN_SIZE);
    int* pattern_lengths = (int*)malloc(pattern_count * sizeof(int));
    int* pattern_hashes = (int*)malloc(pattern_count * sizeof(int));
    
    for (int i = 0; i < pattern_count; i++) {
        strcpy(&patterns_flat[i * MAX_PATTERN_SIZE], patterns[i].pattern);
        pattern_lengths[i] = patterns[i].length;
        pattern_hashes[i] = patterns[i].hash;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_text, text_length);
    cudaMalloc((void**)&d_patterns, pattern_count * MAX_PATTERN_SIZE);
    cudaMalloc((void**)&d_pattern_lengths, pattern_count * sizeof(int));
    cudaMalloc((void**)&d_pattern_hashes, pattern_count * sizeof(int));
    cudaMalloc((void**)&d_match_positions, MAX_DISPLAY * sizeof(int));
    cudaMalloc((void**)&d_match_pattern_ids, MAX_DISPLAY * sizeof(int));
    cudaMalloc((void**)&d_match_count, sizeof(int));
    cudaMalloc((void**)&d_pattern_match_counts, pattern_count * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_text, text, text_length, cudaMemcpyHostToDevice);
    cudaMemcpy(d_patterns, patterns_flat, pattern_count * MAX_PATTERN_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pattern_lengths, pattern_lengths, pattern_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pattern_hashes, pattern_hashes, pattern_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_match_count, &match_count, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_pattern_match_counts, 0, pattern_count * sizeof(int));

    // Timer to measure execution time
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (text_length + blockSize - 1) / blockSize;
    printf("Launching kernel with grid size: %d, block size: %d\n", gridSize, blockSize);
    
    rk_kernel<<<gridSize, blockSize>>>(d_text, d_patterns, d_pattern_lengths, d_pattern_hashes,
                                   d_match_positions, d_match_pattern_ids, d_match_count,
                                   d_pattern_match_counts, text_length, pattern_count, d, q);

    // Synchronize and measure time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy results from device to host
    cudaMemcpy(match_positions, d_match_positions, MAX_DISPLAY * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(match_pattern_ids, d_match_pattern_ids, MAX_DISPLAY * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&match_count, d_match_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(pattern_match_counts, d_pattern_match_counts, pattern_count * sizeof(int), cudaMemcpyDeviceToHost);

    // Check for CUDA errors
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaError));
    }

    // Output results
    printf("\nSample matches found:\n");
    for (int i = 0; i < match_count && i < MAX_DISPLAY; i++) {
        printf("Pattern '%s' found at index %d\n", 
               patterns[match_pattern_ids[i]].pattern, match_positions[i]);
    }
    
    if (match_count > MAX_DISPLAY) {
        printf("... (showing only first %d matches out of %d)\n", MAX_DISPLAY, match_count);
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
    free(match_positions);
    free(match_pattern_ids);
    free(pattern_match_counts);
    
    cudaFree(d_text);
    cudaFree(d_patterns);
    cudaFree(d_pattern_lengths);
    cudaFree(d_pattern_hashes);
    cudaFree(d_match_positions);
    cudaFree(d_match_pattern_ids);
    cudaFree(d_match_count);
    cudaFree(d_pattern_match_counts);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}