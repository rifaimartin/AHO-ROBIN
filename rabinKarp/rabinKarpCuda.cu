#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define MAX_TEXT_SIZE 2000000
#define MAX_PATTERN_SIZE 512  // Reduced from 1024
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

__global__ void rk_kernel(char *text, char *patterns, int *pattern_lengths, int *pattern_hashes, 
                          int *match_positions, int *match_pattern_ids, int *batch_match_count,
                          int *pattern_match_counts, int text_length, int pattern_count, 
                          int batch_offset, int d, int q) {
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
                // Get global pattern index
                int global_pattern_idx = batch_offset + p;
                
                // Increment per-pattern match count
                atomicAdd(&pattern_match_counts[global_pattern_idx], 1);
                
                // Store position and pattern ID for display
                int idx = atomicAdd(batch_match_count, 1);
                if (idx < MAX_DISPLAY) {
                    match_positions[idx] = i;
                    match_pattern_ids[idx] = global_pattern_idx;
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

int read_patterns_file(const char* filename, PatternInfo* patterns)
{
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error opening patterns file %s\n", filename);
        exit(1);
    }

    // Allocate a buffer for reading patterns
    char* buffer = (char*)malloc(MAX_PATTERNS * MAX_PATTERN_SIZE);
    if (!buffer) {
        printf("Memory allocation failed\n");
        fclose(file);
        exit(1);
    }
    
    // Read the entire file into the buffer
    size_t bytes_read = fread(buffer, 1, MAX_PATTERNS * MAX_PATTERN_SIZE - 1, file);
    buffer[bytes_read] = '\0';  // Null-terminate
    fclose(file);

    // Remove newline if present
    int len = strlen(buffer);
    if (len > 0 && (buffer[len-1] == '\n' || buffer[len-1] == '\r')) {
        buffer[len-1] = '\0';
        len--;
    }
    if (len > 0 && buffer[len-1] == '\r') {
        buffer[len-1] = '\0';
    }

    // Parse the comma-separated patterns
    int pattern_count = 0;
    char* token = strtok(buffer, ",");
    while (token != NULL && pattern_count < MAX_PATTERNS) {
        // Check pattern length to avoid buffer overflow
        if (strlen(token) >= MAX_PATTERN_SIZE) {
            printf("Warning: Pattern %d is too long (>= %d chars). Truncating.\n", 
                  pattern_count, MAX_PATTERN_SIZE);
            token[MAX_PATTERN_SIZE-1] = '\0';
        }
        strcpy(patterns[pattern_count].pattern, token);
        patterns[pattern_count].length = strlen(token);
        token = strtok(NULL, ",");
        pattern_count++;
    }

    free(buffer);
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
    
    // Allocate device memory for text
    char *d_text;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_text, text_length));
    CHECK_CUDA_ERROR(cudaMemcpy(d_text, text, text_length, cudaMemcpyHostToDevice));
    
    // Allocate host memory for final results
    int *match_positions = (int*)malloc(MAX_DISPLAY * sizeof(int));
    int *match_pattern_ids = (int*)malloc(MAX_DISPLAY * sizeof(int));
    int total_match_count = 0;
    int *pattern_match_counts = (int*)calloc(pattern_count, sizeof(int));
    
    // Allocate device memory for pattern match counts
    int *d_pattern_match_counts;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_pattern_match_counts, pattern_count * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(d_pattern_match_counts, 0, pattern_count * sizeof(int)));
    
    // Process patterns in batches to avoid memory issues
    const int BATCH_SIZE = 128;  // Adjust this based on your GPU memory
    int num_batches = (pattern_count + BATCH_SIZE - 1) / BATCH_SIZE;
    
    // Timer to measure total execution time
    cudaEvent_t start, stop;
    float total_milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    
    printf("Launching kernel with %d batches\n", num_batches);
    
    for (int batch = 0; batch < num_batches; batch++) {
        int batch_offset = batch * BATCH_SIZE;
        int current_batch_size = BATCH_SIZE;
        
        // Adjust batch size for the last batch if needed
        if (batch_offset + current_batch_size > pattern_count) {
            current_batch_size = pattern_count - batch_offset;
        }
        
        // Prepare pattern data for this batch
        char* patterns_flat = (char*)malloc(current_batch_size * MAX_PATTERN_SIZE);
        int* pattern_lengths = (int*)malloc(current_batch_size * sizeof(int));
        int* pattern_hashes = (int*)malloc(current_batch_size * sizeof(int));
        
        for (int i = 0; i < current_batch_size; i++) {
            strcpy(&patterns_flat[i * MAX_PATTERN_SIZE], patterns[batch_offset + i].pattern);
            pattern_lengths[i] = patterns[batch_offset + i].length;
            pattern_hashes[i] = patterns[batch_offset + i].hash;
        }
        
        // Allocate device memory for this batch
        char *d_patterns;
        int *d_pattern_lengths, *d_pattern_hashes;
        int *d_match_positions, *d_match_pattern_ids, *d_batch_match_count;
        int batch_match_count = 0;
        
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_patterns, current_batch_size * MAX_PATTERN_SIZE));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_pattern_lengths, current_batch_size * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_pattern_hashes, current_batch_size * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_match_positions, MAX_DISPLAY * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_match_pattern_ids, MAX_DISPLAY * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_batch_match_count, sizeof(int)));
        
        // Copy data to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_patterns, patterns_flat, current_batch_size * MAX_PATTERN_SIZE, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_pattern_lengths, pattern_lengths, current_batch_size * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_pattern_hashes, pattern_hashes, current_batch_size * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_batch_match_count, &batch_match_count, sizeof(int), cudaMemcpyHostToDevice));
        
        // Launch kernel
        int blockSize = 256;
        int gridSize = (text_length + blockSize - 1) / blockSize;
        
        rk_kernel<<<gridSize, blockSize>>>(d_text, d_patterns, d_pattern_lengths, d_pattern_hashes,
                                      d_match_positions, d_match_pattern_ids, d_batch_match_count,
                                      d_pattern_match_counts, text_length, current_batch_size, 
                                      batch_offset, d, q);
        
        // Check for kernel launch errors
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        // Copy batch match count back
        CHECK_CUDA_ERROR(cudaMemcpy(&batch_match_count, d_batch_match_count, sizeof(int), cudaMemcpyDeviceToHost));
        
        // If this is the first batch with matches, save the positions for display
        if (batch == 0 && batch_match_count > 0) {
            CHECK_CUDA_ERROR(cudaMemcpy(match_positions, d_match_positions, MAX_DISPLAY * sizeof(int), cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaMemcpy(match_pattern_ids, d_match_pattern_ids, MAX_DISPLAY * sizeof(int), cudaMemcpyDeviceToHost));
        }
        
        // Update total match count
        total_match_count += batch_match_count;
        
        // Free batch resources
        free(patterns_flat);
        free(pattern_lengths);
        free(pattern_hashes);
        
        CHECK_CUDA_ERROR(cudaFree(d_patterns));
        CHECK_CUDA_ERROR(cudaFree(d_pattern_lengths));
        CHECK_CUDA_ERROR(cudaFree(d_pattern_hashes));
        CHECK_CUDA_ERROR(cudaFree(d_match_positions));
        CHECK_CUDA_ERROR(cudaFree(d_match_pattern_ids));
        CHECK_CUDA_ERROR(cudaFree(d_batch_match_count));
    }
    
    // Get total execution time
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&total_milliseconds, start, stop));
    
    // Copy final pattern match counts from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(pattern_match_counts, d_pattern_match_counts, pattern_count * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Output results in the same format as the original code
    printf("\nSample matches found:\n");
    for (int i = 0; i < MAX_DISPLAY && i < total_match_count; i++) {
        printf("Pattern '%s' found at index %d\n", 
               patterns[match_pattern_ids[i]].pattern, match_positions[i]);
    }
    
    if (total_match_count > MAX_DISPLAY) {
        printf("... (showing only first %d matches out of %d)\n", MAX_DISPLAY, total_match_count);
    }
    
    // Print per-pattern match counts
    printf("\nMatch counts per pattern:\n");
    int verified_matches = 0;
    for (int i = 0; i < pattern_count; i++) {
        printf("Pattern '%s': %d matches\n", patterns[i].pattern, pattern_match_counts[i]);
        verified_matches += pattern_match_counts[i];
    }
    
    printf("Total matches found across all patterns: %d\n", verified_matches);
    printf("Execution time: %.2f ms\n", total_milliseconds);
    
    // Free memory
    free(text);
    free(match_positions);
    free(match_pattern_ids);
    free(pattern_match_counts);
    
    CHECK_CUDA_ERROR(cudaFree(d_text));
    CHECK_CUDA_ERROR(cudaFree(d_pattern_match_counts));
    
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    
    return 0;
}