#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

// Reduced pattern size to save memory
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

// Function to process a batch of patterns
void process_pattern_batch(char *text, int text_length, PatternInfo *patterns, 
                        int start_idx, int batch_size, int d, int q) {
    
    int actual_batch_size = batch_size;
    if (start_idx + batch_size > MAX_PATTERNS) {
        actual_batch_size = MAX_PATTERNS - start_idx;
    }
    
    printf("Processing batch of %d patterns (starting at index %d)\n", actual_batch_size, start_idx);
    
    // Prepare pattern data for this batch
    char* patterns_flat = (char*)malloc(actual_batch_size * MAX_PATTERN_SIZE);
    int* pattern_lengths = (int*)malloc(actual_batch_size * sizeof(int));
    int* pattern_hashes = (int*)malloc(actual_batch_size * sizeof(int));
    
    for (int i = 0; i < actual_batch_size; i++) {
        strcpy(&patterns_flat[i * MAX_PATTERN_SIZE], patterns[start_idx + i].pattern);
        pattern_lengths[i] = patterns[start_idx + i].length;
        pattern_hashes[i] = patterns[start_idx + i].hash;
    }
    
    // Host memory for results
    int *match_positions = (int*)malloc(MAX_DISPLAY * sizeof(int));
    int *match_pattern_ids = (int*)malloc(MAX_DISPLAY * sizeof(int));
    int match_count = 0;
    int *pattern_match_counts = (int*)calloc(actual_batch_size, sizeof(int));
    
    // Device memory pointers
    char *d_text, *d_patterns;
    int *d_pattern_lengths, *d_pattern_hashes;
    int *d_match_positions, *d_match_pattern_ids, *d_match_count;
    int *d_pattern_match_counts;
    
    // Allocate device memory with error checking
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_text, text_length));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_patterns, actual_batch_size * MAX_PATTERN_SIZE));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_pattern_lengths, actual_batch_size * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_pattern_hashes, actual_batch_size * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_match_positions, MAX_DISPLAY * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_match_pattern_ids, MAX_DISPLAY * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_match_count, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_pattern_match_counts, actual_batch_size * sizeof(int)));
    
    // Copy data to device with error checking
    CHECK_CUDA_ERROR(cudaMemcpy(d_text, text, text_length, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_patterns, patterns_flat, actual_batch_size * MAX_PATTERN_SIZE, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pattern_lengths, pattern_lengths, actual_batch_size * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pattern_hashes, pattern_hashes, actual_batch_size * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_match_count, &match_count, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemset(d_pattern_match_counts, 0, actual_batch_size * sizeof(int)));
    
    // Timer to measure execution time
    cudaEvent_t start, stop;
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    
    // Launch kernel with smaller block size to avoid timeout
    int blockSize = 256;
    int gridSize = (text_length + blockSize - 1) / blockSize;
    printf("Launching kernel with grid size: %d, block size: %d\n", gridSize, blockSize);
    
    rk_kernel<<<gridSize, blockSize>>>(d_text, d_patterns, d_pattern_lengths, d_pattern_hashes,
                                    d_match_positions, d_match_pattern_ids, d_match_count,
                                    d_pattern_match_counts, text_length, actual_batch_size, d, q);
    
    // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Synchronize and measure time
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    
    // Copy results from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(match_positions, d_match_positions, MAX_DISPLAY * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(match_pattern_ids, d_match_pattern_ids, MAX_DISPLAY * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(&match_count, d_match_count, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(pattern_match_counts, d_pattern_match_counts, actual_batch_size * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Output results
    printf("\nSample matches found in this batch:\n");
    for (int i = 0; i < match_count && i < MAX_DISPLAY; i++) {
        printf("Pattern '%s' found at index %d\n", 
              patterns[start_idx + match_pattern_ids[i]].pattern, match_positions[i]);
    }
    
    if (match_count > MAX_DISPLAY) {
        printf("... (showing only first %d matches out of %d)\n", MAX_DISPLAY, match_count);
    }
    
    // Print per-pattern match counts
    printf("\nMatch counts per pattern in this batch:\n");
    int total_matches = 0;
    for (int i = 0; i < actual_batch_size; i++) {
        printf("Pattern '%s': %d matches\n", patterns[start_idx + i].pattern, pattern_match_counts[i]);
        total_matches += pattern_match_counts[i];
    }
    
    printf("Total matches found across all patterns in this batch: %d\n", total_matches);
    printf("Execution time for this batch: %.2f ms\n", milliseconds);
    
    // Free memory
    free(patterns_flat);
    free(pattern_lengths);
    free(pattern_hashes);
    free(match_positions);
    free(match_pattern_ids);
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
}

int main() {
    const char* text_filename = "human_10m_upper.txt";
    const char* pattern_filename = "pattern.txt";
    int d = 256, q = 101;
    
    // Query device properties to understand limitations
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
    printf("Loaded text of length %d from %s\n", text_length, text_filename);
    
    // Read patterns
    PatternInfo patterns[MAX_PATTERNS];
    int pattern_count = read_patterns_file(pattern_filename, patterns);
    
    printf("Loaded %d patterns from %s\n", pattern_count, pattern_filename);
    
    // Precompute hash for each pattern
    for (int i = 0; i < pattern_count; i++) {
        patterns[i].hash = compute_hash_cpu(patterns[i].pattern, patterns[i].length, d, q);
        printf("Pattern %d: %s (length: %d, hash: %d)\n", 
              i, patterns[i].pattern, patterns[i].length, patterns[i].hash);
    }
    
    // Process patterns in batches to avoid memory issues
    const int BATCH_SIZE = 128;  // Adjust this based on your available memory
    int num_batches = (pattern_count + BATCH_SIZE - 1) / BATCH_SIZE;
    
    printf("\nProcessing %d patterns in %d batches of size %d\n\n", 
          pattern_count, num_batches, BATCH_SIZE);
    
    for (int batch = 0; batch < num_batches; batch++) {
        int start_idx = batch * BATCH_SIZE;
        process_pattern_batch(text, text_length, patterns, start_idx, BATCH_SIZE, d, q);
    }
    
    // Free text memory
    free(text);
    
    printf("\nAll batches processed successfully.\n");
    return 0;
}