#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define MAX_TEXT_SIZE 2000000
#define MAX_PATTERN_SIZE 100
#define MAX_DISPLAY 20

__device__ int compute_hash_gpu(char *str, int length, int d, int q) {
    int hash_value = 0;
    for (int i = 0; i < length; ++i) {
        hash_value = (d * hash_value + str[i]) % q;
    }
    return hash_value;
}

__global__ void rk_kernel(char *text, char *pattern, int *match_positions, int *match_count,
                          int text_length, int pattern_length, int pattern_hash, int d, int q, int h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > text_length - pattern_length) return;

    // Compute the hash of the current window
    int hash_value = 0;
    for (int j = 0; j < pattern_length; j++) {
        hash_value = (d * hash_value + text[i + j]) % q;
    }

    if (hash_value == pattern_hash) {
        bool match = true;
        for (int j = 0; j < pattern_length; j++) {
            if (text[i + j] != pattern[j]) {
                match = false;
                break;
            }
        }

        if (match) {
            int idx = atomicAdd(match_count, 1);
            if (idx < MAX_DISPLAY) {
                match_positions[idx] = i;
            }
        }
    }
}

char* read_text_file(const char* filename, int* length) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file\n");
        exit(1);
    }

    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    rewind(file);

    char* buffer = (char*)malloc(size + 1);
    fread(buffer, 1, size, file);
    buffer[size] = '\0';
    fclose(file);

    *length = size;
    return buffer;
}

int compute_hash_cpu(char *str, int length, int d, int q) {
    int hash_value = 0;
    for (int i = 0; i < length; ++i) {
        hash_value = (d * hash_value + str[i]) % q;
    }
    return hash_value;
}

int main() {
    const char* filename = "human_1m_upper.txt";
    char pattern[MAX_PATTERN_SIZE] = "GGGCA";
    int d = 256, q = 101;

    int text_length;
    char* text = read_text_file(filename, &text_length);
    int pattern_length = strlen(pattern);
    int h = 1;
    for (int i = 0; i < pattern_length - 1; i++) {
        h = (h * d) % q;
    }

    // CPU precomputed pattern hash
    int pattern_hash = compute_hash_cpu(pattern, pattern_length, d, q);

    // Allocate device memory
    char *d_text, *d_pattern;
    int *d_match_positions, *d_match_count;
    int *match_positions = (int*)malloc(MAX_DISPLAY * sizeof(int));
    int match_count = 0;

    cudaMalloc((void**)&d_text, text_length);
    cudaMalloc((void**)&d_pattern, pattern_length);
    cudaMalloc((void**)&d_match_positions, MAX_DISPLAY * sizeof(int));
    cudaMalloc((void**)&d_match_count, sizeof(int));

    cudaMemcpy(d_text, text, text_length, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pattern, pattern, pattern_length, cudaMemcpyHostToDevice);
    cudaMemcpy(d_match_count, &match_count, sizeof(int), cudaMemcpyHostToDevice);

    // Timer untuk menghitung waktu eksekusi
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (text_length - pattern_length + blockSize - 1) / blockSize;
    rk_kernel<<<gridSize, blockSize>>>(d_text, d_pattern, d_match_positions, d_match_count,
                                    text_length, pattern_length, pattern_hash, d, q, h);

    // Sinkronisasi dan pengukuran waktu
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Salin hasil dari device ke host
    cudaMemcpy(match_positions, d_match_positions, MAX_DISPLAY * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&match_count, d_match_count, sizeof(int), cudaMemcpyDeviceToHost);

    // Output dalam format yang diminta
    for (int i = 0; i < match_count && i < MAX_DISPLAY; i++) {
        printf("Pattern found at index %d\n", match_positions[i]);
    }
    if (match_count > MAX_DISPLAY) {
        printf("... (showing only first %d matches)\n", MAX_DISPLAY);
    }

    printf("Total matches found: %d\n", match_count);
    printf("Execution time: %.2f ms\n", milliseconds);

    // Free memory
    free(text);
    free(match_positions);
    cudaFree(d_text);
    cudaFree(d_pattern);
    cudaFree(d_match_positions);
    cudaFree(d_match_count);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
