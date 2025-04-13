#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <time.h>

// Ukuran maksimum buffer
#define MAX_TEXT_SIZE 10000
#define MAX_PATTERNS 100
#define MAX_PATTERN_LENGTH 100
#define THREADS_PER_BLOCK 256

// Struktur untuk menyimpan data pola
typedef struct {
    char pattern[MAX_PATTERN_LENGTH];
    int length;
    int hash;
} PatternData;

// Fungsi untuk menghitung hash di CPU
__host__ int computeHash(const char* str, int length, int d, int q) {
    int hash = 0;
    for (int i = 0; i < length; i++) {
        hash = (d * hash + str[i]) % q;
    }
    return hash;
}

// Fungsi CPU untuk verifikasi hasil (untuk mengukur akurasi)
void cpuRabinKarp(const char* text, int text_length, const PatternData* patterns, int pattern_count, 
                  int d, int q, int* results, int* result_count) {
    *result_count = 0;
    
    for (int i = 0; i < text_length; i++) {
        for (int j = 0; j < pattern_count; j++) {
            int pattern_length = patterns[j].length;
            
            if (i + pattern_length > text_length) continue;
            
            // Hitung hash untuk jendela teks saat ini
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

// Kernel CUDA untuk mencari pola dengan algoritma Rabin-Karp
__global__ void rabinKarpKernel(const char* text, int text_length, const PatternData* patterns, 
                               int pattern_count, int d, int q, int* results, int* result_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < text_length; i += stride) {
        // Setiap thread mulai dari indeks berbeda dalam teks
        for (int j = 0; j < pattern_count; j++) {
            int pattern_length = patterns[j].length;
            
            // Pastikan tidak melewati batas teks
            if (i + pattern_length > text_length) continue;
            
            // Hitung hash untuk jendela teks saat ini
            int text_hash = 0;
            for (int k = 0; k < pattern_length; k++) {
                text_hash = (d * text_hash + text[i + k]) % q;
            }
            
            // Jika hash cocok, periksa karakter satu per satu
            if (text_hash == patterns[j].hash) {
                bool match = true;
                for (int k = 0; k < pattern_length; k++) {
                    if (text[i + k] != patterns[j].pattern[k]) {
                        match = false;
                        break;
                    }
                }
                
                if (match) {
                    // Gunakan atomicAdd untuk menghindari race condition saat menambahkan hasil
                    int pos = atomicAdd(result_count, 1);
                    if (pos < MAX_TEXT_SIZE) { // Pastikan tidak melebihi kapasitas buffer hasil
                        results[2*pos] = i;     // Simpan posisi kecocokan
                        results[2*pos+1] = j;   // Simpan indeks pola yang cocok
                    }
                }
            }
        }
    }
}

// Fungsi untuk mengukur waktu eksekusi
double getExecutionTime(clock_t start, clock_t end) {
    return ((double) (end - start)) / CLOCKS_PER_SEC * 1000.0; // Konversi ke milidetik
}

// Fungsi untuk mengecek kecocokan hasil CPU dan GPU
bool compareResults(int* cpu_results, int cpu_count, int* gpu_results, int gpu_count) {
    if (cpu_count != gpu_count) {
        printf("Jumlah hasil tidak sama: CPU=%d, GPU=%d\n", cpu_count, gpu_count);
        return false;
    }
    
    // Menyortir hasil untuk membandingkan (implementasi sederhana)
    for (int i = 0; i < cpu_count; i++) {
        for (int j = i + 1; j < cpu_count; j++) {
            if (cpu_results[2*i] > cpu_results[2*j] || 
                (cpu_results[2*i] == cpu_results[2*j] && cpu_results[2*i+1] > cpu_results[2*j+1])) {
                // Tukar posisi
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
                // Tukar posisi
                int temp1 = gpu_results[2*i];
                int temp2 = gpu_results[2*i+1];
                gpu_results[2*i] = gpu_results[2*j];
                gpu_results[2*i+1] = gpu_results[2*j+1];
                gpu_results[2*j] = temp1;
                gpu_results[2*j+1] = temp2;
            }
        }
    }
    
    // Bandingkan hasil yang telah diurutkan
    for (int i = 0; i < cpu_count; i++) {
        if (cpu_results[2*i] != gpu_results[2*i] || cpu_results[2*i+1] != gpu_results[2*i+1]) {
            printf("Ketidakcocokan pada indeks %d: CPU=(%d,%d), GPU=(%d,%d)\n", 
                   i, cpu_results[2*i], cpu_results[2*i+1], gpu_results[2*i], gpu_results[2*i+1]);
            return false;
        }
    }
    
    return true;
}

int main() {
    // Teks dan pola untuk pencarian
    const char* text = "atgactgaaaatgaacaaattttttggaatagggtccttgaactggcaaaaagtcaactaaaacaagccacttatgaattttttgttttagatgctcgattaattcaaattgagcaaaatacggcgacgatttacctggatcctatgaaagaactcttttgggataaaaatttaaaaccaatcattttaacggctggttttgaggtttataatactgaaattgtcgtgaactatgtctttgaagaagatttagctaaacaagcagtagaagaaccaacttcccaagttctccaagccccacaaaagaatcacctgccacaggttgattcagatttaaatacaaagtatacttttgacaactttgtccaaggtgatgaaaaccgttgggccttttctgcgtcttatgccgttgcggatgctccaggaactacttacaaccctttatttatctggggtggacctgggctcggaaaaactcacttgctaaatgccattggtaatgcggtattgcaaaataatcctaaagcgcgcgtgaagtacatcacagctgaaaatttcatcaatgaatttgttatccatattcgactggatactatggaagaattgaaagaaaaattccgtaatcttgatgttttgctgattgatgacattcaatcgctcgccaaaaaaacattatctggtacgcaagaagagtttttcaatactttcaacgctctttacgataacaacaaacaaatcgtactaaccagtgaccgcacaccagatcacctcgataatctggaacaacgtttggtcacgcgcttcaaatggggcttgacaatcaatatcacgccgcctgattttgaaacacgtgtggcaattttgaccaataaaacgcaagaatacgattttgtgttcccgcaggataccattgaatatcttgctggacaatttgattctaacgtccgtgacctagaaggtgctttaaaggatattagccttgtcgcaagtattaaaaaagtccaaacgattaccgtcgacattgctgctgaggctgtccgagcacgtaaacaagatggtcctaaaatgactgtcattccaattgatgaaattcaaagtcaagttgggaaattctacggtgtcactgttaaggaaatcaaggcaacaaaacgtacgcaagatattgttttagcacgtcaggtagctatgtatttggcacgtgagatgaccgataactcacttccaaaaatcggaaaagagtttggtggtcgtgatcactctaccgttcttcatgcctataacaaaatcaaaaatatgctggctcaagacgatagtctgcgaatcgaaattgaaaccatcaaaaacaagatcagataagtcttgtggataagtatcaaaaaaatctgcgatttatccacaggttattaacaactcttattccgcgatttttctagcgtttttggagttatcaacagtataaacaagacctactactactactaacttaatacttataaattaaaggagttatcttcatgataaaattctcaattaataaatccttctttcttcaagcattgaatgctaccaagagagcaatctcttcgaagaatgccattcctgttttatctactatcaaaattgaagttcaaaatagtaacattaccttaacaggttcaaatggacaaatttcaattgaaaatattattccaacttctaatgaaaatgcaggtcttttgattacatcaccaggtgctatactattagaagcaaatttctttattaatatcgtttcaagtcttccagatgttactttggattttgaagaaattgaacatcaccaaatcgttttgacaagtggtaaatcagaaatcactcttaaagggaaagatgttgacctttacccacgtcttcaagaaatggctactgaaaatcctcttgttattgaaacaaaattattgaagtcaatcattacagaaactgcttttgcagctagtcttcaagaaagtcgtccaatcttgactggtgttcacatggtgcttagcgaccataaagattttaaagcagtagctactgactcacaccgcatgagccaacgcaaattaactttagacaatacttcaaatgactttaacgttgttatcccaagtaaatctcttcgtgaatttgcagcagtcttcacagatgatatcgaaacagtagaagtcttcttctcagacagccaaatgcttttcagaagtgactacattagcttctatacacgtctattagaaggaaattatcctgatacagaccgtttgttgacaaattcatttgaaacagaagtaaccttcaatacaagtgctcttcgttcagctatggaacgtgctcacttgatttcaaatgccacacaaaatggtactgttaaacttgaaattacaaataacagcgtttcagcacacgttaactctcctgaagttggtaaagttaacgaagaattggatattattgatcaatcaggtagtgatttgacaatcagctttaatccaacttacttgattgaagctcttaaagccttgaagagtgaaacagttgttattcgctttatctcaccagttcgtccatttacgttgatgccaggtgatgatgctgaaaactttatccaattaatcacaccggttcgtacaaactaaaattatttaaggaaactttaaggatgaattcgtatactgtaatcatcctaaaacttttctttttcataatctccagcaagagcactgtatagacagtgttcttgctttttatgttataatatttcttgatatgacactttatattattgctaatcctcattcaggaaatcgtagtgcccaagaagttattaaaaggttgaaaaataagttgaatcaagagatagctgtttttctgacgcggtattcagatgatgaagaaaatcaagtcaatgatgttttagagacctttcaacctgagaaagatagattattgattcttggcggtgatggtaccttatccaaagttttatattagctgccagcagatattctattcgcttattatccaatggggtcagggaatgattttgcgcgtgcccttggattgaaaaaagatttaacacaccttattgaatcgactaagcaagctctaaaagaaatcacggtttatacttaccaaaaaggtctagtcttaaacagtttagatttgggttttgcttcttgggttattaatcatgttgaacagtctcagctaaaaactaaactgaacaaatatcatttaggaaaattaacttacatcctaacagcaattcaatgtttgattaagaaaccagcttttgcttcactttgtctagagacagaagcaggggaagtccttgaacttagaaatcagtttttcttttcgctggcaaacaatacttactttggtggtggtgttatgatttggccacatgccacagcttatttggaacagctggattgtgtgtatgctaaaggtgaaactttatggaaacgtgtgcttgtttaattgagattgatggtgaaatcgtgtcatttgacgaaatcacactaacgcctcaaaaacattatatttatttgtaaaggagaaatcatgtacgaaatcggaagtcttgttgagatgaagaaacctcatgcttgtactgttaaagcgactggtaagaaagctaacgtgtgggaagtcgttcgtattggagcagatattaaaattcgttgtacaaactgtgatcatattgtcatgatgagccgtcatgattttgaacgcaagttgaagaaagtaatataatataatatttaaaatcattaagactgcgcatatagcggacttttttaagtgattttatgctataatagttgtgattgaataaatttaacggagactaataaaatatggctttaacagcagggattgttggtttaccaaacgtcggaaaatcaactctttttaatgcgattactaaagcaggtgcggaagctgctaactatcctttcgcgacaatcgatcctaacgttggtatggtagaagttccagatgaacgtctacaaaaattgactgagcttatcacacctaaaaaaacagtcccaacaacatttgaatttactgatattgcaggtatcgtaaaaggggcttctcgtggtgaaggtcttggtaacaaattcctggctaacattcgtgaagttgatgcgattgttcacgttgttcgtgcttttgatgacgaaaacgttatgcgtgaacaagggcgtgaagatgactttgttgacccacttgctgatattgataccattaacttggaattgattcttgctgacttggaatctgttaacaaacgttatgcgcgtgttgaaaaagtagcacgtacacaaaaagacaaagattctgttgcagaatttaacgttcttcaaaaaatcaaaccagtacttgaagatggtaaatcagctcgtacaattgaatttacagaagatgaacaaaaaattgttaaacaattgttccttttgacaacaaaaccagttctttacgtagcaaacgtcgatgaagataaagttgctaatccagatgatatcgaatatgtgaaacaaattcgtgaatttgcagcaactgaaaatgctgaagttgttgttatctcagcacgtgcagaagaagaaatctctgaacttgacgatgaagataaagaattcttccttgaagatcttggtttgactgaatctggtatcgacaaattgactcgtgcagcttaccacttgcttggtcttggaacttatttcactgcaggtgaaaaagaagttcgtgcatggacatttaaacgtggtatcaaagcacctcaagcagctggtatcattcactctgacttcgaacgtggattcatccgtgcaattacaatgtcttatgatgatttaatcaaatacggttcagaaaaagctgtcaaggaagcaggtcgtcttcgtgaagaagggaaagattatgtggttcaagatggtgacatcatggaattccgctttaacgtctaagctaattaataattgaaaatcaaatgaggttggaaaaattttttccagcccttttgacatttgaaaggaaaaataatggtaaaaatgattgttggtctgggaaatccaggcagtaaatatcacgaaacacgtcataatattggatttatggcaattgatcgtctagctaaagatttgaatgtgactttttcagaagacaaaaacttcaaagcagaagttggctctgcctttatcaatggtgaaaaagtgtacttagtcaaaccaactacatttatgaataattcaggaattgcggtacatgctttgttgacttattataatattgccatcgaagattttatggtaatttatgatgacctagatatggaagtcgggcgtattcgtcttcgtcaaaaaggctcagctggtggtcataatggtatcaagtccattattgctcatacaggaacacaagcgtttgaccgtattaaagttgggattggacgtcctaaacaaggtcgctcagtcgttgaccatgtgcttggaaaatttgacaaggatgactatatcacagtcactaacacactagaaaaagttaatgatactgttcaattttatttacaggaagctgattttgtaaaaacaatgcaaaaatttaatgggtaaatctatgaatattattgatttatttagccaaaataagctaatccaaagttggcatgcaggcgtgtctaatcttggtagacagcttattatgggcttgtcaggagcaagtagagctcttgctatagcttcagcttatcaagctaatgaagaaaaaatagtgattatcacatcaactcaaaatgaagttgaaaaattggctagtgatttatctagtttgattggtgaggataaggtttatacgttttttgcggatgatgttgctgctgcagaattcatttttgcttcgatggataaagctcattctcgcttagaagctttgaattttttacaagataaaaatcaatcaggtatcttgattacgagtttagttggtgctcgtgtcttgttgccaagtcctaaaacttattcagaaagtcaattgaattttcttgtcggagatctttacaaccttgacaagattgtaaagatcttgtcaaatgtcggttatcaaaaagtttctcaggttttgaatccaggcgagtttagccgacgtggtgatattgtcgatatctatgagattacagcagattatccgtaccgcttggagttctttggtgatgaagttgatggtattcgtcaatttgatgcacagactcaaaaatctcttagcaatgttgaacaagtaacgatttatccagcggatgagttgattttgtcagaagaggattttgctagagctagtcaagcttttgagaagtatcttgagactgcaaaggacgaccaacaagcttatttgagtgaactgtatgcagcaacacaagagcagtatagacatcaagatattagacgttttttgtctcttttttatgctaaagagtggacacttctcgattatattccaaaagggacaccagtcttctttgatgattttcaaaagctggttgatcgcaacactaagtttgatttagaggttgctaatctattgacagaagatttacagcgtggtaaagctgtgtcaatattggtttacttcgctgatatttataaaaaattacgtcagtaccagccagcgaccttcttttcaaactttcataaaggcttgggaaatctgaaatttgataaattacacaatttcactcaatacccaatgcaagaatttttcaatcaatttccgttgctgattgatgaaattaatcgttatcaaaaatcaaaagctactattttagttcaatcagatacgcagcatggggtagaacgtctgcaagagaacctgcaagaatatggtttggatttaccgatagttggtgcaaatgatttacaagaacatcaagcgcaattagtagttggcgatttatcaaatggtttttactttgctgacgataaaattgttttaatcactgagcacgaaatttatcataaaaaagtcaaacgtcgcatccgtcgttcaaatatcagcaatgcagaacgtttgaaagattacaatgagctttctaaaggtgattatgtggttcaccatgttcatggaattggtcaatttttagggattgaaaccattgaaattcatggggtacatcgtgactatttaacgattcaataccaagggtcttcaacgatttctcttcctgtagaacaaattgaaagtttatcaaaatatgtttctgcagatggtaaagaacccaaaatcaataaattaaatgatgggcgtttccaaaagaccaaacaaaaagtttctaaacaagttgaagacattgctgatgatttgttgaaactatatgcggaacgcaatcaattgaaaggttttgccttctcaccagatgatgagttacaaaaagagtttgatgaagactttgcttacgtagagacagaagaccaacttcgttctataaaagaaatcaaacacgatatggaagaagaaaagccaatggatcgccttttggtcggtgatgttggttttggtaaaacagaggttgccatgcgcgcagcctttaaggctgtaaaagaccacaaacaagtagccgtcttggtaccgactactgttcttgctcaacagcacttcacaaatttctcagagcgttttgagaattatcctgtagcggttgatgtgcttagtcgtttccaaagtaaaaaagagcaggcagccactttggaaaaattgaaaaaaggtcaagttgatattattatcggaactcaccgattgttatcaaaagatgtagagtttgcagatttgggcttgattattatcgatgaagagcaacgttttggcgttaaacacaaggaaaaattaaaagagcttaagactaaagttgatgttttaaccttaacagcaactccaattcctcgtacacttcacatgtctatgcttggtatccgtgatttatctgtgattgaaacaccaccgacaaatcgttaccctgttcagacttatgtcttggaaactaatccaggtctaattcgtgaggcgattattcgtgaaattgatcgtggtggtcaagttttctacgtttacaatcgcgttgacacgattgatcaaaaggtttctgagttacaagaattggtgcctgaagcaagtattggatttgttcatggtcaaatgagtgaaattcagcttgaaaataccttgatggactttatcgaaggtgtttacgatgttctcgttgccacaacgattattgagacgggtgttgatatttctaatgtcaatacacttttcattgaaaatgctgatcacatgggcttgtcaaccttgtatcaacttcgtggtcgtgttggtcgttcaaatcgtattgcttatgcttatctcatgtatcgtccagataagattttgaacgaagtctctgagaaacgtttggatgccatcaaaggatttacagagcttggttctgggttcaaaattgctatgcgtgacttatccattcgtggtgcgggaaatattttgggtgcctcacagagcggctttattgattctgtcggttttgagatgtattctcaattattagaagaagcgattgctaagaaacaaggtaaatcccaagcacgtcgtaagagtaatgctgaaattaacctccaaatcgatgcttatcttccaagcgagtacattgatgatgaacgtcaaaaaatcgagatttacaaacgtatccgtgaaattgatagtcttaaagattaccaaagtttacaagatgaattgattgaccgttttggagaatacccagatcaggtagcttaccttcttgagattggtcttgtgaaatcctacatggacaatgcctttacagaacttgttgaacgtaaagataataatcttttggtgcgttttgaaaaagcttcgctacaacattatttgactcaagattattttgaggctctttctaagacagatttgaaagctcgaatcggtgagaatcaaggtagaattgacattacctttaatgtcaaaggtaaaaaggactatgaaattttagaagaactccaaaaatttggtagcatgctagctaagattaaagcaagaaaaactaaagaagattaattgtcaaagccttttgacagtagaaagctagtttagcaaagagattttatggtaaaatttttggtaaagtttagactgatattatcaaggaggtcaaaatggaacttgaacgcagattagcttaattgagaagagaaaagaatctctctcaagaagagttagcagaaaaactttatgtttcacgtcaaacaatttctaattgggagagggacaagacttacccagatattaattcacttttgttgatggctaattattttgatgtttccttagatcatctaattaaaggagatgttgatattatgaaacatcaggttgaccagtcgcaatttaagaaatggcttatccttggaggaatttcttggtttattttttcagttgcctttggaacacgttatttatttgataaagggcaagtagttgctattttgactttattagcattgccagttgcctattctctttttcaaatattatatatcgtaaaaaatcgcgaacttcagacctatacagacattctcaatttttttaatccagagaaaaaagttaagataaacttttttagagaagtatggtttgtctttagtttgttaactatcacttggtttattttctttgttggagttgccgtttctaccctagttttttattaaaacattgtctatctaactttgtagaactaagagagtgcaaactctctttttattacattatcttaacaaacagattgaaaacatctattttctgaaaaaaagtgataaaataaaaaagattgcgaggtaaaatattatgagattagataaatatttaaaagtatcccgtattataaaacgccgtccagttgctaaagaagtagctgataaagggcgtattaaagtaaatggtatcttagctaaatcatcaactgatttgaaaattaacgacgaagttgaaattcgttttggaaataaattactgacagtgtgcgttttggaaatgaaagatagtaccaaaaaagaagatgccactaagatgtatgaaattattaatgaaacaaggatagaagcggatggagaagcctaatatcgttcaattgaacaatcaatacatcaatgatgaaaataccaaaaaacgttacatggaagaagaaaatcgtaaacgtaatcgttttatgggatggattttgattgttgttatgttgttgtttattttgccaacttacaacctagttaaaagttaccaaaccttacaagagcgtaaagataaagtcgtgtcacttcagaaaacttacgacaatcttgttgtagaaacagacgctgaaaaagtcttggcaaaccgcttaaaagacgagaactacgttgaaaagtatgctcgtgcaaaatattatttatcacgctcaggcgaaacggtttacccattgccaaatctattaccaaaataatatggaaaatttaatcgaaactattgaaaaatttttagcttattctgatgaaaagctagaagaattggctgaagaaaatcaaaaattacgagaagaaaatcaccaatcaactaaaaaagaatgaaaggaggagccaatgaaaaaactattcgcagtgatgctaataccatttttcttgacttctctttctgctgttagtaccgaaaaagatgtagacttaagcaacgcagcaaaatatcaattgacagctagtgtggtagcatcgacaacttactttgatactgtgccatcaaaccctgtccttgctaaaactagcgaggtgtataaagatagtaatctgacagttgtttctagaacaatcgctgctaatagtcaattggatattgataacattctgattaacgataattcgcaacctgtttttgagttgtctgacggtaattatatcgaggctagtcgccaattgatatatgatgatgtcattcttagtcaagtgcccctttcagcgaaatattggttaaaagatgattttcaagtttatcaattaccatatgttattggaacacaagaagaacaaacagatttaactgcttatcaagcagtaactgtttcacaaaaagcaacaactcatcatggaacttattacaaagttgatggtaaaggttggattagcgaagaagctctttctagtactggtaatcgtatggaaaaagttcagcaggttctcaatcagaaatacaataaaggtaactattccatctacgttaaacaactgtcaacccgagaaacagcagggattaatcctgataccacaatgtacgcagctagtattgccaaattagcgacactatattatgttgaagaaaaaattcaggatggcagtgttaagatggatgataaattgaaatacactgctgatgtgaacacttttgccaaagcttataacccaagtggtagcggaaaaattagcaagacagctgataataaggattacagcattgaagatttattaaaagctgtttgtcagaactctgataacgtagggacaaatattttagggtattatgttgctaaacaatatggtgaccatttcacttcagatattagtgcgataaccaatactaattttgatatgaagacacgcgaaatgtcttctaaaacagctgctgatttaatggaagctatttatcagcaaaatggcgaagtcatttcgtatttgtcatctactgcttttgacaatgcacgtatctctaaagatattaatgttcaagtagctcataaaattggtgatgcttacgactacagacatgacgttgctattgtttatgctgatcaaccgttcatcttatcaatttttaccaataatgcttcttacgacgatatttcaaatatcgctaatgacgtttacaatattttaaaataagtgatatcttatgtatgataaattagtaaaagaaattcaaaaaaaggcttgttttgccaaacacaaaaaagttttaatagctgtatcaggtggagtagattctatgaatctgctacactttttgtatacataccgaaaaaatcttggaatagaaattgctattgctcacgtgaatcataaacaacgtcttgaatcagatactgaggaagcttatttgcgagaatgggcaaaagagcaccaaattccaattcatgtagcctacttttctgggactttttcagaaacagctgcgcgtgattttcgttacaaatttttaaaagagctgatgagtaactatcattatacagcacttgtaacagcacatcatgctgatgatcaagcagagacgatttttatgcgtctgttacgcggcagtcgtttacgacatttatcagcaatgcaaactgtgaatgcatttggtgatggtgaactgattcgtccatttttgacctattcaaaaaaagagctacctaatatttttcattttgaagatgacagcaatcgctctttagattttttgagaaatcgtattaggaatcgctattttccaatgctagaggaagagaatcctaaatttaaacaagcccttcgttatctgggaaaggaaagtcagcaacttttccaagcttttcaggatttaacaaagaatattgatacgacaaattgttctcaatttttagctcagactcctgcagttcaaacaattctttttcaaaattaccttgaaaactttcctgaattgcaagtgtctcatggacaatttgaggaaatgttgcatcttttaagaaataaatctaatgctaactatcatgtgaaatcagattactggctgtttaaagattatgaaacatttcaagtgaaaaaaatcagccctaagaccgatggggaactagctcaaaagatgctagaatataacagtatagtcaattatggtcagtttcaatttcaatttgttagcaaagataaggaaggcatagccgtttacagtttaaatcctattcttttgagaaaaagacaggaaggtgaccgcattgattttggagatttttcgaaaaaattgcgtaggctttttatcgatgagaaaattccaagtcaagaacgtgctaatgccattatcggcgagcaggacggaaaaattatttttgtccaggttgctgataaaacttatttgagaaaaccttctaaacatggtataatggagggcaaattgtatattgaaaaaaatagaaataggtgatgtcatgcttgaacaagatattgaaaaagtgttgtattctgaagaggagattattgccaaaaccaaagaattaggcacacaacttacaaaagactatgaaggtaaaaatcctctattaattggtgttttgaaaggttctgtaccttttatggcagaattgattaaacatatcgatacacatgtcgaaattgattttatggttgtctctagctaccacggcggagtaaaaagcagtggtgaagtaaaaatccttaaagacgttgatacgaatattgaaggtcgtgatgttgtctttatcgaagatattatcgatacaggccgtactctaaaataccttcgtgatatgttcaaataccgtaaagcaaattcagtaaaaattgcaaccctttttgataaaccagaaggtcgtgttgttgacatcgaagctgattatgtctgctacgatgttccaaacgaattcattgtagggtttggtcttgactatgctgaacgttaccgtaatttaccatatgttggcgttttgaaaaaagaaatctattcaaattagtagaaagtcgttagttatatatgaagaataacaaaaataatggctttgttaaaaattcctttatttatattctggttattattgcggctattaccgcatttcaatactatttgagaggaacaagtacacaaagccagcaaattaattactcgactttgattaaagaaattaaagcgggtgatattaaatcgattacttaccagccaaacggtagcatcattgaagtaagtggtaagtattcgaaacctaaaaagacaaaatcaagtgcctctcttcctttcttagaaggaaataactcaacatcagttactgagtttacatcaattattttaccaagtgattcttctattgatacacttactaaagcagctgaaaaagctgatgttgatgtaaacattaagcaagaaagttcaagcggagcttggctatctttggcactaacttatcttccactcatcttagtgattgccttcttcttcatgatgatgaatcaaggtggtaatggtgcacgtggtgcgatgagttttggtaagaatcgagctaagtcacaaaataaaggtgatgttaaagtaagattctctgatgttgctggagcagaagaagaaaaacaagaacttgttgaagttgttgatttcttgaaaaatcctaagaaatataaagcacttggtgctcgtatcccagctggtgtccttcttgaaggccctccaggaacaggtaaaacattacttgcgaaagctgttgctggtgaggctggcgtcccattcttctcaatttcaggttctgattttgttgaaatgtttgtgggtgtcggtgcgagtcgagttcgctcactctttgaagatgctaaaaaagctgaacgtgcgattatctttatcgatgaaattgatgctgttggtcgcagacgtggtgctggtatgggtggcggaaatgatgaacgcgaacaaactcttaaccaactccttattgaaatggatggttttgaaggaaacgaaaacattatcgttattgccgcaactaaccgtagtgatgtcttagaccctgctcttttacgtccaggacgttttgaccgtaagattcttgtaggaagtccagacgttaagggacgtgaagctattcttcgagtacacgctaaaaacaaaccattggcagaagatgttaacttaaaagtcgttgcacaacaaacacctggatttgtcggtgctgatcttgaaaatgtcttgaatgaagctgccttggttgctgcacgtcgtaacaaaactaaaattgatgcttcagatattgatgaagctgaagatagagttattgcaggtccttctaagaaagaccgtacgatttcacaacgtgaacgcgaaatggttgcttaccatgaagctggacacacaatcgttggtttagtgctctcaagtgcccgcgttgtccataaagttacaattgtccctcgtggacgtgctggtggatatatgattgctcttccaaaagaagatcaaatgttacactctaagtctgatcttaaagaacagttggctggtttgatgggtggtcgtgttgctgaagaaatcatcttcaatgctcaaacaactggtgcatcaaacgacttcgaacaagcaactcaattagctcgtgcaatggtaactgaatatggtatgagtgaaaaactaggtcctgttcaatatgaaggtaaccatgctatcatggctggtcaagtttctccggaaaaatcatattcatctcaaacagcacaaatgattgatgatgaagttcgtgcgcttctaaacgaagcacgtaataaagcagctgatatcatcaataataatcgagaaactcacaaacttattgcagaagcacttcttaaatatgaaacacttgatgctgcacaaattaaatctatctatgaaacaggtaaggttcctgaagaattagaaaatgatactgaagaggcacacgccctttcatatgatgaactcaaagaaaagatggatgattctgaagaataatctgcattaattatatgtagaaagctatcagaaacgcccaagaaggctcgtagagccttctttttttattaatagataaaaaagatatgagttgaataatatgctaaaaatgagccaaacagcagccagtgtaaatgatgttagaataaatcaaaaaaacttaaaaaacatattgacagtatagatgaagtttagtatactaaataagctgttgtttgaggcagcaaatatttcaaaaatgctttgaaaaaagttcaaaaaacttcttgacaaagcgttgaagaaatgataaactaatatagctgttgttcgagagggtgacagacacaagtttgaaaagagtttgaaactttcttgaaaaaagatgttgacaacgcgtttcaaatttgatagaatatagaagttgtctcgagtaagacaaagacctttgaaaactgaacaatacgaaccaaacgtgcgggttacggaagtaacctgtcaaaaaaacgataaatctgtcagtgacagaatgagaacaagattaaatgagagtttgatcctggctcaggacgaacgctggcggcgtgcctaatacatgcaagtagaacgctgaagactttagcttgctaaagttggaagagttgcgaacgggtgagtaacgcgtaggtaacctgcctactagcgggggataactattggaaacgatagctaataccgcataacagcatttaacccatgttagatgcttgaaaggagcaattgcttcactagtagatggacctgcgttgtattagctagttggtgaggtaacggctcaccaaggcgacgatacatagccgacctgagagggtgatcggccacactgggactgagacacggcccagactcctacgggaggcagcagtagggaatcttcggcaatgggggcaaccctgaccgagcaacgccgcgtgagtgaagaaggttttcggatcgtaaagctctgttgtaagagaagaacgtgtgtgagagtggaaagttcacacagtgacggtaacttaccagaaagggacggctaactacgtgccagcagccgcggtaatacgtaggtcccgagcgttgtccggatttattgggcgtaaagcgagcgcaggcggtttaataagtctgaagttaaaggcagtggcttaaccattgttcgctttggaaactgttagacttgagtgcagaaggggagagtggaattccatgtgtagcggtgaaatgcgtagatatatggaggaacaccggtggcgaaagcggctctctggtctgtaactgacgctgaggctcgaaagcgtggggagcaaacaggattagataccctggtagtccacgccgtaaacgatgagtgctaggtgttaggccctttccggggcttagtgccgcagctaacgcattaagcactccgcctggggagtacgaccgcaaggttgaaactcaaaggaattgacgggggcccgcacaagcggtggagcatgtggtttaattcgaagcaacgcgaagaaccttaccaggtcttgacatcccgatgctattcctagagataggaagtttcttcggaacatcggtgacaggtggtgcatggttgtcgtcagctcgtgtcgtgagatgttgggttaagtcccgcaacgagcgcaacccctattgttagttgccatcattaagttgggcactctagcgagactgccggtaataaaccggaggaaggtggggatgacgtcaaatcatcatgccccttatgacctgggctacacacgtgctacaatggttggtacaacgagtcgcgagtcggtgacggcaagcaaatctcttaaagccaatctcagttcggattgtaggctgcaactcgcctacatgaagtcggaatcgctagtaatcgcggatcagcacgccgcggtgaatacgttcccgggccttgtacacaccgcccgtcacaccacgagagtttgtaacacccgaagtcggtgaggtaaccttttaggagccagccgcctaaggtgggatagatgattggggtgaagtcgtaacaaggtagccgtatcggaaggtgcggctggatcacctcctttctaaggataaacggaagcacgtttgggtattgtttagttttgagaggtcttgtggggccttagctcagctgggagagcgcctgctttgcacgcaggaggtcagcggttcgatcccgctaggctccattgaatcgaaagattcaaaagattgtccattgaaaattgaatatctatatcaaattccacgattcaagaaattgaattgtagatagtaacaagaaataaaccgaaacgctgtgatttaatgagtttaaggtcaacagaccaaaataaggttaagttaataagggcgcacggtggatgccttggcactagaagccgatgaaggacgtgactaacgacgaaatgctttggggagttgtaagtaaacattgatccagagatgtccgaatgggggaacccggcatgtaatgcatgtcactcattactgttaaggtaatgaagaggaagacgcagtgaactgaaacatctaagtagctgcaggaagagaaagcaaacgcgattgccttagtagcggcgagcgaaatggcaagagggcaaaccgatgtgtttacacatcggggttgtaggactgcgacgtgggacgacaagattatagaagaattacctgggaaggtaagccaaagagagtaacagcctcgtattcgaaatagtctttaaccctagcagtatcctgagtacggcgagacacgagaaatctcgtcggaatctgggaggaccatctcctaaccctaaatactctctagtgaccgatagtgaaccagtaccgtgagggaaaggtgaaaagcaccccgggaggggagtgaaatagaacctgaaaccgtgtgcctacaacaagttcgagcccgttaatgggtgagagcgtgccttttgtagaatgaaccggcgagttacgatatgatgcgaggttaagttgaagagacggagccgtagggaaaccgagtcttaatagggcgcattagtatcatgtcgtagacccgaaaccatgtgacctacccatgagcagggtgaaggtgaggtaaaactcactggaggcccgaaccagggcacgttgaaaagtgcttggatgacttgtgggtagcggagaaattccaaacgaacttggagatagctggttctctccgaaatagctttagggctagcgtcgatgttaagtctcttggaggtagagcactgtttgattgaggggtccatcccggattaccaatatcagataaactccgaatgccaatgagatataatcggcagtcagactgcgagtgctaagatccgtagtcgaaagggaaacagcccagaccaccagctaaggtcccaaaatatatgttaagtggaaaaggatgtggggttgcacagacaactaggatgttagcttagaagcagctattcattcaaagagtgcgtaatagctcactagtcgagtgaccctgcgccgaaaatgtaccggggctaaaacatattaccgaagctgtggataccttttaggtatggtaggagagcgttctatgtgtgaagaaggtgtaccgtgaggagcgctggaacgcatagaagtgagaatgccggtatgagtagcgaaagacaggtgagaatcctgtccaccgtatgactaaggtttccaggggaaggctcgtcctccctgggttagtcgggacctaaggagagaccgaaaggtgtatccgatggacaacaggttgatattcctgtactagagtatatagtgatggagggacgcagtaggctaactaaagcgtgcgattggaagtgcacgtctaagcagtgaggtgtgatatgagtcaaatgcttatatctctaacattgagctgtgatggggagcgaagttaagtagcgaagttagtgatgtcacactgccaagaaaagcttctagcgttaattatactctacccgtaccgcaaaccgacacaggtagtcgaggcgagtagcctcaggtgagcgagagaactctcgttaaggaactcggcaaaatggccccgtaacttcgggagaaggggcgctggctttaagtcagccgcagtgaataggcccaagcaactgtttatcaaaaacacagctctctgctaaatcgtaagatgatgtatagggggtgacgcctgcccggtgctggaaggttaagaggagcgcttagcattagcgaaggtgtgaattgaagccccagtaaacggcggccgtaactataacggtcctaaggtagcgaaattccttgtcgggtaagttccgacccgcacgaaaggcgtaatgatttgggcactgtctcaacgagagactcggtgaaattttagtacctgtgaagatgcaggttacccgcgacaggacggaaagaccccatggagctttactgcagtttgatattgagtatctgtaccacatgtacaggataggtaggagcctatgaaatcgggacgctagtttcggtggaggcgttgttgggatactacccttgtgttatggctactctaacctagataggctatccctatcggagacagtgtctgacgggcagtttgactggggcggtcgcctcctaaaaggtaacggaggcgcccaaaggttccctcagaatggttggaaatcattcgcagagtgtaaaggtataagggagcttgactgcgagagctacaactcgagcagggacgaaagtcgggcttagtgatccggtggttccgcatggaagggccatcgctcaacggataaaagctaccctggggataacaggcttatctcccccaagagttcacatcgacggggaggtttggcacctcgatgtcggctcgtcgcatcctggggctgtagtcggtcccaagggttgggctgttcgcccattaaagcggcacgcgagctgggttcagaacgtcgtgagacagttcggtccctatccgtcgcgggcgtaggaaatttgagaggatctgctcctagtacgagaggaccagagtggacttaccgctggtgtaccagttgtcttgccaaaggcatcgctgggtagctatgtagggaagggataaacgctgaaagcatctaagtgtgaagcccacctcaagatgagatttcccatgattttatatcagtaagagccctgagagatgatcaggtagataggttagaagtgtaagtgtggtgacacatgtagcggactaatactaatagctcgaggacttatccaaaaaagaaacgagtcaatattgacagcgcgtaggtttcttgttagaatatataggtattcaattttgattggataatcaatcatagttaagtgacgatagcctaggagatacacctgttcccatgccgaacacagaagttaagccctagcacgcctgatgtagttgggggttgccccctgttagatatggtagtcgcttagcaaaagggagtttagctcagctgggagagcatctgccttacaagcagagggtcagcggttcgatcccgttaactcccattaacaatggtatataaaccaaaaggtcccgtggtgtagcggttatcacgtcgccctgtcacggcgaagatcgcgggttcgattcccgtcgggaccgtttgactcgttagctcagttggtagagcatctgacttttaatcagagggtcactggttcgagcccagtacgggtcatattaatatgcgggtttggcggaattggcagacgcaccagatttaggatctggcgcttaacggcgtgggggttcaagtcccttaacccgcattaaataagaagttgccggcttagctcagttggtagagcatctgatttgtaatcagagggtcgcgtgttcaagtcatgtagccggcatataggatactctttatgcgaacgtagttcagtggtagaacatcaccttgccaaggtgggggtcgcgggttcgaaccccgtcgttcgcttgttagaggccggggtggcggaactggcagacgcacaggacttaaaatcctgcgaagggtaaccttcgtaccggttcgattccggtcctcggcagtatacaatttaatattacgatgcacccttggctcaactggatagagtacctgactacgaatcaggcggttgcaggttcgaatcctgcagggtgcataatttcgggaagtagctcagcttggtagagcacttggtttgggaccaaggggtcgcaggttcgaatcctgtcttcccgatgtgaattctcgaaattcacaaagacctttgaaaactgaacaatacgaaccaaacgtgcgggttacggaagtaacctgtcaaaaaaacgataaatctgtcagtgacagaatgagaacaagattaaatgagagtttgatcctggctcaggacgaacgctggcggcgtgcctaatacatgcaagtagaacgctgaagactttagcttgctaaagttggaagagttgcgaacgggtgagtaacgcgtaggtaacctgcctactagcgggggataactattggaaacgatagctaataccgcataacagcatttaacccatgttagatgcttgaaaggagcaattgcttcactagtagatggacctgcgttgtattagctagttggtgaggtaacggctcaccaaggcgacgatacatagccgacctgagagggtgatcggccacactgggactgagacacggcccagactcctacgggaggcagcagtagggaatcttcggcaatgggggcaaccctgaccgagcaacgccgcgtgagtgaagaaggttttcggatcgtaaagctctgttgtaagagaagaacgtgtgtgagagtggaaagttcacacagtgacggtaacttaccagaaagggacggctaactacgtgccagcagccgcggtaatacgtaggtcccgagcgttgtccggatttattgggcgtaaagcgagcgcaggcggtttaataagtctgaagttaaaggcagtggcttaaccattgttcgctttggaaactgttagacttgagtgcagaaggggagagtggaattccatgtgtagcggtgaaatgcgtagatatatggaggaacaccggtggcgaaagcggctctctggtctgtaactgacgctgaggctcgaaagcgtggggagcaaacaggattagataccctggtagtccacgccgtaaacgatgagtgctaggtgttaggccctttccggggcttagtgccgcagctaacgcattaagcactccgcctggggagtacgaccgcaaggttgaaactcaaaggaattgacgggggcccgcacaagcggtggagcatgtggtttaattcgaagcaacgcgaagaaccttaccaggtcttgacatcccgatgctattcctagagataggaagtttcttcggaacatcggtgacaggtggtgcatggttgtcgtcagctcgtgtcgtgagatgttgggttaagtcccgcaacgagcgcaacccctattgttagttgccatcattaagttgggcactctagcgagactgccggtaataaaccggaggaaggtggggatgacgtcaaatcatcatgccccttatgacctgggctacacacgtgctacaatggttggtacaacgagtcgcgagtcggtgacggcaagcaaatctcttaaagccaatctcagttcggattgtaggctgcaactcgcctacatgaagtcggaatcgctagtaatcgcggatcagcacgccgcggtgaatacgttcccgggccttgtacacaccgcccgtcacaccacgagagtttgtaacacccgaagtcggtgaggtaaccttttaggagccagccgcctaaggtgggatagatgattggggtgaagtcgtaacaaggtagccgtatcggaaggtgcggctggatcacctcctttctaaggataaacggaagcacgtttgggtattgtttagttttgagaggtcttgtggggccttagctcagctgggagagcgcctgctttgcacgcaggaggtcagcggttcgatcccgctaggctccattgaatcgaaagattcaaaagattgtccattgaaaattgaatatctatatcaaattccacgattcaagaaattgaattgtagatagtaacaagaaataaaccgaaacgctgtgatttaatgagtttaaggtcaacagaccaaaataaggttaagttaataagggcgcacggtggatgccttggcactagaagccgatgaaggacgtgactaacgacgaaatgctttggggagttgtaagtaaacattgatccagagatgtccgaatgggggaacccggcatgtaatgcatgtcactcattactgttaaggtaatgaagaggaagacgcagtgaactgaaacatctaagtagctgcaggaagagaaagcaaacgcgattgccttagtagcggcgagcgaaatggcaagagggcaaaccgatgtgtttacacatcggggttgtaggactgcgacgtgggacgacaagattatagaagaattacctgggaaggtaagccaaagagagtaacagcctcgtattcgaaatagtctttaaccctagcagtatcctgagtacggcgagacacgagaaatctcgtcggaatctgggaggaccatctcctaaccctaaatactctctagtgaccgatagtgaaccagtaccgtgagggaaaggtgaaaagcaccccgggaggggagtgaaatagaacctgaaaccgtgtgcctacaacaagttcgagcccgttaatgggtgagagcgtgccttttgtagaatgaaccggcgagttacgatatgatgcgaggttaagttgaagagacggagccgtagggaaaccgagtcttaatagggcgcattagtatcatgtcgtagacccgaaaccatgtgacctacccatgagcagggtgaaggtgaggtaaaactcactggaggcccgaaccagggcacgttgaaaagtgcttggatgacttgtgggtagcggagaaattccaaacgaacttggagatagctggttctctccgaaatagctttagggctagcgtcgatgttaagtctcttggaggtagagcactgtttgattgaggggtccatcccggattaccaatatcagataaactccgaatgccaatgagatataatcggcagtcagactgcgagtgctaagatccgtagtcgaaagggaaacagcccagacca";
    const char* patterns[] = {"tgcgata", "tgtg", "aaag", "cctct", "aagg"};
    int num_patterns = 5;
    
    int text_length = strlen(text);
    int d = 256; // Ukuran alfabet
    int q = 101; // Bilangan prima untuk hash
    
    // Alokasi memori untuk data pola di host
    PatternData* h_patterns = (PatternData*)malloc(num_patterns * sizeof(PatternData));
    
    // Inisialisasi data pola
    for (int i = 0; i < num_patterns; i++) {
        strncpy(h_patterns[i].pattern, patterns[i], MAX_PATTERN_LENGTH);
        h_patterns[i].length = strlen(patterns[i]);
        h_patterns[i].hash = computeHash(patterns[i], h_patterns[i].length, d, q);
    }
    
    // Alokasi memori di GPU
    char* d_text;
    PatternData* d_patterns;
    int* d_results;
    int* d_result_count;
    
    cudaMalloc((void**)&d_text, text_length * sizeof(char));
    cudaMalloc((void**)&d_patterns, num_patterns * sizeof(PatternData));
    cudaMalloc((void**)&d_results, 2 * MAX_TEXT_SIZE * sizeof(int)); // Untuk menyimpan (posisi, indeks pola)
    cudaMalloc((void**)&d_result_count, sizeof(int));
    
    // Salin data ke GPU
    cudaMemcpy(d_text, text, text_length * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_patterns, h_patterns, num_patterns * sizeof(PatternData), cudaMemcpyHostToDevice);
    cudaMemset(d_result_count, 0, sizeof(int));
    
    // Konfigurasi kernel
    int blocks = (text_length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // Ukur waktu eksekusi GPU
    clock_t gpu_start, gpu_end;
    gpu_start = clock();
    
    // Jalankan kernel
    rabinKarpKernel<<<blocks, THREADS_PER_BLOCK>>>(d_text, text_length, d_patterns, num_patterns, d, q, d_results, d_result_count);
    
    // Tunggu sampai kernel selesai
    cudaDeviceSynchronize();
    
    gpu_end = clock();
    double gpu_time = getExecutionTime(gpu_start, gpu_end);
    
    // Periksa error CUDA
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Salin hasil kembali ke host
    int h_gpu_result_count;
    int* h_gpu_results = (int*)malloc(2 * MAX_TEXT_SIZE * sizeof(int));
    
    cudaMemcpy(&h_gpu_result_count, d_result_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gpu_results, d_results, 2 * h_gpu_result_count * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Ukur waktu eksekusi CPU untuk perbandingan
    clock_t cpu_start, cpu_end;
    int h_cpu_result_count;
    int* h_cpu_results = (int*)malloc(2 * MAX_TEXT_SIZE * sizeof(int));
    
    cpu_start = clock();
    cpuRabinKarp(text, text_length, h_patterns, num_patterns, d, q, h_cpu_results, &h_cpu_result_count);
    cpu_end = clock();
    double cpu_time = getExecutionTime(cpu_start, cpu_end);
    
    // Tampilkan hasil
    printf("========= HASIL PENCARIAN =========\n");
    printf("Pola yang ditemukan:\n");
    for (int i = 0; i < h_gpu_result_count; i++) {
        int position = h_gpu_results[2*i];
        int pattern_idx = h_gpu_results[2*i+1];
        printf("Pattern \"%s\" ditemukan pada posisi %d\n", 
               patterns[pattern_idx], position);
    }
    
    // Bandingkan hasil CPU dan GPU untuk mengukur akurasi
    bool accurate = compareResults(h_cpu_results, h_cpu_result_count, h_gpu_results, h_gpu_result_count);
    
    // Tampilkan statistik performa
    printf("\n========= STATISTIK PERFORMA =========\n");
    printf("Waktu eksekusi CPU: %.3f ms\n", cpu_time);
    printf("Waktu eksekusi GPU: %.3f ms\n", gpu_time);
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    printf("Akurasi: %s\n", accurate ? "100% (Semua hasil cocok)" : "Tidak akurat (Ada perbedaan hasil)");
    printf("Jumlah kecocokan ditemukan: %d\n", h_gpu_result_count);
    
    // Bersihkan memori
    free(h_patterns);
    free(h_gpu_results);
    free(h_cpu_results);
    cudaFree(d_text);
    cudaFree(d_patterns);
    cudaFree(d_results);
    cudaFree(d_result_count);
    
    return 0;
}