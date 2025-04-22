#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_TEXT_SIZE 2000000  // Increased for 1M file
#define MAX_PATTERN_SIZE 1024
#define MAX_PATTERNS 1024
#define MAX_DISPLAY 20  // Maximum number of matches to display

typedef struct {
    char pattern[MAX_PATTERN_SIZE];
    int length;
    int hash;
} PatternInfo;

/*
 * Compute hash value of a string of given length
 */
int compute_hash(char *str, int length, int d, int q)
{
    int i = 0;
    int hash_value = 0;

    for (i = 0; i < length; ++i) {
        hash_value = (d * hash_value + str[i]) % q;
    }

    return hash_value;
}

/*
 * Read patterns from a file
 * Format: comma-separated list of patterns (e.g., "GGGCA,TCGTG,AAAAA")
 */
int read_patterns_file(const char* filename, PatternInfo* patterns)
{
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error opening patterns file %s\n", filename);
        exit(1);
    }

    // Allocate a much larger buffer for reading patterns
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
        strcpy(patterns[pattern_count].pattern, token);
        patterns[pattern_count].length = strlen(token);
        token = strtok(NULL, ",");
        pattern_count++;
    }

    free(buffer);
    return pattern_count;
}

/*
 * Rabin-Karp string matching algorithm for a single pattern
 * Returns the number of matches found
 */
int rk_matcher_single(char *text, int text_length, PatternInfo *pattern_info, 
                      int d, int q, int *display_count)
{
    int i, j;
    int pattern_length = pattern_info->length;
    int pattern_hash = pattern_info->hash;
    int text_hash = 0;
    int h = 1;
    int matches = 0;

    // Calculate h = d^(pattern_length-1) % q
    for (i = 0; i < pattern_length - 1; i++) {
        h = (h * d) % q;
    }

    // Calculate hash value for first window of text
    text_hash = compute_hash(text, pattern_length, d, q);

    // Slide the pattern over text one by one
    for (i = 0; i <= text_length - pattern_length; i++) {
        // Check if hash values match
        if (pattern_hash == text_hash) {
            // Check for exact character match
            for (j = 0; j < pattern_length; j++) {
                if (text[i + j] != pattern_info->pattern[j])
                    break;
            }
            if (j == pattern_length) {
                matches++;
                // Only display a limited number of matches to avoid flooding the console
                if (*display_count < MAX_DISPLAY) {
                    printf("Pattern '%s' found at index %d\n", pattern_info->pattern, i);
                    (*display_count)++;
                    if (*display_count == MAX_DISPLAY) {
                        printf("... (showing only first %d matches)\n", MAX_DISPLAY);
                    }
                }
            }
        }

        // Calculate hash value for next window of text
        if (i < text_length - pattern_length) {
            // Remove leading digit, add trailing digit
            text_hash = (d * (text_hash - text[i] * h) + text[i + pattern_length]) % q;
            
            // We might get negative value, convert it to positive
            if (text_hash < 0)
                text_hash += q;
        }
    }
    
    return matches;
}

/*
 * Rabin-Karp for multiple patterns
 */
int rk_matcher_multi(char *text, int text_length, PatternInfo *patterns, int pattern_count, int d, int q)
{
    int total_matches = 0;
    int display_count = 0;

    for (int p = 0; p < pattern_count; p++) {
        int matches = rk_matcher_single(text, text_length, &patterns[p], d, q, &display_count);
        total_matches += matches;
        printf("Pattern '%s': %d matches\n", patterns[p].pattern, matches);
    }
    
    return total_matches;
}

/*
 * Function to read text file content
 */
char* read_text_file(const char* filename, int* length)
{
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

int main(int argc, char *argv[])
{
    int d = 256; // Number of characters in the alphabet
    int q = 101; // A prime number for hash calculation
    char* text;
    int text_length;
    clock_t start_time, end_time;
    double execution_time;
    
    const char* text_filename = "human_10m_upper.txt";
    const char* pattern_filename = "pattern.txt";
    
    printf("Reading text file: %s\n", text_filename);
    
    // Read the text file content
    text = read_text_file(text_filename, &text_length);
    printf("Read %d characters from file\n", text_length);
    
    // Read patterns
    PatternInfo patterns[MAX_PATTERNS];
    int pattern_count = read_patterns_file(pattern_filename, patterns);
    
    printf("Loaded %d patterns from %s\n", pattern_count, pattern_filename);
    
    // Precompute hash for each pattern
    for (int i = 0; i < pattern_count; i++) {
        patterns[i].hash = compute_hash(patterns[i].pattern, patterns[i].length, d, q);
        printf("Pattern %d: %s (length: %d, hash: %d)\n", 
               i, patterns[i].pattern, patterns[i].length, patterns[i].hash);
    }
    
    // Measure execution time
    start_time = clock();
    
    // Run the Rabin-Karp algorithm for multiple patterns
    int matches = rk_matcher_multi(text, text_length, patterns, pattern_count, d, q);
    
    end_time = clock();
    execution_time = ((double) (end_time - start_time)) / CLOCKS_PER_SEC * 1000.0; // milliseconds
    
    printf("Total matches found across all patterns: %d\n", matches);
    printf("Execution time: %.2f ms\n", execution_time);
    
    // Clean up
    free(text);
    
    return 0;
}