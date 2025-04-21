#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_TEXT_SIZE 2000000  // Increased for 1M file

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
 * Rabin-Karp string matching algorithm
 * Returns the number of matches found
 */
int rk_matcher(char *text, int text_length, char *pattern, int d, int q)
{
    int i, j;
    int pattern_length = strlen(pattern);
    int pattern_hash = 0;
    int text_hash = 0;
    int h = 1;
    int matches = 0;
    int display_count = 0;
    const int MAX_DISPLAY = 20;  // Maximum number of matches to display

    // Calculate h = d^(pattern_length-1) % q
    for (i = 0; i < pattern_length - 1; i++) {
        h = (h * d) % q;
    }

    // Calculate hash value for pattern and first window of text
    pattern_hash = compute_hash(pattern, pattern_length, d, q);
    text_hash = compute_hash(text, pattern_length, d, q);

    // Slide the pattern over text one by one
    for (i = 0; i <= text_length - pattern_length; i++) {
        // Check if hash values match
        if (pattern_hash == text_hash) {
            // Check for exact character match
            for (j = 0; j < pattern_length; j++) {
                if (text[i + j] != pattern[j])
                    break;
            }
            if (j == pattern_length) {
                matches++;
                // Only display a limited number of matches to avoid flooding the console
                if (display_count < MAX_DISPLAY) {
                    printf("Pattern found at index %d\n", i);
                    display_count++;
                    if (display_count == MAX_DISPLAY && matches > MAX_DISPLAY) {
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
    char pattern[100];
    int d = 256; // Number of characters in the alphabet
    int q = 101; // A prime number for hash calculation
    char* text;
    int text_length;
    clock_t start_time, end_time;
    double execution_time;
    
    const char* filename = "human_1m_upper.txt";
  
    strcpy(pattern, "TCGTG");
    printf("Reading file: %s\n", filename);
    printf("Searching for pattern: %s\n", pattern);
    
    // Read the file content
    text = read_text_file(filename, &text_length);
    printf("Read %d characters from file\n", text_length);
    
    // Measure execution time
    start_time = clock();
    
    // Run the Rabin-Karp algorithm
    int matches = rk_matcher(text, text_length, pattern, d, q);
    
    end_time = clock();
    execution_time = ((double) (end_time - start_time)) / CLOCKS_PER_SEC * 1000.0; // milliseconds
    
    printf("Total matches found: %d\n", matches);
    printf("Execution time: %.2f ms\n", execution_time);
    
    // Clean up
    free(text);
    
    return 0;
}