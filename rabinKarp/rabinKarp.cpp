#include <stdio.h>
#include <string.h>
#include <math.h>

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
int rk_matcher(char *text, char *pattern, int d, int q)
{
    int i, j;
    int text_length = strlen(text);
    int pattern_length = strlen(pattern);
    int pattern_hash = 0;
    int text_hash = 0;
    int h = 1;
    int matches = 0;

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
                printf("Pattern found at index %d\n", i);
                matches++;
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

int main(int argc, char *argv[])
{
    char text[100], pattern[100];
    int d = 256; // Number of characters in the alphabet
    int q = 101; // A prime number for hash calculation
    
    if (argc > 2) {
        // Use command line arguments
        strcpy(text, argv[1]);
        strcpy(pattern, argv[2]);
    } else {
        // Use default values
        strcpy(text, "TATTTCTGGTCCACCTGATTAATAAGAGCTTCCTCTACAATAGAATTTGTCAGGTAGGAATGTGTGTAGACTATAAAGGTCCTCACTCTCTTTATCTACCCT");
        strcpy(pattern, "TGTG");
    }
    
    int matches = rk_matcher(text, pattern, d, q);
    printf("Total matches found: %d\n", matches);
    
    return 0;
}