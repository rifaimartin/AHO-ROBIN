#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <unordered_map>
#include <map>
#include <ctime>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <algorithm>

using namespace std;

// Constants
#define MAX_STATES 16384
#define MAX_CHARS 256
#define MAX_PATTERN_LENGTH 100
#define MAX_DISPLAY 20
#define THREADS_PER_BLOCK 256
#define MAX_RESULTS 1000000

// CUDA error checking
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(1);
    }
}

// CPU Trie node implementation for building the automaton
struct TrieNode {
    std::map<char, TrieNode*> children;
    vector<int> output;

    TrieNode() {}
    
    ~TrieNode() {
        for (auto& pair : children) {
            delete pair.second;
        }
    }
};

// Flat GPU representation for PFAC - simpler than Aho-Corasick
struct FlatGPUTrie {
    int* transitions;      // Transitions table [state][character] -> next state
    int* is_match;         // Is this state a match state?
    int* match_pattern;    // Pattern ID for match states
    int num_states;        // Total number of states
};

void insertPattern(TrieNode* root, const string& pattern, int patternIndex) {
    TrieNode* current = root;

    for (char c : pattern) {
        if (current->children.find(c) == current->children.end()) {
            current->children[c] = new TrieNode();
        }
        current = current->children[c];
    }

    current->output.push_back(patternIndex);
}

// Convert CPU trie to flat GPU representation for PFAC
void convertToFlatGPUTrie(TrieNode* root, FlatGPUTrie& trie) {
    // First count the number of states
    queue<TrieNode*> q;
    unordered_map<TrieNode*, int> nodeToState;
    
    q.push(root);
    nodeToState[root] = 0;
    int numStates = 1;
    
    while (!q.empty()) {
        TrieNode* node = q.front();
        q.pop();
        
        for (auto& pair : node->children) {
            TrieNode* child = pair.second;
            if (nodeToState.find(child) == nodeToState.end()) {
                nodeToState[child] = numStates++;
                q.push(child);
            }
        }
    }
    
    printf("Counted %d states in the trie\n", numStates);
    trie.num_states = numStates;
    
    // Allocate memory for the flat representation
    trie.transitions = new int[numStates * MAX_CHARS];
    trie.is_match = new int[numStates];
    trie.match_pattern = new int[numStates];
    
    // Initialize all transitions to -1 (no transition)
    for (int i = 0; i < numStates; i++) {
        for (int j = 0; j < MAX_CHARS; j++) {
            trie.transitions[i * MAX_CHARS + j] = -1;
        }
        trie.is_match[i] = 0;
        trie.match_pattern[i] = -1;
    }
    
    // Set transitions and match states
    for (auto& pair : nodeToState) {
        TrieNode* node = pair.first;
        int stateId = pair.second;
        
        // Set transitions
        for (auto& childPair : node->children) {
            char c = childPair.first;
            int childStateId = nodeToState[childPair.second];
            trie.transitions[stateId * MAX_CHARS + (unsigned char)c] = childStateId;
        }
        
        // Set match info
        if (!node->output.empty()) {
            trie.is_match[stateId] = 1;
            trie.match_pattern[stateId] = node->output[0]; // Use first pattern ID for matches
        }
    }
}

// Simple kernel to test CUDA functionality
__global__ void testKernel(int* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        output[0] = 42;
    }
}

// PFAC kernel - each thread handles one starting position
__global__ void pfacKernel(
    const char* text, int textLength,
    const int* transitions, const int* is_match, const int* match_pattern,
    int numStates, 
    int* pattern_matches, int* match_count,
    int* match_positions, int* match_pattern_ids, int maxMatches,
    const int* pattern_lengths, int patternCount) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process different starting positions in the text
    for (int pos = tid; pos < textLength; pos += stride) {
        int state = 0; // Start from root
        
        // Process each character from this position
        for (int i = pos; i < textLength; i++) {
            unsigned char c = text[i];
            
            // Get next state
            state = transitions[state * MAX_CHARS + c];
            
            // If no valid transition, stop processing from this position
            if (state == -1) break;
            
            // Check if this is a match state
            if (is_match[state]) {
                int patternId = match_pattern[state];
                
                if (patternId >= 0 && patternId < patternCount) {
                    // Increment per-pattern match counter
                    atomicAdd(&pattern_matches[patternId], 1);
                    
                    // Record for display (limited number)
                    int idx = atomicAdd(match_count, 1);
                    if (idx < maxMatches) {
                        int matchPos = i - pattern_lengths[patternId] + 1;
                        match_positions[idx] = matchPos;
                        match_pattern_ids[idx] = patternId;
                    }
                }
            }
        }
    }
}

// Read text file content
string read_text_file(const char* filename, int* length) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        cout << "Error opening file: " << filename << endl;
        exit(1);
    }
    
    // Get file size
    file.seekg(0, ios::end);
    *length = file.tellg();
    file.seekg(0, ios::beg);
    
    // Read entire file into string
    string content(*length, '\0');
    file.read(&content[0], *length);
    
    file.close();
    return content;
}

// Read patterns from file (supports multiple formats)
vector<string> read_patterns_file(const char* filename) {
    vector<string> patterns;
    ifstream file(filename);
    
    if (!file.is_open()) {
        cout << "Error opening patterns file: " << filename << endl;
        exit(1);
    }
    
    // Read entire file
    stringstream buffer;
    buffer << file.rdbuf();
    string content = buffer.str();
    file.close();
    
    // Try to determine pattern format
    bool hasCommas = content.find(',') != string::npos;
    bool hasNewlines = content.find('\n') != string::npos;
    
    if (hasCommas) {
        // Process as comma-separated
        stringstream ss(content);
        string pattern;
        while (getline(ss, pattern, ',')) {
            // Trim whitespace
            pattern.erase(0, pattern.find_first_not_of(" \t\r\n"));
            pattern.erase(pattern.find_last_not_of(" \t\r\n") + 1);
            
            if (!pattern.empty()) {
                patterns.push_back(pattern);
            }
        }
    } else if (hasNewlines) {
        // Process as newline-separated
        stringstream ss(content);
        string pattern;
        while (getline(ss, pattern)) {
            // Trim whitespace
            pattern.erase(0, pattern.find_first_not_of(" \t\r\n"));
            pattern.erase(pattern.find_last_not_of(" \t\r\n") + 1);
            
            if (!pattern.empty()) {
                patterns.push_back(pattern);
            }
        }
    } else {
        // Process as whitespace-separated
        stringstream ss(content);
        string pattern;
        while (ss >> pattern) {
            if (!pattern.empty()) {
                patterns.push_back(pattern);
            }
        }
    }
    
    printf("Read %zu patterns from file\n", patterns.size());
    return patterns;
}

// Simple CPU PFAC implementation for verification
void runCPUPFAC(const string& text, const vector<string>& patterns, vector<int>& matches) {
    matches.resize(patterns.size(), 0);
    
    // Brute force approach for comparison
    for (size_t i = 0; i < text.length(); i++) {
        for (size_t p = 0; p < patterns.size(); p++) {
            if (i + patterns[p].length() <= text.length()) {
                bool match = true;
                for (size_t j = 0; j < patterns[p].length(); j++) {
                    if (text[i + j] != patterns[p][j]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    matches[p]++;
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    const char* text_filename = argc > 1 ? argv[1] : "human_10m_upper.txt";
    const char* pattern_filename = argc > 2 ? argv[2] : "pattern.txt";
    int text_length = 0;
    
    printf("Reading text file: %s\n", text_filename);
    
    // Read the file content
    string text = read_text_file(text_filename, &text_length);
    printf("Read %d characters from file\n", text_length);
    
    // Optional: Use smaller dataset for testing if requested
    bool doVerification = false;
    if (argc > 3 && string(argv[3]) == "verify") {
        doVerification = true;
        if (text_length > 1000000) {
            printf("Verification requested - using first 1M characters for CPU verification\n");
            text = text.substr(0, 1000000);
            text_length = 1000000;
        }
    }
    
    // Read patterns from file
    vector<string> patterns = read_patterns_file(pattern_filename);
    printf("Loaded %zu patterns from %s\n", patterns.size(), pattern_filename);
    
    for (size_t i = 0; i < min(size_t(10), patterns.size()); i++) {
        printf("Pattern %zu: %s (length: %zu)\n", i, patterns[i].c_str(), patterns[i].length());
    }
    if (patterns.size() > 10) {
        printf("... (showing only first 10 patterns)\n");
    }
    
    // First test CUDA functionality with a simple kernel
    printf("Testing CUDA functionality...\n");
    int* d_test;
    int h_test = 0;
    cudaError_t testErr = cudaMalloc(&d_test, sizeof(int));
    if (testErr != cudaSuccess) {
        printf("CUDA malloc failed for test: %s\n", cudaGetErrorString(testErr));
        return 1;
    }
    
    cudaMemcpy(d_test, &h_test, sizeof(int), cudaMemcpyHostToDevice);
    testKernel<<<1, 1>>>(d_test);
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        printf("CUDA sync failed for test: %s\n", cudaGetErrorString(syncErr));
        return 1;
    }
    
    cudaMemcpy(&h_test, d_test, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Test result: %d (should be 42)\n", h_test);
    cudaFree(d_test);
    
    if (h_test != 42) {
        printf("CUDA test failed! Not proceeding with main algorithm.\n");
        return 1;
    }
    
    // Build the trie on CPU
    printf("Building pattern trie...\n");
    TrieNode* root = new TrieNode();
    for (size_t i = 0; i < patterns.size(); ++i) {
        insertPattern(root, patterns[i], i);
    }
    
    // Convert to flat GPU structure
    printf("Converting to flat GPU trie...\n");
    FlatGPUTrie trie;
    convertToFlatGPUTrie(root, trie);
    
    printf("Created flat trie with %d states\n", trie.num_states);
    
    // Create pattern lengths array
    int* patternLengths = new int[patterns.size()];
    for (size_t i = 0; i < patterns.size(); i++) {
        patternLengths[i] = patterns[i].length();
    }
    
    // Optionally run CPU version for verification
    vector<int> cpuMatches;
    if (doVerification) {
        printf("Running CPU PFAC for verification...\n");
        clock_t cpuStart = clock();
        runCPUPFAC(text, patterns, cpuMatches);
        clock_t cpuEnd = clock();
        double cpuTime = (double)(cpuEnd - cpuStart) / CLOCKS_PER_SEC * 1000.0;
        printf("CPU PFAC completed in %.2f ms\n", cpuTime);
    }
    
    // Allocate GPU memory for trie
    int* d_transitions;
    int* d_is_match;
    int* d_match_pattern;
    
    cudaCheckError(cudaMalloc(&d_transitions, trie.num_states * MAX_CHARS * sizeof(int)));
    cudaCheckError(cudaMalloc(&d_is_match, trie.num_states * sizeof(int)));
    cudaCheckError(cudaMalloc(&d_match_pattern, trie.num_states * sizeof(int)));
    
    // Copy trie to device
    cudaCheckError(cudaMemcpy(d_transitions, trie.transitions, 
                          trie.num_states * MAX_CHARS * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_is_match, trie.is_match, 
                          trie.num_states * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_match_pattern, trie.match_pattern, 
                          trie.num_states * sizeof(int), cudaMemcpyHostToDevice));
    
    // Allocate remaining GPU memory
    char* d_text;
    int* d_pattern_matches;      // Per-pattern match counter
    int* d_match_count;          // Global match counter
    int* d_match_positions;      // Match positions for display
    int* d_match_pattern_ids;    // Pattern IDs for display
    int* d_pattern_lengths;      // Pattern lengths
    
    cudaCheckError(cudaMalloc(&d_text, text_length * sizeof(char)));
    cudaCheckError(cudaMalloc(&d_pattern_matches, patterns.size() * sizeof(int)));
    cudaCheckError(cudaMalloc(&d_match_count, sizeof(int)));
    cudaCheckError(cudaMalloc(&d_match_positions, MAX_DISPLAY * sizeof(int)));
    cudaCheckError(cudaMalloc(&d_match_pattern_ids, MAX_DISPLAY * sizeof(int)));
    cudaCheckError(cudaMalloc(&d_pattern_lengths, patterns.size() * sizeof(int)));
    
    // Initialize counters
    int* pattern_matches = new int[patterns.size()]();
    int initial_match_count = 0;
    
    cudaCheckError(cudaMemcpy(d_text, text.c_str(), text_length * sizeof(char), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_pattern_matches, pattern_matches, patterns.size() * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_match_count, &initial_match_count, sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_pattern_lengths, patternLengths, patterns.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record the start event
    cudaEventRecord(start);
    
    // Configure kernel launch parameters
    int blockSize = THREADS_PER_BLOCK;
    int gridSize = min(1024, (text_length + blockSize - 1) / blockSize);
    
    printf("Launching PFAC kernel with grid size: %d, block size: %d\n", gridSize, blockSize);
    
    // Launch PFAC kernel
    pfacKernel<<<gridSize, blockSize>>>(
        d_text, text_length,
        d_transitions, d_is_match, d_match_pattern,
        trie.num_states,
        d_pattern_matches, d_match_count,
        d_match_positions, d_match_pattern_ids, MAX_DISPLAY,
        d_pattern_lengths, patterns.size()
    );
    
    // Check for kernel launch errors
    cudaError_t kernelError = cudaGetLastError();
    if (kernelError != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(kernelError));
        return 1;
    }
    
    // Wait for kernel to finish
    printf("Waiting for kernel to complete...\n");
    
    cudaError_t syncError = cudaDeviceSynchronize();
    if (syncError != cudaSuccess) {
        printf("Synchronization error: %s\n", cudaGetErrorString(syncError));
        // Continue anyway to clean up resources
    } else {
        // Record the stop event
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        // Calculate kernel execution time
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        // Copy results back
        int totalMatchCount;
        int matchPositions[MAX_DISPLAY];
        int matchPatternIds[MAX_DISPLAY];
        
        cudaCheckError(cudaMemcpy(&totalMatchCount, d_match_count, sizeof(int), cudaMemcpyDeviceToHost));
        cudaCheckError(cudaMemcpy(matchPositions, d_match_positions, 
                                min(totalMatchCount, MAX_DISPLAY) * sizeof(int), cudaMemcpyDeviceToHost));
        cudaCheckError(cudaMemcpy(matchPatternIds, d_match_pattern_ids, 
                                min(totalMatchCount, MAX_DISPLAY) * sizeof(int), cudaMemcpyDeviceToHost));
        cudaCheckError(cudaMemcpy(pattern_matches, d_pattern_matches, patterns.size() * sizeof(int), cudaMemcpyDeviceToHost));
        
        // Output results
        printf("\nResults:\n");
        for (int i = 0; i < min(totalMatchCount, MAX_DISPLAY); i++) {
            int patternId = matchPatternIds[i];
            if (patternId >= 0 && patternId < (int)patterns.size()) {
                printf("Pattern '%s' found at position %d\n", 
                       patterns[patternId].c_str(), matchPositions[i]);
            } else {
                printf("Invalid pattern ID found: %d\n", patternId);
            }
        }
        
        if (totalMatchCount > MAX_DISPLAY) {
            printf("... (showing only first %d of %d matches)\n", MAX_DISPLAY, totalMatchCount);
        }
        
        // Output match counts for each pattern
        int totalMatches = 0;
        printf("\nMatch counts by pattern:\n");
        for (size_t i = 0; i < patterns.size(); i++) {
            printf("Pattern '%s': %d matches\n", patterns[i].c_str(), pattern_matches[i]);
            totalMatches += pattern_matches[i];
        }
        
        printf("\nTotal matches found across all patterns: %d\n", totalMatches);
        printf("Execution time: %.2f ms\n", milliseconds);
        
        // If verification was done, compare results
        if (doVerification) {
            printf("\nVerification Results:\n");
            int cpuTotalMatches = 0;
            bool mismatch = false;
            
            for (size_t i = 0; i < patterns.size(); i++) {
                cpuTotalMatches += cpuMatches[i];
                if (cpuMatches[i] != pattern_matches[i]) {
                    printf("MISMATCH: Pattern '%s': CPU=%d, GPU=%d\n", 
                           patterns[i].c_str(), cpuMatches[i], pattern_matches[i]);
                    mismatch = true;
                }
            }
            
            printf("\nCPU total matches: %d, GPU total matches: %d\n", cpuTotalMatches, totalMatches);
            if (mismatch) {
                printf("WARNING: CPU and GPU results do not match! Check implementation.\n");
            } else {
                printf("VERIFICATION SUCCESSFUL: CPU and GPU results match exactly!\n");
            }
        }
    }
    
    // Free CPU memory
    delete root;
    delete[] patternLengths;
    delete[] pattern_matches;
    delete[] trie.transitions;
    delete[] trie.is_match;
    delete[] trie.match_pattern;
    
    // Free GPU memory
    cudaFree(d_transitions);
    cudaFree(d_is_match);
    cudaFree(d_match_pattern);
    cudaFree(d_text);
    cudaFree(d_pattern_matches);
    cudaFree(d_match_count);
    cudaFree(d_match_positions);
    cudaFree(d_match_pattern_ids);
    cudaFree(d_pattern_lengths);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}