#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <unordered_map>
#include <ctime>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>

using namespace std;

// Maximum number of states in the Aho-Corasick automaton
#define MAX_STATES 4096
// Maximum number of characters in the alphabet (ASCII)
#define MAX_CHARS 256
// Maximum number of patterns
#define MAX_PATTERNS 100
// Maximum pattern length
#define MAX_PATTERN_LENGTH 100
// Maximum matches to display
#define MAX_DISPLAY 20
// Number of threads per block
#define THREADS_PER_BLOCK 256

// State representation for GPU
struct GPUTrieState {
    int transitions[MAX_CHARS];  // Transition function
    int fail;                    // Failure function
    int output[MAX_PATTERNS];    // Output function (pattern indices)
    int output_count;           // Number of patterns in output
};

// CUDA error checking
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// CPU implementation for building the Aho-Corasick automaton
struct TrieNode {
    unordered_map<char, TrieNode*> children;
    TrieNode* fail;
    vector<int> output;

    TrieNode() : fail(nullptr) {}
    
    // Destructor to clean up all nodes
    ~TrieNode() {
        for (auto& pair : children) {
            delete pair.second;
        }
    }
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

void buildFailTransitions(TrieNode* root) {
    queue<TrieNode*> nodeQueue;

    // Set fail transitions for depth 1 nodes to root
    for (auto& pair : root->children) {
        pair.second->fail = root;
        nodeQueue.push(pair.second);
    }

    while (!nodeQueue.empty()) {
        TrieNode* current = nodeQueue.front();
        nodeQueue.pop();

        for (auto& pair : current->children) {
            char c = pair.first;
            TrieNode* child = pair.second;
            nodeQueue.push(child);

            // Find the node to fail to
            TrieNode* temp = current->fail;
            while (temp != root && temp->children.find(c) == temp->children.end()) {
                temp = temp->fail;
            }

            if (temp->children.find(c) != temp->children.end()) {
                child->fail = temp->children[c];
            } else {
                child->fail = root;
            }

            // Do NOT merge output from fail node, as this causes double counting
            // The fail transitions will be followed at runtime instead
            // This was causing multiple counts for the same match
        }
    }
}

// Convert the CPU trie to GPU-friendly structure
void convertTrieToGPU(TrieNode* root, GPUTrieState* gpuStates, int& numStates, 
                      unordered_map<TrieNode*, int>& nodeToState) {
    
    // Initialize all transitions to -1 (no transition)
    for (int i = 0; i < MAX_STATES; i++) {
        for (int j = 0; j < MAX_CHARS; j++) {
            gpuStates[i].transitions[j] = -1;
        }
        gpuStates[i].fail = 0; // Default fail to root
        gpuStates[i].output_count = 0;
    }
    
    // Assign state IDs to nodes with BFS
    queue<TrieNode*> q;
    q.push(root);
    nodeToState[root] = 0; // Root is state 0
    numStates = 1;
    
    while (!q.empty()) {
        TrieNode* node = q.front();
        q.pop();
        int stateId = nodeToState[node];
        
        // Add its children
        for (auto& pair : node->children) {
            char c = pair.first;
            TrieNode* child = pair.second;
            
            // If this child hasn't been assigned a state ID yet
            if (nodeToState.find(child) == nodeToState.end()) {
                nodeToState[child] = numStates++;
                q.push(child);
            }
            
            // Set transition
            gpuStates[stateId].transitions[(unsigned char)c] = nodeToState[child];
        }
        
        // Copy only the direct output of this node, not from fail transitions
        gpuStates[stateId].output_count = min((int)node->output.size(), MAX_PATTERNS);
        for (int i = 0; i < gpuStates[stateId].output_count; i++) {
            gpuStates[stateId].output[i] = node->output[i];
        }
        
        // Set fail transition if not root
        if (node != root) {
            gpuStates[stateId].fail = nodeToState[node->fail];
        }
    }
}

// Read text file content
string read_text_file(const char* filename, int* length) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "Error opening file: " << filename << endl;
        exit(1);
    }
    
    // Read entire file into string
    stringstream buffer;
    buffer << file.rdbuf();
    string content = buffer.str();
    
    file.close();
    *length = content.size();
    return content;
}

// Read patterns from file (comma-separated)
vector<string> read_patterns_file(const char* filename) {
    vector<string> patterns;
    ifstream file(filename);
    
    if (!file.is_open()) {
        cout << "Error opening patterns file: " << filename << endl;
        exit(1);
    }
    
    string line;
    if (getline(file, line)) {
        // Parse comma-separated patterns
        stringstream ss(line);
        string pattern;
        
        while (getline(ss, pattern, ',')) {
            if (!pattern.empty()) {
                patterns.push_back(pattern);
            }
        }
    } else {
        cout << "Pattern file is empty!" << endl;
    }
    
    file.close();
    
    // Check if we have too many patterns for our GPU structure
    if (patterns.size() > MAX_PATTERNS) {
        cout << "Warning: Found " << patterns.size() << " patterns, but can only use the first " 
             << MAX_PATTERNS << " due to GPU memory constraints." << endl;
        patterns.resize(MAX_PATTERNS);
    }
    
    return patterns;
}

// CUDA kernel for Aho-Corasick search
__global__ void ahoCorasickKernel(char* text, int textLength, GPUTrieState* states, 
                                int numStates, int* matches, int* matchCount, 
                                int* matchPositions, int* matchPatternIds, int maxMatches, 
                                int* patternLengths, int patternCount) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int startPos = tid; startPos < textLength; startPos += stride) {
        int currentState = 0; // Start from the root
        
        for (int i = startPos; i < textLength; i++) {
            unsigned char c = (unsigned char)text[i];
            
            // Follow failure links until we find a match or reach root
            while (currentState != 0 && states[currentState].transitions[c] == -1) {
                currentState = states[currentState].fail;
            }
            
            // Check if there's a valid transition
            if (states[currentState].transitions[c] != -1) {
                currentState = states[currentState].transitions[c];
            }
            
            // Check for pattern matches in the current state
            for (int j = 0; j < states[currentState].output_count; j++) {
                int patternIndex = states[currentState].output[j];
                int patternLength = patternLengths[patternIndex];
                int matchPos = i - patternLength + 1;
                
                // Only process matches that start at our assigned position
                if (matchPos == startPos) {
                    // Increment per-pattern match count
                    atomicAdd(&matches[patternIndex], 1);
                    
                    // Record for display
                    int idx = atomicAdd(matchCount, 1);
                    if (idx < maxMatches) {
                        matchPositions[idx] = matchPos;
                        matchPatternIds[idx] = patternIndex;
                    }
                }
            }
            
            // Now check all fail states for matches too
            int failState = states[currentState].fail;
            while (failState != 0) {
                for (int j = 0; j < states[failState].output_count; j++) {
                    int patternIndex = states[failState].output[j];
                    int patternLength = patternLengths[patternIndex];
                    int matchPos = i - patternLength + 1;
                    
                    // Only process matches that start at our assigned position
                    if (matchPos == startPos) {
                        // Increment per-pattern match count
                        atomicAdd(&matches[patternIndex], 1);
                        
                        // Record for display
                        int idx = atomicAdd(matchCount, 1);
                        if (idx < maxMatches) {
                            matchPositions[idx] = matchPos;
                            matchPatternIds[idx] = patternIndex;
                        }
                    }
                }
                failState = states[failState].fail;
            }
            
            // If we've gone beyond our responsibility for this thread, stop
            if (i > startPos && currentState == 0) {
                break; // No match possible anymore, so move to next starting position
            }
        }
    }
}

int main() {
    const char* text_filename = "human_10m_upper.txt";
    const char* pattern_filename = "pattern.txt";
    int text_length = 0;
    
    printf("Reading text file: %s\n", text_filename);
    
    // Read the file content
    string text = read_text_file(text_filename, &text_length);
    printf("Read %d characters from file\n", text_length);
    
    // Read patterns from file
    vector<string> patterns = read_patterns_file(pattern_filename);
    printf("Loaded %d patterns from %s\n", (int)patterns.size(), pattern_filename);
    
    for (int i = 0; i < (int)patterns.size() && i < 10; i++) {
        printf("Pattern %d: %s (length: %d)\n", i, patterns[i].c_str(), (int)patterns[i].length());
    }
    if (patterns.size() > 10) {
        printf("... (showing only first 10 patterns)\n");
    }
    
    // Build the Aho-Corasick automaton on CPU
    TrieNode* root = new TrieNode();
    for (int i = 0; i < (int)patterns.size(); ++i) {
        insertPattern(root, patterns[i], i);
    }
    buildFailTransitions(root);
    
    // Convert to GPU structure
    GPUTrieState* cpuStates = new GPUTrieState[MAX_STATES];
    int numStates = 0;
    unordered_map<TrieNode*, int> nodeToState;
    convertTrieToGPU(root, cpuStates, numStates, nodeToState);
    
    printf("Created automaton with %d states\n", numStates);
    
    // Create pattern lengths array
    int* patternLengths = new int[patterns.size()];
    for (size_t i = 0; i < patterns.size(); i++) {
        patternLengths[i] = patterns[i].length();
    }
    
    // Allocate GPU memory
    char* d_text;
    GPUTrieState* d_states;
    int* d_matches;  // Array to count matches per pattern
    int* d_matchCount;  // Total match count
    int* d_matchPositions;  // Array to store match positions for display
    int* d_matchPatternIds;  // Array to store pattern IDs for display
    int* d_patternLengths;
    
    cudaCheckError(cudaMalloc(&d_text, text_length * sizeof(char)));
    cudaCheckError(cudaMalloc(&d_states, numStates * sizeof(GPUTrieState)));
    cudaCheckError(cudaMalloc(&d_matches, patterns.size() * sizeof(int)));
    cudaCheckError(cudaMalloc(&d_matchCount, sizeof(int)));
    cudaCheckError(cudaMalloc(&d_matchPositions, MAX_DISPLAY * sizeof(int)));
    cudaCheckError(cudaMalloc(&d_matchPatternIds, MAX_DISPLAY * sizeof(int)));
    cudaCheckError(cudaMalloc(&d_patternLengths, patterns.size() * sizeof(int)));
    
    // Initialize arrays
    int* patternMatches = new int[patterns.size()]();  // Initialize to zeros
    int initialMatchCount = 0;
    
    // Copy data to GPU
    cudaCheckError(cudaMemcpy(d_text, text.c_str(), text_length * sizeof(char), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_states, cpuStates, numStates * sizeof(GPUTrieState), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_matches, patternMatches, patterns.size() * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_matchCount, &initialMatchCount, sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_patternLengths, patternLengths, patterns.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record the start event
    cudaEventRecord(start);
    
    // Launch kernel
    int blockSize = THREADS_PER_BLOCK;
    int gridSize = (text_length + blockSize - 1) / blockSize;
    printf("Launching kernel with grid size: %d, block size: %d\n", gridSize, blockSize);
    
    ahoCorasickKernel<<<gridSize, blockSize>>>(
        d_text, text_length, d_states, numStates, 
        d_matches, d_matchCount, d_matchPositions, d_matchPatternIds,
        MAX_DISPLAY, d_patternLengths, patterns.size()
    );
    
    // Check for kernel launch errors
    cudaCheckError(cudaGetLastError());
    
    // Wait for kernel to finish
    cudaCheckError(cudaDeviceSynchronize());
    
    // Record the stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate kernel execution time
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy results back
    int totalMatchCount;
    int matchPositions[MAX_DISPLAY];
    int matchPatternIds[MAX_DISPLAY];
    
    cudaCheckError(cudaMemcpy(&totalMatchCount, d_matchCount, sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(matchPositions, d_matchPositions, 
                            min(totalMatchCount, MAX_DISPLAY) * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(matchPatternIds, d_matchPatternIds, 
                            min(totalMatchCount, MAX_DISPLAY) * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(patternMatches, d_matches, patterns.size() * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Check for CUDA errors
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaError));
    }
    
    // Output results
    for (int i = 0; i < min(totalMatchCount, MAX_DISPLAY); i++) {
        printf("Pattern found at index %d\n", matchPositions[i]);
    }
    
    if (totalMatchCount > MAX_DISPLAY) {
        printf("... (showing only first %d matches)\n", MAX_DISPLAY);
    }
    
    // Output match counts for each pattern
    int totalMatches = 0;
    for (size_t i = 0; i < patterns.size(); i++) {
        printf("Pattern '%s': %d matches\n", patterns[i].c_str(), patternMatches[i]);
        totalMatches += patternMatches[i];
    }
    
    printf("Total matches found across all patterns: %d\n", totalMatches);
    printf("Execution time: %.2f ms\n", milliseconds);
    
    // Free memory
    delete[] cpuStates;
    delete[] patternLengths;
    delete[] patternMatches;
    delete root;
    
    cudaFree(d_text);
    cudaFree(d_states);
    cudaFree(d_matches);
    cudaFree(d_matchCount);
    cudaFree(d_matchPositions);
    cudaFree(d_matchPatternIds);
    cudaFree(d_patternLengths);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}