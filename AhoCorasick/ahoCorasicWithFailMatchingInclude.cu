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

            // Merge output from fail node 
            // child->output.insert(child->output.end(), child->fail->output.begin(), child->fail->output.end());
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
        
        // Copy output
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
            
            // Check for pattern matches
            if (states[currentState].output_count > 0) {
                for (int j = 0; j < states[currentState].output_count; j++) {
                    int patternIndex = states[currentState].output[j];
                    int matchPos = i - patternLengths[patternIndex] + 1;
                    
                    // Record the match
                    int idx = atomicAdd(matchCount, 1);
                    if (idx < maxMatches) {
                        matchPositions[idx] = matchPos;
                        matchPatternIds[idx] = patternIndex;
                    }
                    
                    // Count the match for this pattern
                    atomicAdd(&matches[patternIndex], 1);
                }
            }
            
            // If we've gone beyond our responsibility for this thread, stop
            if (i > startPos && states[currentState].output_count == 0 && currentState == 0) {
                break; // No match possible anymore, so move to next starting position
            }
        }
    }
}

int main() {
    clock_t start_time = clock();
    
    const char* text_filename = "human_10m_upper.txt";
    const char* pattern_filename = "pattern.txt";
    int text_length = 0;
    
    cout << "Reading text file: " << text_filename << endl;
    
    // Read the file content
    string text = read_text_file(text_filename, &text_length);
    cout << "Read " << text_length << " characters from file" << endl;
    
    // Read patterns from file
    vector<string> patterns = read_patterns_file(pattern_filename);
    cout << "Loaded " << patterns.size() << " patterns from " << pattern_filename << ":" << endl;
    
    for (const auto& pattern : patterns) {
        cout << "  - " << pattern << endl;
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
    
    cout << "Created automaton with " << numStates << " states" << endl;
    
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
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record the start event
    cudaEventRecord(start, 0);
    
    // Launch kernel
    int numBlocks = (text_length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    ahoCorasickKernel<<<numBlocks, THREADS_PER_BLOCK>>>(
        d_text, text_length, d_states, numStates, 
        d_matches, d_matchCount, d_matchPositions, d_matchPatternIds,
        MAX_DISPLAY, d_patternLengths, patterns.size()
    );
    
    // Check for kernel launch errors
    cudaCheckError(cudaGetLastError());
    
    // Wait for kernel to finish
    cudaCheckError(cudaDeviceSynchronize());
    
    // Record the stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    // Calculate kernel execution time
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);
    
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
    
    // Display match positions (up to MAX_DISPLAY)
    int display_count = 0;
    for (int i = 0; i < min(totalMatchCount, MAX_DISPLAY); i++) {
        cout << "Pattern '" << patterns[matchPatternIds[i]] << "' found at index " << matchPositions[i] << endl;
        display_count++;
    }
    
    if (totalMatchCount > MAX_DISPLAY) {
        cout << "... (showing only first " << MAX_DISPLAY << " matches)" << endl;
    }
    
    // Output match counts for each pattern
    cout << "\nMatch counts per pattern:" << endl;
    int totalMatches = 0;
    for (size_t i = 0; i < patterns.size(); i++) {
        cout << "Pattern '" << patterns[i] << "': " << patternMatches[i] << " matches" << endl;
        totalMatches += patternMatches[i];
    }
    
    // Calculate total execution time
    clock_t end_time = clock();
    double execution_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC * 1000.0; // milliseconds
    
    cout << "Total matches found across all patterns: " << totalMatches << endl;
    cout << "GPU kernel execution time: " << gpu_time << " ms" << endl;
    cout << "Total execution time (including CPU preprocessing): " << execution_time << " ms" << endl;
    
    // Clean up
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