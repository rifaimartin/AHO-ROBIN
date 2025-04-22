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
// Constants that won't change
#define MAX_STATES 8192  // Increased from 4096 to handle more complex automaton
#define MAX_CHARS 256
#define MAX_PATTERN_LENGTH 100
#define MAX_DISPLAY 20
#define THREADS_PER_BLOCK 256
#define MAX_FAIL_ITERATIONS 100
#define MAX_TEXT_CHUNK 1000

// This will be set dynamically based on pattern count
int MAX_PATTERNS = 1024; // Default, will be adjusted based on actual pattern count

// State representation for GPU with dynamic pattern size
struct GPUTrieState {
    int transitions[MAX_CHARS];  // Transition function
    int fail;                    // Failure function
    int* output;                 // Output function (pattern indices) - dynamically allocated
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
        
        // Allocate output array dynamically based on MAX_PATTERNS
        gpuStates[i].output = new int[MAX_PATTERNS];
        for (int j = 0; j < MAX_PATTERNS; j++) {
            gpuStates[i].output[j] = -1; // Initialize to invalid pattern ID
        }
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
                if (numStates >= MAX_STATES) {
                    printf("ERROR: Exceeded maximum number of states (%d). Increase MAX_STATES.\n", MAX_STATES);
                    exit(1);
                }
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

// Verify the GPU automaton structure
void verifyAutomaton(GPUTrieState* states, int numStates) {
    printf("Verifying automaton structure with %d states...\n", numStates);
    
    int warnings = 0;
    int errors = 0;
    
    for (int i = 0; i < numStates; i++) {
        int hasValidTransition = 0;
        for (int j = 0; j < MAX_CHARS; j++) {
            if (states[i].transitions[j] != -1) {
                if (states[i].transitions[j] < 0 || states[i].transitions[j] >= numStates) {
                    printf("Error: State %d has invalid transition on char %d to state %d\n", 
                           i, j, states[i].transitions[j]);
                    errors++;
                } else {
                    hasValidTransition = 1;
                }
            }
        }
        
        if (!hasValidTransition && i != 0) {
            printf("Warning: State %d has no valid transitions!\n", i);
            warnings++;
        }
        
        if (states[i].fail < 0 || states[i].fail >= numStates) {
            printf("Error: State %d has invalid fail state: %d\n", i, states[i].fail);
            errors++;
        }
        
        if (states[i].output_count < 0 || states[i].output_count > MAX_PATTERNS) {
            printf("Error: State %d has invalid output_count: %d\n", i, states[i].output_count);
            errors++;
        }
    }
    
    printf("Verification complete. Found %d warnings and %d errors.\n", warnings, errors);
    
    if (errors > 0) {
        printf("ERROR: Automaton structure is invalid. Fix before proceeding.\n");
        exit(1);
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
    
    // Set the global MAX_PATTERNS value based on actual pattern count
    // Add some padding for safety
    MAX_PATTERNS = patterns.size() + 16;
    printf("Setting MAX_PATTERNS to %d based on %zu patterns found\n", 
           MAX_PATTERNS, patterns.size());
    
    return patterns;
}

// Simple kernel to test CUDA functionality
__global__ void testKernel(int* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        output[0] = 42;
    }
}

// CUDA kernel for Aho-Corasick search with unified memory for output arrays
__global__ void ahoCorasickKernel(char* text, int textLength, GPUTrieState* states, 
                                int numStates, int* matches, int* matchCount, 
                                int* matchPositions, int* matchPatternIds, int maxMatches, 
                                int* patternLengths, int patternCount, int** stateOutputs) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int startPos = tid; startPos < textLength; startPos += stride) {
        int currentState = 0; // Start from the root
        
        // Process at most MAX_TEXT_CHUNK characters from this starting position
        for (int i = startPos; i < textLength && i < startPos + MAX_TEXT_CHUNK; i++) {
            unsigned char c = (unsigned char)text[i];
            
            // Follow failure links until we find a match or reach root - with iteration limit
            int failIterations = 0;
            while (currentState != 0 && currentState < numStates && 
                  (c < MAX_CHARS && states[currentState].transitions[c] == -1) && 
                   failIterations < MAX_FAIL_ITERATIONS) {
                
                int nextState = states[currentState].fail;
                // Basic bounds checking
                if (nextState < 0 || nextState >= numStates) {
                    currentState = 0; // Reset to root if invalid fail state
                    break;
                }
                
                currentState = nextState;
                failIterations++;
            }
            
            // If we hit the iteration limit, reset to root state
            if (failIterations == MAX_FAIL_ITERATIONS) {
                currentState = 0;
            }
            
            // Check if there's a valid transition - with bounds checking
            if (currentState < numStates && c < MAX_CHARS && states[currentState].transitions[c] != -1) {
                int nextState = states[currentState].transitions[c];
                // Validate the next state
                if (nextState >= 0 && nextState < numStates) {
                    currentState = nextState;
                } else {
                    currentState = 0; // Reset to root if invalid transition
                }
            }
            
            // Safety check for current state
            if (currentState < 0 || currentState >= numStates) {
                currentState = 0;
                continue;
            }
            
            // Check for pattern matches in the current state - with bounds checking
            int outputCount = states[currentState].output_count;
            if (outputCount > 0 && outputCount <= patternCount) {
                int* outputs = stateOutputs[currentState];
                for (int j = 0; j < outputCount; j++) {
                    int patternIndex = outputs[j];
                    
                    // Validate pattern index
                    if (patternIndex >= 0 && patternIndex < patternCount) {
                        int patternLength = patternLengths[patternIndex];
                        int matchPos = i - patternLength + 1;
                        
                        // Only process matches that start at our assigned position
                        if (matchPos == startPos) {
                            // Increment per-pattern match count
                            atomicAdd(&matches[patternIndex], 1);
                            
                            // Record for display - safely limit the maximum number of matches recorded
                            int idx = atomicAdd(matchCount, 1);
                            if (idx < maxMatches) {
                                matchPositions[idx] = matchPos;
                                matchPatternIds[idx] = patternIndex;
                            }
                        }
                    }
                }
            }
            
            // Safety check for current state (again)
            if (currentState < 0 || currentState >= numStates) {
                currentState = 0;
                continue;
            }
            
            // Now check all fail states for matches too - with iteration limit and bounds checking
            int failState = states[currentState].fail;
            int failChainIteration = 0;
            
            while (failState != 0 && failState > 0 && failState < numStates && failChainIteration < MAX_FAIL_ITERATIONS) {
                int failOutputCount = states[failState].output_count;
                
                if (failOutputCount > 0 && failOutputCount <= patternCount) {
                    int* failOutputs = stateOutputs[failState];
                    for (int j = 0; j < failOutputCount; j++) {
                        int patternIndex = failOutputs[j];
                        
                        // Validate pattern index
                        if (patternIndex >= 0 && patternIndex < patternCount) {
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
                    }
                }
                
                int nextFailState = states[failState].fail;
                // Validate the next fail state
                if (nextFailState < 0 || nextFailState >= numStates) {
                    break; // Exit the loop on invalid fail state
                }
                
                failState = nextFailState;
                failChainIteration++;
            }
            
            // If we've gone beyond our responsibility or reached root state, stop
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
    
    // Verify automaton structure
    verifyAutomaton(cpuStates, numStates);
    
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
    int** d_stateOutputs;  // Array of pointers to output arrays for each state
    
    // Alokasi memori untuk struktur utama
    cudaCheckError(cudaMalloc(&d_text, text_length * sizeof(char)));
    cudaCheckError(cudaMalloc(&d_states, numStates * sizeof(GPUTrieState)));
    cudaCheckError(cudaMalloc(&d_matches, patterns.size() * sizeof(int)));
    cudaCheckError(cudaMalloc(&d_matchCount, sizeof(int)));
    cudaCheckError(cudaMalloc(&d_matchPositions, MAX_DISPLAY * sizeof(int)));
    cudaCheckError(cudaMalloc(&d_matchPatternIds, MAX_DISPLAY * sizeof(int)));
    cudaCheckError(cudaMalloc(&d_patternLengths, patterns.size() * sizeof(int)));
    
    // Setup state outputs (pointer array to output arrays)
    cudaCheckError(cudaMalloc(&d_stateOutputs, numStates * sizeof(int*)));
    int** h_stateOutputs = new int*[numStates];
    
    // Create and copy output arrays for each state
    for (int i = 0; i < numStates; i++) {
        int* d_output;
        cudaCheckError(cudaMalloc(&d_output, MAX_PATTERNS * sizeof(int)));
        cudaCheckError(cudaMemcpy(d_output, cpuStates[i].output, MAX_PATTERNS * sizeof(int), cudaMemcpyHostToDevice));
        h_stateOutputs[i] = d_output;
    }
    
    // Copy the array of pointers to GPU
    cudaCheckError(cudaMemcpy(d_stateOutputs, h_stateOutputs, numStates * sizeof(int*), cudaMemcpyHostToDevice));
    
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
    
    // Launch kernel with limited grid size to prevent resource overload
    int blockSize = THREADS_PER_BLOCK;
    int gridSize = min(1024, (text_length + blockSize - 1) / blockSize);
    printf("Launching kernel with grid size: %d, block size: %d\n", gridSize, blockSize);
    
    ahoCorasickKernel<<<gridSize, blockSize>>>(
        d_text, text_length, d_states, numStates, 
        d_matches, d_matchCount, d_matchPositions, d_matchPatternIds,
        MAX_DISPLAY, d_patternLengths, patterns.size(), d_stateOutputs
    );
    
    // Check for kernel launch errors
    cudaError_t kernelError = cudaGetLastError();
    if (kernelError != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(kernelError));
        // Clean up and exit
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
        return 1;
    }
    
    // Wait for kernel to finish with timeout detection
    printf("Waiting for kernel to complete...\n");
    cudaError_t syncError = cudaDeviceSynchronize();
    if (syncError != cudaSuccess) {
        printf("Synchronization error: %s\n", cudaGetErrorString(syncError));
        // Clean up and exit
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
        return 1;
    }
    
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
        printf("Pattern '%s': %d matches\n", patterns[i].c_str(), patternMatches[i]);
        totalMatches += patternMatches[i];
    }
    
    printf("\nTotal matches found across all patterns: %d\n", totalMatches);
    printf("Execution time: %.2f ms\n", milliseconds);
    
    // Free memory - clean up dynamically allocated arrays
    for (int i = 0; i < MAX_STATES; i++) {
        delete[] cpuStates[i].output;
    }
    delete[] cpuStates;
    delete[] patternLengths;
    delete[] patternMatches;
    delete root;
    
    // Free GPU memory
    for (int i = 0; i < numStates; i++) {
        cudaFree(h_stateOutputs[i]);
    }
    delete[] h_stateOutputs;
    
    cudaFree(d_stateOutputs);
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