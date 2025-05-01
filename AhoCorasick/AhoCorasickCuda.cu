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
#define MAX_STATES 16384  // Increased for larger automatons
#define MAX_CHARS 256
#define MAX_PATTERN_LENGTH 100
#define MAX_DISPLAY 20
#define THREADS_PER_BLOCK 256
#define WORK_CHUNK_SIZE 64    // Number of characters processed per thread per iteration
#define MAX_RESULTS 1000000   // Max number of matches to record

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
    std::map<char, TrieNode*> children;  // Using map for ordered transitions
    TrieNode* fail;
    vector<int> output;

    TrieNode() : fail(nullptr) {}
    
    ~TrieNode() {
        for (auto& pair : children) {
            delete pair.second;
        }
    }
};

// Flat GPU representation
struct FlatGPUAutomaton {
    int* state_info;       // [fail_state|output_start|trans_start]
    int* transitions;      // Packed [char1|state1|char2|state2|...]
    int* outputs;          // Packed pattern indices
    int* trans_counts;     // Number of transitions per state
    int* output_counts;    // Number of outputs per state
    int num_states;        // Total number of states
    int total_transitions; // Total number of transitions
    int total_outputs;     // Total number of outputs
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
            TrieNode* failState = current->fail;
            
            while (failState != nullptr) {
                // Try to find a transition for character c from the fail state
                if (failState->children.find(c) != failState->children.end()) {
                    child->fail = failState->children[c];
                    break;
                }
                
                // If no transition and not at root, try the fail state's fail state
                if (failState == root) {
                    child->fail = root;
                    break;
                }
                
                failState = failState->fail;
            }
            
            // DO NOT merge outputs from fail state - we'll check fail states at runtime
        }
    }
}

// Calculate the total number of transitions and outputs in the automaton
void calculateAutomatonSize(TrieNode* root, int& numStates, int& totalTransitions, int& totalOutputs) {
    numStates = 0;
    totalTransitions = 0;
    totalOutputs = 0;
    
    queue<TrieNode*> q;
    unordered_map<TrieNode*, int> nodeToState;
    
    q.push(root);
    nodeToState[root] = numStates++;
    
    while (!q.empty()) {
        TrieNode* node = q.front();
        q.pop();
        
        // Count transitions
        totalTransitions += node->children.size();
        
        // Count outputs
        totalOutputs += node->output.size();
        
        // Add children to queue
        for (auto& pair : node->children) {
            TrieNode* child = pair.second;
            if (nodeToState.find(child) == nodeToState.end()) {
                nodeToState[child] = numStates++;
                q.push(child);
            }
        }
    }
    
    printf("Calculated automaton size: %d states, %d transitions, %d outputs\n", 
           numStates, totalTransitions, totalOutputs);
}

// Convert CPU trie to optimized flat GPU representation
void convertToFlatGPUAutomaton(TrieNode* root, FlatGPUAutomaton& automaton) {
    int numStates = 0;
    int totalTransitions = 0;
    int totalOutputs = 0;
    
    // First calculate required sizes
    calculateAutomatonSize(root, numStates, totalTransitions, totalOutputs);
    
    // Allocate memory for the flat automaton
    automaton.num_states = numStates;
    automaton.total_transitions = totalTransitions;
    automaton.total_outputs = totalOutputs;
    
    automaton.state_info = new int[numStates * 3];
    automaton.transitions = new int[totalTransitions * 2];
    automaton.outputs = new int[totalOutputs];
    automaton.trans_counts = new int[numStates];
    automaton.output_counts = new int[numStates];
    
    // Initialize counts to zero
    memset(automaton.trans_counts, 0, numStates * sizeof(int));
    memset(automaton.output_counts, 0, numStates * sizeof(int));
    
    // BFS to assign states
    queue<TrieNode*> q;
    unordered_map<TrieNode*, int> nodeToState;
    
    q.push(root);
    nodeToState[root] = 0;
    
    int currentStateID = 0;
    int transitionIndex = 0;
    int outputIndex = 0;
    
    while (!q.empty()) {
        TrieNode* node = q.front();
        q.pop();
        
        int stateID = nodeToState[node];
        
        // Set transition and output counts
        automaton.trans_counts[stateID] = node->children.size();
        automaton.output_counts[stateID] = node->output.size();
        
        // Set fail state
        if (node == root) {
            automaton.state_info[stateID * 3] = 0; // Root fails to itself
        } else {
            automaton.state_info[stateID * 3] = nodeToState[node->fail];
        }
        
        // Set output start index
        automaton.state_info[stateID * 3 + 1] = outputIndex;
        
        // Copy outputs
        for (int outputID : node->output) {
            automaton.outputs[outputIndex++] = outputID;
        }
        
        // Set transition start index
        automaton.state_info[stateID * 3 + 2] = transitionIndex / 2;
        
        // Copy transitions
        for (auto& pair : node->children) {
            char c = pair.first;
            TrieNode* child = pair.second;
            
            // Assign state ID to child if not already assigned
            if (nodeToState.find(child) == nodeToState.end()) {
                nodeToState[child] = ++currentStateID;
                q.push(child);
            }
            
            // Store transition
            automaton.transitions[transitionIndex++] = (unsigned char)c;
            automaton.transitions[transitionIndex++] = nodeToState[child];
        }
    }
    
    // Verify
    if (currentStateID + 1 != numStates) {
        printf("Error: State count mismatch! Expected %d, got %d\n", numStates, currentStateID + 1);
    }
    if (transitionIndex != totalTransitions * 2) {
        printf("Error: Transition count mismatch! Expected %d, got %d\n", totalTransitions * 2, transitionIndex);
    }
    if (outputIndex != totalOutputs) {
        printf("Error: Output count mismatch! Expected %d, got %d\n", totalOutputs, outputIndex);
    }
}

// Simple kernel to test CUDA functionality
__global__ void testKernel(int* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        output[0] = 42;
    }
}

// Improved Aho-Corasick kernel that correctly counts matches
__global__ void improvedAhoCorasickKernel(
    const char* text, int textLength, 
    const int* state_info, const int* transitions, const int* outputs,
    const int* trans_counts, const int* output_counts,
    int numStates,
    int* pattern_matches, int* match_count,
    int* match_positions, int* match_pattern_ids, int maxMatches,
    const int* pattern_lengths, int patternCount) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Each thread is responsible for matches that START at its assigned positions
    for (int startPos = tid; startPos < textLength; startPos += stride) {
        int state = 0;

        // Process characters starting from this thread's assigned position
        for (int i = startPos; i < min(startPos + WORK_CHUNK_SIZE, textLength); i++) {
            unsigned char c = text[i];
            bool found = false;
            
            // Try direct transitions first
            int transCount = trans_counts[state];
            int transStart = state_info[state * 3 + 2];

            for (int j = 0; j < transCount; j++) {
                if (transitions[(transStart + j) * 2] == c) {
                    state = transitions[(transStart + j) * 2 + 1];
                    found = true;
                    break;
                }
            }

            // If no direct transition, follow fail path
            if (!found) {
                int currentState = state;
                while (currentState != 0 && !found) {
                    currentState = state_info[currentState * 3]; // follow fail link
                    
                    transCount = trans_counts[currentState];
                    transStart = state_info[currentState * 3 + 2];

                    for (int j = 0; j < transCount; j++) {
                        if (transitions[(transStart + j) * 2] == c) {
                            state = transitions[(transStart + j) * 2 + 1];
                            found = true;
                            break;
                        }
                    }
                }
                
                if (!found) state = 0; // Reset to root if no transition found
            }

            // Check for pattern matches at this position
            // First, check the current state
            int outCount = output_counts[state];
            if (outCount > 0) {
                int outStart = state_info[state * 3 + 1];
                for (int j = 0; j < outCount; j++) {
                    int patternId = outputs[outStart + j];
                    if (patternId >= 0 && patternId < patternCount) {
                        int matchPos = i - pattern_lengths[patternId] + 1;
                        
                        // IMPORTANT: Only count matches that START at our assigned position
                        if (matchPos == startPos) {
                            atomicAdd(&pattern_matches[patternId], 1);
                            int idx = atomicAdd(match_count, 1);
                            if (idx < maxMatches) {
                                match_positions[idx] = matchPos;
                                match_pattern_ids[idx] = patternId;
                            }
                        }
                    }
                }
            }
            
            // Now check all fail states for matches too
            int failState = state;
            while (failState != 0) {
                failState = state_info[failState * 3]; // Move to fail state
                
                outCount = output_counts[failState];
                if (outCount > 0) {
                    int outStart = state_info[failState * 3 + 1];
                    for (int j = 0; j < outCount; j++) {
                        int patternId = outputs[outStart + j];
                        if (patternId >= 0 && patternId < patternCount) {
                            int matchPos = i - pattern_lengths[patternId] + 1;
                            
                            // Only count matches that START at our assigned position
                            if (matchPos == startPos) {
                                atomicAdd(&pattern_matches[patternId], 1);
                                int idx = atomicAdd(match_count, 1);
                                if (idx < maxMatches) {
                                    match_positions[idx] = matchPos;
                                    match_pattern_ids[idx] = patternId;
                                }
                            }
                        }
                    }
                }
            }
            
            // If we're at root and processed at least one character, we can stop
            // as no future matches can start at our position
            if (i > startPos && state == 0) break;
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

// CPU implementation for verification
void runCPUAhoCorasick(const string& text, const vector<string>& patterns, vector<int>& matches) {
    // Create trie
    TrieNode* root = new TrieNode();
    for (size_t i = 0; i < patterns.size(); ++i) {
        insertPattern(root, patterns[i], i);
    }
    buildFailTransitions(root);
    
    // Initialize match counts
    matches.resize(patterns.size(), 0);
    
    // Search text
    TrieNode* state = root;
    for (size_t i = 0; i < text.length(); ++i) {
        char c = text[i];
        
        // Find next state
        while (state != root && state->children.find(c) == state->children.end()) {
            state = state->fail;
        }
        
        if (state->children.find(c) != state->children.end()) {
            state = state->children[c];
        }
        
        // Check for matches
        TrieNode* current = state;
        while (current != root) {
            for (int patternId : current->output) {
                matches[patternId]++;
            }
            current = current->fail;
        }
    }
    
    delete root;
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
    
    // Build the Aho-Corasick automaton on CPU
    printf("Building Aho-Corasick automaton...\n");
    TrieNode* root = new TrieNode();
    for (size_t i = 0; i < patterns.size(); ++i) {
        insertPattern(root, patterns[i], i);
    }
    buildFailTransitions(root);
    
    // Convert to flat GPU structure
    printf("Converting to flat GPU structure...\n");
    FlatGPUAutomaton automaton;
    convertToFlatGPUAutomaton(root, automaton);
    
    printf("Created flat automaton with %d states, %d transitions, %d outputs\n", 
           automaton.num_states, automaton.total_transitions, automaton.total_outputs);
    
    // Create pattern lengths array
    int* patternLengths = new int[patterns.size()];
    for (size_t i = 0; i < patterns.size(); i++) {
        patternLengths[i] = patterns[i].length();
    }
    
    // Optionally run CPU version for verification
    vector<int> cpuMatches;
    if (doVerification) {
        printf("Running CPU Aho-Corasick for verification...\n");
        clock_t cpuStart = clock();
        runCPUAhoCorasick(text, patterns, cpuMatches);
        clock_t cpuEnd = clock();
        double cpuTime = (double)(cpuEnd - cpuStart) / CLOCKS_PER_SEC * 1000.0;
        printf("CPU Aho-Corasick completed in %.2f ms\n", cpuTime);
    }
    
    // Allocate GPU memory for flat automaton
    int* d_state_info;
    int* d_transitions;
    int* d_outputs;
    int* d_trans_counts;
    int* d_output_counts;
    
    cudaCheckError(cudaMalloc(&d_state_info, automaton.num_states * 3 * sizeof(int)));
    cudaCheckError(cudaMalloc(&d_transitions, automaton.total_transitions * 2 * sizeof(int)));
    cudaCheckError(cudaMalloc(&d_outputs, automaton.total_outputs * sizeof(int)));
    cudaCheckError(cudaMalloc(&d_trans_counts, automaton.num_states * sizeof(int)));
    cudaCheckError(cudaMalloc(&d_output_counts, automaton.num_states * sizeof(int)));
    
    // Copy automaton to device
    cudaCheckError(cudaMemcpy(d_state_info, automaton.state_info, 
                          automaton.num_states * 3 * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_transitions, automaton.transitions, 
                          automaton.total_transitions * 2 * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_outputs, automaton.outputs, 
                          automaton.total_outputs * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_trans_counts, automaton.trans_counts, 
                          automaton.num_states * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_output_counts, automaton.output_counts, 
                          automaton.num_states * sizeof(int), cudaMemcpyHostToDevice));
    
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
    
    // Use smaller grid size for better stability
    int blockSize = THREADS_PER_BLOCK;
    int gridSize = (text_length + blockSize - 1) / blockSize;  // Reduced grid size
    
    printf("Launching improved Aho-Corasick kernel with grid size: %d, block size: %d\n", gridSize, blockSize);
    
    // Set timeout to prevent kernel hangs
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 10);  // Set a limit on synchronization depth
    
    // Use the improved kernel
    improvedAhoCorasickKernel<<<gridSize, blockSize>>>(
        d_text, text_length, 
        d_state_info, d_transitions, d_outputs,
        d_trans_counts, d_output_counts,
        automaton.num_states,
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
    
    // Wait for kernel to finish with timeout safety
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
    delete[] automaton.state_info;
    delete[] automaton.transitions;
    delete[] automaton.outputs;
    delete[] automaton.trans_counts;
    delete[] automaton.output_counts;
    
    // Free GPU memory
    cudaFree(d_state_info);
    cudaFree(d_transitions);
    cudaFree(d_outputs);
    cudaFree(d_trans_counts);
    cudaFree(d_output_counts);
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