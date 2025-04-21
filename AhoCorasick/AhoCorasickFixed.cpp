#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <unordered_map>
#include <ctime>
#include <fstream>
#include <sstream>

using namespace std;

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
            child->output.insert(child->output.end(), child->fail->output.begin(), child->fail->output.end());
        }
    }
}

vector<pair<int, int>> search(const string& text, const vector<string>& patterns) {
    vector<pair<int, int>> matches; // Store start and end indices of matches

    TrieNode* root = new TrieNode();
    for (int i = 0; i < (int)patterns.size(); ++i) {
        insertPattern(root, patterns[i], i);
    }

    buildFailTransitions(root);

    TrieNode* current = root;
    
    // Maximum number of matches to display
    const int MAX_DISPLAY = 20;
    int display_count = 0;

    for (int i = 0; i < (int)text.size(); ++i) {
        char c = text[i];

        // Follow failure links until we find a match or reach root
        while (current != root && current->children.find(c) == current->children.end()) {
            current = current->fail;
        }

        // Transition to the next state
        if (current->children.find(c) != current->children.end()) {
            current = current->children[c];
        } else {
            current = root;
        }

        // Check if current state contains any output
        for (int patternIndex : current->output) {
            int startIdx = i - (patterns[patternIndex].size() - 1);
            matches.push_back(make_pair(startIdx, i));
            
            // Display match info with limit
            if (display_count < MAX_DISPLAY) {
                cout << "Pattern found at index " << startIdx << endl;
                display_count++;
                
                if (display_count == MAX_DISPLAY && matches.size() > MAX_DISPLAY) {
                    cout << "... (showing only first " << MAX_DISPLAY << " matches)" << endl;
                }
            }
        }
    }

    delete root; // Free allocated memory
    return matches;
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

int main() {
    clock_t start_time = clock();
    
    const char* filename = "human_1m_upper.txt";
    int text_length = 0;
    
    cout << "Reading file: " << filename << endl;
    
    // Read the file content
    string text = read_text_file(filename, &text_length);
    cout << "Read " << text_length << " characters from file" << endl;
    
    // Define patterns to search for
    vector<string> patterns = {"GGA"};
    cout << "Searching for pattern: " << patterns[0] << endl;
    
    // Run the Aho-Corasick algorithm
    vector<pair<int, int>> result = search(text, patterns);
    
    // Calculate execution time
    clock_t end_time = clock();
    double execution_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC * 1000.0; // milliseconds
    
    cout << "Total matches found: " << result.size() << endl;
    cout << "Execution time: " << execution_time << " ms" << endl;
    
    return 0;
}