#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
using namespace std;

typedef pair<int, int> pii; // (posisi, indeks pola)

// Fungsi untuk menghitung hash dari string
int compute_hash(const string& str, int length, int d, int q) {
    int hash_value = 0;
    for (int i = 0; i < length; i++) {
        hash_value = (d * hash_value + str[i]) % q;
    }
    return hash_value;
}

// Fungsi Rabin-Karp untuk multiple pattern
vector<pii> search(const string& text, const vector<string>& patterns) {
    vector<pii> result;
    int d = 256; // Ukuran alfabet
    int q = 101; // Bilangan prima
    
    int text_length = text.length();
    int num_patterns = patterns.size();
    
    // Hitung hash untuk semua pola dan simpan dalam map: hash -> indeks pola
    unordered_map<int, vector<int>> pattern_hash_map;
    for (int i = 0; i < num_patterns; i++) {
        int pattern_length = patterns[i].length();
        if (pattern_length > text_length) continue;
        
        int pattern_hash = compute_hash(patterns[i], pattern_length, d, q);
        pattern_hash_map[pattern_hash].push_back(i);
    }
    
    // Untuk setiap panjang pola yang berbeda, lakukan pencarian Rabin-Karp
    for (int i = 0; i < num_patterns; i++) {
        int pattern_length = patterns[i].length();
        if (pattern_length > text_length) continue;
        
        // Cek apakah panjang pola ini sudah diproses
        bool already_processed = false;
        for (int j = 0; j < i; j++) {
            if (patterns[j].length() == pattern_length) {
                already_processed = true;
                break;
            }
        }
        if (already_processed) continue;
        
        // Hitung h = d^(pattern_length-1) % q
        int h = 1;
        for (int j = 0; j < pattern_length - 1; j++) {
            h = (h * d) % q;
        }
        
        // Hitung hash untuk jendela pertama teks
        int text_hash = compute_hash(text, pattern_length, d, q);
        
        // Cek kecocokan untuk jendela pertama
        if (pattern_hash_map.find(text_hash) != pattern_hash_map.end()) {
            for (int pattern_idx : pattern_hash_map[text_hash]) {
                if (patterns[pattern_idx].length() == pattern_length) {
                    bool match = true;
                    for (int j = 0; j < pattern_length; j++) {
                        if (text[j] != patterns[pattern_idx][j]) {
                            match = false;
                            break;
                        }
                    }
                    if (match) {
                        result.push_back(make_pair(0, pattern_idx));
                    }
                }
            }
        }
        
        // Slide jendela satu per satu
        for (int j = 0; j <= text_length - pattern_length; j++) {
            if (j > 0) {
                // Perbarui hash untuk jendela baru
                text_hash = (d * (text_hash - text[j-1] * h) + text[j+pattern_length-1]) % q;
                if (text_hash < 0) text_hash += q;
            }
            
            // Cek kecocokan jika hash cocok
            if (j > 0 && pattern_hash_map.find(text_hash) != pattern_hash_map.end()) {
                for (int pattern_idx : pattern_hash_map[text_hash]) {
                    if (patterns[pattern_idx].length() == pattern_length) {
                        bool match = true;
                        for (int k = 0; k < pattern_length; k++) {
                            if (text[j+k] != patterns[pattern_idx][k]) {
                                match = false;
                                break;
                            }
                        }
                        if (match) {
                            result.push_back(make_pair(j, pattern_idx));
                        }
                    }
                }
            }
        }
    }
    
    return result;
}

int main() {
    vector<string> patterns = {"TGCGATA", "TGTG", "AAAG", "CCTCT", "AAGG"};
    string text = "TATTTCTGGTCCACCTGATTAATAAGAGCTTCCTCTACAATAGAATTTGTCAGGTAGGAATGTGTGTAGACTATAAAGGTCCTCACTCTCTTTATCTACCCT";
    
    vector<pii> result = search(text, patterns);
    
    cout << "Pola yang ditemukan:" << endl;
    for (const auto& match : result) {
        cout << "Pattern \"" << patterns[match.second] << "\" ditemukan pada posisi " 
             << match.first << endl;
    }
    
    return 0;
}