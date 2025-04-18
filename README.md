# Parallel String Matching Algorithm Comparison: Aho-Corasick vs Rabin-Karp using CUDA

This project implements and compares two popular string matching algorithms (Aho-Corasick and Rabin-Karp) in parallel using CUDA for high-performance DNA sequence analysis. The implementation focuses on leveraging GPU acceleration to significantly improve the processing speed of large-scale genomic data.

# Overview
String matching is a fundamental operation in bioinformatics, especially for DNA sequence analysis. Traditional sequential approaches often struggle with scalability when dealing with large datasets. This project explores how parallelization using NVIDIA's CUDA can enhance the performance of two fundamentally different string-matching approaches:

* Aho-Corasick: A finite state machine-based algorithm for efficient multi-pattern matching
* Rabin-Karp: A hashing-based algorithm that uses rolling hash functions for pattern identification

# Results
Our research demonstrates significant performance improvements using GPU parallelization:

| Algorithm    | Dataset Size | Sequential Time (ms) | CUDA Time (ms) | Speedup |
|--------------|--------------|----------------------|----------------|---------|
| Aho-Corasick | -        | TBD                  | TBD            | TBD     |
| Rabin-Karp   | -        | TBD                  | TBD            | TBD     |

## Technical Insights

- Aho-Corasick's finite state machine structure creates irregular memory access patterns, potentially leading to thread divergence in CUDA implementations
- Rabin-Karp's hash computation is highly parallelizable but faces challenges in managing hash collisions efficiently
- Memory management strategies significantly impact performance for both algorithms
- Algorithm selection should consider pattern quantity, distribution, and specific DNA sequence characteristics

### Contributors

* Christoffel H. Moekoe (christoffelhm@gmail.com)
* Muhammad Rifai (rifaimartinjham@gmail.com)
* Ilham I. Saputra (pewililham13@gmail.com)
* Sofia K. Hanim (Kartikasofia35@gmail.com)


License
This project is part of academic research at UPH Tangerang, Indonesia.