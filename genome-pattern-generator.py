import random
import argparse

def generate_genome_patterns(pattern_length=5, num_patterns=1000, output_file=None):
    """
    Generate random DNA patterns of specified length
    
    Args:
        pattern_length: Length of each pattern
        num_patterns: Number of patterns to generate
        output_file: File to save the patterns
    """
    if output_file is None:
        output_file = f"pattern_{num_patterns}.txt"
        
    nucleotides = ['A', 'C', 'G', 'T']
    patterns = set()  # Use a set to avoid duplicates
    
    # Generate unique patterns
    while len(patterns) < num_patterns:
        pattern = ''.join(random.choice(nucleotides) for _ in range(pattern_length))
        patterns.add(pattern)
    
    # Convert to list and sort for deterministic output
    patterns_list = sorted(list(patterns))
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(','.join(patterns_list))
    
    print(f"Generated {len(patterns_list)} unique patterns of length {pattern_length}")
    print(f"Saved to {output_file}")
    
    # Print sample of patterns
    sample_size = min(8, len(patterns_list))
    print(f"\nSample of {sample_size} patterns:")
    for i in range(sample_size):
        print(patterns_list[i], end=",")
    print("...")

def main():
    parser = argparse.ArgumentParser(description='Generate random DNA patterns')
    parser.add_argument('--length', type=int, default=5, help='Length of each pattern')
    parser.add_argument('--counts', nargs='+', type=int, 
                        default=[8, 16, 32, 64, 128, 256, 512, 1024], 
                        help='List of pattern counts to generate')
    
    args = parser.parse_args()
    
    # Generate patterns for each count value
    for count in args.counts:
        output_file = f"pattern_{count}.txt"
        generate_genome_patterns(args.length, count, output_file)

if __name__ == "__main__":
    main()