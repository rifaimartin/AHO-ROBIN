def extract_first_million(input_file, output_file, length=1000000, make_uppercase=True):
    """
    Extract first 'length' base pairs from FASTA file, skipping header lines.
    Optionally convert all bases to uppercase or lowercase.
    
    Args:
        input_file (str): Path to input FASTA file
        output_file (str): Path to output file
        length (int): Number of base pairs to extract
        make_uppercase (bool): If True, convert all bases to uppercase; if False, keep original case
    """
    import os
    
    # Verify file exists and print full path
    full_input_path = os.path.abspath(input_file)
    print(f"Looking for file: {full_input_path}")
    if not os.path.exists(full_input_path):
        print(f"ERROR: Input file not found: {full_input_path}")
        return
    else:
        print(f"File found. Size: {os.path.getsize(full_input_path) / (1024*1024):.2f} MB")
    
    count = 0
    line_count = 0
    try:
        with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
            # Write an informative header to the output file
            fout.write(f"# First {length} base pairs extracted from {input_file}\n")
            
            print("Starting to process file...")
            for line in fin:
                line_count += 1
                if line_count % 1000 == 0:
                    print(f"Processed {line_count} lines, extracted {count} bases so far...")
                
                if line.startswith('>'):
                    print(f"Found header line: {line.strip()}")
                    continue  # Skip header lines
                
                # Remove whitespace and newlines
                seq = line.strip()
                
                # Convert to uppercase if requested
                if make_uppercase:
                    seq = seq.upper()
                
                remaining = length - count
                
                if len(seq) <= remaining:
                    fout.write(seq)
                    count += len(seq)
                else:
                    fout.write(seq[:remaining])
                    count += remaining
                    break
                    
            print(f"Extraction complete. Extracted {count} base pairs to {output_file}")
    except Exception as e:
        print(f"ERROR during extraction: {str(e)}")

# File paths - adjust these to match your actual file paths
input_file = 'GCA_000001405.29_GRCh38.p14_genomic.fna'
output_file = 'human_1m_upper.txt'

# Extract 1 million base pairs and convert all to uppercase
print("Starting extraction script...")
extract_first_million(input_file, output_file, make_uppercase=True)
print("Script execution finished.")