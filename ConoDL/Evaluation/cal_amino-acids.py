from collections import Counter


input_file = 'input_sequence.txt'
output_file = 'output_amino-acid.txt'
amino_acids = "ACDEFGHIKLMNPQRSTVWY"

total_amino_acids = 0
amino_acid_counts = {amino_acid: 0 for amino_acid in amino_acids}

with open(input_file, 'r') as f:
    sequences = f.readlines()
    for sequence in sequences:
        sequence = sequence.strip()
        amino_acid_counter = Counter(sequence)
        total_amino_acids += len(sequence)
        for amino_acid in amino_acids:
            amino_acid_counts[amino_acid] += amino_acid_counter[amino_acid]

with open(output_file, 'w') as output:
    output.write("amino acid,count\n")
    for amino_acid, count in amino_acid_counts.items():
        output.write(f"{amino_acid},{count}\n")
