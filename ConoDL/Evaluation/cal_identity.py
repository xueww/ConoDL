'''Function to calculate the identity of input sequence.'''

from Bio import pairwise2
import random


file_path = 'input_sequence.txt'
try:
    with open(file_path, 'r') as file:
        lines = file.readlines()
except FileNotFoundError:
    print(f"file '{file_path}' no find")
    exit()
except Exception as e:
    print(f"error：{e}")
    exit()


sequences = [line.strip() for line in lines]

random_sequences = random.sample(sequences, k=200)

output_file = 'output_identity.txt'

with open(output_file, 'w') as output:
    for i in range(len(random_sequences)):
        max_similarity = 0
        for j in range(len(random_sequences)):
            if i != j:
                alignment = pairwise2.align.globalxx(random_sequences[i], random_sequences[j], one_alignment_only=True)
                similarity = alignment[0].score / max(len(random_sequences[i]), len(random_sequences[j]))
                if similarity > max_similarity:
                    max_similarity = similarity
        result = f" {i+1} \t {max_similarity}\n"
        output.write(result)