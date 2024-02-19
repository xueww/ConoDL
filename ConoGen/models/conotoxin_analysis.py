'''Function to analyze the input sequence and extract the cysteine scaffold'''

import string


def analyze_toxin_sequence(sequence):
    scaffold = ""   
    current_scaffold = ""   
    for char in sequence:
        if char == "C":
            current_scaffold += "C"
        else:
            if current_scaffold:
                scaffold += current_scaffold.replace("-", "") + "-"
                current_scaffold = ""
    if current_scaffold:    
        scaffold += current_scaffold.replace("-", "")
    scaffold = scaffold.strip("-")  
    return scaffold


def get_scaffold_type(scaffold):    
    scaffold_types = {
        "CC-C-C": "I",
        "CCC-C-C-C": "II",
        "CC-C-C-CC": "III",
        "CC-C-C-C-C": "IV",
        "CC-CC": "V",
        "C-C-CC-C-C": "VI/VII",
        "C-C-C-C-C-C-C-C-C-C": "VIII",
        "C-C-C-C-C-C": "IX",
        "CC-C.[PO]C": "X",
        "C-C-CC-CC-C-C": "XI",
        "C-C-C-C-CC-C-C": "XII",
        "C-C-C-CC-C-C-C": "XIII",
        "C-C-C-C": "XIV",
        "C-C-CC-C-C-C-C": "XV",
        "C-C-CC": "XVI",
        "C-C-CC-C-CC-C": "XVII",
        "C-C-CC-CC": "XVIII",
        "C-C-C-CCC-C-C-C": "XIX",
        "C-CC-C-CC-C-C-C-C": "XX",
        "CC-C-C-C-CC-C-C-C": "XXI",
        "C-C-C-C-C-C-C-C": "XXII",
        "C-C-C-CC-C": "XXIII",
        "C-CC-C": "XXIV",
        "C-C-C-C-CC": "XXV",
        "C-C-C-C-CC-CC": "XXVI",
        "C-C-C-CCC-C-C": "XXVII",
        "C-C-C-CC-C-C-C-C": "XXVIII",
        "CCC-C-CC-C-C": "XXIX",
        "C-C-CCC-C-C-C-CC": "XXX",
        "C-CC-C-C-C": "XXXII",
        "C-C-C-C-C-C-C-C-C-C-C-C": "XXXIII"
         # Add definitions for other scaffold types
    }
    if scaffold in scaffold_types:
        return scaffold_types[scaffold]
    else:
        return "-"


def count_cysteine(sequence):
    return sequence.count("C")

def compute_sequence_length(sequence):
    letters = [char for char in sequence if char in string.ascii_letters]
    return len(letters)


def process_toxin_file(toxin_sequence):
    scaffold = analyze_toxin_sequence(toxin_sequence)
    scaffold_type = get_scaffold_type(scaffold)
    cysteine_count = count_cysteine(toxin_sequence)
    length = compute_sequence_length(toxin_sequence)
    return toxin_sequence, length, scaffold, scaffold_type, cysteine_count


