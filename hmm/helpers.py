"""
helper functions
"""

# mappings for codon
coding_start_mapping = {'ATG': [1, 2, 3], 'ATC': [4, 5, 6], 'ATA': [7, 8, 9], 'ATT': [10, 11, 12],
                        'GTG': [13, 14, 15], 'GTT': [16, 17, 18], 'TTG': [19, 20, 21]}
coding_stop_mapping = {'TAG': [22, 23, 24], 'TAA': [25, 26, 27], 'TGA': [28, 29, 30]}
reverse_start_mapping = {'CAT': [31, 32, 33], 'AAT': [34, 35, 36], 'CAC': [37, 38, 39], 'CAA': [40, 41, 42],
                         'TAT': [43, 44, 45], 'CAG': [46, 47, 48], 'GAT': [49, 50, 51]}
reverse_stop_mapping = {'CTA': [52, 53, 54], 'TTA': [55, 56, 57], 'TCA': [58, 59, 60]}


def translate_codon_to_state(codon, label):
    """given a codon and its label, return its state"""
    codon = codon.upper()
    mapping = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
    if label == 'noncoding':
        return 0
    elif label == 'coding_start':
        return coding_start_mapping[codon]
    elif label == 'coding_stop':
        return coding_stop_mapping[codon]
    elif label == 'reverse_start':
        return reverse_start_mapping[codon]
    elif label == 'reverse_stop':
        return reverse_stop_mapping[codon]
    elif label == 'coding':
        return [61, 62, 63]
    elif label == 'reverse':
        return [64, 65, 67]


def translate_state_to_ann(state):
    """given a state, return its corresponding annotation"""
    if state == 0:
        return 'N'
    elif 1 <= state <= 30 or 61 <= state <= 63:
        return 'C'
    elif 31 <= state <= 60 or state > 63:
        return 'R'


def translate_annotations_to_states(obs, ann):
    """translate annotations to indices of hidden states"""
    i, n = 0, len(obs)
    states = [None] * n
    while i < n:
        if ann[i] == 'N':
            states[i] = 0
            if i + 3 < n:
                if ann[i+1: i+4] == "CCC":
                    states[i+1: i+4] = translate_codon_to_state(obs[i+1: i+4], "coding_start")
                    i += 4
                elif ann[i+1: i+4] == "RRR":
                    states[i+1: i+4] = translate_codon_to_state(obs[i+1: i+4], "reverse_stop")
                    i += 4
                else:
                    i += 1
            else:
                i += 1
        elif ann[i] == 'C':
            if i+3 < n and ann[i+3] == 'N':
                states[i: i+3] = translate_codon_to_state(obs[i: i+3], "coding_stop")
            else:
                states[i: i+3] = translate_codon_to_state(obs[i: i+3], "coding")
            i += 3
        elif ann[i] == 'R':
            if i+3 < n and ann[i+3] == 'N':
                states[i: i+3] = translate_codon_to_state(obs[i: i+3], "reverse_start")
            else:
                states[i: i+3] = translate_codon_to_state(obs[i: i+3], "reverse")
            i += 3
    return states


def translate_states_to_annotations(indices):
    return ''.join([translate_state_to_ann(i) for i in indices])


def translate_observations_to_indices(obs):
    """translate actg to indices"""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    return [mapping[symbol.upper()] for symbol in obs]


def translate_indices_to_observations(indices):
    """translate indices back to actg"""
    mapping = ['A', 'C', 'G', 'T']
    return ''.join(mapping[idx] for idx in indices)


