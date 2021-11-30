from hmm import hmm
from compare_anns import print_all


def read_fasta_file(filename):
    """
    Reads the given FASTA file f and returns a dictionary of sequences.
    Lines starting with ';' in the FASTA file are ignored.
    """
    sequences_lines = {}
    current_sequence_lines = None
    with open(filename) as fp:
        for line in fp:
            line = line.strip()
            if line.startswith(';') or not line:
                continue
            if line.startswith('>'):
                sequence_name = line.lstrip('>')
                current_sequence_lines = []
                sequences_lines[sequence_name] = current_sequence_lines
            else:
                if current_sequence_lines is not None:
                    current_sequence_lines.append(line)
    sequences = {}
    for name, lines in sequences_lines.items():
        sequences[name] = ''.join(lines)
    return sequences


def validation(genomes, annotations, id):
    """for a given id, train a model use the remaining 4 genomes and use this model to make predictions and calculate ac"""
    model = hmm()
    for i in range(5):
        if i != id:
            print(f"Training by counting. Processing genome{i+1}...")
            model.training_by_counting(genomes[i], annotations[i])
    print(f"Using model{id+1} to predict annotations for genome{id+1}...")
    true_ann = annotations[id]
    pred_ann = model.predict(genomes[id])
    total_ac = print_all(true_ann, pred_ann)
    return model, total_ac


def find_best_model():
    """find the best model by running cross validation on first 5 genomes and store its params for later use"""
    # read in and store the first 5 genome and their annotations for training
    genomes, annotations = [None] * 5, [None] * 5
    for i in range(1, 6):
        genomes[i-1] = read_fasta_file(f"data/genome{i}.fa")[f"genome{i}"]
        annotations[i-1] = read_fasta_file(f"data/true-ann{i}.fa")[f"true-ann{i}"]
    # run cross validation to select the best model
    models, acs = [None] * 5, [0] * 5
    for i in range(5):
        print(f"Validation {i+1}...")
        models[i], acs[i] = validation(genomes, annotations, i)
    # select the best model based on acs
    best_ac, best_ind = max((ac, ind) for (ind, ac) in enumerate(acs))
    best_model = models[best_ind]
    best_model.save_model()
    print(f"\nThe best model is model{best_ind+1}")


def predict_ann_and_save():
    """for genomes 6-10, save their annotations predicted by the trained hmm model"""
    for i in range(6, 11):
        genome = read_fasta_file(f"data/genome{i}.fa")[f"genome{i}"]
        model = hmm()
        pred_anns = model.predict(genome)
        with open(f"data/prediction/pred-ann{i}.fa", "w") as f:
            f.write(f">pred-ann{i}\n")
            f.write(pred_anns + "\n")


if __name__ == "__main__":
    # find the best hmm model by running cross validation on the first 5 genomes
    # find_best_model()
    # use the best model we found to predict annotations for genome 6-10
    predict_ann_and_save()


