import numpy as np
import torch

"""
concepts: a list of indexes
distribution: probability distribution with regard to concepts
vocab: concept vocabulary
"""


def rid_repetition(concepts, distribution):
    new_concept = []
    new_distr = []
    for i, cpt in enumerate(concepts):
        if cpt not in new_concept:
            if cpt != "<pad>":
                new_concept.append(cpt)
                new_distr.append(distribution[i])
        else:
            index = new_concept.index(cpt)
            new_distr[index] += distribution[i]
    return new_concept, new_distr


def print_state(concepts, distribution_tensor, vocab):
    if torch.cuda.is_available():
        distribution = distribution_tensor.cpu().numpy()
    else:
        distribution = distribution_tensor.numpy()
    concepts = [vocab.itos[cpt] for cpt in concepts]
    concepts, distribution = rid_repetition(concepts, distribution)
    return concepts, distribution
