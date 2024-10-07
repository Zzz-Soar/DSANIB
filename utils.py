import os
import random
import numpy as np
import torch
import dgl
import logging
from functools import partial
from dgllife.utils import smiles_to_bigraph

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23,
               "X": 24,"Z": 25}

CHARPROTLEN = 25


CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}


CHARISOSMILEN = 64

def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=290):
    X = np.zeros(MAX_SMI_LEN, dtype=np.int64())
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind[ch]
    return X

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def graph_collate_func(x):
    # N = len(x)
    d, p, y = zip(*x)
    d = dgl.batch(d)
    # compoundint = torch.from_numpy(label_smiles(
    #     d_seq, CHARISOSMISET, 100))
    # compound_max = 290
    # compound_new = torch.zeros((N, compound_max), dtype=torch.long)
    # for i, pair in enumerate(d_seq):
    #     compoundstr = pair
    #     compoundint = torch.from_numpy(label_smiles(
    #         compoundstr, CHARISOSMISET, compound_max))
    #     compound_new[i] = compoundint
    return d, torch.tensor(np.array(p)), torch.tensor(y)

def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)


def proteinMap(sequence, max_length=1200):
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in sequence category encoding, skip and treat as " f"padding."
            )
    return encoding

def drugMap(molecule, max_drug_nodes=290):

    actual_node_feats = molecule.ndata.pop('h')
    num_actual_nodes = actual_node_feats.shape[0]
    num_virtual_nodes = max_drug_nodes - num_actual_nodes
    virtual_node_bit = torch.zeros([num_actual_nodes, 1])
    actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
    molecule.ndata['h'] = actual_node_feats

    virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)
    molecule.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
    molecule = molecule.add_self_loop()

    return  molecule
