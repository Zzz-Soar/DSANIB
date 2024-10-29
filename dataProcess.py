import torch.utils.data as data
import torch
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from utils import proteinMap, drugMap

# the data process for DSANIB 
class DTIDataset(data.Dataset):
    def __init__(self, list_IDs, df):
        self.list_IDs = list_IDs
        self.df = df
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]

        sequence = self.df.iloc[index]['Protein']
        protein_Initial = proteinMap(sequence)

        molecule = self.df.iloc[index]['SMILES']
        molecule_comp = self.fc(smiles=molecule, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        drug_Initial = drugMap(molecule_comp)

        label = self.df.iloc[index]["Y"]

        return drug_Initial, protein_Initial, label
