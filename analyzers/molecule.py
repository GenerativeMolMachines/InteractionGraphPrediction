from collections import defaultdict
from typing import List, Dict
import pickle
from structure import Structure
from rdkit import Chem
import selfies as sf
import numpy as np
from rdkit.Chem import Descriptors


# SMILES, ex. : C1=CC=CC=C1
# SELFIES, ex. : [C][=C][C][=C][C][=C][Ring1][=Branch1]
# SMARTS, ex. : [C:1]=[O,N:2]>>*[C:1][*:2]


class Molecule(Structure):
    def __init__(self, name: str, sequence: str, format: str = "smiles"):  # smiles, selfies, smarts
        super().__init__(name, sequence)
        self.molecule = Chem.MolFromSmiles(self.sequence)
        self.format = format
        self.atom_dict = defaultdict(lambda: {})
        self.bond_dict = defaultdict(lambda: {})
        self.fingerprint_dict = defaultdict(lambda: {})
        self.edge_dict = defaultdict(lambda: {})
        if format != "smiles":
            self.change_format(format)

    def get_descriptors(self, desc_names: List[str]) -> Dict[str, int]:  # fingerprints(Morgan, MACCS)
        pass

    def change_format(self, new_format: str):
        if new_format == "smiles":
            self.format = "smiles"
            self.sequence = Chem.MolToSmiles(self.molecule)
        elif new_format == "selfies":
            self.format = "selfies"
            self.sequence = sf.encoder(self.sequence)
        elif new_format == "smarts":
            self.format = "smarts"
            self.sequence = Chem.MolToSmarts(self.molecule)

    def create_adjacency(self):
        adjacency = Chem.GetAdjacencyMatrix(self.molecule)
        return np.array(adjacency)

    def create_atoms(self):
        atoms = [a.GetSymbol() for a in self.molecule.GetAtoms()]
        for a in self.molecule.GetAromaticAtoms():
            i = a.GetIdx()
            atoms[i] = (atoms[i], 'aromatic')
        atoms = [self.atom_dict[a] for a in atoms]
        return np.array(atoms)

    def create_ij_bond_dict(self):
        i_jbond_dict = defaultdict(lambda: [])
        for b in self.molecule.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            bond = self.bond_dict[str(b.GetBondType())]
            i_jbond_dict[i].append((j, bond))
            i_jbond_dict[j].append((i, bond))
        return i_jbond_dict

    def extract_fingerprints(self, atoms, i_jbond_dict, radius):
        if (len(atoms) == 1) or (radius == 0):
            fingerprints = [self.fingerprint_dict[a] for a in atoms]
        else:
            nodes = atoms
            i_jedge_dict = i_jbond_dict

            for _ in range(radius):

                fingerprints = []
                for i, j_edge in i_jedge_dict.items():
                    neighbors = [(nodes[j], edge) for j, edge in j_edge]
                    fingerprint = (nodes[i], tuple(sorted(neighbors)))
                    fingerprints.append(self.fingerprint_dict[fingerprint])
                nodes = fingerprints
                _i_jedge_dict = defaultdict(lambda: [])
                for i, j_edge in i_jedge_dict.items():
                    for j, edge in j_edge:
                        both_side = tuple(sorted((nodes[i], nodes[j])))
                        edge = self.edge_dict[(both_side, edge)]
                        _i_jedge_dict[i].append((j, edge))
                i_jedge_dict = _i_jedge_dict

        return np.array(fingerprints)


if __name__ == "__main__":
    mol = Molecule("Cyclone", "C1=CC=CC=C1")
    print(mol.create_adjacency())
