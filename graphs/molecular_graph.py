from typing import List

from analyzers import Molecule, Structure
from graph import Edge, Node, Graph


class Bond(Edge):
    def __init__(self, vertex1: Node, vertex2: Node):
        super().__init__(vertex1, vertex2)


class Atom(Node):
    def __init__(self, name: str):
        super().__init__(name, None, None)


class MolecularGraph(Graph):
    def __init__(self, nodes: List[Atom] = None, edges: List[Bond] = None, molecule: Molecule = None):
        if molecule is None:
            super().__init__(nodes, edges)
        else:
            mol = molecule.molecule
            self.molecule = molecule
            nodes_tmp = []
            edges_tmp = []
            atoms = mol.GetAtoms()
            bonds = mol.GetBonds()
            for atom in atoms:
                nodes.append(Atom(atom.GetSymbol()))
            for bond in bonds:
                edges.append(Bond(Atom(bond.GetBeginAtom()), Atom(bond.GetEndAtom())))
            super().__init__(nodes_tmp, edges_tmp)

