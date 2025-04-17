from . import Structure


class Peptide(Structure):
    def __init__(self, name: str, sequence: str):
        super().__init__(name, sequence)
