from . import Structure


class Antibody(Structure):
    def __init__(self, name: str, sequence: str):
        super().__init__(name, sequence)
