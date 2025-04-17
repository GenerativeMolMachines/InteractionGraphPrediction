from structure import Structure


class RNA(Structure):
    def __init__(self, name: str, sequence: str):
        super().__init__(name, sequence)