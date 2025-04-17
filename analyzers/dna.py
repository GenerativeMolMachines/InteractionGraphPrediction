from structure import Structure
from rna import RNA


class DNA(Structure):
    def __init__(self, name: str, sequence: str):
        super().__init__(name, sequence)

    def transcription(self):
        return RNA(name=self.name + "; from dna", sequence=self.sequence.replace("T", "U"))
