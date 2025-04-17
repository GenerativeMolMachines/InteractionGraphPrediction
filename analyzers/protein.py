from Bio.SeqUtils.ProtParam import ProteinAnalysis
from structure import Structure
from typing import List, Dict


class Protein(Structure):
    def __init__(self, name: str, sequence: str):
        super().__init__(name, sequence)
        self.protein = ProteinAnalysis(sequence)

    def get_descriptors(self, desc_names: List[str]) -> Dict[str, int]:
        res: Dict[str, int] = {}
        if "mw" in desc_names:
            res["mw"] = self.protein.molecular_weight()
        if "gravy" in desc_names:
            res["gravy"] = self.protein.gravy()
        if "second_structure" in desc_names:
            helix, turn, sheet = self.protein.secondary_structure_fraction()
            res["helix"] = helix
            res["turn"] = turn
            res["sheet"] = sheet
        return res
