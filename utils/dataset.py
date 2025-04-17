from typing import List
from neo4j._data import Record

from analyzers import Structure
from graphs.graph import Edge, Node


class InteractionDataset:
    def __init__(self):
        self.col1 = []  # object or seq
        self.col2 = []  # object or seq
        self.interaction = []  # 0/1

    def add_records(self, records: List[Record]):
        for record in records:
            rec = record["p"].nodes
            v = rec[0]
            u = rec[1]
            v_lbs = list(v._labels)
            u_lbs = list(u._labels)
            v_properties = dict(v)
            u_properties = dict(u)
            vertex1 = Node(v_properties["name"], v_lbs,
                           Structure(v_properties["name"], v_properties["content"]))  # CHANGE THIS
            vertex2 = Node(u_properties["name"], u_lbs,
                           Structure(u_properties["name"], u_properties["content"]))  # CHANGE THIS
            self.col1.append(vertex1)
            self.col2.append(vertex2)
            self.interaction.append(1)

    def add_records_from_cols(self, to_col1, to_col2, interacted):
        for i in range(len(to_col1)):
            self.col1.append(to_col1[i])
            self.col2.append(to_col2[i])
            self.interaction.append(interacted[i])