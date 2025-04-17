from typing import List, Tuple
from analyzers import Structure, Protein


class Node:
    def __init__(self, name: str, labels: List[str], structure: Structure):
        self.name = name
        self.labels = list(labels)
        self.structure = structure

    def __str__(self):
        return f"name={self.name}, labels={self.labels}, seq={self.structure.sequence}."


class Edge:
    def __init__(self, vertex1: Node, vertex2: Node, content: str = ""):
        self.vertex1 = vertex1
        self.vertex2 = vertex2
        self.content = content


class Graph:
    def __init__(self, nodes: List[Node], edges: List[Edge]):
        self.nodes = nodes
        self.edges = edges