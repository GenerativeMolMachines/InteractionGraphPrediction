from typing import List, Tuple

from neo4j import Record

from analyzers import Structure, Protein
from graphs.graph import Node, Edge


class DBGraph:
    def __init__(self, nodes: List[Node], edges: List[Edge]):
        self.nodes = nodes
        self.edges = edges

    @staticmethod
    def graph_from_records(records: List[Record]):
        nodes = set()
        edges = []
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
            nodes.add(vertex1)
            nodes.add(vertex2)
            edges.append(Edge(vertex1, vertex2))
        return DBGraph(list(nodes), edges)
