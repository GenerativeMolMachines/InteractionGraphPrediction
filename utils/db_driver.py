from typing import List, Tuple
from neo4j import GraphDatabase, RoutingControl
from neo4j._data import Record


class DBDriver:
    def __init__(self, uri: str, username: str, password: str):
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

    def get_different_type(self, type1: str = None, type2: str = None, limit: int = 100):
        t1 = ""
        if type1 is not None:
            t1 += "t:" + type1
        t2 = ""
        if type2 is not None:
            t2 += "h:" + type2
        request_text = "MATCH p=(" + t1 + ")-[r:interacts_with]->(" + t2 + ")  RETURN p LIMIT " + str(limit)
        return self._execute(request_text)

    def _execute(self, request_text: str) -> List[Record]:
        records, _, _ = self.driver.execute_query(
            request_text,
            database_="neo4j", routing_=RoutingControl.READ)
        return records

    def get_records(self, elements: int = 100) -> List[Record]:
        request_text = "MATCH p=(sr)-[r:interacts_with]->(e) WHERE e.content IS NOT NULL and sr.content IS NOT NULL  RETURN p LIMIT " + str(
            elements)
        return self._execute(request_text)

    def get_interactions_by_id(self, id1: int, id2: int = None, limit: int = 100):
        cond = "WHERE Id(sr) = " + str(id1)
        if id2 is not None:
            cond += "and Id(sr) = " + str(id2)
        request_text = "MATCH p=(sr)-[r:interacts_with]->(e) " + cond + "  RETURN p LIMIT " + str(
            limit)
        return self._execute(request_text)

    def get_elements(self, ids: List[int] = None, type: str = None, limit: int = 100):
        t = "t"
        if type is not None:
            t += ":" + type
        cond = "Where Id(t) in [" + ",".join([str(i) for i in ids]) + "]"

        request_text = "MATCH (" + t + ")" + cond + " RETURN n LIMIT " + str(limit)
        return self._execute(request_text)

    def find_not_connected_pairs(self, type1: str = None, type2: str = None, limit: int = 100):
        tp1, tp2 = "tp1", "tp2"
        if type1 is not None:
            tp1 += ":" + type1
        if type2 is not None:
            tp2 += ":" + type2
        request_text = "MATCH (" + tp1 + "), (" + tp2 + """) WHERE NOT EXISTS {MATCH (tp1)-[]->(tp2)}
         and tp1.content IS NOT NULL and tp2.content IS NOT NULL and tp1 <> tp2
         RETURN tp1, tp2  LIMIT """ + str(limit)

        print(self._execute(request_text)[0]["tp1"])

        return self._execute(request_text)

    def delete(self):
        pass

    def update(self):
        pass
