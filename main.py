import os
from graphs.db_graph import DBGraph
from utils.db_driver import DBDriver

DB_URL = os.environ["DB_URL"]
DB_USER = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]


if __name__ == "__main__":
    driver = DBDriver(
        uri=DB_URL,
        username=DB_USER,
        password=DB_PASSWORD,
    )
    graph = DBGraph.graph_from_records(driver.find_not_connected_pairs())
    print(graph)
    print(graph.edges)
    print(graph.nodes)
