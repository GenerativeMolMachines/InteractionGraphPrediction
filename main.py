from graphs.db_graph import DBGraph
from utils.db_driver import DBDriver


if __name__ == "__main__":
    driver = DBDriver("neo4j://77.234.216.102:7687", "akrukov", "fuel-carlo-michael-regular-betty-3025")
    graph = DBGraph.graph_from_records(driver.find_not_connected_pairs())
    print(graph)
    print(graph.edges)
    print(graph.nodes)
