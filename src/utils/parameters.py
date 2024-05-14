dfa_db_path = "utils/dfa_db"
TIMEOUT_SECONDS = 600
FEATURE_SIZE = 22
GNN_EMBEDDING_SIZE = 32

# edge_types = {k:v for (v, k) in enumerate(["self", "normal-to-temp", "temp-to-normal", "AND", "OR"])}
edge_types = {k:v for (v, k) in enumerate(["self", "normal-to-temp", "temp-to-normal", "AND"])}
# feature_inds = {"rejecting": -1, "accepting": -2, "temp": -3, "normal": -4, "init": -5, "AND": -6, "OR": -7, "distance_normalized": -8}
# feature_inds = {"rejecting": -1, "accepting": -2, "temp": -3, "normal": -4, "init": -5, "AND": -6, "distance_normalized": -7}
feature_inds = {"rejecting": -1, "accepting": -2, "temp": -3, "normal": -4, "init": -5, "AND": -6}
