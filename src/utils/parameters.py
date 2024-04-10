dfa_db_path = "utils/dfa_db"
TIMEOUT_SECONDS = 600
FEATURE_SIZE = 22
GNN_EMBEDDING_SIZE = 32

edge_types = {k:v for (v, k) in enumerate(["self", "normal-to-temp", "temp-to-normal", "AND", "OR"])}