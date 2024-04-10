import dgl
import ring
import numpy as np
import networkx as nx
from dfa import DFA, dfa2dict
from utils.parameters import FEATURE_SIZE, edge_types
from copy import deepcopy
from pysat.solvers import Solver

DGL5_COMPAT = True

feature_inds = {"rejecting": -1, "accepting": -2, "temp": -3, "normal": -4, "init": -5, "AND": -6, "OR": -7}

class DFABuilder(object):
    def __init__(self, propositions, dfa_n_conjunctions, dfa_n_disjunctions, device=None):
        super(DFABuilder, self).__init__()
        self.propositions = propositions
        self.device = device
        self.dfa_n_conjunctions = dfa_n_conjunctions
        self.dfa_n_disjunctions = dfa_n_disjunctions

    # To make the caching work.
    def __ring_key__(self):
        return "DFABuilder"

    def __call__(self, dfa_binary_seq, library="dgl"):
        return self._to_graph(dfa_binary_seq, library)

    @ring.lru(maxsize=1000000)
    def _to_graph(self, dfa_binary_seq, library="dgl"):
        l = dfa_binary_seq.shape[0]
        dfa_goal_bin = dfa_binary_seq.reshape(self.dfa_n_conjunctions, self.dfa_n_disjunctions, l//(self.dfa_n_conjunctions*self.dfa_n_disjunctions))
        nxg_goal = []
        nxg_goal_or_nodes = []
        for i, dfa_clause_bin in enumerate(dfa_goal_bin):
            nxg_clause = []
            nxg_init_nodes = []
            for j, dfa_bin in enumerate(dfa_clause_bin):
                dfa_binary_str = "".join(str(int(i)) for i in dfa_bin.tolist())
                dfa_int = int(dfa_binary_str, 2)
                if dfa_int > 0:
                    dfa = DFA.from_int(dfa_int, self.propositions)
                    nxg, init_node = self.dfa2nxg(dfa)
                    nxg = nxg.reverse(copy=True)
                    nxg = nx.relabel_nodes(nxg, lambda x: str(i) + "_" + str(j) + "_" + x, copy=True)
                    nxg_clause.append(nxg)
                    nxg_init_nodes.append(str(i) + "_" + str(j) + "_" + init_node)

            if nxg_clause != []:
                composed_nxg_clause = nx.compose_all(nxg_clause)
            else:
                composed_nxg_clause = nx.DiGraph()

            or_node = str(i) + "_OR"
            composed_nxg_clause.add_node(or_node, feat=np.array([[0.0] * FEATURE_SIZE]))
            composed_nxg_clause.nodes[or_node]["feat"][0][feature_inds["OR"]] = 1.0
            for nxg_init_node in nxg_init_nodes:
                composed_nxg_clause.add_edge(nxg_init_node, or_node, type=edge_types["OR"])
            nxg_goal.append(composed_nxg_clause)
            nxg_goal_or_nodes.append(or_node)

        if nxg_goal != []:
            composed_nxg_goal = nx.compose_all(nxg_goal)
        else:
            composed_nxg_goal = nx.DiGraph()

        and_node = "AND"
        composed_nxg_goal.add_node(and_node, feat=np.array([[0.0] * FEATURE_SIZE]))
        nx.set_node_attributes(composed_nxg_goal, np.array([0.0]), "is_root")
        composed_nxg_goal.nodes[and_node]["is_root"] = np.array([1.0])
        composed_nxg_goal.nodes[and_node]["feat"][0][feature_inds["AND"]] = 1.0

        for or_node in nxg_goal_or_nodes:
            composed_nxg_goal.add_edge(or_node, and_node, type=edge_types["AND"])

        for node in composed_nxg_goal.nodes:
            composed_nxg_goal.add_edge(node, node, type=edge_types["self"])
        
        nxg = composed_nxg_goal

        if (library == "networkx"):
            return nxg

        # convert the Networkx graph to dgl graph and pass the 'feat' attribute
        if DGL5_COMPAT:
            # print(nx.get_node_attributes(nxg, 'feat'))
            g = dgl.from_networkx(nxg, node_attrs=np.array(["feat", "is_root"]), edge_attrs=np.array(["type"]), device=self.device)
        else:
            g = dgl.DGLGraph()
            g.from_networkx(nxg, node_attrs=np.array(["feat", "is_root"]), edge_attrs=np.array(["type"]), device=self.device)
        return g

    def _get_guard_embeddings(self, guard):
        embeddings = []
        try:
            guard = guard.replace(" ", "").replace("(", "").replace(")", "").replace("\"", "")
        except:
            return embeddings
        if (guard == "true"):
            return embeddings
        guard = guard.split("&")
        cnf = []
        seen_atoms = []
        for c in guard:
            atoms = c.split("|")
            clause = []
            for atom in atoms:
                try:
                    index = seen_atoms.index(atom if atom[0] != "~" else atom[1:])
                except:
                    index = len(seen_atoms)
                    seen_atoms.append(atom if atom[0] != "~" else atom[1:])
                clause.append(index + 1 if atom[0] != "~" else -(index + 1))
            cnf.append(clause)
        models = []
        with Solver(bootstrap_with=cnf) as s:
            models = list(s.enum_models())
        if len(models) == 0:
            return embeddings
        for model in models:
            temp = [0.0] * FEATURE_SIZE
            for a in model:
                if a > 0:
                    atom = seen_atoms[abs(a) - 1]
                    temp[self.propositions.index(atom)] = 1.0
            embeddings.append(temp)
        return embeddings

    def _get_onehot_guard_embeddings(self, guard):
        is_there_onehot = False
        is_there_all_zero = False
        onehot_embedding = [0.0] * FEATURE_SIZE
        onehot_embedding[feature_inds["temp"]] = 1.0 # Since it will be a temp node
        full_embeddings = self._get_guard_embeddings(guard)
        for embed in full_embeddings:
            # discard all non-onehot embeddings (a one-hot embedding must contain only a single 1)
            if embed.count(1.0) == 1:
                # clean the embedding so that it's one-hot
                is_there_onehot = True
                var_idx = embed.index(1.0)
                onehot_embedding[var_idx] = 1.0
            elif embed.count(0.0) == len(embed):
                is_there_all_zero = True
        if is_there_onehot or is_there_all_zero:
            return [onehot_embedding]
        else:
            return []

    def _is_sink_state(self, node, nxg):
        for edge in nxg.edges:
            if node == edge[0] and node != edge[1]: # If there is an outgoing edge to another node, then it is not an accepting state
                return False
        return True

    def dfa_dict2nxg(self, dfa_dict, init_node, minimize=False):

        init_node = str(init_node)

        nxg = nx.DiGraph()

        accepting_states = []
        for start, (accepting, transitions) in dfa_dict.items():
            start = str(start)
            nxg.add_node(start)
            if accepting:
                accepting_states.append(start)
            for action, end in transitions.items():
                if nxg.has_edge(start, str(end)):
                    existing_label = nxg.get_edge_data(start, str(end))['label']
                    nxg.add_edge(start, str(end), label='{} | {}'.format(existing_label, action))
                else:
                    nxg.add_edge(start, str(end), label=action)

        # nxg = nx.ego_graph(nxg, init_node, radius=5)
        accepting_states = list(set(accepting_states).intersection(set(nxg.nodes)))

        return init_node, accepting_states, nxg


    def _format(self, init_node, accepting_states, nxg):
        rejecting_states = []
        for node in nxg.nodes:
            if self._is_sink_state(node, nxg) and node not in accepting_states:
                rejecting_states.append(node)

        for node in nxg.nodes:
            nxg.nodes[node]["feat"] = np.array([[0.0] * FEATURE_SIZE])
            nxg.nodes[node]["feat"][0][feature_inds["normal"]] = 1.0
            if node in accepting_states:
                nxg.nodes[node]["feat"][0][feature_inds["accepting"]] = 1.0
            if node in rejecting_states:
                nxg.nodes[node]["feat"][0][feature_inds["rejecting"]] = 1.0

        nxg.nodes[init_node]["feat"][0][feature_inds["init"]] = 1.0

        edges = deepcopy(nxg.edges)

        new_node_name_base_str = "temp_"
        new_node_name_counter = 0

        for e in edges:
            guard = nxg.edges[e]["label"]
            nxg.remove_edge(*e)
            if e[0] == e[1]:
                continue # We define self loops below
            onehot_embedding = self._get_onehot_guard_embeddings(guard) # It is ok if we receive a cached embeddding since we do not modify it
            if len(onehot_embedding) == 0:
                continue
            new_node_name = new_node_name_base_str + str(new_node_name_counter)
            new_node_name_counter += 1
            nxg.add_node(new_node_name, feat=np.array(onehot_embedding))
            nxg.add_edge(e[0], new_node_name, type=edge_types["normal-to-temp"])
            nxg.add_edge(new_node_name, e[1], type=edge_types["temp-to-normal"])

        return nxg, init_node

    def dfa2nxg(self, dfa):
        dfa_dict, init_state = dfa2dict(dfa)
        init_node, accepting_states, nxg = self.dfa_dict2nxg(dfa_dict, init_state)
        dfa_nxg, init_node = self._format(init_node, accepting_states, nxg)
        return dfa_nxg, init_node

def draw(G, formula):
    from networkx.drawing.nx_agraph import graphviz_layout
    import matplotlib.pyplot as plt

    # colors = ["black", "red"]
    # edge_color = [colors[i] for i in nx.get_edge_attributes(G,'type').values()]

    plt.title(formula)
    pos=graphviz_layout(G, prog='dot')
    # labels = nx.get_node_attributes(G,'token')
    labels = G.nodes
    nx.draw(G, pos, with_labels=True, arrows=True, node_shape='s', edgelist=list(nx.get_edge_attributes(G,'type')), node_size=500, node_color="white") #edge_color=edge_color
    plt.show()