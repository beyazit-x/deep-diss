import dgl
import ring
import torch
import numpy as np
import networkx as nx
from dfa import DFA
from utils.parameters import FEATURE_SIZE, edge_types

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

    def __call__(self, dfa_int_seq):
        return self._to_graph(dfa_int_seq)

    @ring.lru(maxsize=1000000)
    def _to_graph(self, dfa_int_seq):
        l = dfa_int_seq.shape[0]
        dfa_goal_int_seq = dfa_int_seq.reshape(self.dfa_n_conjunctions, self.dfa_n_disjunctions, l//(self.dfa_n_conjunctions*self.dfa_n_disjunctions))
        nxg_goal = []
        nxg_goal_or_nodes = []
        rename_goal = []
        for i, dfa_clause_int_seq in enumerate(dfa_goal_int_seq):
            nxg_clause = []
            nxg_init_nodes = []
            rename_clause = []
            for j, dfa_int_seq in enumerate(dfa_clause_int_seq):
                dfa_int_str = "".join(str(int(i)) for i in dfa_int_seq.tolist())
                dfa_int = int(dfa_int_str)
                if dfa_int > 0:
                    nxg, init_node = self.dfa_int2nxg(dfa_int)
                    nxg_clause.append(nxg)
                    rename_clause.append(str(j) + "_")
                    nxg_init_nodes.append(str(j) + "_" + init_node)

            if nxg_clause != []:
                composed_nxg_clause = nx.union_all(nxg_clause, rename=rename_clause)
                or_node = "OR"
                composed_nxg_clause.add_node(or_node, feat=np.array([[0.0] * FEATURE_SIZE]))
                composed_nxg_clause.nodes[or_node]["feat"][0][feature_inds["OR"]] = 1.0
                for nxg_init_node in nxg_init_nodes:
                    composed_nxg_clause.add_edge(nxg_init_node, or_node, type=edge_types["OR"])
                nxg_goal.append(composed_nxg_clause)
                rename_goal.append(str(i) + "_")
                nxg_goal_or_nodes.append(str(i) + "_" + or_node)

        if nxg_goal != []:
            composed_nxg_goal = nx.union_all(nxg_goal, rename=rename_goal)
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

        return self._get_dgl_graph(nxg)

    def _get_dgl_graph(self, nxg):

        edges = list(nxg.edges)
        nodes = list(nxg.nodes)
        edge_types_attributes = nx.get_edge_attributes(nxg, "type")

        U, V, _type = zip(*[(nodes.index(edge[0]), nodes.index(edge[1]), edge_types_attributes[edge]) for edge in edges])
        _feat, _is_root = zip(*[(nxg.nodes[node]["feat"], nxg.nodes[node]["is_root"]) for node in nodes])

        U = torch.from_numpy(np.array(U))
        V = torch.from_numpy(np.array(V))
        _type = torch.from_numpy(np.array(_type))
        _feat = torch.from_numpy(np.array(_feat))
        _is_root = torch.from_numpy(np.array(_is_root))

        g = dgl.graph((U, V), device=self.device)
        g.ndata["feat"] = _feat
        g.ndata["is_root"] = _is_root
        g.edata["type"] = _type

        return g

    def dfa_to_formatted_nxg(self, dfa):

        nxg = nx.DiGraph()
        new_node_name_counter = 0
        new_node_name_base_str = "temp_"

        for s in dfa.states():
            start = str(s)
            nxg.add_node(start)
            nxg.nodes[start]["feat"] = np.array([[0.0] * FEATURE_SIZE])
            nxg.nodes[start]["feat"][0][feature_inds["normal"]] = 1.0
            if dfa._label(s): # is accepting?
                nxg.nodes[start]["feat"][0][feature_inds["accepting"]] = 1.0
            elif sum(s != dfa._transition(s, a) for a in dfa.inputs) == 0: # is rejecting?
                nxg.nodes[start]["feat"][0][feature_inds["rejecting"]] = 1.0
            embeddings = {}
            for a in dfa.inputs:
                e = dfa._transition(s, a)
                if s == e:
                    continue # We define self loops later when composing graphs
                end = str(e)
                if end not in embeddings.keys():
                    embeddings[end] = np.zeros(FEATURE_SIZE)
                    embeddings[end][feature_inds["temp"]] = 1.0 # Since it is a temp node
                embeddings[end][self.propositions.index(a)] = 1.0
            for end in embeddings.keys():
                new_node_name = new_node_name_base_str + str(new_node_name_counter)
                new_node_name_counter += 1
                nxg.add_node(new_node_name, feat=np.array(np.array([embeddings[end]])))
                nxg.add_edge(new_node_name, start, type=edge_types["temp-to-normal"])
                nxg.add_edge(end, new_node_name, type=edge_types["normal-to-temp"])

        init_node = str(dfa.start)
        nxg.nodes[init_node]["feat"][0][feature_inds["init"]] = 1.0

        return nxg, init_node

    @ring.lru(maxsize=1000000)
    def dfa_int2nxg(self, dfa_int):
        dfa = DFA.from_int(dfa_int, self.propositions)
        nxg, init_node = self.dfa_to_formatted_nxg(dfa)
        return nxg, init_node

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

