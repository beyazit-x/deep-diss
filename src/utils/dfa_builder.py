import dgl
import numpy as np
import networkx as nx

edge_types = ["none", "self", "normal-to-temp", "temp-to-normal"] # We had to add "none" here because of the way typed_adj_matrix is constructed. Does that break things?
DGL5_COMPAT = True

"""
A class that can take an LTL formula and generate a minimal DFA of it. This
code can generate graphs in either Networkx or DGL formats. And uses caching to remember recently
generated graphs.
"""
class DFABuilder(object):
    def __init__(self, propositions, use_mean_guard_embed, use_onehot_guard_embed, device):
        super(DFABuilder, self).__init__()
        self.props = propositions
        self.use_mean_guard_embed = use_mean_guard_embed
        self.use_onehot_guard_embed = use_onehot_guard_embed
        self.device = device

    def __call__(self, nxg, library="dgl"):
        """print(formula)
        print("Number of nodes:", len(nxg.nodes))
        print("Number of edges:", len(nxg.edges))
        for i in nxg.nodes:
            print("Node:", i)
            print(nxg.nodes[i]["feat"].shape, type(nxg.nodes[i]["feat"]), nxg.nodes[i]["feat"], type(nxg.nodes[i]["feat"][0][0]))
            print(nxg.nodes[i]["is_root"], type(nxg.nodes[i]["is_root"]))
        for i in nxg.edges:
            print(nxg.edges[i]["type"], type(nxg.edges[i]["type"]))"""

        nxg = nxg.reverse(copy=True)
        if (library == "networkx"):
            return nxg

        # convert the Networkx graph to dgl graph and pass the 'feat' attribute
        if DGL5_COMPAT:
            # print(nx.get_node_attributes(nxg, 'feat'))
            g = dgl.from_networkx(nxg, node_attrs=np.array(["feat", "is_root"]), edge_attrs=np.array(["type"]), device=self.device) # dgl does not support string attributes (i.e., token)
        else:
            g = dgl.DGLGraph()
            g.from_networkx(nxg, node_attrs=np.array(["feat", "is_root"]), edge_attrs=np.array(["type"]), device=self.device) # dgl does not support string attributes (i.e., token)
        return g
