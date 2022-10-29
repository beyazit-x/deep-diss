from gnns.graph_registry import *
from gnns.graphs.GNN import *

register(id="GCN_2x32_MEAN", entry_point="gnns.graphs.GCN.GCN", hidden_dims=[32, 32])

register(id="GCN_4x32_MEAN", entry_point="gnns.graphs.GCN.GCN", hidden_dims=[32, 32, 32, 32])

register(id="GCN_32_MEAN", entry_point="gnns.graphs.GCN.GCN", hidden_dims=[32])

register(id="GCN_32_ROOT", entry_point="gnns.graphs.GCN.GCNRoot", hidden_dims=[32])

register(id="GCN_2x32_ROOT", entry_point="gnns.graphs.GCN.GCNRoot", hidden_dims=[32, 32])

register(id="GCN_4x32_ROOT", entry_point="gnns.graphs.GCN.GCNRoot", hidden_dims=[32, 32, 32, 32])

register(id="GCN_8x32_ROOT", entry_point="gnns.graphs.GCN.GCNRoot", hidden_dims=[32] * 8)

register(id="GCN_16x32_ROOT", entry_point="gnns.graphs.GCN.GCNRoot", hidden_dims=[32] * 16)

register(id="GCN_32x32_ROOT", entry_point="gnns.graphs.GCN.GCNRoot", hidden_dims=[32] * 32)

register(id="GCN_64x32_ROOT", entry_point="gnns.graphs.GCN.GCNRoot", hidden_dims=[32] * 64)

register(id="GCN_4x64_ROOT", entry_point="gnns.graphs.GCN.GCNRoot", hidden_dims=[64] * 4)

register(id="GCN_8x64_ROOT", entry_point="gnns.graphs.GCN.GCNRoot", hidden_dims=[64] * 8)

register(id="GCN_2x32_ROOT_SHARED", entry_point="gnns.graphs.GCN.GCNRootShared", hidden_dim=32, num_layers=2)

register(id="GCN_4x32_ROOT_SHARED", entry_point="gnns.graphs.GCN.GCNRootShared", hidden_dim=32, num_layers=4)

register(id="GCN_8x32_ROOT_SHARED", entry_point="gnns.graphs.GCN.GCNRootShared", hidden_dim=32, num_layers=8)

register(id="GCN_16x32_ROOT_SHARED", entry_point="gnns.graphs.GCN.GCNRootShared", hidden_dim=32, num_layers=16)

register(id="GCN_32x32_ROOT_SHARED", entry_point="gnns.graphs.GCN.GCNRootShared", hidden_dim=32, num_layers=32)

register(id="GCN_64x32_ROOT_SHARED", entry_point="gnns.graphs.GCN.GCNRootShared", hidden_dim=32, num_layers=64)

register(id="GCN_4x64_ROOT_SHARED", entry_point="gnns.graphs.GCN.GCNRootShared", hidden_dim=64, num_layers=4)

register(id="GCN_8x64_ROOT_SHARED", entry_point="gnns.graphs.GCN.GCNRootShared", hidden_dim=64, num_layers=8)

register(id="RGCN_2x32_ROOT", entry_point="gnns.graphs.RGCN.RGCNRoot", hidden_dims=[32, 32])

register(id="RGCN_4x16_ROOT", entry_point="gnns.graphs.RGCN.RGCNRoot", hidden_dims=[16] * 4)

register(id="RGCN_4x32_ROOT", entry_point="gnns.graphs.RGCN.RGCNRoot", hidden_dims=[32] * 4)

register(id="RGCN_8x32_ROOT", entry_point="gnns.graphs.RGCN.RGCNRoot", hidden_dims=[32] * 8)

register(id="RGCN_8x8_ROOT", entry_point="gnns.graphs.RGCN.RGCNRoot", hidden_dims=[8] * 8)

register(id="RGCN_4x32_ROOT_SHARED", entry_point="gnns.graphs.RGCN.RGCNRootShared", hidden_dim=32, num_layers=4)

register(id="RGCN_8x32_ROOT_SHARED", entry_point="gnns.graphs.RGCN.RGCNRootShared", hidden_dim=32, num_layers=8)

register(id="RGCN_16x32_ROOT_SHARED", entry_point="gnns.graphs.RGCN.RGCNRootShared", hidden_dim=32, num_layers=16)

register(id="RGCN_16x16_ROOT_SHARED", entry_point="gnns.graphs.RGCN.RGCNRootShared", hidden_dim=16, num_layers=16)

register(id="RGCN_32x32_ROOT_SHARED", entry_point="gnns.graphs.RGCN.RGCNRootShared", hidden_dim=32, num_layers=32)

register(id="RGCN_64x32_ROOT_SHARED", entry_point="gnns.graphs.RGCN.RGCNRootShared", hidden_dim=32, num_layers=64)
