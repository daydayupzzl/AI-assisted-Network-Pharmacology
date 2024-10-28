import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool as gep
import yaml

class BaseGraphkmerDTA(torch.nn.Module):
    def __init__(self, config):
        super(BaseGraphkmerDTA, self).__init__()

        # Load hyperparameters from config
        n_output = config.get('n_output', 1)
        num_features_pro = config.get('num_features_pro', 54)
        num_features_mol = config.get('num_features_mol', 78)
        km_feature = config.get('km_feature', 8420)
        output_dim = config.get('output_dim', 128)
        dropout = config.get('dropout', 0.2)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.mol_convs = self.create_gcn_layers(num_features_mol, [num_features_mol, num_features_mol * 2, num_features_mol * 4])
        self.mol_fc_layers = self.create_fc_layers([num_features_mol * 4, 1024, output_dim])

        self.pro_convs = self.create_gcn_layers(num_features_pro, [num_features_pro, num_features_pro * 2, num_features_pro * 4])
        self.pro_fc_layers = self.create_fc_layers([num_features_pro * 4, 1024, output_dim])

        self.km_fc_layers = self.create_fc_layers([km_feature, 1024, 512, output_dim])

        self.combined_fc_layers = self.create_fc_layers([3 * output_dim, 1024, 512, n_output], apply_dropout=True)

    def create_gcn_layers(self, input_dim, hidden_dims):
        return nn.ModuleList([GCNConv(input_dim if i == 0 else hidden_dims[i-1], hidden_dims[i]) for i in range(len(hidden_dims))])

    def create_fc_layers(self, layer_dims, apply_dropout=False):
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            layers.append(self.relu)
            if apply_dropout:
                layers.append(self.dropout)
        return nn.Sequential(*layers)

    def process_graph(self, x, edge_index, conv_layers, fc_layers, batch):
        for conv in conv_layers:
            x = self.relu(conv(x, edge_index))
        x = gep(x, batch)
        x = fc_layers(x)
        return x

    def process_kmer(self, km):
        return self.km_fc_layers(km)

    def forward(self, data_mol, data_pro):
        mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch
        target_x, target_edge_index, target_batch = data_pro.x, data_pro.edge_index, data_pro.batch
        km = data_pro.kmer

        x = self.process_graph(mol_x, mol_edge_index, self.mol_convs, self.mol_fc_layers, mol_batch)

        xk = self.process_kmer(km)

        xt = self.process_graph(target_x, target_edge_index, self.pro_convs, self.pro_fc_layers, target_batch)

        xc = torch.cat((x, xt, xk), 1)
        out = self.combined_fc_layers(xc)
        return out

class GraphkmerDTA(BaseGraphkmerDTA):
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        super(GraphkmerDTA, self).__init__(config)

class GraphkmerDTA_DPI(BaseGraphkmerDTA):
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        super(GraphkmerDTA_DPI, self).__init__(config)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data_mol, data_pro):
        out = super(GraphkmerDTA_DPI, self).forward(data_mol, data_pro)
        return self.sigmoid(out)
