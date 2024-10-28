import os
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint
from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric import data as DATA
import torch

def data_to_csv(csv_file, datalist):
    with open(csv_file, 'w') as f:
        f.write('compound_iso_smiles,target_sequence,target_key,affinity\n')
        for data in datalist:
            f.write(','.join(map(str, data)) + '\n')
class DTADataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis', xd=None, y=None, transform=None,
                 pre_transform=None, smile_graph=None, target_key=None, target_graph=None, target_sequence=None,
                 kmer_data=None, model_st=None):
        super(DTADataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.kmer_data = kmer_data
        self.process(xd, target_key, y, smile_graph, target_graph, target_sequence, model_st)

    @property
    def processed_file_names(self):
        return [f'{self.dataset}_data_mol.pt', f'{self.dataset}_data_pro.pt']

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, target_key, y, smile_graph, target_graph, target_sequence, model_st):
        data_list_mol, data_list_pro = [], []
        for smiles, tar_key, labels in tqdm(zip(xd, target_key, y), desc="Processing Data", unit="sample", total=len(xd)):
            c_size, features, edge_index = smile_graph[smiles]
            target_size, target_features, target_edge_index, target_edge_weight = target_graph[tar_key]

            GCNData_mol = DATA.Data(x=torch.Tensor(features),
                                    edge_index=torch.LongTensor(edge_index).t(),
                                    y=torch.FloatTensor([labels]))
            GCNData_mol.c_size = torch.LongTensor([c_size])

            GCNData_pro = DATA.Data(x=torch.Tensor(target_features),
                                    edge_index=torch.LongTensor(target_edge_index).t(),
                                    edge_weight=torch.FloatTensor(target_edge_weight),
                                    y=torch.FloatTensor([labels]))
            GCNData_pro.target_size = torch.LongTensor([target_size])

            if model_st in ['GraphDTA', 'GraphDPI']:
                GCNData_pro.sequence = torch.LongTensor(np.array([seq_cat(target_sequence[tar_key]).astype(int)]))

            if self.kmer_data is not None and tar_key in self.kmer_data:
                GCNData_pro.kmer = torch.FloatTensor(np.array([self.kmer_data[tar_key]]))
            else:
                GCNData_pro.kmer = torch.FloatTensor(np.array([getmarkov(target_sequence[tar_key])]))

            data_list_mol.append(GCNData_mol)
            data_list_pro.append(GCNData_pro)

        if self.pre_filter is not None:
            data_list_mol = [data for data in data_list_mol if self.pre_filter(data)]
            data_list_pro = [data for data in data_list_pro if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list_mol = [self.pre_transform(data) for data in data_list_mol]
            data_list_pro = [self.pre_transform(data) for data in data_list_pro]

        self.data_mol, self.data_pro = data_list_mol, data_list_pro

    def __len__(self):
        return len(self.data_mol)

    def __getitem__(self, idx):
        return self.data_mol[idx], self.data_pro[idx]

def train(model, device, train_loader, optimizer, epoch):
    print(f'Training on {len(train_loader.dataset)} samples...')
    model.train()
    loss_fn = torch.nn.MSELoss()
    for batch_idx, data in enumerate(train_loader):
        data_mol, data_pro = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output = model(data_mol, data_pro)
        loss = loss_fn(output, data_mol.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train epoch: {epoch} [{batch_idx * len(data_mol)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]	Loss: {loss.item():.6f}')

def predicting(model, device, loader):
    model.eval()
    total_preds, total_labels = torch.Tensor(), torch.Tensor()
    print(f'Make prediction for {len(loader.dataset)} samples...')
    with torch.no_grad():
        for data in loader:
            data_mol, data_pro = data[0].to(device), data[1].to(device)
            output = model(data_mol, data_pro)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    return batchA, batchB

def getmarkov(input):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    amino_acid_indices = {aa: i for i, aa in enumerate(amino_acids)}
    data0 = np.bincount([amino_acid_indices[ch] for ch in input if ch in amino_acid_indices], minlength=20)
    data0 = (data0 - np.mean(data0)) / np.std(data0)

    data1 = np.zeros((20, 20))
    for i in range(len(input) - 1):
        if input[i] in amino_acid_indices and input[i + 1] in amino_acid_indices:
            data1[amino_acid_indices[input[i]], amino_acid_indices[input[i + 1]]] += 1
    data1 = (data1 - np.mean(data1)) / np.std(data1)

    data2 = np.zeros((20, 20, 20))
    for i in range(len(input) - 2):
        if input[i] in amino_acid_indices and input[i + 1] in amino_acid_indices and input[i + 2] in amino_acid_indices:
            data2[amino_acid_indices[input[i]], amino_acid_indices[input[i + 1]], amino_acid_indices[input[i + 2]]] += 1
    data2 = (data2 - np.mean(data2)) / np.std(data2)

    return np.concatenate((data0, data1.flatten(), data2.flatten()))

def smiles2erg(s):
    try:
        mol = Chem.MolFromSmiles(s)
        return np.array(GetErGFingerprint(mol))
    except:
        return np.zeros((315,))

def seq_cat(prot):
    seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
    max_seq_len = 1000
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict.get(ch, 0)
    return x