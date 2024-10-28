import pandas as pd
import json, pickle
from collections import OrderedDict

from src.tools.load_or_create import load_or_create_smile_graph, load_or_create_target_graph, load_or_create_kmer_data
from src.tools.residue_graph import valid_target
from src.tools.utils import *


def create_dataset_for_5folds(dataset, fold=0, model_st=None):
    dataset_path = os.path.join('data', dataset)
    train_fold_path = os.path.join(dataset_path, 'folds', 'train_fold_setting1.txt')
    ligands_path = os.path.join(dataset_path, 'ligands_can.txt')
    proteins_path = os.path.join(dataset_path, 'proteins.txt')
    affinity_path = os.path.join(dataset_path, 'Y')
    msa_path = os.path.join(dataset_path, 'aln')
    contact_path = os.path.join(dataset_path, 'pconsc4')

    train_fold_origin = json.load(open(train_fold_path))
    ligands = json.load(open(ligands_path), object_pairs_hook=OrderedDict)
    proteins = json.load(open(proteins_path), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(affinity_path, 'rb'), encoding='latin1')

    valid_fold = train_fold_origin[fold]
    train_folds = [item for i, sublist in enumerate(train_fold_origin) if i != fold for item in sublist]

    drugs = [Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True) for d in ligands]
    prots = list(proteins.values())
    prot_keys = list(proteins.keys())

    if dataset == 'davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)

    valid_train_count = 0
    valid_valid_count = 0
    entries = {'train': [], 'valid': []}

    rows, cols = np.where(~np.isnan(affinity))
    for opt, fold in zip(['train', 'valid'], [train_folds, valid_fold]):
        filtered_rows, filtered_cols = rows[fold], cols[fold]
        for pair_ind in range(len(filtered_rows)):
            if not valid_target(prot_keys[filtered_cols[pair_ind]], dataset):
                continue
            ls = [
                drugs[filtered_rows[pair_ind]],
                prots[filtered_cols[pair_ind]],
                prot_keys[filtered_cols[pair_ind]],
                affinity[filtered_rows[pair_ind], filtered_cols[pair_ind]]
            ]
            entries[opt].append(ls)
            if opt == 'train':
                valid_train_count += 1
            else:
                valid_valid_count += 1
        csv_file = os.path.join('data', f"{dataset}_fold_{fold}_{opt}.csv")
        data_to_csv(csv_file, entries[opt])

    print(f"dataset: {dataset}")
    print(f"fold: {fold}")
    print(f"train entries: {len(train_folds)}, effective train entries: {valid_train_count}")
    print(f"valid entries: {len(valid_fold)}, effective valid entries: {valid_valid_count}")

    smile_graph = load_or_create_smile_graph(dataset, drugs)

    target_graph, target_sequence = load_or_create_target_graph(dataset, prot_keys, proteins, contact_path, msa_path)

    kmer_data = load_or_create_kmer_data(dataset, target_sequence)

    print(f"effective drugs: {len(smile_graph)}, effective proteins: {len(target_graph)}")
    if len(smile_graph) == 0 or len(target_graph) == 0:
        raise Exception('No protein or drug, run the script for datasets preparation.')

    df_train = pd.read_csv(os.path.join('data', f"{dataset}_fold_{fold}_train.csv"))
    train_drugs, train_prot_keys, train_Y = df_train['compound_iso_smiles'], df_train['target_key'], df_train['affinity']
    train_dataset = DTADataset(
        root='data', dataset=f"{dataset}_train", xd=train_drugs.values, y=train_Y.values,
        target_key=train_prot_keys.values, smile_graph=smile_graph, target_graph=target_graph,
        target_sequence=target_sequence, kmer_data=kmer_data, model_st=model_st
    )

    df_valid = pd.read_csv(os.path.join('data', f"{dataset}_fold_{fold}_valid.csv"))
    valid_drugs, valid_prot_keys, valid_Y = df_valid['compound_iso_smiles'], df_valid['target_key'], df_valid['affinity']
    valid_dataset = DTADataset(
        root='data', dataset=f"{dataset}_valid", xd=valid_drugs.values, y=valid_Y.values,
        target_key=valid_prot_keys.values, smile_graph=smile_graph, target_graph=target_graph,
        target_sequence=target_sequence, kmer_data=kmer_data, model_st=model_st
    )

    return train_dataset, valid_dataset