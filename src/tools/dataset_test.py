import pandas as pd
import json, pickle
from collections import OrderedDict

from src.tools.load_or_create import load_or_create_smile_graph, load_or_create_kmer_data, load_or_create_target_graph
from src.tools.residue_graph import valid_target
from src.tools.utils import *


def create_dataset_for_test(dataset, model_st):
    dataset_path = os.path.join('data', dataset)
    test_fold_path = os.path.join(dataset_path, 'folds', 'test_fold_setting1.txt')
    ligands_path = os.path.join(dataset_path, 'ligands_can.txt')
    proteins_path = os.path.join(dataset_path, 'proteins.txt')
    affinity_path = os.path.join(dataset_path, 'Y')
    msa_path = os.path.join(dataset_path, 'aln')
    contact_path = os.path.join(dataset_path, 'pconsc4')

    test_fold = json.load(open(test_fold_path))
    ligands = json.load(open(ligands_path), object_pairs_hook=OrderedDict)
    proteins = json.load(open(proteins_path), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(affinity_path, 'rb'), encoding='latin1')

    drugs = [Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True) for d in ligands]
    prots = list(proteins.values())
    prot_keys = list(proteins.keys())

    if dataset == 'davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)

    valid_test_count = 0
    rows, cols = np.where(~np.isnan(affinity))
    rows, cols = rows[test_fold], cols[test_fold]
    temp_test_entries = []

    for pair_ind in range(len(rows)):
        if not valid_target(prot_keys[cols[pair_ind]], dataset):
            continue
        temp_test_entries.append([
            drugs[rows[pair_ind]],
            prots[cols[pair_ind]],
            prot_keys[cols[pair_ind]],
            affinity[rows[pair_ind], cols[pair_ind]]
        ])
        valid_test_count += 1

    csv_file = os.path.join('data', f"{dataset}_test.csv")
    data_to_csv(csv_file, temp_test_entries)
    print(f"dataset: {dataset}")
    print(f"test entries: {len(test_fold)}, effective test entries: {valid_test_count}")

    smile_graph = load_or_create_smile_graph(dataset, drugs)

    target_graph, target_sequence = load_or_create_target_graph(dataset, prot_keys, proteins, contact_path, msa_path)

    kmer_data = load_or_create_kmer_data(dataset, target_sequence)

    print(f"effective drugs: {len(smile_graph)}, effective proteins: {len(target_graph)}")
    if len(smile_graph) == 0 or len(target_graph) == 0:
        raise Exception('No protein or drug, run the script for datasets preparation.')

    df_test = pd.read_csv(csv_file)
    test_drugs, test_prot_keys, test_Y = df_test['compound_iso_smiles'], df_test['target_key'], df_test['affinity']
    test_dataset = DTADataset(
        root='data', dataset=f"{dataset}_test", xd=test_drugs.values, y=test_Y.values,
        target_key=test_prot_keys.values, smile_graph=smile_graph, target_graph=target_graph,
        target_sequence=target_sequence, kmer_data=kmer_data, model_st=model_st
    )

    return test_dataset