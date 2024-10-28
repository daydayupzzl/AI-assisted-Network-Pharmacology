import pandas as pd

from src.tools.load_or_create import load_or_create_smile_graph, load_or_create_target_graph
from src.tools.residue_graph import valid_target
from src.tools.utils import *


def create_dataset_for_repurpose(X_repurpose, target, dataset, target_name, model_st, version):

    process_dir = os.path.join('./data/')
    pro_distance_dir = os.path.join(process_dir, dataset, 'pconsc4')
    aln_dir = os.path.join(process_dir, dataset, 'aln')

    drugs = [Chem.MolToSmiles(Chem.MolFromSmiles(d), isomericSmiles=True) for d in X_repurpose]
    drug_smiles = X_repurpose
    prots = [target for _ in X_repurpose]
    prot_keys = [f"{target_name}_{i}" for i in range(len(X_repurpose))]

    valid_test_count = 0
    temp_test_entries = []
    for pair_ind in range(len(X_repurpose)):
        if not valid_target(target_name, dataset):
            continue
        ls = [
            drugs[pair_ind],
            prots[pair_ind],
            prot_keys[pair_ind],
            1
        ]
        temp_test_entries.append(ls)
        valid_test_count += 1

    csv_file = os.path.join('data', dataset, f"{version}_test.csv")
    data_to_csv(csv_file, temp_test_entries)
    print(f"test entries: {len(X_repurpose)}, effective test entries: {valid_test_count}")

    smile_file = os.path.join('work', 'smile_graph', f"{dataset}_smile_graph.pkl")
    smile_graph = load_or_create_smile_graph(dataset, drugs, smile_file)

    target_graph, target_sequence = load_or_create_target_graph(prot_keys, target, target_name, dataset, pro_distance_dir, aln_dir)

    print(f"effective drugs: {len(smile_graph)}, effective proteins: {len(target_graph)}")
    if len(smile_graph) == 0 or len(target_graph) == 0:
        raise Exception('No protein or drug, run the script for datasets preparation.')

    df_test = pd.read_csv(csv_file)
    test_drugs, test_prot_keys, test_prot_sequence, test_Y = (
        df_test['compound_iso_smiles'], df_test['target_key'], df_test['target_sequence'], df_test['affinity']
    )
    test_drugs, test_prot_keys, test_prot_sequence, test_Y = (
        np.asarray(test_drugs), np.asarray(test_prot_keys), np.asarray(test_prot_sequence), np.asarray(test_Y)
    )
    test_dataset = DTADataset(
        root='data', dataset=f"{dataset}_test", xd=test_drugs, y=test_Y,
        target_key=test_prot_keys, smile_graph=smile_graph, target_graph=target_graph,
        target_sequence=target_sequence, model_st=model_st
    )

    return test_dataset