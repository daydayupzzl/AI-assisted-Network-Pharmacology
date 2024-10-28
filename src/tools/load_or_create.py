import pickle
from src.tools.mol_graph import smile_to_graph
from src.tools.residue_graph import target_to_graph, valid_target
from src.tools.utils import *
from tqdm import tqdm

def load_or_create_smile_graph(dataset, compound_iso_smiles):
    smile_file = os.path.join('work', 'smile_graph', f"{dataset}_smile_graph.pkl")
    if os.path.isfile(smile_file):
        with open(smile_file, 'rb') as f:
            return pickle.load(f)
    else:
        smile_graph = {}
        with tqdm(total=len(compound_iso_smiles), desc="Processing drug graph") as pbar:
            for smile in compound_iso_smiles:
                smile_graph[smile] = smile_to_graph(smile)
                pbar.update(1)
        with open(smile_file, 'wb') as f:
            pickle.dump(smile_graph, f)
        return smile_graph

def load_or_create_target_graph(dataset, prot_keys, proteins, contact_path, msa_path):
    target_graph_file = os.path.join('work', 'target_graph', f"{dataset}_target_graph.pkl")
    if os.path.isfile(target_graph_file):
        with open(target_graph_file, 'rb') as f:
            target_graph = pickle.load(f)
        target_sequence = {key: proteins[key] for key in prot_keys}
    else:
        target_graph = {}
        target_sequence = {}
        with tqdm(total=len(prot_keys), desc="Processing protein") as pbar:
            for key in prot_keys:
                if not valid_target(key, dataset):
                    continue
                target_graph[key] = target_to_graph(key, proteins[key], contact_path, msa_path)
                target_sequence[key] = proteins[key]
                pbar.update(1)
        with open(target_graph_file, 'wb') as f:
            pickle.dump(target_graph, f)
    return target_graph, target_sequence

def load_or_create_kmer_data(dataset, target_sequence):
    kmer_data_file = os.path.join('work', f"preprocessed_kmer_data_{dataset}.pkl")
    if os.path.isfile(kmer_data_file):
        print("Loading preprocessed K-mer data...")
        with open(kmer_data_file, 'rb') as f:
            return pickle.load(f)
    else:
        kmer_data = {}
        for tar_key in tqdm(target_sequence.keys(), desc="Preprocessing K-mer Data", unit="sequence"):
            kmer_data[tar_key] = getmarkov(target_sequence[tar_key])
        with open(kmer_data_file, 'wb') as f:
            pickle.dump(kmer_data, f)
        return kmer_data
