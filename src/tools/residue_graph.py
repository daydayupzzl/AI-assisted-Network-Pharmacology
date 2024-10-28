from src.tools.utils import *

from src.tools.normalize import *
from src.tools.residue_feature import *
def valid_target(key, dataset):
    contact_dir = 'data/' + dataset + '/pconsc4'
    aln_dir = 'data/' + dataset + '/aln'
    contact_file = os.path.join(contact_dir, key + '.npy')
    aln_file = os.path.join(aln_dir, key + '.aln')
    if os.path.exists(contact_file) and os.path.exists(aln_file):
        return True
    else:
        return False

def PSSM_calculation(aln_file, pro_seq):
    pfm_mat = np.zeros((len(pro_res_table), len(pro_seq)))
    with open(aln_file, 'r') as f:
        line_count = len(f.readlines())
        for line in f.readlines():
            if len(line) != len(pro_seq):
                continue
            count = 0
            for res in line:
                if res not in pro_res_table:
                    count += 1
                    continue
                pfm_mat[pro_res_table.index(res), count] += 1
                count += 1
    pseudocount = 0.8
    ppm_mat = (pfm_mat + pseudocount / 4) / (float(line_count) + pseudocount)
    pssm_mat = ppm_mat
    return pssm_mat

def seq_feature(pro_seq):
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        pro_hot[i,] = one_of_k_encoding(pro_seq[i], pro_res_table)
        pro_property[i,] = residue_features(pro_seq[i])
    return np.concatenate((pro_hot, pro_property), axis=1)

def target_feature(aln_file, pro_seq):
    pssm = PSSM_calculation(aln_file, pro_seq)
    other_feature = seq_feature(pro_seq)
    return np.concatenate((np.transpose(pssm, (1, 0)), other_feature), axis=1)

def target_to_feature(target_key, target_sequence, aln_dir):
    aln_file = os.path.join(aln_dir, target_key + '.aln')
    feature = target_feature(aln_file, target_sequence)
    return feature

def target_to_graph(target_key, target_sequence, contact_dir, aln_dir):
    target_edge_index = []
    target_edge_distance = []
    target_size = len(target_sequence)
    contact_file = os.path.join(contact_dir, target_key + '.npy')
    contact_map = np.load(contact_file)
    contact_map += np.matrix(np.eye(contact_map.shape[0]))
    index_row, index_col = np.where(contact_map >= 0.5)
    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])
        target_edge_distance.append(contact_map[i, j])
    target_feature = target_to_feature(target_key, target_sequence, aln_dir)
    target_edge_index = np.array(target_edge_index)
    return target_size, target_feature, target_edge_index, np.array(target_edge_distance)