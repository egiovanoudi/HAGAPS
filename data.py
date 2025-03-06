import numpy as np
import pandas as pd
import os
from itertools import groupby
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch


def encode_sequence(seq):
    mapping = {'A': 1, 'T': 2, 'U': 2, 'C': 3, 'G': 4}
    return [mapping[nucleotide] for nucleotide in seq]

def phosphodiester_bond(sequence):
    # Create edges based on adjacency
    source = list(range(len(sequence) - 1))
    target = list(range(1, len(sequence)))
    edge_weights = np.ones(len(sequence) - 1)
    edges = [source, target]
    return edges, edge_weights

def hydrogen_bond(structure_file):
    sources = []
    targets = []
    edge_weights = []

    with open(structure_file, 'r') as file:
        for line in file:
            source, target, prob = line.split()
            sources.append(int(source)-1)       # 0-based indexing
            targets.append(int(target)-1)
            edge_weights.append(float(prob))
    edges = [sources, targets]
    return edges, edge_weights

def process_structure_folder(struct_path):
    base_pairing_data = []

    # Iterate through all files in the folder
    for struct_file in os.listdir(struct_path):
        file_path = os.path.join(struct_path, struct_file)
        edges, edge_weights = hydrogen_bond(file_path)
        base_pairing_data.append({'pas': struct_file, 'hydrogen_edges': edges, 'hydrogen_edge_weights': edge_weights})
    return pd.DataFrame(base_pairing_data)

def prepare_data(dataset, struct_path):
    edges = []
    edge_weights = []

    encoded_seq = [torch.tensor(encode_sequence(seq)) for seq in dataset['sequence']]
    encoded_seq = pad_sequence(encoded_seq, batch_first=True, padding_value=0)
    max_len = encoded_seq.shape[1]
    dataset['encoded_sequence'] = encoded_seq.tolist()

    for seq in dataset['sequence']:
        edge, edge_weight = phosphodiester_bond(seq)
        edges.append(edge)
        edge_weights.append(edge_weight)
    dataset['phosphodiester_edges'] = edges
    dataset['phosphodiester_edge_weights'] = edge_weights

    base_pairing_data = process_structure_folder(struct_path)
    dataset = pd.merge(dataset, base_pairing_data, on='pas')

    # Concatenate the columns
    dataset['edges'] = dataset.apply(lambda row: [a + b for a, b in zip(row['phosphodiester_edges'], row['hydrogen_edges'])], axis=1)
    dataset['edge_weights'] = dataset.apply(lambda row: list(row['phosphodiester_edge_weights']) + list(row['hydrogen_edge_weights']), axis=1)
    return dataset, max_len

def create_dataloader(dataset, batch_size, shuffle, device):
    data_list = []

    for _, batch in dataset.iterrows():
        pas = batch['pas']
        gene = batch['gene']
        x = torch.tensor(batch['encoded_sequence']).to(device)
        edges = torch.tensor(batch['edges']).to(device)
        edge_weights = torch.tensor(batch['edge_weights']).to(device)
        y = torch.tensor(batch['usage']).to(device)
        data = Data(x=x, edge_index=edges, edge_attr=edge_weights, y=y, pas=pas, gene=gene)
        data_list.append(data)

    # Sort data by gene to ensure they stay together
    data_list.sort(key=lambda d: d.gene)

    # Group by gene
    grouped_data = [list(group) for _, group in groupby(data_list, key=lambda d: d.gene)]
    dataloader = DataLoader(grouped_data, batch_size=batch_size, shuffle=shuffle, collate_fn=grouped_collate)
    return dataloader

def grouped_collate(batch_list):
    """ Custom collation function that groups data instances before batching. """
    
    grouped_batches = [Batch.from_data_list(group) for group in batch_list]
    return grouped_batches


