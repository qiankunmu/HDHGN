import ast
import javalang.parse
import torch
from torch_geometric.data import Dataset, Data
import os

from utils import pre_walk_tree, pre_walk_tree_java


class HDHGData(Data):
    def __init__(self, x=None, edge_types=None, **kwargs):
        super(HDHGData, self).__init__(x, edge_types=edge_types, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_in_indexs':
            return torch.tensor([[self.x.size(0)], [self.edge_types.size(0)]])
        elif key == 'edge_out_indexs':
            return torch.tensor([[self.edge_types.size(0)], [self.x.size(0)]])
        elif key == 'edge_in_out_indexs':
            return torch.tensor([[self.edge_types.size(0)], [self.x.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)


class HDHGNDataset(Dataset):
    def __init__(self, root, paths_file_path, vocab):
        self.paths_file_path = paths_file_path
        self.vocab = vocab
        self.processed_file_names_list = []

        paths_file = open(self.paths_file_path)
        for i, file_path in enumerate(paths_file):
            self.processed_file_names_list.append(f"processed_data_{i}.pt")
        paths_file.close()

        super().__init__(root, transform=None, pre_transform=None, pre_filter=None)

    @property
    def processed_file_names(self):
        return self.processed_file_names_list

    def process(self):
        paths_file = open(self.paths_file_path)
        for i, file_path in enumerate(paths_file):
            file = open(file_path[0:-1], encoding="utf-8")
            code = file.read()

            root = ast.parse(code)
            index, edge_index, types, features, edge_types, edge_in_indexs_s, edge_in_indexs_t, edge_out_indexs_s, edge_out_indexs_t, edge_in_out_indexs_s, edge_in_out_indexs_t, edge_in_out_head_tail = pre_walk_tree(
                root, 0, 0)
            types_encoded = [self.vocab.vocab["types"].word2id[t] for t in types]
            types_encoded = torch.tensor(types_encoded, dtype=torch.long)
            features_encoded = [self.vocab.vocab[types[i]].word2id.get(f, 1) for (i, f) in enumerate(features)]
            features_encoded = torch.tensor(features_encoded, dtype=torch.long)
            edge_types_encoded = [self.vocab.vocab["edge_types"].word2id.get(e, 1) for e in edge_types]
            edge_types_encoded = torch.tensor(edge_types_encoded, dtype=torch.long)
            edge_in_indexs_encoded = torch.tensor([edge_in_indexs_s, edge_in_indexs_t], dtype=torch.long)
            edge_out_indexs_encoded = torch.tensor([edge_out_indexs_s, edge_out_indexs_t], dtype=torch.long)
            edge_in_out_indexs_encoded = torch.tensor([edge_in_out_indexs_s, edge_in_out_indexs_t], dtype=torch.long)
            edge_in_out_head_tail_encoded = torch.tensor(edge_in_out_head_tail, dtype=torch.long)
            labels = torch.tensor([self.vocab.vocab["labels"].word2id[file_path[34:40]]], dtype=torch.long)

            d = HDHGData(x=features_encoded, types=types_encoded, edge_types=edge_types_encoded,
                         edge_in_indexs=edge_in_indexs_encoded, edge_out_indexs=edge_out_indexs_encoded,
                         edge_in_out_indexs=edge_in_out_indexs_encoded,
                         edge_in_out_head_tail=edge_in_out_head_tail_encoded, labels=labels)

            torch.save(d, os.path.join(self.processed_dir, f"processed_data_{i}.pt"))

        paths_file.close()

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        d = torch.load(os.path.join(self.processed_dir, f'processed_data_{idx}.pt'))
        return d

class HDHGNDataset_java(Dataset):
    def __init__(self, root, paths_file_path, vocab):
        self.paths_file_path = paths_file_path
        self.vocab = vocab
        self.processed_file_names_list = []

        paths_file = open(self.paths_file_path)
        for i, file_path in enumerate(paths_file):
            self.processed_file_names_list.append(f"processed_data_{i}.pt")
        paths_file.close()

        super().__init__(root, transform=None, pre_transform=None, pre_filter=None)

    @property
    def processed_file_names(self):
        return self.processed_file_names_list

    def process(self):
        paths_file = open(self.paths_file_path)
        for i, file_path in enumerate(paths_file):
            file = open(file_path[0:-1], encoding="utf-8")
            code = file.read()

            root = javalang.parse.parse(code)
            index, edge_index, types, features, edge_types, edge_in_indexs_s, edge_in_indexs_t, edge_out_indexs_s, edge_out_indexs_t, edge_in_out_indexs_s, edge_in_out_indexs_t, edge_in_out_head_tail = pre_walk_tree_java(
                root, 0, 0)
            types_encoded = [self.vocab.vocab["types"].word2id[t] for t in types]
            types_encoded = torch.tensor(types_encoded, dtype=torch.long)
            features_encoded = [self.vocab.vocab[types[i]].word2id.get(f, 1) for (i, f) in enumerate(features)]
            features_encoded = torch.tensor(features_encoded, dtype=torch.long)
            edge_types_encoded = [self.vocab.vocab["edge_types"].word2id.get(e, 1) for e in edge_types]
            edge_types_encoded = torch.tensor(edge_types_encoded, dtype=torch.long)
            edge_in_indexs_encoded = torch.tensor([edge_in_indexs_s, edge_in_indexs_t], dtype=torch.long)
            edge_out_indexs_encoded = torch.tensor([edge_out_indexs_s, edge_out_indexs_t], dtype=torch.long)
            edge_in_out_indexs_encoded = torch.tensor([edge_in_out_indexs_s, edge_in_out_indexs_t], dtype=torch.long)
            edge_in_out_head_tail_encoded = torch.tensor(edge_in_out_head_tail, dtype=torch.long)
            labels = torch.tensor([self.vocab.vocab["labels"].word2id[file_path[32:38]]], dtype=torch.long)

            d = HDHGData(x=features_encoded, types=types_encoded, edge_types=edge_types_encoded,
                         edge_in_indexs=edge_in_indexs_encoded, edge_out_indexs=edge_out_indexs_encoded,
                         edge_in_out_indexs=edge_in_out_indexs_encoded,
                         edge_in_out_head_tail=edge_in_out_head_tail_encoded, labels=labels)

            torch.save(d, os.path.join(self.processed_dir, f"processed_data_{i}.pt"))

        paths_file.close()

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        d = torch.load(os.path.join(self.processed_dir, f'processed_data_{idx}.pt'))
        return d
