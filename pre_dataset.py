import torch
from torch.utils.data import DataLoader, TensorDataset


class MyDataset:   
    def __init__(self, data, labels, adjacency_matrix, data_rc, labels_rc, adjacency_matrix_rc):
        self.data = data
        self.labels = labels
        self.adjacency_matrix = adjacency_matrix
        self.data_rc = data_rc
        self.labels_rc = labels_rc
        self.adjacency_matrix_rc = adjacency_matrix_rc

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 实现获取单个样本的逻辑，返回数据和标签
        return (
            self.data[index],
            self.labels[index],
            self.adjacency_matrix[index],
            self.data_rc[index],
            self.labels_rc[index],
            self.adjacency_matrix_rc[index],
        )
    
    def split(self, split_ratio=[0.8, 0.0, 0.1], seed=None):
        total_size = len(self.data)
        indices = list(range(total_size))
        if seed is not None:
            torch.manual_seed(seed)
        torch.randperm(total_size)

        train_size = int(split_ratio[0] * total_size)
        val_size = int(split_ratio[1] * total_size)
        test_size = total_size - train_size - val_size

        train_indices = indices[:train_size]
        val_indices = indices[train_size:(train_size + val_size)]
        test_indices = indices[-test_size:]

        train_data = MyDataset(
            self.data[train_indices],
            self.labels[train_indices],
            self.adjacency_matrix[train_indices],
            self.data_rc[train_indices],
            self.labels_rc[train_indices],
            self.adjacency_matrix_rc[train_indices]

        )

        val_data = MyDataset(
            self.data[val_indices],
            self.labels[val_indices],
            self.adjacency_matrix[val_indices],
            self.data_rc[val_indices],
            self.labels_rc[val_indices],
            self.adjacency_matrix_rc[val_indices]
        )

        test_data = MyDataset(
            self.data[test_indices],
            self.labels[test_indices],
            self.adjacency_matrix[test_indices],
            self.data_rc[test_indices],
            self.labels_rc[test_indices],
            self.adjacency_matrix_rc[test_indices]
        )

        return train_data, val_data, test_data



