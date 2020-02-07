import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class EMIDataset(Dataset):
    def __init__(self, data_dir: str, train: bool, num_classes: int, normalize: bool):
        whole_set = pd.read_csv(data_dir, sep='\s+', header=None).values.astype(dtype=np.float32)
        data = {i: [] for i in range(num_classes)}
        [data[l[-1] - 1].append(l) for l in whole_set]
        
        train_data, test_data = [], []
        for i in range(num_classes):
            train_n = round(0.8 * len(data[i]))
            test_n = len(data[i]) - train_n
            train_data.extend(data[i][:train_n])
            test_data.extend(data[i][-test_n:])
        train_data, test_data = np.vstack(train_data), np.vstack(test_data)
        
        # (train_data[:, -1] - 1)
        #                      ^  : trans label from [1, 5] to [0, 4]
        train_sigs, train_labels = train_data[:, :-1], (train_data[:, -1] - 1).astype(np.long)
        test_sigs, test_labels = test_data[:, :-1], (test_data[:, -1] - 1).astype(np.long)
        
        if normalize:
            # scaler = StandardScaler()
            scaler = MinMaxScaler(feature_range=(-1, 1))
            train_sigs = scaler.fit_transform(train_sigs)
            test_sigs = scaler.transform(test_sigs)
        
        self.signals, self.labels = (train_sigs, train_labels) if train else (test_sigs, test_labels)
        self.len = len(self.labels)
    
    def __getitem__(self, index):
        # index %= self.len
        signal, label = self.signals[index], self.labels[index]
        return torch.from_numpy(signal), label
    
    def __len__(self):
        return self.len


if __name__ == '__main__':
    train_set = EMIDataset(
        data_dir='../emi_sig/data.txt',
        train=True,
        num_classes=5,
        normalize=True
    )
    print(train_set[0])
    test_set = EMIDataset(
        data_dir='../emi_sig/data.txt',
        train=False,
        num_classes=5,
        normalize=True
    )
    print(test_set[0])
