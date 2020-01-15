import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class EMIDataset(Dataset):
    def __init__(self, data_dir: str, train: bool, num_classes: int, normalize: bool):
        whole_set = pd.read_csv(data_dir, sep='\s+', header=None).values.astype(dtype=np.float32)
        data = {i: [] for i in range(num_classes)}
        [data[l[-1] - 1].append(l) for l in whole_set]

        picked = []
        for i in range(num_classes):
            train_n = round(0.75 * len(data[i]))
            test_n = len(data[i]) - train_n
            picked.extend(data[i][:train_n] if train else data[i][-test_n:])
        picked = np.vstack(picked)
        picked[:, -1] -= 1
        # np.random.shuffle(picked)
        self.signals, self.labels = picked[:, :-1], picked[:, -1].astype(np.long)
        if normalize:
            self.signals = (self.signals - np.mean(self.signals, axis=0)) / np.std(self.signals, axis=0)
        self.len = len(self.labels)
        
    def __getitem__(self, index):
        # index %= self.len
        signal, label = self.signals[index], self.labels[index]
        return torch.from_numpy(signal), label
    
    def __len__(self):
        return self.len


if __name__ == '__main__':
    train_set = EMIDataset(
        data_dir='emi_sig/data.txt',
        train=True,
        num_classes=5,
        normalize=True
    )
    print(train_set[0])
    test_set = EMIDataset(
        data_dir='emi_sig/data.txt',
        train=False,
        num_classes=5,
        normalize=True
    )
    print(test_set[0])

