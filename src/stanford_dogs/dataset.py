import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.datasets import ImageFolder

class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.subset)

    
def get_dataloaders(
        train_transform,
        test_transform,
        batch_size: int,
        num_workers: int,
        seed: int,
        data_dir,
        val_split: float= 0.1,
    ):

    data = ImageFolder(
        root =  data_dir
    )

    val_size = test_size = int(val_split* len(data))
    train_size = len(data) - val_size - test_size

    generator = torch.Generator().manual_seed(seed) # for reproducibility
    train_sub , val_sub, test_sub = random_split(data, [train_size, val_size, test_size], generator=generator)

    train_dataset = TransformedSubset(train_sub, transform= train_transform)
    val_dataset = TransformedSubset(val_sub, transform= test_transform)
    test_dataset = TransformedSubset(test_sub, transform= test_transform)

    train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle= True, num_workers= num_workers, pin_memory= True)
    val_loader = DataLoader(val_dataset, batch_size= batch_size, shuffle= False, num_workers= num_workers, pin_memory= True)
    test_loader = DataLoader(test_dataset, batch_size= batch_size, shuffle= False, num_workers= num_workers, pin_memory= True)

    return train_loader, val_loader, test_loader