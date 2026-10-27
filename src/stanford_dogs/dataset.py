import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import ImageFolder

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
    train_indices , val_indices, test_indices = random_split(range(len(data)), [train_size, val_size, test_size], generator=generator)

    train_dataset = Subset(ImageFolder(
        root =  data_dir, 
        transform= train_transform,),
        train_indices
    )

    val_dataset = Subset(ImageFolder(
        root =  data_dir, 
        transform= test_transform,),
        val_indices
    )

    test_dataset = Subset(ImageFolder(
        root =  data_dir, 
        transform= test_transform,),
        test_indices
    )

    train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle= True, num_workers= num_workers, pin_memory= True)
    val_loader = DataLoader(val_dataset, batch_size= batch_size, shuffle= False, num_workers= num_workers, pin_memory= True)
    test_loader = DataLoader(test_dataset, batch_size= batch_size, shuffle= False, num_workers= num_workers, pin_memory= True)

    return train_loader, val_loader, test_loader