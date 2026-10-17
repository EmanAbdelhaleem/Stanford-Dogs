from torchvision import transforms
from stanford_dogs.config import IMG_SIZE

def get_transforms(img_size: int=IMG_SIZE):

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            size = img_size,
            scale = (0.2,1.0),
            ratio = (0.75, 1.50)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform, test_transform
