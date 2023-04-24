import torch
import torchvision
from torchvision.datasets import ImageFolder
from augmentations import augmentation, ContrastiveAugmentation
import torchvision.transforms as transforms
import os

class initialize_dataset:
    def __init__(self, image_resolution=224, batch_size=128, dataset_path="generated_image/spot/dcgan"):
        self.image_resolution = image_resolution
        self.batch_size = batch_size
        self.dataset_path = dataset_path

    def load_dataset(self, transform=False):
        path = self.dataset_path

        if transform:
            transform = augmentation(image_resolution=self.image_resolution)
        else:
            transform = transforms.Compose([transforms.Resize((self.image_resolution, self.image_resolution)),
                                            transforms.ToTensor(),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        train_path = os.path.join(path, "train")
        test_path = os.path.join(path, "test")

        train_dataset = ImageFolder(root=train_path, transform=transform)
        test_dataset = ImageFolder(root=test_path, transform=transform)

        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                      batch_size=self.batch_size,
                                                      shuffle=True)

        return train_dataloader, test_dataloader
