import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

class initialize_dataset:
    def __init__(self, image_resolution=224, batch_size=128, root_dir="./dataset/dcgan"):
        self.image_resolution= image_resolution
        self.batch_size=batch_size
        self.root_dir = root_dir

    def load_dataset(self, transform=False):
        if transform:
            transform = transforms.Compose([
                transforms.Resize((self.image_resolution, self.image_resolution)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        else:
            transform = transforms.Compose([
                transforms.Resize((self.image_resolution, self.image_resolution)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        train_path = self.root_dir + "/train"
        test_path = self.root_dir + "/test"

        train_dataset = ImageFolder(root=train_path, transform=transform)
        test_dataset = ImageFolder(root=test_path, transform=transform)

        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                      batch_size=self.batch_size,
                                                      shuffle=True)

        return train_dataloader, test_dataloader
