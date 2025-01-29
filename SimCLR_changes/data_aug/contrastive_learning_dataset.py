from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
import os
from PIL import Image
from torch.utils.data import Dataset


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms
    
    @staticmethod
    def get_simclr_pipeline_transform_greyscale(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.Grayscale(num_output_channels=1),
                                              transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views):
        print(os.path.exists("/storage3/michel"))
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                          download=True), 
                          
                          'stylegan': lambda: CustomImageDataset(image_folder= os.path.join(self.root_folder, "stylegan-oriented"),
                                        transform=ContrastiveLearningViewGenerator(
                                            self.get_simclr_pipeline_transform_greyscale(size = 28),
                                            n_views),
                                        n_views=n_views),
                          
                          'dead_leaves': lambda: CustomImageDataset(image_folder=os.path.join(self.root_folder, "dead_leaves-mixed"),
                                        transform=ContrastiveLearningViewGenerator(
                                            self.get_simclr_pipeline_transform_greyscale(size = 28),
                                            n_views),
                                        n_views=n_views),
                          'shaders21k_grey': lambda: CustomImageDataset(image_folder=os.path.join(self.root_folder, "shaders21k"),
                                        transform=ContrastiveLearningViewGenerator(
                                            self.get_simclr_pipeline_transform_greyscale(size = 28),
                                            n_views),
                                        n_views=n_views),
                          'fractaldb': lambda: CustomImageDataset(image_folder=os.path.join(self.root_folder, "FractalDB"),
                                        transform=ContrastiveLearningViewGenerator(
                                            self.get_simclr_pipeline_transform_greyscale(size = 28),
                                            n_views),
                                        n_views=n_views),
                          'fmnist': lambda: datasets.FashionMNIST(self.root_folder, train=True, transform=ContrastiveLearningViewGenerator(
                                            self.get_simclr_pipeline_transform_greyscale(size = 28),
                                            n_views), download=True), 
                              
                          'shaders21k_rgb' : lambda: CustomImageDataset(image_folder=os.path.join(self.root_folder, "shaders21k"),
                                        transform=ContrastiveLearningViewGenerator(
                                            self.get_simclr_pipeline_transform(size = 32),
                                            n_views),
                                        n_views=n_views),
                          'dead_leaves_rgb': lambda: CustomImageDataset(image_folder=os.path.join(self.root_folder, "dead_leaves-mixed"),
                                        transform=ContrastiveLearningViewGenerator(
                                            self.get_simclr_pipeline_transform(size = 32),
                                            n_views),
                                        n_views=n_views),
                          'stylegan_rgb': lambda: CustomImageDataset(image_folder= os.path.join(self.root_folder, "stylegan-oriented"),
                                        transform=ContrastiveLearningViewGenerator(
                                            self.get_simclr_pipeline_transform(size = 32),
                                            n_views),
                                        n_views=n_views),
                        }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()


class CustomImageDataset(Dataset):
    def __init__(self, image_folder, transform=None, n_views=2):
        """
        Args:
            image_folder (str): Path to the folder containing images.
            transform (callable, optional): A function/transform to apply to each image.
            n_views (int): Number of augmented views to generate for each image.
        """
        self.image_folder = image_folder
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg'))]
        self.transform = transform
        self.n_views = n_views

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path) 
        if self.transform:
            return [self.transform(image) for _ in range(self.n_views)]
        else:
            return image