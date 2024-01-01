from PIL import ImageFilter
from torchvision import transforms
import random


class GaussianBlur(object):

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
augmentation = [
    transforms.ToPILImage(mode='RGB'),
    transforms.RandomResizedCrop((16,32), scale=(0.3, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)
    ], p=0.8),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
]

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self):
        self.base_transform = transforms.Compose(augmentation)
        self.tensor_transform = transforms.Compose([transforms.ToTensor()])

    def __call__(self, x):
        q = self.tensor_transform(x)
        k = self.base_transform(x)
        q = q.numpy().reshape(-1)
        k = k.numpy().reshape(-1) 
        return [q, k]

#####################################################################    
augmentation_emnist = [
    transforms.RandomResizedCrop((14,28), scale=(0.3, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)
    ], p=0.8),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
]


class TwoCropsTransform_emnist:
    """Take two random crops of one image as the query and key."""

    def __init__(self):
        self.base_transform = transforms.Compose(augmentation_emnist)
        self.tensor_transform = transforms.Compose([transforms.ToTensor()])

    def __call__(self, x):
        q = self.tensor_transform(x)
        k = self.base_transform(x)
        q = q.numpy().reshape(-1)
        k = k.numpy().reshape(-1) 
        return [q, k]