from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
from PIL import Image
from RandAugment import RandAugment
from RandAugment.augmentations import CutoutDefault

# Data transformation with augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]), 
    'strong': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_transforms['strong'].transforms.insert(0, RandAugment(3, 4))
data_transforms['strong'].transforms.append(CutoutDefault(112))

class TransformTwice:
    def __init__(self, transform, transform2):
        self.transform = transform
        self.transform2 = transform2

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform2(inp)
        out3 = self.transform2(inp)
        return out1, out2, out3

# Dataset
class LT_Dataset(Dataset):
    
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        class_num_list = [0] * 1000
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
                class_num_list[int(line.split()[1])] += 1
        print(sorted(class_num_list, reverse=True))
        print (f"#Num Samples: {sum(class_num_list)}")
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, index

# Load datasets
def load_data(data_root, phase, labelled=True):

    txt = os.path.join(data_root, f'ImageNet_LT_{phase}.txt')

    print('Loading data from %s' % (txt))

    if phase not in ['train', 'val']:
        transform = data_transforms['test']
    else:
        if labelled:
            transform = data_transforms[phase]
        else:
             transform = TransformTwice(data_transforms[phase], data_transforms['strong'])

    print('Use data transformation:', transform)

    set_ = LT_Dataset(data_root, txt, transform)

    return set_
        
    
    
