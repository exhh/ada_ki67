import os.path
import sys

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image

from .data_loader import register_data_params, register_dataset_obj
from .data_loader import DatasetParams

@register_data_params('IHC')
class IHCParams(DatasetParams):
    num_channels = 3
    image_size   = 500
    mean         = 0.5
    std          = 0.5
    num_cls      = 1
    target_transform = None


@register_dataset_obj('IHC')
class IHC(data.Dataset):

    def __init__(self, root, split='train', transform=None,
                 target_transform=None):
        self.root = root
        sys.path.append(root)
        self.split = split
        self.ids = self.collect_ids()
        self.transform = transform
        self.target_transform = target_transform
        self.num_cls = 1
        self.contourname = 'Contours'
        self.decayparam = {'scale': 1.0, 'alpha': 3.0, 'r': 15.0}

    def collect_ids(self):
        im_dir = os.path.join(self.root, 'images', self.split)
        ids = []
        for filename in os.listdir(im_dir):
            if filename.endswith('.png'):
                ids.append(filename[:-4])
        return ids

    def img_path(self, id):
        fmt = 'images/{}/{}.png'
        path = fmt.format(self.split, id)
        return os.path.join(self.root, path)

    def label_path(self, id):
        fmt = 'labels/{}/{}_label.png'
        path = fmt.format(self.split, id)
        return os.path.join(self.root, path)

    def label_path_mat(self, id):
        fmt = 'images/{}/{}_withcontour.mat'
        path = fmt.format(self.split, id)
        return os.path.join(self.root, path)

    def joint_transform(self, image, mask):
        if self.split == 'train':
            angle = transforms.RandomRotation.get_params(degrees=(-30, 30))
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask = torch.from_numpy(np.array(mask, np.int64, copy=False)).unsqueeze(0)
        return image, mask

    def __getitem__(self, index):
        id = self.ids[index]
        imagename = self.img_path(id)
        imagename_noext = imagename.rsplit('/',1)[1].split('.')[0]
        img = Image.open(imagename).convert('RGB')

        labelname = self.label_path(id)
        target = Image.open(labelname[0]).convert('L')

        image, label = self.joint_transform(img, target)
        if self.split == 'train':
            return image, label
        else:
            return image, label, imagename_noext

    def __len__(self):
        return len(self.ids)


if __name__ == '__main__':
    cs = PNET('/x/IHC')
