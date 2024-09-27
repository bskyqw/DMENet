import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2
class test_dataset:
    def __init__(self, image_root, gt_root, depth_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.bmp')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if
                       f.endswith('.bmp') or f.endswith('.tiff') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.ToTensor()
        self.depths_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()
        ])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        depth_path = self.depths[self.index]
        if depth_path.endswith('.tiff'):
            depth = self.load_and_normalize_16bit_depth(depth_path)
        else:
            depth = self.binary_loader(depth_path)
        depth = self.depths_transform(depth).unsqueeze(0)


        name = self.images[self.index].split('/')[-1]
        image_for_post = self.rgb_loader(self.images[self.index])
        image_for_post = image_for_post.resize(gt.size)

        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        if name.endswith('.bmp'):
            name = name.split('.bmp')[0] + '.png'

        self.index += 1
        self.index = self.index % self.size
        return image, gt, depth, name, np.array(image_for_post)

    def load_and_normalize_16bit_depth(self, depth_path):
        image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        min_val, max_val = np.min(image), np.max(image)
        image_normalized = ((image - min_val) / (max_val - min_val)) * (65535 - 0) + 0
        image_8bit = (image_normalized / 65535.0) * 255
        image_8bit = image_8bit.astype(np.uint8)
        return Image.fromarray(image_8bit)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

