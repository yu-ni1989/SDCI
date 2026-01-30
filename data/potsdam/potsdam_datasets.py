import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F

PARENT_PATH = os.path.split(os.path.realpath(__file__))[0]
GID_CAT_NAME = open(os.path.join(PARENT_PATH, "cls_potsdam6.txt")).read().splitlines()

GID_PALETTE = [
    [0, 0, 255],
    [255, 0, 0],
    [255, 255, 0],
    [0, 255, 0],
    [0, 255, 255],
]


def load_gid_img_name_list(dataset_path):
    with open(dataset_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

class potsdamDataset(Dataset):

    def __init__(self,
                 img_name_list_path,
                 data_root,
                 img_transform=None):
        self.img_name_list = load_gid_img_name_list(img_name_list_path)
        self.data_root = data_root
        self.img_transform = img_transform
        self.background = [GID_CAT_NAME[0]]
        self.categories = GID_CAT_NAME[1:]
        self.category_number = len(self.categories)
        self.palette = np.array(GID_PALETTE)


    def _create_color_to_index_map(self):

        color_map = {}
        for index, bgr_color in enumerate(self.palette):
            rgb_color = tuple(bgr_color[::-1])
            color_map[rgb_color] = index
        return color_map

    def _convert_rgb_to_index(self, rgb_image):

        rgb_array = np.array(rgb_image)
        index_map = np.zeros(rgb_array.shape[:2], dtype=np.int64)
        for rgb_color, index in self.color_to_index.items():
            mask = np.all(rgb_array == rgb_color, axis=-1)
            index_map[mask] = index

        return torch.from_numpy(index_map)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        combined_path_str = self.img_name_list[idx]
        img_path = os.path.join(self.data_root, combined_path_str + '.tif')
        folder_name, base_name = os.path.split(combined_path_str)
        img = Image.open(img_path).convert('RGB')
        ori_img = torch.tensor(np.array(img)).permute(2, 0, 1) / 255.
        if self.img_transform is not None:
            img = self.img_transform(img)

        return ori_img, img, combined_path_str
