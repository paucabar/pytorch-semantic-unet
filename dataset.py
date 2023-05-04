import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class BinaryDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, cache=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self._do_cache = cache
        self._cache = {}
    def __len__(self):
        return len(self.images)
    def _load_image(self, path: str):
        return np.array(Image.open(path)).astype(np.float32)
    def _get_image(self, path: str):
        if self._do_cache:
            image = self._cache.get(path)
            if image is None:
                image = self._load_image(path)
                self._cache[path] = image
            return image.copy() # Copy to avoid overwriting by augmentations
        else:
            return self._load_image(path)
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = self._get_image(img_path)
        mask = self._get_image(mask_path) # open labelled image
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return image, mask
