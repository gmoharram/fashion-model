"""Data utility functions."""
import os
import torch.utils.data as data
from PIL import Image

from main_code.constants import INPUT_HEIGHT, INPUT_WIDTH
from main_code.utils import transform_image

from IPython.core.debugger import set_trace


class FashionDataset(data.Dataset):
    def __init__(
        self, image_paths_file, input_height=INPUT_HEIGHT, input_width=INPUT_WIDTH
    ):
        self.root_dir_name = os.path.dirname(image_paths_file)
        self.input_size = (input_height, input_width)

        with open(image_paths_file) as f:
            self.image_names = f.read().splitlines()

    def __getitem__(self, key):
        if isinstance(key, slice):
            # get the start, stop, and step from the slice
            return [self[ii] for ii in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            # handle negative indices
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)
            # get the data from direct index
            return self.get_item_from_index(key)
        else:
            raise TypeError("Invalid argument type.")

    def __len__(self):
        return len(self.image_names)

    def get_item_from_index(self, index):
        img_id = self.image_names[index].replace(".png", "")

        input_img = Image.open(
            os.path.join(self.root_dir_name, "inputs", img_id + ".png")
        ).convert("RGB")
        input_img = transform_image(input_img)

        target_img = Image.open(
            os.path.join(self.root_dir_name, "targets", img_id + "_t.png")
        ).convert("RGB")
        target_img = transform_image(target_img)

        return input_img, target_img
