import numpy as np
import torch
from torchvision import transforms

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    accelerator = "gpu"
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    accelerator = "mps"
else:
    device = torch.device("cpu")

from main_code.constants import INPUT_HEIGHT, INPUT_WIDTH

### Transformations ###

to_tensor = transforms.ToTensor()
resizer = transforms.Resize(
    size=(INPUT_HEIGHT, INPUT_WIDTH), interpolation=transforms.InterpolationMode.NEAREST
)
# center_crop = transforms.CenterCrop(512)


def transform_image(img):
    """Apply transformations to image after opening with PIL."""

    # img = center_crop(img)
    img = resizer(img)
    img = to_tensor(img)

    return img
