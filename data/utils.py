import torchvision.transforms.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop, RandomHorizontalFlip
import PIL
import PIL.Image

def _get_img_from_path(img_path, transform=None):
    with open(img_path, 'rb') as f:
        img = PIL.Image.open(f).convert('RGB')
    if transform is not None:
        img = transform(img)
    return img

def _convert_image_to_rgb(image):
    return image.convert("RGB")

class TargetPad:
    """
    Pad the image if its aspect ratio is above a target ratio.
    Pad the image to match such target ratio
    """

    def __init__(self, target_ratio: float, size: int):
        """
        :param target_ratio: target ratio
        :param size: preprocessing output dimension
        """
        self.size = size
        self.target_ratio = target_ratio

    def __call__(self, image):
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
            return image
        scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')

def image_transform(config, mode='train'):
    IMAGENET_STATS = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    img_size = config['img_size']

    if mode == 'train':
        return Compose([RandomResizedCrop(size=img_size, scale=(0.75, 1.33)),
                                   RandomHorizontalFlip(),
                                   ToTensor(),
                                   Normalize(**IMAGENET_STATS)])
    elif mode == 'val':
        return Compose([Resize((img_size, img_size)), ToTensor(),
                               Normalize(**IMAGENET_STATS)])

def targetpad_transform(target_ratio: float, dim: int):
    """
        CLIP-like preprocessing transform computed after using TargetPad pad
        :param target_ratio: target ratio for TargetPad
        :param dim: image output dimension
        :return: CLIP-like torchvision Compose transform
        """
    return Compose([
        TargetPad(target_ratio, dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])