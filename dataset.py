from torch.utils.data import Dataset
import json
from PIL import Image

class NYC_V2(Dataset):
    def __init__(self, filename, transform=None):
        super(NYC_V2, self).__init__()
        f = open(filename)
        self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, mask = self.data[index]
        image, mask = Image.open(image), Image.open(mask)
        if self.transform:
            image, mask = self.transform(image, mask, size=128)

        return image, mask