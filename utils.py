import pandas as pd
import json
import torch.nn.functional as TF
import random
import torch
from torchvision.transforms import transforms

def create_data_list(path, filename):
    df = pd.read_csv(filename, header=None)
    image, mask = df[0].to_numpy(), df[1].to_numpy()
    image = path + "/"+ image
    mask = path + "/"+  mask
    data = []

    for i, j in zip(image, mask):
        data.append((i, j))

    with open('train.json', "w", encoding='utf-8') as f:
        json.dump(data, f)


def transform(image, mask, size=256):
    resize = transforms.Resize(size=(size, size))
    hflip = transforms.RandomHorizontalFlip(p=1)
    vflip = transforms.RandomVerticalFlip(p=1)
    totensor = transforms.ToTensor()


    #resizing image
    image = resize(image)
    mask = resize(mask)

    #Horizontal Flipping
    if random.random() > 0.5:
        image = hflip(image)
        mask = hflip(mask)

    #Vertical Flipping
    if random.random() > 0.5:
        image = vflip(image)
        mask = vflip(mask)

    #Converting Image to torch tensor.
    image = totensor(image)
    mask = totensor(mask)

    return image, mask

def add_result(result):
    with open('results_v1.txt', 'a') as f:
        f.write(result + "\n")
    f.close()

def save_checkpoint(epoch, model):
    state = {'epoch': epoch,
             'model': model}
    filename = 'HeatMapModel_v1.pth.tar'
    torch.save(state, filename)