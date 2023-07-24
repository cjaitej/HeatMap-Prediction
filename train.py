from dataset import NYC_V2
from torch.utils.data import DataLoader
from model import UNET
from torch import optim
from utils import *
import torch
import torch.nn as nn

def train(checkpoint):
    if checkpoint == None:
        model = UNET(in_c=3, out_c=1)
        start_epoch = 0
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']
    model = model.to(device=device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    average_loss = 0

    print(f" -- Initiating the Training Process -- ")
    print(f"Epoch: {start_epoch}: ")

    for epoch in range(start_epoch, epochs):
        for i, (image, mask) in enumerate(train_gen):
            image = image.to(device)
            mask = mask.to(device)
            pred_image = model(image)

            #calculating Loss
            loss = criterion(pred_image, mask)

            #Feedback
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            average_loss = average_loss + loss

            del image, mask, pred_image

            if i%20 == 0:
                print("=", end="")
        save_checkpoint(epoch=epoch, model=model)
        add_result(f"Epoch: {epoch} | Average Loss: {average_loss/(i + 1)}")
        print(f"   Epoch: {epoch} | Average Loss: {average_loss/(i + 1)}")
        average_loss = 0



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = "HeatMapModel_v1.pth.tar"
    batch_size = 32
    iterations = 10000
    workers = 4
    epochs = 1000
    lr = 0.0001
    file_name = "train.json"

    train_dataset = NYC_V2(filename=file_name, transform=transform)

    train_gen = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=True,
        pin_memory=True
    )

    train(checkpoint)