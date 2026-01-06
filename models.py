import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten=nn.Flatten()
        self.network=nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )

    def forward(self,x):
        x=self.flatten(x)
        logits=self.network(x)
        return logits   
def test_accuracy(model,dataloader):
    n_corrects=0


    device=next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        for image_batch,label_batch in dataloader:

            image_batch=image_batch.to(device)
            label_bacth=label_batch.to(device)

            logits_batch=model(image_batch)

            predicts_batch=logits_batch.argmax(dim=1)
            n_corrects+=(label_bacth==predicts_batch).sum().item()

    accuracy=n_corrects/len(dataloader.dataset)

    return accuracy

def train(model,dataloader,loss_fn,optimizer):

    device=next(model.parameters()).device
    model.train()
    for image_batch,label_batch in dataloader:

        image_batch=image_batch.to(device)
        label_bacth=label_batch.to(device)

        logits_batch=model(image_batch)

        loss=loss_fn(logits_batch,label_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()

def test(model,dataloader,loss_fn):
    loss_total=0.0

    device=next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for image_batch,label_batch in dataloader:
            image_batch=image_batch.to(device)
            label_bacth=label_batch.to(device)


            logits_batch=model(image_batch)

            loss=loss_fn(logits_batch,label_batch)
            loss_total+=loss.item()

    return loss_total/len(dataloader)