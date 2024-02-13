import torch
import torch.nn.functional as F

def train(model, train_loader, optimizer, device):
    model.to(device)
    model.train()

    total_loss = 0
    for data in train_loader:
        data.to(device)

        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y.squeeze().long())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += float(loss)

    return total_loss / len(train_loader)


def test(model, loader, device):
     model.to(device)
     model.eval()

     total_correct = 0
     for data in loader:
         data.to(device)

         pred = model(data.x, data.edge_index, data.batch).argmax(dim=-1)
         total_correct += int((pred == data.y.squeeze().long()).sum())

     return total_correct / len(loader.dataset)
