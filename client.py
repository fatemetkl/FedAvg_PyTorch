from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.CNN import Net as CNNNet
from models.TNN import Net as TNet

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(
    net: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float]:
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(epochs):
        correct, total, running_loss = 0, 0, 0.0
        n_batches = len(train_loader)
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = net(images)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(preds.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total

        print(
            f"Epoch: {epoch}, Client Training Loss: {running_loss/n_batches}," f"Client Training Accuracy: {accuracy}",
        )
    return accuracy, total


def validate(
    net: nn.Module,
    validation_loader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float, int]:
    """Validate the network on the entire validation set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        n_batches = len(validation_loader)
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            preds = net(images)
            loss += criterion(preds, labels).item()
            _, predicted = torch.max(preds.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total

    print(
        f"Client Validation Loss: {loss/n_batches}," f"Client Validation Accuracy: {accuracy}",
    )
    return loss / n_batches, accuracy, total



class Client():

    def __init__(self, cid:int, train_loader:DataLoader, val_loader:DataLoader, config:Dict[str, Any]) -> None:
        self.cid = cid
        self.train_loader = train_loader
        self.validation_loader = val_loader
        self.device =  DEVICE
        self.config = config
        if config["model_struct"]=="CNN":
            model = CNNNet().to(DEVICE)
        else:
            model = TNet().to(DEVICE)
        self.model = model
    

    def client_update(self, server_parameters: Dict[str, Any] , config:Dict[str, Any]) -> Tuple[Dict[str, Any], int]:

        # update local parameters
        self.model.load_state_dict(server_parameters)
        # fit the local model on local training data
        print (f"Client {self.cid} training")
        accuracy, num_train_examples = train(self.model, self.train_loader, epochs=config["local_epochs"], device=self.device)      

        # Return the local model parameters and sample num
        return (
            self.model.state_dict(),
            num_train_examples
        )


    def validate(self, server_parameters: Dict[str, Any] , config:Dict[str, Any]) -> Tuple[float, float, int]:
        self.model.load_state_dict(server_parameters)
        loss, accuracy, num_val_examples = validate(self.model, self.validation_loader, device=self.device)

        return (
            loss,
            accuracy,
            num_val_examples,
        )