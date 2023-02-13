from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.CNN import Net as CNNNet

def test(
    net: nn.Module,
    test_loader: DataLoader,
    round_number: int,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float]:
    """Test the global network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        n_batches = len(test_loader)
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = net(images)
            loss += criterion(preds, labels).item()
            _, predicted = torch.max(preds.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total

    print(
        f" Round: {round_number}", f" Test Loss: {loss/n_batches}," f"Test Accuracy: {accuracy}",
    )
    return loss / n_batches, accuracy

def FedAvg_aggregate(server_dict: Dict[str, Any], all_client_params: List[Dict[str, Any]], client_sample_num: List[int]) -> Dict[str, Any]:

    # Calculate number of all the samples of clients in this round
    all_samples_num = sum(client_sample_num)

    for k in server_dict.keys():
        server_dict[k] = torch.stack([all_client_params[i][k].float()*float(client_sample_num[i]/all_samples_num)  for i in range(len(all_client_params))],0).sum(0)
    
    return server_dict
    
    
    