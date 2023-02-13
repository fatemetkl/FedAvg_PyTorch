import argparse
import yaml
from typing import Any, Dict, List, Tuple
import random

import torch
import torch.nn as nn

from data import load_data_IID, load_data_nonIID
from client import Client
from models.CNN import Net as CNNNet
from models.TNN import Net as TNet
import server

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(config: Dict[str, Any]) -> None:
    n_clients = config["n_clients"]
    # Pull and create distributed data by creating n_clients data loader
    if config["IID"]:
        trainloaders, valloaders, testloader = load_data_IID(config)
        Data_setting = "IID"
    else:
        trainloaders, valloaders, testloader = load_data_nonIID(config)
        Data_setting = "non_IID"
    # Create n_clients clients and distribute dataloaders
    clients_list = []
    for cid in range(n_clients):
        new_client = Client(cid, trainloaders[cid], valloaders[cid],config)
        clients_list.append(new_client)

    #start the server 
    #initialize a server model
    if config["model_struct"]=="CNN":
        server_model = CNNNet()
        goal_accuracy = 0.99
    else:
        server_model = TNet()
        goal_accuracy = 0.97

    # Main loop -> max server rounds
    all_accuracy = []
    reached = False
    for round in range(config["n_server_rounds"]):
        if not reached:
            # Randomly select m clients
            m = max(int(n_clients*config["client_rate"]),1)
            S_t = random.sample(range(0, n_clients-1), m)
            print("Sampled clients for this round: ", S_t)
            local_param_list = []
            client_train_examples = []
            for client in [clients_list[indx] for indx in S_t]:
                # Update clients with the model and do local training
                client_parameters, num_examples = client.client_update(server_model.state_dict(), config)
                local_param_list.append(client_parameters)
                client_train_examples.append(num_examples)
            # Updating the server model by weighted aggregation
            server_model_params = server_model.state_dict()
            new_server_params =  server.FedAvg_aggregate(server_model_params, local_param_list, client_train_examples)
            server_model.load_state_dict(new_server_params)

            # Test the updated server model on the test data
            server_loss ,server_accuracy = server.test(server_model, testloader, round, DEVICE)
            all_accuracy.append(server_accuracy)
            if server_accuracy >= goal_accuracy:
                print("Reached the accuracy goal")
                reached = True


    # Saving all the accuracy and loss results in a file
    E = config["local_epochs"]
    B = config["batch_size"]
    model_structure = config["model_struct"]
    file_name = f"{model_structure}_model_{Data_setting}_data_{E}E_{B}B.txt"
    with open("results/"+file_name, 'w') as f:
        for line in all_accuracy:
            f.write(f"{line}\n")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Server Main")
    parser.add_argument(
        "--config_path",
        action="store",
        type=str,
        help="Path to configuration file.",
        default="config.yaml",
    )
    args = parser.parse_args()

    # load configuration dictionairy
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    main(config)