# Reproducing the results of the paper "Communication-Efficient Learning of Deep Networks from Decentralized Data [1]"
## How to run the code:
Set all the hyper-parameters and network settings in the ``config.yaml`` file. Then run “python trainer.py”. You can also include the path to the config file in the terminal after the run command. Make sure the packages in``requirements.txt`` are installed.  
Accuracy results for each round are saved inside a “.txt” file in the results folder. The name of the .txt file shows the experimental settings of that file. Using these saved accuracy results we can also create the plots in the paper.  
The accuracy results for all the settings in the tables are saved inside the `` results/old`` directory.
## Discussion on the results:  
Experimental settings in Table 2 test the impact of increasing client local computations via changing local epoch and batch size.
Our results also confirm that as we increase the client computation either by decreasing B or increasing E or both, the number of communication rounds needed to reach a good target accuracy decreases. This is due to the increased computation per client and having more local SGD per client which makes the algorithm a lot faster, especially for IID settings. We also see a considerable improvement in the training speed (fewer rounds) in the more challenging Non-IID setting. In [1], they have defined the expected number of updates per client per round with U that increases with more client computation.    

Results highly suggest that there is room for improvement in Non-IID setting especially according to 2NN network results.
Sources of randomness in the results:  
There are several sources of randomness that affect the target round, making it a bit less or more than the values reported in the paper.
The randomness in choosing clients in each round of FedAvg. Choosing only 10% of clients probably is one of the reasons some of the results are different from the values reported in the paper. Also, there is randomness in distributing the data, which might affect the speed of convergence to some degree.  

## Experimental settings:
All the experimental settings are set exactly according to the paper.
Reported results are the number of rounds in which the test accuracy hits a target value. For the CNN network, it is 0.99, and for the 2NN network, it is 0.97.
In all the settings we set C=0.1 which means 10% of all the 100 clients randomly chosen for federation in each round. E, B, the network structure, and IDDness are specified in the config file. If you set the IID variable in the config file to 1, the data is distributed equally but randomly among clients. Each client gets 600 samples.
If you set the IID value to 0, the data distribution would be Non-IID, where most of the clients get only samples of 2 digits. In the Non-IID setting each client gets 2 sherds of size 300, which means again each client has 600 samples.
In the config file, the max round is set to 1000 for CNN and set to 2000 for the 2NN network (mentioned in the paper appendix).
All the learning rates in all experiments are set to 0.01 unless otherwise stated.
Round results are sometimes less and sometimes more but all follow the same pattern reported in the original paper as explained previously. I mostly used 0.01 as the learning rate but this value probably needs to be optimized for each of the settings as also mentioned in the paper. I decided to set the learning rate to 0.01 based on some initial results but did not do any hyperparameter tuning as it was time-consuming and was not asked by the instructions. But I assume in many cases the optimal learning rate would be less than 0.01. In many of the settings, a smaller learning rate value would result in fewer rounds.

## Code structure:
``\models`` : implementation of 2NN (TNN.py) and CNN (CNN.py) models
``\results\old``: accuracy results of all the settings are stored.  
``trainer.py``: main file where the coordination between the clients and the server happens. data.py: includes the methods for distributing the data (IID and Non-IID).   
``Client.py``: includes the Client class, train function, and validate function.   
``Server.py``: includes the server functionalities such as FedAvg weighted aggregation
And testing.   
Code is tested on GPU and CPU.

for lr 0.1 0.01 and 0.001 is used for optimization, optimized in one setting and used for all that network structure
for CNN best value -> 0.1
for 2NN -> 0.01


all the training data size: 60000
divided between 100 clients -> each 600 data examples

## References
```
[1] McMahan B, Moore E, Ramage D, Hampson S, y Arcas BA. Communication-efficient learning of deep networks from decentralized data. InArtificial intelligence and statistics 2017 Apr 10 (pp. 1273-1282). PMLR.

```
