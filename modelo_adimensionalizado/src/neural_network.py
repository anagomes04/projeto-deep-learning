import torch 
from torch import nn

class NeuralNetwork(nn.Module):

    activation_fn: callable

    def __init__(self, arq, activation_fn):

        super(NeuralNetwork, self).__init__()

        self.activation_fn = activation_fn

        self.neural_network = nn.ModuleList()
        
        for i in range(len(arq)-1):
            
            self.neural_network.append(nn.Linear(arq[i], arq[i+1]))
            nn.init.xavier_normal_(self.neural_network[i].weight)

    
    def forward(self, x):

        for layer in self.neural_network[:-1]: x = self.activation_fn(layer(x))
        x = self.neural_network[-1](x)  
        return x
    
    def save(self, local: str):

        torch.save(self.state_dict(), local)

    def load(self, local: str):

        self.load_state_dict(torch.load(local))