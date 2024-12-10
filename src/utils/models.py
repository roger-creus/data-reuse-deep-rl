import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class QNetwork(nn.Module):
    def __init__(self, env, input_channels=4, lstm_hidden_size=128):
        super().__init__()
        self.lstm_hidden_size = lstm_hidden_size
        
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(input_channels, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.Flatten()
        )
        
        self.mlp = nn.Sequential(
            nn.ReLU(),
            layer_init(nn.Linear(3136, 512)),
            nn.ReLU()
        )
        
        if lstm_hidden_size == -1:
            lstm_hidden_size = 512
        else:
            self.lstm = nn.LSTM(512, lstm_hidden_size)
            for name, param in self.lstm.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param, 1.0)
                    
        self.q_func = nn.Sequential(
            nn.ReLU(),
            layer_init(nn.Linear(lstm_hidden_size, env.action_space.n)),
        )

    def forward(self, x, lstm_state=None, done=None, from_pixels=True):
        if self.lstm_hidden_size == -1:
            x = x / 255.0
            x = self.cnn(x)
            x = self.mlp(x)
            x = self.q_func(x)
            return x
        else:
            x, lstm_state = self.get_states(x, lstm_state, done, from_pixels)
            return x, lstm_state
    
    def get_states(self, x, lstm_state, done, from_pixels=True):
        if from_pixels:
            x = x / 255.0
            x = self.cnn(x)
            x = self.mlp(x)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = x.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state
    
    def get_representation(self, x):
        x = x / 255.0
        neurons = []
        clss_to_hook = nn.Conv2d
        for module in self.cnn.children():
            x = module(x)
            if isinstance(module, clss_to_hook):
                neurons.append(
                    ('cnn', F.relu(x.clone()).detach())
                )
                
        clss_to_hook = nn.Linear
        for module in self.mlp.children():
            x = module(x)
            if isinstance(module, clss_to_hook):
                neurons.append(
                    ('mlp', F.relu(x.clone()).detach())
                )

        dead_neurons = self.calculate_dead_neurons(neurons)
        return x, dead_neurons

    def get_Q(self, hidden):
        return self.q_func(hidden)
    
    def calculate_dead_neurons(self, neurons):
        total = {
            "cnn" : 0,
            "mlp" : 0
        }
        dead = {
            "cnn" : 0,
            "mlp" : 0
        }
        
        for layer, ns in neurons:
            score = ns.mean(dim=0)
            mask = score <= 0.0
            total[layer] += torch.numel(mask)
            dead[layer] += (mask.sum().item())
        
        fraction_dead = {
            "cnn" : dead["cnn"]/total["cnn"] * 100,
            "mlp" : dead["mlp"]/total["mlp"] * 100
        }
        return fraction_dead