import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQNAgent:
    def __init__(self, seed_num):
        seed = seed_num
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        # features as input
        self.state_shape = 12
        self.layer_width = 54
        self.dropout = 0.3
        self.memory = deque(maxlen=100000)
        self.gamma = 0.99  # discount rate
        self.epsilon = 1   # exploration rate
        self.epsilon_min = 0.1 # minimum exploration rate
        # initialise models and optimisation techniques
        self.model = self.build_model().to(self.device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.01)

    # build model architecture
    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_shape, self.layer_width),
            nn.ReLU(),
            nn.Dropout(self.dropout), 

            nn.Linear(self.layer_width, self.layer_width),
            nn.ReLU(),
            nn.Dropout(self.dropout), 

            nn.Linear(self.layer_width, self.layer_width),
            nn.ReLU(),
            nn.Dropout(self.dropout), 

            nn.Linear(self.layer_width, self.layer_width),
            nn.ReLU(),
            
            nn.Linear(self.layer_width, 1)
        )
        return model

    # remember experiences
    def remember(self, state, reward, next_state):
        self.memory.append((state, reward, next_state))

    # epsilon greedy action selection
    def act(self, next_states):
        next_states = torch.FloatTensor(next_states).to(self.device)
        if np.random.rand() <= self.epsilon:
            return random.randrange(next_states.size(0))
        act_values = self.model(next_states).detach().cpu().numpy()
        return np.argmax(act_values[:, 0])

    # batch replay with exploration decay
    def replay(self, batch_size, epsilon_decay=0.999):
        # sample batch
        batch_sample = random.sample(self.memory, batch_size)
        y_values = []
        x_values = []
        # get states from batch
        for state, reward, next_state in batch_sample:
            # create a list of the states so we can test the accuracy of the model's prediction of the reward from each state
            x_values.append(state)
            # formatting
            next_state = torch.FloatTensor(np.array(next_state)).unsqueeze(0).to(self.device)
            # prediction of future reward from next state to include in target state reward
            estimate_optimal_q_value = self.model(next_state).detach().cpu().numpy()[0]
            # target values to compare with model's prediction accuracy (including the discounted future reward)
            y_values.append(reward + self.gamma * int(estimate_optimal_q_value))
    
        x_values = torch.FloatTensor(np.array(x_values)).to(self.device)
        y_values = torch.FloatTensor(np.array(y_values)).unsqueeze(1).to(self.device)
        
        # zero out gradients so don't include old gradients
        self.optimizer.zero_grad()
        # model predictions of state's value using state values
        model_prediction = self.model(x_values)
        # compare prediciton with actual value from that state
        loss = self.loss_fn(model_prediction, y_values)
        # backwards propagation
        loss.backward()
        # parameter update
        self.optimizer.step()
        
        # exploration decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= epsilon_decay

