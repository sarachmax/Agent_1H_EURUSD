# -*- coding: utf-8 -*-
"""
Created on Mon May 28 21:00:56 2018

@author: sarac
"""
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import LSTM
from keras.layers import Dropout 


class DQNAgent:
    def __init__(self, state_size, inp_shape):
        self.memory = deque(maxlen=4000)
        self.input_shape = inp_shape
        self.learning_rate = 0.001
        self.state_size = state_size
        self.epsilon = 1.0  # exploration rate
        self.gamma = 0.95   # discount rate 
        self.action_size = 3    
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        
    
    
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()

        model.add(LSTM(120, return_sequences = True, input_shape = self.input_shape, activation='sigmoid'))
        model.add(Dropout(0.2)) 

        model.add(LSTM(72, return_sequences = True,  activation='sigmoid'))
        model.add(Dropout(0.2)) 

        model.add(LSTM(24, activation='sigmoid'))
        model.add(Dropout(0.2)) 

        model.add(Dense(self.action_size, activation='tanh'))

        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
   
    def act(self, state, train = True, random_action = False):
        if train : 
            if np.random.rand() <= self.epsilon or random_action :
                print('random action')
                return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        print('act : ', act_values)
        return np.argmax(act_values[0]) #return action 
        
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done :
                target[0][action] = reward
            else : 
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma*np.amax(t)
                
            self.model.fit(state, target, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min : 
            self.epsilon *= self.epsilon_decay
        return target
        
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
        
    