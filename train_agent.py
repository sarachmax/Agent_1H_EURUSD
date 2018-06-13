# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 12:12:11 2018

@author: sarac
"""

from EURUSD_AGENT import DQNAgent
import time
import numpy as np
import pandas as pd 
#import random

EPISODES = 300
MARGIN = 1000

start_index = 12702    #2010.01.04 
end_index = 56055  #2016.12.30 
dataset = pd.read_csv('EURUSD_1H.csv')
train_data = dataset.iloc[start_index:end_index,2:6]

train_data = np.array(train_data)
state_size = 24 
num_data = state_size
feed_data = [] 
all_index = end_index-start_index
for i in range(num_data, all_index):
    feed_data.append(train_data[i-num_data:i,:])
feed_data = np.array(feed_data)

input_shape = (feed_data.shape[1], feed_data.shape[2])

class Environment:
    def __init__(self, data, num_index, size):
        self.state_data = data
        self.state_index = 0 
        self.end_index = num_index-1
        self.loss_limit = 0.3 # force sell 
        self.profit_limit = 0.05
        self.state_size = size-1
        

        self.profit = 0
        self.reward = 0
        self.mem_reward = 0
        
        # portfolio 
        self.cost_price = 0 
        self.mem_action = 0    
    
    def reset(self):
        self.state_index = 0 
        self.profit = 0
        self.reward = 0 
        self.cost_price = 0 
        self.mem_action = 0
        self.mem_reward = 0
        init_state = self.state_data[self.state_index]
        return [init_state]
    
    def get_action(self, action):
        if action == 1 :
            # buy 
            return 1
        elif action == 2 : 
            # sell 
            return -1
        else : 
            # noaction 
            return 0 
    
    def calculate_reward(self, action):
        action = self.get_action(action)
        current_price = self.state_data[self.state_index, self.state_size, 3]
        if action == self.mem_action :
            self.profit = action*(current_price - self.cost_price)
            self.reward = self.mem_reward + self.profit
            print('new/mem action : ', action, ' / ', self.mem_action)
        else :  
            if action == 0 : 
                self.profit = self.mem_action*(current_price - self.cost_price)    
            else :
                self.profit = current_price*(-0.001) + self.mem_action*(current_price - self.cost_price)
            self.reward = self.profit + self.mem_reward
            self.mem_reward = self.reward 
            self.cost_price = current_price
            self.profit = 0
            print('new/mem action : ', action, ' / ', self.mem_action)
            self.mem_action = action
    
    def done_check(self):
        if self.cost_price != 0 : 
            loss = -self.loss_limit*self.cost_price
        else : 
            loss = -self.loss_limit*self.state_data[self.state_index,  self.state_size, 3]
        if self.state_index + 1 == self.end_index :
            if self.reward > 0 : 
                if self.reward <= 0.05*self.state_data[self.state_index,  self.state_size, 3]:
                    self.reward = -1
            print('Full End !')
            return True 
        elif self.reward <= loss : 
            print('------------------------------------------------------------')
            print('loss limit: ', loss)
            print('reward : ', self.reward)
            print('Cut Loss !')
            self.reward = -3
            return True
        else :
            return False
        
        
    def step(self,action):
        skip = 6   
        self.state_index += skip
        if self.state_index >= self.end_index-1 : 
            self.state_index = self.end_index-1 
        ns = self.state_data[self.state_index]
        if (self.profit >= self.profit_limit*self.cost_price and self.profit > 0 ) or (self.profit < -(self.profit_limit*self.cost_price)):
            if self.get_action(action) != 0 :
                self.calculate_reward(0) 
                done_action = 0
                print("CLOSE POSITION BY LIMITS !")
            else : 
                self.calculate_reward(action)
                done_action = action
                print("CLOSR POSITION BY AGENT !")
        else : 
            self.calculate_reward(action)
            done_action = action
        done = self.done_check()
        self.envi_status(done_action)
        return [ns], self.reward*MARGIN, done_action, done   
    
    def envi_status(self, action):
        print("--------------------------------------------------------------")
        print('Index : ', self.state_index, '/', self.end_index)
        print('Reward : ', self.reward*MARGIN)
        print('LastAction : ', self.mem_action)
        print('CurrentProfit : ', self.profit*MARGIN)
        print('Agent Action : ', self.get_action(action))
        print('Progress : ', self.state_index, '/', self.end_index, ' : ', self.state_index/self.end_index*100 ,'%')
        print("--------------------------------------------------------------")
        
#########################################################################################################
# Train     
#########################################################################################################       
if __name__ == "__main__":
    
    agent = DQNAgent(state_size, input_shape)
    # agent.save("agent_model.h5")
    
    num_index = all_index - state_size
    env = Environment(feed_data, num_index, state_size)
    
    batch_size = 12 
    best_reward = -300
     
    
    for e in range(EPISODES):
        
        state = env.reset()
        state = np.reshape(state, (1, feed_data.shape[1], feed_data.shape[2]))
        for t in range(end_index-start_index):
            start_time = time.time()
            action = agent.act(state)
             
            next_state, reward, action, done = env.step(action)
            next_state = np.reshape(next_state, (1, feed_data.shape[1], feed_data.shape[2]))
            agent.remember(state, action, reward, next_state, done)
            state = next_state 
            if done:
                agent.update_target_model()
                print('----------------------------- Episode Result -----------------------')
                print("episode: {}/{}, time: {}, e: {:.4}"
                      .format(e+1, EPISODES, t, agent.epsilon))
                print('----------------------------- End Episode --------------------------')
                if reward >= best_reward :
                    best_reward = reward
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            
            end_time = time.time()
            used_time = end_time - start_time 
            print('Episode : ', e, '/', EPISODES, '   time : ', used_time)
            
        agent.save("agent_model.h5")
        
    #agent.save("agent_model.h5")

    print('train done')
    print('BEST RESULT ==================================')
    print("best reward : ", best_reward)
    
            
