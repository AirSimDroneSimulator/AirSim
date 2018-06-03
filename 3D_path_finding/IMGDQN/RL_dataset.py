import numpy as np


class Dataset:
    def __init__(self,memory_length = 1024,memory_size=4):
        self.memory_length = memory_length
        self.memory_size = memory_size
        self.memory_counter = 0
        self.memory = np.zeros((self.memory_length,self.memory_size))
    
    def add(self,data):
        if data.ndim == 1:
            data = np.array([data])
            
        for i in range(data.shape[0]):
            self.memory[self.memory_counter % self.memory_length,:]=data[i]
            self.memory_counter+=1
            
            
        return self.memory_counter
        
    def next_batch(self,batch_size):
        index_size = min(self.memory_counter,self.memory_length)
        
        index = np.random.choice(index_size,batch_size)
        
        return self.memory[index,:]
        
    def memory_init(self):
        self.memory_counter = 0
        return 
    
        
    