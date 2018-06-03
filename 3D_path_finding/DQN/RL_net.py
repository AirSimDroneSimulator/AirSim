import numpy as np
import keras
from DQN.RL_dataset import Dataset

def pack_example(x):
	return x

def unpack_example(x):
	return x

init_settings = {}
init_settings["learning_rate"] = 0.01
init_settings["reward_decay"] = 0.9
init_settings["e_greedy"] = 0.9
init_settings["e_greedy_increasement"] = 0
init_settings["memory_length"] = 512
init_settings["batch_size"] = 32
init_settings["epochs"] = 1
init_settings["replace_target_iter"] = 256
init_settings["model"] = None
init_settings["pack_fun"] = pack_example
init_settings["unpack_fun"] = unpack_example

class DQNClass:
    '''
    def __init__(
        self,
        n_actions,#action
        n_features,#feature
        learning_rate=0.01,
        reward_decay=0.9,
        e_greedy=0.9,
        e_greedy_increasement=0,
        memory_length=512,
        batch_size=32,
        epochs=1,
        replace_target_iter=256,
        model = None
    ):
    '''
    def __init__(self, settings):
        self.settings = init_settings.copy()
        self.settings.update(settings)
        
        self.n_actions = self.settings["n_actions"]      #actions number
        self.n_features = self.settings["n_features"]    #feature number
        self.lr = self.settings["learning_rate"]         
        self.gamma = self.settings["reward_decay"]       
        self.epsilon_max = self.settings["e_greedy"]     
        self.epsilon = 0 
        self.e_greedy_increasement = self.settings["e_greedy_increasement"]
        self.memory_length = self.settings["memory_length"]
        self.batch_size = self.settings["batch_size"]
        self.epochs = self.settings["epochs"]
        self.replace_target_iter = self.settings["replace_target_iter"]
        
        
        self.pack = self.settings["pack_fun"]
        self.unpack = self.settings["unpack_fun"]
        
        self.training_counter = 0
        self.dataset = Dataset(memory_length = self.memory_length,memory_size = self.n_features*2+2)
        if self.e_greedy_increasement == 0:
            self.epsilon = self.epsilon_max
        
        if self.settings["model"] == None:
            model = keras.models.Sequential()
            model.add(keras.layers.Dense(100, activation='relu', input_dim=self.n_features))
            model.add(keras.layers.Dense(100, activation='relu'))
            model.add(keras.layers.Dense(100, activation='relu'))
            model.add(keras.layers.Dense(100, activation='relu'))
            model.add(keras.layers.Dense(self.n_actions))
            model.compile(keras.optimizers.Adam(lr=self.lr), 'mse')
            self.model = model
        else:
            self.model = self.settings["model"]
        self.target_model = keras.models.clone_model(self.model)
        
   
    def choose_action(self,observation,test = False):
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon or test:
            actions_value = self.model.predict(observation)
            action = np.argmax(actions_value)
            #print (self.epsilon,'arg')
        else:
            action = np.random.randint(0, self.n_actions)
            #print("rand")
        return action
    
    def add_data(self,s,a,r,s_):
        transition = np.hstack((s, [a, r], s_))

        counter = self.dataset.add(transition)
        return counter
    
    def learn(self,times = 1):
        res = ''
        
        if self.training_counter % (self.replace_target_iter) == 0:
            #self.target_model = keras.models.clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())
            res += 'target net replaced'
            print(res)
        
        self.training_counter+=1
                
        for i in range(times):
            
            tdata = self.dataset.next_batch(self.batch_size*times)
            
            s = tdata[:,0:self.n_features]
            a = tdata[:,self.n_features]
            r = tdata[:,self.n_features+1]
            s_ = tdata[:,-self.n_features:]
            #print (s_)
            q_next = self.target_model.predict(s_)
            q_value = self.target_model.predict(s)
            
            q_target = q_value
            
            batch_index = range(self.batch_size*times)
            action_index = a.astype(int)
            q_target[batch_index,action_index] = r + self.gamma * np.max(q_next,axis = 1)
            
            train_X = s
            train_Y = q_target
            self.epsilon = min(self.epsilon+self.e_greedy_increasement,self.epsilon_max)
            ver = 0
            
            
            self.model.fit(train_X,train_Y,epochs = self.epochs,verbose=ver)
        #self.model.train_on_batch(train_X,train_Y)
        return res
        
        
        
        
        
        
        
        
        