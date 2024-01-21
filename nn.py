import numpy as np
import gym

class NeuralNetwork():
    def __init__(self,layers) -> None:
        self.w = []
        self.b = []
        for layeri in range(len(layers)):
            if (layeri + 1) < len(layers):
                wrows = layers[layeri+1]
                wcols = layers[layeri]
                bsize = layers[layeri+1]
                w = np.random.random((wcols,wrows))
                b = np.random.random((1,bsize))
                self.w.append(w)
                self.b.append(b)
    
    def feed_forward(self,sample):
        result = sample
        for wi in range(len(self.w)):
            result = np.dot(result, self.w[wi]) +self.b[wi]
            if wi != len(self.w) - 1:
                result = self.ReLU(result)
        return result
    
    def get_structure(self):
        structure_w = []
        structure_b = []
        for w in self.w:
            structure_w.append(w.shape)
        for b in self.b:
            structure_b.append(b.shape)
        return structure_w, structure_b
    def flatten(self):
        flatten_w = []
        flatten_b = []
        for w in self.w:
            flatten_w += list(w.flatten())
        for b in self.b:
            flatten_b += list(b.flatten())
        return flatten_w,flatten_b
    def ReLU(self,x):
        return x * (x > 0)
    def rebuild(self,fw,sw,fb,sb):
        start = 0
        for i in range(len(sw)):
            end = start + sw[i][0]*sw[i][1]
            array = np.array(fw[start:end])
            self.w[i] = array.reshape(sw[i])
            start = end
        start = 0
        for i in range(len(sb)):
            end = start + sb[i][0]*sb[i][1]
            array = np.array(fb[start:end])
            self.b[i] = array.reshape(sb[i])
            start = end
class ES():
    def __init__(self,w_size,b_size) -> None:
        self.wu = np.random.random((w_size,1))
        self.bu = np.random.random((b_size,1))

    def adapt(self,wu_pop,bu_pop,fitness):
        mean_difference = (fitness - np.mean(fitness))
        if np.all(mean_difference != 0):
            direction_per_solution = mean_difference/np.std(fitness)
        else:
            direction_per_solution = np.zeros(len(fitness))
        direction_per_solution = np.array([direction_per_solution])
        print(direction_per_solution)
        self.wu += 0.001 * np.dot(wu_pop.T,direction_per_solution.T)
        self.bu += 0.001 * np.dot(bu_pop.T,direction_per_solution.T)
    def create_solutions(self,pop_size):
        new_solutions_wu = []
        new_solutions_bu = []
        for wu in self.wu:
            new_solutions_wu.append(np.random.normal(wu,0.1,pop_size))
        new_solutions_wu = np.array(new_solutions_wu).T
        for bu in self.bu:
            new_solutions_bu.append(np.random.normal(bu,0.1,pop_size))
        new_solutions_bu = np.array(new_solutions_bu).T
        return new_solutions_wu,new_solutions_bu
    
def run_episode(name,policy):
    env = gym.make(name)
    observation, info = env.reset(seed=42)
    total_reward = 0
    while True:
        
        observation, reward, terminated, truncated, info = env.step(np.argmax(policy.feed_forward(observation)))
        if terminated or truncated:
            env.close()
            return total_reward
        total_reward += 1

game = 'CartPole-v1'
env = gym.make(game, render_mode="human")
input_space = env.observation_space.shape[0]
action_space = env.action_space.n
nn = NeuralNetwork([input_space,5,5,action_space])

sw,sb = nn.get_structure()
fw,fb = nn.flatten()
es = ES(len(fw),len(fb))

for i in range(100):
    new_solutions_wu,new_solutions_bu = es.create_solutions(500)
    total_fitness = []
    for i in range(len(new_solutions_wu)):
        nn.rebuild(new_solutions_wu[i],sw,new_solutions_bu[i],sb)
        fitness = run_episode(game,nn)
        total_fitness.append(fitness)
    
    es.adapt(new_solutions_wu,new_solutions_bu,np.array(total_fitness))
    print(np.max(total_fitness))
    print(np.mean(total_fitness))


