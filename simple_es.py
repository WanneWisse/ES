#get y value
def f(x):
    #return 2*x**2
    return x * np.sin(x**2) + 1

import numpy as np

x = np.linspace(-3,3,100)
y = f(x)

import matplotlib.pyplot as plt

xpoints = x
ypoints = y

# Initialize figure and axis
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
scatter = ax.scatter([], [])  # Empty scatter plot
line, = ax.plot(xpoints, ypoints, color='red')  # Empty line plot

def es():
    u = 1.0
    std = 0.5
    pop_size = 100
    elite_size = 10
    generations = 100
    for _ in range(generations):
        pop = np.random.normal(u,std,pop_size)
        #hillclimb problem
        fitness_pop = -f(pop)
        #print(fitness_pop)
        #print((fitness_pop - np.mean(fitness_pop))/np.std(fitness_pop))
        direction_per_solution = (fitness_pop - np.mean(fitness_pop))/np.std(fitness_pop)
        u = u + 0.01 * np.dot(pop.T,direction_per_solution)
        print(u)
        
        # sorted_fitness_pop = np.argsort(fitness_pop)[::-1]
        # elite_pop = pop[sorted_fitness_pop][0:elite_size]

        #plot code
        scatter.set_offsets(np.column_stack((pop, -fitness_pop)))
        plt.pause(0.5)

        #u = np.sum(elite_pop)/elite_size
    
es()

plt.ioff()  # Turn off interactive mode after plotting is done
plt.show()


