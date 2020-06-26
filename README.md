# evolutionary
reinforcement learning with openAI gym cartpole and evolutionary optimizations like genetic algorithms and particle swarms

The cartpole simulation can move either left or right, and terminates as soon as the pendulum has fallen past a certain angle. 

There is a counter that counts up and gets higher the longer the simulation lasts (i.e. the longer the pendulum stays relatively upright). 

The goal of this algorithm is thus to output the right sequences of left/right signals to the cart, (with the angle as input) in order to keep the pendulum upright. 

The algorithm has no idea it is a pendulum. 

The neural network used is a 3-layer network with weights initialized randomly, and then trained via the reward function. 

The counter acts as a reward function, i.e. the longer the pendulum has stayed upright, the higher the reward for that set of weights network. 

The optimization algorithm used is a self-written Genetic Algorithm. So network weights with higher rewards will be promoted as stronger strains in the genetic algorithm, and network weights with lower rewards will not. 
