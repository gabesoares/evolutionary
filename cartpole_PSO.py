import gym
import numpy as np


# HYPERPARAMETERS
GENERATIONS = 500  # max number of generations
hidden_neurons = 2  # number of neurons in hidden layer
input_size = 4  # 4 states as the inputs to the NN
swarm_size = 10  # number of chromosomes in each population
Pc = 0.6  # crossover rate
Pm = 0.1  # mutation rate
crossed_parents = 10  # number of chromosomes that go through crossover, needs to be even

# need to set up cartpole environment

def main():
    env = gym.make('CartPole-v0')
    env.seed(0)
    ob_space = env.observation_space
    obs = env.reset()  # obs holds the state variables [x xdot theta theta_dot]
    # might need to scale data so doesn't saturate the neurons, can use z-scaling or other,
    # makes sure the data has zero mean and unit variance
    reward = 0
    success_num = 0
    current_fitness = np.zeros(swarm_size)

    chromosome_pool = generate_swarm()  # initial pool of chromosomes
    best_particle_positions = chromosome_pool
    best_particle_fitnesses = np.zeros(swarm_size)

    best_swarm_position = best_particle_positions[0][:]
    best_swarm_fitness = [0]

    for generation in range(GENERATIONS):

        # update mating pool via selection, crossover, and mutation
        # keep the best X chromosomes from the previous generation
        if generation != 0:  # i.e. this is not the first initial population

            # update best swarm position and fitness

            for i in range(swarm_size):
                if current_fitness[i] > best_swarm_fitness:  # if the current particle has a better fitness than the best
                    best_swarm_fitness = current_fitness[i]  # update the new best fitness
                    best_swarm_position = chromosome_pool[i][:]  # update the new best swarm position

            # update best individual particle positions
            for i in range(swarm_size):
                if current_fitness[i] > best_particle_fitnesses[i]:
                    best_particle_fitnesses[i] = current_fitness[i]
                    best_particle_positions[i][:] = chromosome_pool[i][:]

            # update particle velocity

            #  2. selection
            r = np.random.uniform(0, 1, swarm_size)  # create list of random numbers for selection

            a = np.array([0])
            # a.shape = (1, 1)
            cp_augmented = np.concatenate((a, cp), axis=0)
            hist, bin_edges = np.histogram(r, cp_augmented)
            selected_population = []
            empty = 1

            for i in range(swarm_size):
                for j in range(hist[i]):
                    if empty == 1:
                        selected_population = chromosome_pool[i][:]
                        empty = 0
                    else:
                        selected_population = np.vstack((selected_population, chromosome_pool[i][:]))

            # might be a good thing to check if keeping the top 4 or 2 works better

            #  3. crossover
            r = np.random.uniform(0, 1, swarm_size)  # create list of random numbers for crossover

            indices = np.argwhere(r > Pc)
            parent_count = 0
            empty = 1;
            i = 0
            crossover_pairs = []

            while i < swarm_size:
                if r[i] < Pc:
                    # include ith chromosome
                    if empty == 1:
                        crossover_pairs = selected_population[i][:]
                        crossover_pairs.shape = (1, 10)
                        empty = 0
                    else:
                        crossover_pairs = np.vstack((crossover_pairs, selected_population[i][:]))

                    if np.size(crossover_pairs, 0) == crossed_parents:
                        break  # we have enough parents for crossover
                i = i + 1
                if i == (swarm_size) and np.size(crossover_pairs, 0) < crossed_parents:
                    i = 0  # we don't have enough yet, so restart the loop
                    #  generate new random number list
                    r = np.random.uniform(0, 1, population_size)  # create list of random numbers for crossover

            offspring = crossover(crossover_pairs)

            #  4. mutation
            r = np.random.uniform(0, 1, population_size*(hidden_neurons*input_size + hidden_neurons))
            offspring.shape = (1, population_size*(hidden_neurons*input_size + hidden_neurons))

            for m in range(np.size(r, 0)):
                if r[m] < Pm:
                    random_value = np.random.uniform(-1.0, 1.0, 1)
                    offspring[0][m] += random_value

            offspring.shape = (population_size, hidden_neurons * input_size + hidden_neurons)
            #  5. new mating pool finalized

            chromosome_pool = offspring

        for iteration in range(swarm_size):  # episode
            observations = []
            actions = []
            rewards = []
            while True:  # run each action which is much less than episode length

                # function to determine correct action given observation
                # it will only produce a PROBABILITY of moving left or right, this is a STOCHASTIC policy
                # we will then sample from this distribution using random # [0,1]
                act = nn_forward(obs, chromosome_pool[iteration])  # current chromosome in the generation
                if act >= 0.5:
                    act = 1
                else:
                    act = 0
                # act = np.asscalar(act)

                observations.append(obs)
                actions.append(act)
                rewards.append(reward)

                env.render()

                next_obs, reward, done, info = env.step(act)
                z = sum(rewards)

                done = obs[0] < -2.4 \
                       or obs[0] > 2.4 \
                       or obs[2] < -45 * 2 * 3.14159 / 360 \
                       or obs[2] > 45 * 2 * 3.14159 / 360
                done = bool(done)

                if done:
                    obs = env.reset()
                    print('Generation: ', generation)
                    print('Chromosome: ', iteration)
                    print('Fitness: ', sum(rewards))
                    reward = -1
                    current_fitness[iteration] = sum(rewards)
                    break
                else:
                    obs = next_obs

            if sum(rewards) >= 300:
                success_num += 1
                if success_num >= 100:
                    print('Iteration: ', iteration)
                    print('Clear!!')
                    current_fitness[iteration] = sum(rewards)
                    break
            else:
                success_num = 0

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]

def nn_forward(x, chromosome): # takes the state as input
    # condition the chromosome into matrix shape to do neural net forward pass
    # chromosome.shape = (hidden_neurons+1, input_size)  # + 1 to include the row of output weights from hidden layer to output
    w1 = chromosome[0:hidden_neurons*input_size]
    w1.shape = (hidden_neurons, input_size)
    w2 = chromosome[hidden_neurons*input_size:]

    #  w1 and w2 now ready to do forward pass
    h = np.dot(w1, x)
    h[h < 0] = 0  # ReLU nonlinearity, could switch to smthg else later (tanh, sigmoid, etc..)
    logp = np.dot(w2, h)  # log probability
    p = sigmoid(logp) #  probability of moving right. sigmoid nonlinearity squashes output to [0,1]
    return p

def generate_swarm():  # generates initial population for GA
    # need 10 initial chromosomes
    swarm = []
    # population.shape = (hidden_neurons+1, input_size)
    for i in range(swarm_size):
        # initialize the weights (no biases) using Xavier initialization
        w1 = np.random.randn(hidden_neurons * input_size) / np.sqrt(input_size)
        w1.shape = (1, hidden_neurons * input_size)
        w2 = np.random.randn(hidden_neurons) / np.sqrt(hidden_neurons)
        w2.shape = (1, hidden_neurons)

        a = np.concatenate((w1, w2), axis=1)
        if i == 0:
            swarm = a
        else:
            swarm = np.vstack((swarm, a))

    return swarm

def crossover(crossover_pairs):  # performs crossover on the current population
    selector = np.random.random_integers(1, swarm_size)  # used to select which parents to crossover from existing population
    selector = 1
    offspring = np.zeros([np.size(crossover_pairs, 0), np.size(crossover_pairs, 1)])
    for i in range(int(crossed_parents/2)):
        temp = crossover_pairs[2*i][selector:]
        temp2 = crossover_pairs[2*i+1][selector:]

        offspring[2*i][selector:] = temp2
        a = 2*i + 1
        offspring[a][selector:] = temp

    for i in range(np.size(crossover_pairs, 0)):
        offspring[i][0] = crossover_pairs[i][0]

    return offspring

if __name__ == '__main__':
    main()