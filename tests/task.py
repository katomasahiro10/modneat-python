from matplotlib.pyplot import hist
from modneat import visualize
import os
import copy
import time

class xor:
    # The XOR inputs and expected corresponding outputs for fitness evaluation
    def __init__(self, network_type):
        self.network_type = network_type
        self.xor_inputs  = [(1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]
        self.xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]

    def eval_fitness(self, net):
        """
        Arguments:
            net: The feed-forward neural network generated from genome
        Returns:
            The fitness score - the higher score the means the better 
            fit organism. Maximal score: 16.0
        """
        net.reset()
        history = []
        error_sum = 0.0
        history.append(copy.deepcopy(net.__dict__))

        for xi, xo in zip(self.xor_inputs, self.xor_outputs):
            output = net.activate(xi)
            history.append(copy.deepcopy(net.__dict__))
            error_sum += abs(output[0] - xo[0])
        # Calculate amplified fitness
        fitness = (4 - error_sum) ** 2
        return fitness, history

    def eval_genomes(self, genomes, config):
        """
        Arguments:
            genomes: The list of genomes from population in the 
                    current generation
            config: The configuration settings with algorithm
                    hyper-parameters
        """
        for genome_id, genome in genomes:
            genome.fitness = 4.0
            net = self.network_type.create(genome, config)
            genome.fitness, genome.history = self.eval_fitness(net)
    
    def eval_single_genome(self, genome, config):
        """
        Arguments:
            genome: A genome for the evaluation
            config: The configuration settings with algorithm
                    hyper-parameters
        Return value:
            fitness: single float value
            history: information of the genome
        """
        genome.fitness = 4.0
        net = self.network_type.create(genome, config)
        fitness, history = self.eval_fitness(net)
        return fitness, history
    
    def show_results(self, best_genome, config, stats, out_dir):
        # Display the best genome among generations.
        print('\n ************************* Finish evolution *************************  \n')
        print('\nBest genome:\n{!s}'.format(best_genome))

        # Show output of the most fit genome against training data.
        print('\nOutput:')
        net = self.network_type.create(best_genome, config)
        for xi, xo in zip(self.xor_inputs, self.xor_outputs):
            output = net.activate(xi)
            print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

        # Check if the best genome is an adequate XOR solver
        best_genome_fitness, best_genome_history = self.eval_fitness(net)
        if best_genome_fitness > config.fitness_threshold:
            print("\n\nSUCCESS: The XOR problem solver found!!!")
        else:
            print("\n\nFAILURE: Failed to find XOR problem solver!!!")

        # Visualize the experiment results
        node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
        visualize.draw_net(config, best_genome, False, node_names=node_names, directory=out_dir)
        visualize.plot_stats(stats, ylog=False, view=False, filename=os.path.join(out_dir, 'avg_fitness.png'))
        visualize.plot_species(stats, view=False, filename=os.path.join(out_dir, 'speciation.png'))

class non_static(xor):
    # The XOR inputs and expected corresponding outputs for fitness evaluation
    def __init__(self, network_type):
        self.network_type = network_type
        self.xor_inputs  = [(1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.0, 1.0)]
        self.xor_outputs = [   (1.0,),     (0.33,),     (-0.33,),     (-1.0,)]

# This class show the usage of env.gym for modneat
class cartpole_v0:
    def __init__(self, network_type):
        self.env = gym.make('CartPole-v0')
        self.network_type = network_type

    def eval_fitness(self,net):
        fitness = 0.0
        net.reset()
        observation = self.env.reset() #range of the observation is 4.
        for step in range(200):
            action = 0 if net.activate(observation)[0] <= 0.5 else 1
            observation, reward, done, info = self.env.step(action)
            fitness += reward
            if done == True:
                break
        return fitness, 0

    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            net = self.network_type.create(genome, config)
            genome.fitness, genome.history = self.eval_fitness(net) 
    
    def show_results(self, best_genome, config, stats, out_dir):
        # Display the best genome among generations.
        print('\n ************************* Finish evolution *************************  \n')
        print('\nBest genome:\n{!s}'.format(best_genome))

        # Show output of the most fit genome against training data.
        print('\nOutput:')
        net = self.network_type.create(best_genome, config)

        # Check if the best genome is an adequate XOR solver
        best_genome_fitness, best_genome_history = self.eval_fitness(net)
        if best_genome_fitness > config.fitness_threshold:
            print("\n\nSUCCESS: The XOR problem solver found!!!")
        else:
            print("\n\nFAILURE: Failed to find XOR problem solver!!!")

        # Visualize the experiment results
        node_names = {-1:'Cart Position', -2: 'Cart Velocity', -3: 'Pole Angle', -4:'Pole Velocity At Tip', 0:'output'}
        visualize.draw_net(config, best_genome, False, node_names=node_names, directory=out_dir)
        visualize.plot_stats(stats, ylog=False, view=False, filename=os.path.join(out_dir, 'avg_fitness.png'))
        visualize.plot_species(stats, view=False, filename=os.path.join(out_dir, 'speciation.png'))

class time_xor:
    def __init__(self, network_type):
        self.network_type = network_type
        self.xor_inputs  = [(1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]
        self.xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]

    def eval_fitness(self,net):
        #1回のeval_fitnessに0.1秒かかるタスク
        net.reset()
        history = []
        error_sum = 0.0
        history.append(copy.deepcopy(net.__dict__))

        for xi, xo in zip(self.xor_inputs, self.xor_outputs):
            output = net.activate(xi)
            history.append(copy.deepcopy(net.__dict__))
            error_sum += abs(output[0] - xo[0])
        # Calculate amplified fitness
        fitness = (4 - error_sum) ** 2
        time.sleep(0.1)
        return fitness, history

    def eval_genomes(self, genomes, config):
        """
        Arguments:
            genomes: The list of genomes from population in the 
                    current generation
            config: The configuration settings with algorithm
                    hyper-parameters
        """
        for genome_id, genome in genomes:
            genome.fitness = 4.0
            net = self.network_type.create(genome, config)
            genome.fitness, genome.history = self.eval_fitness(net)

    def eval_single_genome(self, genome, config):
        """
        Arguments:
            genome: A genome for the evaluation
            config: The configuration settings with algorithm
                    hyper-parameters
        Return value:
            fitness: single float value
            history: information of the genome
        """
        genome.fitness = 4.0
        net = self.network_type.create(genome, config)
        fitness, history = self.eval_fitness(net)
        return fitness, history
    
    def show_results(self, best_genome, config, stats, out_dir):
        pass