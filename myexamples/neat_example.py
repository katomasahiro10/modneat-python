#
# This file provides source code of XOR experiment using on NEAT-Python library
#

# The Python standard library import
import sys
import os
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# The NEAT-Python library imports
import modneat
# The helper used to visualize experiment results
import visualize
import task

NETWORK_TYPE = modneat.nn.FeedForwardNetwork
GENOME_TYPE = modneat.DefaultGenome

TASK = task.xor()

# The current working directory
local_dir = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(local_dir, './config/config.ini')

# The directory to store outputs
out_dir = os.path.join(local_dir, 'out')

def run_experiment(config_file):
    """
    The function to run XOR experiment against hyper-parameters 
    defined in the provided configuration file.
    The winner genome will be rendered as a graph as well as the
    important statistics of neuroevolution process execution.
    Arguments:
        config_file: the path to the file with experiment 
                    configuration
    """
    # Load configuration.
    config = modneat.Config(GENOME_TYPE, modneat.DefaultReproduction,
                         modneat.DefaultSpeciesSet, modneat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = modneat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(modneat.StdOutReporter(True))
    stats = modneat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(modneat.Checkpointer(5, filename_prefix='out/neat-checkpoint-'))

    # Run for up to 100 generations.
    best_genome = p.run(TASK.eval_genomes, 100)

    # Display the best genome among generations.
    print('\nBest genome:\n{!s}'.format(best_genome))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    net = NETWORK_TYPE.create(best_genome, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    # Check if the best genome is an adequate XOR solver
    best_genome_fitness = eval_fitness(net)
    if best_genome_fitness > config.fitness_threshold:
        print("\n\nSUCCESS: The XOR problem solver found!!!")
    else:
        print("\n\nFAILURE: Failed to find XOR problem solver!!!")

    # Visualize the experiment results
    node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    visualize.draw_net(config, best_genome, False, node_names=node_names, directory=out_dir)
    visualize.plot_stats(stats, ylog=False, view=False, filename=os.path.join(out_dir, 'avg_fitness.svg'))
    visualize.plot_species(stats, view=False, filename=os.path.join(out_dir, 'speciation.svg'))

def clean_output():
    if os.path.isdir(out_dir):
        # remove files from previous run
        shutil.rmtree(out_dir)

    # create the output directory
    os.makedirs(out_dir, exist_ok=False)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.

    # Clean results of previous run if any or init the ouput directory
    clean_output()

    # Run the experiment
    run_experiment(CONFIG_PATH)
