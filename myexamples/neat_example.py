import sys
import os
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# The NEAT-Python library imports
import modneat
# The helper used to visualize experiment results
import task

NETWORK_TYPE = modneat.nn.FeedForwardNetwork
GENOME_TYPE = modneat.DefaultGenome

TASK = task.xor(network_type = NETWORK_TYPE)

# The current working directory
local_dir = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(local_dir, './config/config.ini')

# The directory to store outputs
out_dir = os.path.join(local_dir, 'out')

def run_experiment(config_file):
    """
    Arguments:
        config_file: the path to the file with experiment configuration
    """
    # Load configuration.
    config = modneat.Config(GENOME_TYPE, modneat.DefaultReproduction,
                         modneat.DefaultSpeciesSet, modneat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = modneat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(modneat.StdOutReporter(True))
    p.add_reporter(modneat.FileOutReporter(True, out_dir + '/neat_result.txt'))
    stats = modneat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(modneat.Checkpointer(5, filename_prefix='out/neat-checkpoint-'))

    # Run for up to 100 generations.
    best_genome = p.run(TASK.eval_genomes, 100)
    TASK.show_results(best_genome, config, stats, out_dir)


def clean_output():
    if os.path.isdir(out_dir):
        # remove files from previous run
        shutil.rmtree(out_dir)

    # create the output directory
    os.makedirs(out_dir, exist_ok=False)


if __name__ == '__main__':
    # Clean results of previous run if any or init the ouput directory
    clean_output()

    # Run the experiment
    run_experiment(CONFIG_PATH)
