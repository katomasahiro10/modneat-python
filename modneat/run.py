from datetime import datetime
import argparse
import sys
import os
import shutil

# The NEAT-Python library imports
import modneat
from modneat import parallel

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, help='testdoc', default='ModFeedForward')
    parser.add_argument('--config', type=str, help='', default='./configs/modexhebb_genome.ini')
    parser.add_argument('--checkpoint_interval', type=int, help='', default=100)
    parser.add_argument('--checkpoint_load', type=str, help='', default='')
    parser.add_argument('--savedir', type=str, help='', default='./results')
    parser.add_argument('--task', type=str, help='', default='non_static')
    parser.add_argument('--generation', type=int, help='', default=100)
    parser.add_argument('--run_id', type=int, help='', default=0)
    parser.add_argument('--num_workers', type=int, help='', default=0)
    parser.add_argument('--description', type=str, help='description of an experiment', default='No description')

    args = parser.parse_args()
    return args

def run_experiment(config_file, num_workers):
    """
    Arguments:
        config_file: the path to the file with experiment configuration
    """


    # Load configuration.
    config = modneat.Config(GENOME_TYPE, modneat.DefaultReproduction,
                         modneat.DefaultSpeciesSet, modneat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    if(CHECKPOINT_LOAD_PATH == ''):
        p = modneat.Population(config)
    else:
        p = modneat.Checkpointer.restore_checkpoint(CHECKPOINT_LOAD_PATH)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(modneat.StdOutReporter(True))
    p.add_reporter(modneat.FileOutReporter(True, out_dir + '/results.txt'))
    stats = modneat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(modneat.Checkpointer(savedir=out_dir, stats=stats, generation_interval=CHECKPOINT_INTERVAL,\
                                        time_interval_seconds=None))

    # Run for up to args.generations.
    if(num_workers == 0 or num_workers == 1):
        best_genome = p.run(TASK.eval_genomes, GENERATION)
    else:
        if(hasattr(TASK, 'eval_single_genome')):
            parallel_evaluator = parallel.ParallelEvaluator(num_workers=num_workers, eval_function=TASK.eval_single_genome)
            best_genome = p.run(parallel_evaluator.evaluate, GENERATION)
        else:
            print(f"Error: {TASK} has no method 'eval_single_genome'.")
            print("please implement 'eval_single_genome' func for multithreading.")
            sys.exit()
    TASK.show_results(best_genome, config, stats, out_dir)


def clean_output():
    if os.path.isdir(out_dir):
        # remove files from previous run
        shutil.rmtree(out_dir)

    # create the output directory
    os.makedirs(out_dir, exist_ok=False)
    os.makedirs(out_dir + '/checkpoints', exist_ok=False)
    os.makedirs(out_dir + '/settings', exist_ok=False)


if __name__ == '__main__':
    exec_command = ' '.join(sys.argv)
    # Get args
    args = create_parser()

    # Load task module from current working directory
    local_dir = os.getcwd()
    sys.path.append(local_dir)
    print("実行ディレクトリ: ", local_dir)
    try:
        import task
    except ImportError as e:
        print(f"Error in run.py: {e} \nCould not import task module.")
        sys.exit()

    NETWORK_TYPE = eval('modneat.nn.' + args.network)
    GENOME_TYPE = NETWORK_TYPE.genome_type()
    TASK = eval('task.' + args.task + '(network_type = NETWORK_TYPE)')
    CONFIG_PATH = os.path.join(local_dir, args.config)
    GENERATION = args.generation
    CHECKPOINT_INTERVAL = args.checkpoint_interval
    CHECKPOINT_LOAD_PATH = args.checkpoint_load
    NUM_WORKERS = args.num_workers

    # The directory to store outputs
    if(CHECKPOINT_LOAD_PATH == ''):
        out_dir = os.path.join(local_dir, args.savedir, args.task + '_' + args.network + '_' + str(args.run_id))
    else:
        out_dir = os.path.join(local_dir, args.savedir, '[CONTINUED]' + args.task + '_' + args.network + '_' + str(args.run_id))
    
    now = datetime.now()
    datestring = now.strftime("%Y%m%d-%H%M%S")
    out_dir = out_dir + '-' + datestring

    # Clean results of previous run if any or init the ouput directory
    clean_output()

    # Save experiment settings
    shutil.copyfile(CONFIG_PATH, out_dir + '/settings/config.ini')
    shutil.copyfile('./task.py', out_dir + '/settings/task.py')
    with open(out_dir + "/description.txt", "w") as file:
        file.write(args.description)
    with open(out_dir + "/settings/command", "w") as file:
        file.write('COMMAND: ' + exec_command + '\n')
        file.write('ARGS: ' + str(args).replace(",", "\n"))

    # Run the experiment
    run_experiment(CONFIG_PATH, NUM_WORKERS)
