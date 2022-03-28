from federal.FLnodes import *
from federal.simulation.CVH import CVHRun


def main():
    # parser = PruningFLParser()
    # args = parser.parse()
    # GLOBAL_LOGGER.info("All initialized. Experiment is {}. Client selection = {}. "
    #       "Num users = {}. Seed = {}. Max round = {}. "
    #       "Target density = {}".format(CIFAR10_NAME, args.use_adaptive, args.initial_pruning, args.client_selection,
    #                                    workers, args.seed, MAX_ROUND, args.target_density))
    dic = {"cs": 10}
    args = argparse.Namespace(**dic)
    fl_runner = CVHRun(args)
    fl_runner.download_dict()
    fl_runner.download_cp_rate()
    fl_runner.upload_download_ranks()
    fl_runner.valid_acc()

    
