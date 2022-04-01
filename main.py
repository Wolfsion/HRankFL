# from federal.simulation.simMain import main
# from federal.simulation.trainMain import main as train_main
from federal.simulation.suff_train import main
#from visual.test_unit import main
import sys
from signal import signal, SIGPIPE, SIG_DFL

if __name__ == "__main__":
    # 让 python 忽略 SIGPIPE 信号，并且不抛出异常
    # signal(SIGPIPE, SIG_DFL)
    main()
