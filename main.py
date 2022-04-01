# from federal.simulation.simMain import main
# from federal.simulation.trainMain import main as train_main
from federal.simulation.suff_train import main
from visual.test_unit import main
import signal
signal.signal(signal.SIGPIPE, signal.STG_IIGN)  # 忽略SIGPIPE信号

if __name__ == "__main__":
    main()
