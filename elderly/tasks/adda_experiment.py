import argparse
import logging

from experiment import main, elderly_expt_args
from ml.tasks.adda_experiment import typical_adda_train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train arguments')
    expt_conf = vars(elderly_expt_args(parser).parse_args())

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("[%(name)s] [%(levelname)s] %(message)s"))
    console.setLevel(logging.DEBUG)
    logging.getLogger("ml").addHandler(console)

    if expt_conf['expt_id'] == 'debug':
        hyperparameters = {
            'target': ['valence', 'arousal'],
            'batch_size': [2],
            'model_type': ['panns'],
            'checkpoint_path': ['../cnn14.pth'],
            'window_size': [0.02],
            'window_stride': [0.01],
            'n_waves': [1],
            'epoch_rate': [0.2],
            'mixup_alpha': [0.1],
        }
    else:
        hyperparameters = {
            'target': ['valence', 'arousal'],
            'batch_size': [8],
            'model_type': ['panns'],
            'checkpoint_path': ['../cnn14.pth'],
            'window_size': [0.08],
            'window_stride': [0.02],
            'n_waves': [1],
            'epoch_rate': [1.0],
            'mixup_alpha': [0.0, 0.2, 0.4],
        }

    main(expt_conf, hyperparameters, typical_adda_train)

    if expt_conf['expt_id'] == 'debug':
        import shutil
        shutil.rmtree('./mlruns')