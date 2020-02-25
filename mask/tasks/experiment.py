import argparse
import itertools
import json
import logging
import pprint
from copy import deepcopy
from datetime import datetime as dt
from pathlib import Path
from typing import Dict

import mlflow
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from librosa.core import load
from ml.src.dataloader import set_dataloader, set_ml_dataloader
from ml.src.dataset import ManifestWaveDataSet
from ml.tasks.base_experiment import BaseExperimentor, typical_train, base_expt_args

DATALOADERS = {'normal': set_dataloader, 'ml': set_ml_dataloader}
LABEL2INT = {'clear': 0, 'mask': 1, '?': -1}
AVAILABLE_TARGET = ['normal', 'mean', 'dev']


def mask_expt_args(parser):
    parser = base_expt_args(parser)
    expt_parser = parser.add_argument_group("Mask Experiment arguments")
    expt_parser.add_argument('--target', default='normal', choices=AVAILABLE_TARGET)
    expt_parser.add_argument('--n-waves', help='Number of wave files to make one instance', type=int, default=1)
    expt_parser.add_argument('--shuffle-order', action='store_true', default=False,
                             help='Shuffle wave orders on multiple waves or not')

    return parser


def label_func(row):
    return LABEL2INT[row[1]]


def set_load_func(data_dir, sr, n_waves):
    const_length = sr * 1

    def one_wave_load_func(path):
        wave = load(f'{data_dir}/{path[0]}', sr=sr)[0]

        assert wave.shape[0] == const_length, f'{wave.shape[0]}, {const_length}'
        return wave.reshape((1, -1))

    def multi_waves_load_func(row):
        waves = np.zeros((const_length * n_waves,), dtype=np.float32)
        for i, path in enumerate(row[0].split(',')):
            wave = load(f'{data_dir}/{path}', sr=sr)[0]
            waves[i * const_length:(i + 1) * const_length] = wave

        assert waves.shape[0] == const_length * n_waves, f'{waves.shape[0]}, {const_length * n_waves}'
        return waves.reshape((1, -1))

    if n_waves == 1:
        return one_wave_load_func
    elif n_waves > 1:
        return multi_waves_load_func
    else:
        raise NotImplementedError


def set_data_paths(expt_conf, phases) -> Dict:
    manifest_df = pd.read_csv(expt_conf['manifest_path'])
    if phases == ['train', 'val', 'infer']:
        for phase, part in zip(['train', 'val', 'infer'], ['train', 'devel', 'test']):
            phase_df = manifest_df[manifest_df['file_name'].str.startswith(part)]
            phase_df.to_csv(Path(expt_conf['manifest_path']).parent / f'{phase}_manifest.csv', index=False, header=None)
            expt_conf[f'{phase}_path'] = str(Path(expt_conf['manifest_path']).parent / f'{phase}_manifest.csv')
    elif phases == ['train', 'infer']:
        infer_df = manifest_df[manifest_df['file_name'].str.startswith('test')]
        train_devel_df = manifest_df[~manifest_df.index.isin(infer_df.index)]
        infer_df.to_csv(Path(expt_conf['manifest_path']).parent / f'infer_manifest.csv', index=False, header=None)
        train_devel_df.to_csv(Path(expt_conf['manifest_path']).parent / f'train_manifest.csv', index=False, header=None)

        expt_conf[f'infer_path'] = str(Path(expt_conf['manifest_path']).parent / f'infer_manifest.csv')
        expt_conf[f'train_path'] = str(Path(expt_conf['manifest_path']).parent / f'train_manifest.csv')

    return expt_conf


def dump_dict(path, dict_):
    with open(path, 'w') as f:
        json.dump(dict_, f, indent=4)


def main(expt_conf, hyperparameters, typical_train_func):
    if expt_conf['expt_id'] == 'timestamp':
        expt_conf['expt_id'] = dt.today().strftime('%Y-%m-%d_%H:%M')

    expt_dir = (Path(__file__).resolve().parents[1] / 'output' / f"{expt_conf['expt_id']}")
    expt_dir.mkdir(exist_ok=True)

    logging.basicConfig(level=logging.DEBUG, format="[%(name)s] [%(levelname)s] %(message)s",
                        filename=expt_dir / 'expt.log')
    expt_conf['log_dir'] = str(expt_dir / 'tensorboard')

    expt_conf['class_names'] = [0, 1]
    val_metrics = ['loss', 'uar']

    expt_conf['sample_rate'] = 16000

    dataset_cls = ManifestWaveDataSet

    manifest_df = pd.read_csv(expt_conf['manifest_path'])
    expt_conf = set_data_paths(expt_conf, phases=['train', 'val', 'infer'])

    patterns = list(itertools.product(*hyperparameters.values()))
    val_results = pd.DataFrame(np.zeros((len(patterns), len(hyperparameters) + len(val_metrics))),
                               columns=list(hyperparameters.keys()) + val_metrics)

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(hyperparameters)

    groups = None

    def experiment(pattern, expt_conf):
        for i, param in enumerate(hyperparameters.keys()):
            expt_conf[param] = pattern[i]

        expt_conf['model_path'] = str(expt_dir / f"{'_'.join([str(p).replace('/', '-') for p in pattern])}.pth")
        expt_conf['log_id'] = f"{'_'.join([str(p).replace('/', '-') for p in pattern])}"
        wav_path = Path(expt_conf['manifest_path']).resolve().parents[1] / 'wav'
        load_func = set_load_func(wav_path, expt_conf['sample_rate'], expt_conf['n_waves'])

        with mlflow.start_run():
            mlflow.set_tag('target', expt_conf['target'])
            result_series, val_pred = typical_train_func(expt_conf, load_func, label_func, dataset_cls, groups, val_metrics)

            mlflow.log_params({hyperparameter: value for hyperparameter, value in zip(hyperparameters.keys(), pattern)})
            mlflow.log_artifacts(expt_dir)

        return result_series, val_pred

    # For debugging
    if expt_conf['n_jobs'] == 1:
        result_pred_list = [experiment(pattern, deepcopy(expt_conf)) for pattern in patterns]
    else:
        n_jobs = expt_conf['n_jobs']
        expt_conf['n_jobs'] = 0
        result_pred_list = Parallel(n_jobs=n_jobs, verbose=0)(
            [delayed(experiment)(pattern, deepcopy(expt_conf)) for pattern in patterns])

    val_results.iloc[:, :len(hyperparameters)] = patterns
    result_list = [result for result, pred in result_pred_list]
    val_results.iloc[:, len(hyperparameters):] = result_list
    pp.pprint(val_results)
    pp.pprint(val_results.iloc[:, len(hyperparameters):].describe())

    val_results.to_csv(expt_dir / 'val_results.csv', index=False)
    print(f"Devel results saved into {expt_dir / 'val_results.csv'}")
    for (_, pred), pattern in zip(result_pred_list, patterns):
        pattern_name = f"{'_'.join([str(p).replace('/', '-') for p in pattern])}"
        pd.DataFrame(pred).to_csv(expt_dir / f'{pattern_name}_val_pred.csv', index=False)
        dump_dict(expt_dir / f'{pattern_name}.txt', expt_conf)

    # Train with train + devel dataset
    if expt_conf['train_with_all']:
        best_trial_idx = val_results['uar'].argmax()

        best_pattern = patterns[best_trial_idx]
        for i, param in enumerate(hyperparameters.keys()):
            expt_conf[param] = best_pattern[i]
        dump_dict(expt_dir / 'best_parameters.txt', {p: v for p, v in zip(hyperparameters.keys(), best_pattern)})

        expt_conf['model_path'] = str(expt_dir / f"{'_'.join([str(p).replace('/', '-') for p in best_pattern])}.pth")
        expt_conf = set_data_paths(expt_conf, phases=['train', 'infer'])
        wav_path = Path(expt_conf['manifest_path']).resolve().parents[1] / 'wav'
        load_func = set_load_func(wav_path, expt_conf['sample_rate'], expt_conf['n_waves'])
        experimentor = BaseExperimentor(expt_conf, load_func, label_func, dataset_cls)

        pred = experimentor.experiment_without_validation(seed_average=expt_conf['n_seed_average'])

        sub_name = f"sub_{'_'.join([str(p).replace('/', '-') for p in best_pattern])}.csv"
        if (expt_dir / sub_name).is_file():
            sub_df = pd.read_csv(expt_dir / sub_name)
        else:
            sub_df = manifest_df[manifest_df['file_name'].str.startswith('test')][['file_name']]

        sub_df[expt_conf['target']] = pd.Series(pred).apply(lambda x: list(LABEL2INT.keys())[x])
        sub_df.to_csv(expt_dir / sub_name, index=False)
        print(f"Submission file is saved in {expt_dir / sub_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mask experiment arguments')
    expt_conf = vars(mask_expt_args(parser).parse_args())

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("[%(name)s] [%(levelname)s] %(message)s"))
    console.setLevel(logging.DEBUG)
    logging.getLogger("ml").addHandler(console)

    if 'debug' in expt_conf['expt_id']:
        hyperparameters = {
            'batch_size': [1],
            'model_type': ['panns'],
            'checkpoint_path': ['../cnn14.pth'],
            'window_size': [0.01],
            'window_stride': [0.002],
            'n_waves': [1],
            'epoch_rate': [0.05],
            'mixup_alpha': [0.1],
            'sample_balance': ['same'],
            'time_drop_rate': [0.0],
            'freq_drop_rate': [0.0],
        }
    else:
        hyperparameters = {
            'batch_size': [32],
            'model_type': ['panns'],
            'checkpoint_path': ['../cnn14.pth'],
            'window_size': [[0.2, 0.4, 0.6]],
            'window_stride': [0.005, 0.01, 0.02],
            'n_waves': [1],
            'epoch_rate': [1.0],
            'mixup_alpha': [0.0],
            'sample_balance': ['same'],
            'time_drop_rate': [0.0],
            'freq_drop_rate': [0.0],
        }

    main(expt_conf, hyperparameters, typical_train)

    if expt_conf['expt_id'] == 'debug':
        import shutil
        shutil.rmtree('./mlruns')