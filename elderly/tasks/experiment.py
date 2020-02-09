import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from librosa.core import load
from ml.src.dataloader import set_dataloader, set_ml_dataloader
from ml.tasks.base_experiment import base_expt_args, BaseExperimentor, CrossValidator, SeedAverager

DATALOADERS = {'normal': set_dataloader, 'ml': set_ml_dataloader}


def elderly_expt_args(parser):
    parser = base_expt_args(parser)
    expt_parser = parser.add_argument_group("Elderly Experiment arguments")
    expt_parser.add_argument('--target', help='Valence or arousal', choices=['valence', 'arousal'])

    return parser


def set_label_func(target_col):
    def label_func(row):
        return row[target_col]

    return label_func


def set_load_func(data_dir, sr):
    const_sec = 5
    const_length = sr * const_sec

    def load_func(path):
        wave = load(f'{data_dir}/{path[0]}', sr=sr)[0]

        if wave.shape[0] == const_length:
            return wave.reshape((1, -1))

        elif wave.shape[0] > const_length:
            diff = wave.shape[0] - const_length
            wave = wave[diff // 2:-diff // 2]
        else:
            n_pad = (const_length - wave.shape[0]) // 2
            wave = np.pad(wave, n_pad)[:const_length]

        assert wave.shape[0] == const_length, wave.shape[0]
        return wave.reshape((1, -1))


    return load_func


def train_with_all(val_results, hyperparameter_list, expt_conf, experimentor):
    best_trial_idx = val_results['uar'].argmax()
    best_pattern = patterns[best_trial_idx]

    for idx, param in enumerate(hyperparameter_list):
        expt_conf[param] = best_pattern[idx]

    manifest_df = pd.read_csv(expt_conf['manifest_path'])
    infer_df = manifest_df[manifest_df['file_name'].str.startswith('test')]
    train_devel_df = manifest_df[~manifest_df.index.isin(infer_df.index)]
    infer_df.to_csv(Path(expt_conf['manifest_path']).parent / f'infer_manifest.csv', index=False, header=None)
    train_devel_df.to_csv(Path(expt_conf['manifest_path']).parent / f'train_manifest.csv', index=False, header=None)
    experimentor.cfg[f'infer_path'] = str(Path(expt_conf['manifest_path']).parent / f'infer_manifest.csv')
    experimentor.cfg[f'train_path'] = str(Path(expt_conf['manifest_path']).parent / f'train_manifest.csv')

    pred = experimentor.experiment_without_validation()

    sub_df = pd.read_csv(expt_conf['manifest_path'])
    sub_df = sub_df[sub_df['file_name'].str.startswith('test')]
    sub_df['label'] = pd.Series(pred).apply(lambda x: list(LABEL2INT.keys())[x])
    sub_df.columns = ['file_name', 'prediction']
    (Path(__file__).resolve().parents[1] / 'output' / 'sub').mkdir(exist_ok=True)
    sub_name = f"{expt_conf['expt_id']}_{'_'.join(list(map(str, best_pattern)))}.csv"
    sub_df.to_csv(Path(__file__).resolve().parents[1] / 'output' / 'sub' / sub_name, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train arguments')
    expt_conf = vars(elderly_expt_args(parser).parse_args())
    assert expt_conf['train_path'] != '' or expt_conf['val_path'] != '', \
        'You need to select training, validation data file to training, validation in --train-path, --val-path argments'

    hyperparameters = {
        'lr': [0.0001, 0.001],
        # 'batch_size': [16, 64],
    }

    expt_conf['class_names'] = [0, 1, 2]
    expt_conf['sample_rate'] = 16000

    load_func = set_load_func(Path(expt_conf['manifest_path']).resolve().parents[1] / 'wav', expt_conf['sample_rate'])
    target_col = 5 if expt_conf['sample_rate'] == 'valence' else 6
    label_func = set_label_func(target_col)

    val_metrics = ['loss', 'uar', 'f1']

    manifest_df = pd.read_csv(expt_conf['manifest_path'])
    for phase, part in zip(['train', 'val', 'infer'], ['train', 'devel', 'test']):
        phase_df = manifest_df[manifest_df[f'partition'].str.startswith(part)]
        phase_df.to_csv(Path(expt_conf['manifest_path']).parent / f'{phase}_manifest.csv', index=False, header=None)
        expt_conf[f'{phase}_path'] = str(Path(expt_conf['manifest_path']).parent / f'{phase}_manifest.csv')

    patterns = list(itertools.product(*hyperparameters.values()))
    val_results = pd.DataFrame(np.zeros((len(patterns), len(hyperparameters) + len(val_metrics))),
                               columns=list(hyperparameters.keys()) + val_metrics)

    for i, pattern in enumerate(patterns):
        val_results.iloc[i, :len(hyperparameters)] = pattern
        print(f'Pattern: \n{val_results.iloc[i, :len(hyperparameters)]}')

        for idx, param in enumerate(hyperparameters.keys()):
            expt_conf[param] = pattern[idx]

        groups = None
        # TODO below
        if expt_conf['cv_name'] == 'group_k_fold':
            raise NotImplementedError

        if expt_conf['n_seed_average']:
            experimentor = SeedAverager(expt_conf, load_func, label_func, expt_conf['cv_name'], expt_conf['n_splits'],
                                        groups)
        elif expt_conf['cv_name']:
            experimentor = CrossValidator(expt_conf, load_func, label_func, expt_conf['cv_name'], expt_conf['n_splits'],
                                          groups)
        else:
            experimentor = BaseExperimentor(expt_conf, load_func, label_func)

        result_series, pred = experimentor.experiment_with_validation(val_metrics)
        val_results.loc[i, len(hyperparameters):] = result_series

    (Path(__file__).resolve().parents[1] / 'output' / 'metrics').mkdir(exist_ok=True)
    expt_path = Path(__file__).resolve().parents[1] / 'output' / 'metrics' / f"{expt_conf['expt_id']}.csv"
    print(val_results)
    print(val_results.iloc[:, len(hyperparameters):].describe())
    val_results.to_csv(expt_path, index=False)

    # Train with train + devel dataset
    if expt_conf['train_with_all']:
        train_with_all(val_results, list(hyperparameters.keys()), expt_conf, experimentor)
