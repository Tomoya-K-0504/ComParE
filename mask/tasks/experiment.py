import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from librosa.core import load
from ml.models.model_manager import BaseModelManager
from ml.src.dataloader import set_dataloader, set_ml_dataloader
from ml.src.metrics import get_metrics
from ml.src.preprocessor import Preprocessor, preprocess_args
from ml.tasks.train_manager import train_manager_args
from ml.src.dataset import ManifestWaveDataSet

DATALOADERS = {'normal': set_dataloader, 'ml': set_ml_dataloader}
LABEL2INT = {'clear': 0, 'mask': 1}


def expt_args(parser):
    train_manager_args(parser)
    expt_parser = parser.add_argument_group("Experiment arguments")
    expt_parser.add_argument('--expt-id', help='data file for training', default='sth')
    expt_parser.add_argument('--n-seed-average', type=int, help='Seed averaging', default=0)
    expt_parser.add_argument('--retrain-traindevel', action='store_true',
                             help='Whether train with train+devel dataset after hyperparameter tuning')

    return parser


def label_func(row):
    return LABEL2INT[row[1]]


def set_load_func(data_dir, sr):
    def load_func(path):
        wave = load(f'{data_dir}/{path[0]}', sr=sr)[0]
        return wave.reshape((1, -1))

    return load_func


def retrain_traindevel(expt_conf, best_pattern):
    val_metrics = []

    if expt_conf['n_seed_average']:
        pred_list = []
        for seed in range(expt_conf['n_seed_average']):
            expt_conf['seed'] = seed
            pred_list.append(experiment(expt_conf, val_metrics, with_validation=False))
        pred = np.array(pred_list).mean(axis=1)
        assert pred.shape[1] == 1
    else:
        pred = experiment(expt_conf, val_metrics, with_validation=False)

    sub_df = pd.read_csv(expt_conf['manifest_path'])
    sub_df = sub_df[sub_df['file_name'].str.startswith('test')]
    sub_df['label'] = pd.Series(pred).apply(lambda x: list(LABEL2INT.keys())[x])
    sub_df.columns = ['file_name', 'prediction']
    (Path(__file__).resolve().parents[1] / 'output' / 'sub').mkdir(exist_ok=True)
    sub_name = f"{expt_conf['expt_id']}_{'_'.join(list(map(str, best_pattern)))}.csv"
    sub_df.to_csv(Path(__file__).resolve().parents[1] / 'output' / 'sub' / sub_name, index=False)


def experiment(expt_conf, val_metrics, with_validation=True):
    phases = ['train', 'val', 'infer'] if with_validation else ['train', 'infer']

    expt_conf['class_names'] = list(LABEL2INT.keys())
    expt_conf['sample_rate'] = 16000

    load_func = set_load_func(Path(expt_conf['manifest_path']).resolve().parents[1] / 'wav', expt_conf['sample_rate'])

    manifest_df = pd.read_csv(expt_conf['manifest_path'])
    if with_validation:
        for phase, part in zip(phases, ['train', 'devel', 'test']):
            phase_df = manifest_df[manifest_df['file_name'].str.startswith(part)]
            phase_df.to_csv(Path(expt_conf['manifest_path']).parent / f'{phase}_manifest.csv', index=False, header=None)
            expt_conf[f'{phase}_path'] = str(Path(expt_conf['manifest_path']).parent / f'{phase}_manifest.csv')
    else:
        infer_df = manifest_df[manifest_df['file_name'].str.startswith('test')]
        train_devel_df = manifest_df[~manifest_df.index.isin(infer_df.index)]
        infer_df.to_csv(Path(expt_conf['manifest_path']).parent / f'infer_manifest.csv', index=False, header=None)
        train_devel_df.to_csv(Path(expt_conf['manifest_path']).parent / f'train_manifest.csv', index=False, header=None)
        expt_conf[f'infer_path'] = str(Path(expt_conf['manifest_path']).parent / f'infer_manifest.csv')
        expt_conf[f'train_path'] = str(Path(expt_conf['manifest_path']).parent / f'train_manifest.csv')

    dataloaders = {}
    for phase in phases:
        process_func = Preprocessor(expt_conf, phase, expt_conf['sample_rate']).preprocess
        dataset = ManifestWaveDataSet(expt_conf[f'{phase}_path'], expt_conf, load_func, process_func, label_func, phase)
        dataloaders[phase] = DATALOADERS['normal'](dataset, phase, expt_conf)

    train_metrics = get_metrics(['loss', 'uar'])
    val_metrics = get_metrics(val_metrics, target_metric='loss')
    metrics = {'train': train_metrics, 'val': val_metrics}

    model_manager = BaseModelManager(expt_conf['class_names'], expt_conf, dataloaders, metrics)

    metrics = model_manager.train(with_validation=with_validation)
    pred = model_manager.infer()

    [Path(expt_conf[f'{phase}_path']).unlink() for phase in phases]

    if with_validation:
        return metrics['val'], pred
    else:
        return pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train arguments')
    expt_conf = vars(expt_args(preprocess_args(parser)).parse_args())
    assert expt_conf['train_path'] != '' or expt_conf['val_path'] != '', \
        'You need to select training, validation data file to training, validation in --train-path, --val-path argments'

    hyperparameters = {
        'lr': [0.0001, 0.001],
        # 'batch_size': [16, 64],
    }

    val_metrics = ['loss', 'uar', 'f1', 'accuracy']

    patterns = list(itertools.product(*hyperparameters.values()))
    val_results = pd.DataFrame(np.zeros((len(patterns), len(hyperparameters) + len(val_metrics))),
                               columns=list(hyperparameters.keys()) + val_metrics)

    for i, pattern in enumerate(patterns):
        val_results.iloc[i, :len(hyperparameters)] = pattern
        print(f'Pattern: \n{val_results.iloc[i, :len(hyperparameters)]}')

        for idx, param in enumerate(hyperparameters.keys()):
            expt_conf[param] = pattern[idx]

        if expt_conf['n_seed_average']:
            result_list = []
            for seed in range(expt_conf['n_seed_average']):
                expt_conf['seed'] = seed
                result_list.append(experiment(expt_conf, val_metrics))
        else:
            results_metrics, pred = experiment(expt_conf, val_metrics)
            for metric in results_metrics:
                val_results.loc[i, metric.name] = metric.average_meter.best_score

    (Path(__file__).resolve().parents[1] / 'output' / 'metrics').mkdir(exist_ok=True)
    expt_path = Path(__file__).resolve().parents[1] / 'output' / 'metrics' / f"{expt_conf['log_id']}.csv"
    print(val_results)
    print(val_results.iloc[:, len(hyperparameters):].describe())
    val_results.to_csv(expt_path, index=False)

    # Train with train + devel dataset
    if expt_conf['retrain_traindevel']:
        best_trial_idx = val_results['uar'].argmax()
        best_pattern = patterns[best_trial_idx]

        for idx, param in enumerate(hyperparameters.keys()):
            expt_conf[param] = best_pattern[idx]

        retrain_traindevel(expt_conf, best_pattern)
