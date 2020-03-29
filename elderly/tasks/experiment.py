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
from elderly_dataset import ManifestMultiWaveDataSet
from joblib import Parallel, delayed
from librosa.core import load
from ml.src.dataloader import set_dataloader, set_ml_dataloader
from ml.tasks.base_experiment import BaseExperimentor, typical_train, base_expt_args, get_metric_list
from ml.utils.notify_slack import notify_slack
from smoothing import smoothing

DATALOADERS = {'normal': set_dataloader, 'ml': set_ml_dataloader}
LABELS2INT = {'L': 0, 'M': 1, 'H': 2}
AVAILABLE_TARGET = ['valence', 'arousal', 'valence_mean', 'arousal_mean', 'valence_dev', 'arousal_dev']


def elderly_expt_args(parser):
    parser = base_expt_args(parser)
    expt_parser = parser.add_argument_group("Elderly Experiment arguments")
    expt_parser.add_argument('--target', help='Valence or arousal', default='valence', choices=AVAILABLE_TARGET)
    expt_parser.add_argument('--n-waves', help='Number of wave files to make one instance', type=int, default=1)
    expt_parser.add_argument('--shuffle-order', action='store_true', default=False,
                             help='Shuffle wave orders on multiple waves or not')
    expt_parser.add_argument('--n-parallel', default=1, type=int)
    expt_parser.add_argument('--smoothing', default='majority', choices=['majority'])

    return parser


def set_label_func(target):
    col_index = {'valence': 5, 'arousal': 6, 'valence_mean': 8, 'valence_dev': 9, 'arousal_mean': 10, 'arousal_dev': 11}
    assert set(col_index.keys()) == set(AVAILABLE_TARGET)
    def label_func(row):
        return row[col_index[target]]

    return label_func


def cut_pad_wave(wave, const_length):
    if wave.shape[0] == const_length:
        return wave.reshape((1, -1))

    elif wave.shape[0] > const_length:
        diff = wave.shape[0] - const_length
        wave = wave[diff // 2:-diff // 2]
    else:
        n_pad = (const_length - wave.shape[0]) // 2
        wave = np.pad(wave, n_pad)[:const_length]

    return wave


def set_load_func(data_dir, sr, n_waves):
    const_sec = 5
    const_length = sr * const_sec

    def one_wave_load_func(path):
        wave = load(f'{data_dir}/{path[0]}', sr=sr)[0]
        wave = cut_pad_wave(wave, const_length)

        assert wave.shape[0] == const_length, f'{wave.shape[0]}, {const_length}'
        return wave.reshape((1, -1))

    def multi_waves_load_func(row):
        waves = np.zeros((const_length * n_waves, ), dtype=np.float32)
        for i, path in enumerate(row[0].split(',')):
            wave = load(f'{data_dir}/{path}', sr=sr)[0]
            waves[i * const_length:(i + 1) * const_length] = cut_pad_wave(wave, const_length)

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
            phase_df = manifest_df[manifest_df[f'partition'].str.startswith(part)]
            phase_df.to_csv(Path(expt_conf['manifest_path']).parent / f'{phase}_manifest.csv', index=False, header=None)
            expt_conf[f'{phase}_path'] = str(Path(expt_conf['manifest_path']).parent / f'{phase}_manifest.csv')
    elif phases == ['train', 'infer']:
        infer_df = manifest_df[manifest_df['partition'] == 'test']
        train_devel_df = manifest_df[~manifest_df.index.isin(infer_df.index)]
        infer_df.to_csv(Path(expt_conf['manifest_path']).parent / f'infer_manifest.csv', index=False, header=None)
        train_devel_df.to_csv(Path(expt_conf['manifest_path']).parent / f'train_manifest.csv', index=False, header=None)

        expt_conf[f'infer_path'] = str(Path(expt_conf['manifest_path']).parent / f'infer_manifest.csv')
        expt_conf[f'train_path'] = str(Path(expt_conf['manifest_path']).parent / f'train_manifest.csv')

    return expt_conf


def get_cv_groups(expt_conf):
    manifest_df = pd.read_csv(expt_conf['manifest_path'])
    train_val_manifest = manifest_df[manifest_df['partition'] != 'test']
    subjects = pd.DataFrame(train_val_manifest['filename_text'].str.slice(0, -4).unique())
    subjects['group'] = [j % expt_conf['n_splits'] for j in range(len(subjects))]
    subjects = subjects.set_index(0)
    groups = train_val_manifest['filename_text'].str.slice(0, -4).apply(lambda x: subjects.loc[x, 'group'])
    return groups


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

    if expt_conf['task_type'] == 'classify':
        expt_conf['class_names'] = [0, 1, 2]
        metrics_names = {'train': ['loss', 'uar'], 'val': ['loss', 'uar'], 'infer': []}
    else:
        expt_conf['class_names'] = [0]
        metrics_names = {'train': ['loss'], 'val': ['loss'], 'test': ['loss']}
    expt_conf['sample_rate'] = 16000

    dataset_cls = ManifestMultiWaveDataSet
    label_func = set_label_func(expt_conf['target'])

    manifest_df = pd.read_csv(expt_conf['manifest_path'])
    expt_conf = set_data_paths(expt_conf, phases=['train', 'val', 'infer'])

    patterns = list(itertools.product(*hyperparameters.values()))
    # patterns = [pattern for pattern in patterns if pattern[4] > pattern[5]]
    val_results = pd.DataFrame(np.zeros((len(patterns), len(hyperparameters) + len(metrics_names['val']))),
                               columns=list(hyperparameters.keys()) + metrics_names['val'])

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(hyperparameters)

    groups = None
    if expt_conf['cv_name'] == 'group':
        groups = get_cv_groups(expt_conf)

    def experiment(pattern, expt_conf):
        for i, param in enumerate(hyperparameters.keys()):
            expt_conf[param] = pattern[i]

        expt_conf['model_path'] = str(expt_dir / f"{'_'.join([str(p).replace('/', '-') for p in pattern])}.pth")
        expt_conf['log_id'] = f"{'_'.join([str(p).replace('/', '-') for p in pattern])}"
        wav_path = Path(expt_conf['manifest_path']).resolve().parents[1] / 'wav'
        load_func = set_load_func(wav_path, expt_conf['sample_rate'], expt_conf['n_waves'])

        with mlflow.start_run():
            mlflow.set_tag('target', expt_conf['target'])
            result_series, val_pred, _ = typical_train_func(expt_conf, load_func, label_func, process_func=None,
                                                         dataset_cls=dataset_cls, groups=groups)

            mlflow.log_params({hyperparameter: value for hyperparameter, value in zip(hyperparameters.keys(), pattern)})
            mlflow.log_artifacts(expt_dir)

        return result_series, val_pred

    # For debugging
    if expt_conf['n_parallel'] == 1:
        result_pred_list = [experiment(pattern, deepcopy(expt_conf)) for pattern in patterns]
    else:
        expt_conf['n_jobs'] = 0
        result_pred_list = Parallel(n_jobs=expt_conf['n_parallel'], verbose=0)(
            [delayed(experiment)(pattern, deepcopy(expt_conf)) for pattern in patterns])

    val_results.iloc[:, :len(hyperparameters)] = [[str(param) for param in p] for p in patterns]
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
    phases = ['train', 'infer']
    if expt_conf['infer']:
        if expt_conf['task_type'] == 'classify':
            best_trial_idx = val_results['uar'].argmax()
        else:
            best_trial_idx = val_results['loss'].argmin()

        best_pattern = patterns[best_trial_idx]
        for i, param in enumerate(hyperparameters.keys()):
            expt_conf[param] = best_pattern[i]
        dump_dict(expt_dir / 'best_parameters.txt', {p: v for p, v in zip(hyperparameters.keys(), best_pattern)})

        expt_conf['model_path'] = str(expt_dir / f"{'_'.join([str(p).replace('/', '-') for p in best_pattern])}.pth")
        expt_conf = set_data_paths(expt_conf, phases=['train', 'infer'])
        wav_path = Path(expt_conf['manifest_path']).resolve().parents[1] / 'wav'
        load_func = set_load_func(wav_path, expt_conf['sample_rate'], expt_conf['n_waves'])
        experimentor = BaseExperimentor(expt_conf, load_func, label_func, process_func=None, dataset_cls=dataset_cls)

        metrics = {p: get_metric_list(metrics_names[p]) for p in phases}
        _, pred = experimentor.experiment_without_validation(metrics, seed_average=expt_conf['n_seed_average'])

        sub_name = f"sub_{'_'.join([str(p).replace('/', '-') for p in best_pattern])}.csv"
        if (expt_dir / sub_name).is_file():
            sub_df = pd.read_csv(expt_dir / sub_name)
        else:
            sub_df = manifest_df[manifest_df['partition'] == 'test'][['filename_text']].reset_index(drop=True)

        if expt_conf['task_type'] == 'classify':
            sub_df[expt_conf['target']] = pd.Series(pred['infer']).apply(lambda x: list(LABELS2INT.keys())[x])
        else:
            sub_df[expt_conf['target']] = pred['infer']

        sub_df = smoothing(sub_df, expt_conf['target'], expt_conf['smoothing'])
        sub_df.to_csv(expt_dir / sub_name, index=False)
        print(f"Submission file is saved in {expt_dir / sub_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train arguments')
    expt_conf = vars(elderly_expt_args(parser).parse_args())

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("[%(name)s] [%(levelname)s] %(message)s"))
    console.setLevel(logging.DEBUG)
    logging.getLogger("ml").addHandler(console)

    if expt_conf['target'] not in ['arousal', 'valence']:
        expt_conf['task_type'] = 'regress'

    if 'debug' in expt_conf['expt_id']:
        hyperparameters = {
            'lr': [1e-3],
            'batch_size': [1],
            'model_type': ['logmel_cnn'],
            'transform': ['logmel'],
            # 'checkpoint_path': ['../cnn14.pth'],
            'window_size': [0.101],
            'window_stride': [0.1],
            'n_waves': [1],
            'epoch_rate': [0.05],
            'mixup_alpha': [0.0],
            'sample_balance': [[1.0, 1.0, 1.0]],
            'time_drop_rate': [0.0],
            'freq_drop_rate': [0.0],
        }
    else:
        hyperparameters = {
            'lr': [1e-3, 1e-4],
            'batch_size': [16],
            'model_type': ['logmel_cnn'],
            'transform': ['logmel'],
            #'checkpoint_path': ['../cnn14.pth'],
            'window_size': [0.05],
            'window_stride': [0.01],
            'n_waves': [1],
            'epoch_rate': [1.0],
            'mixup_alpha': [0.0],
            'sample_balance': [None],
            'time_drop_rate': [0.0],
            'freq_drop_rate': [0.0],
        }
    if expt_conf['target'] == 'valence':
        hyperparameters['sample_balance'] = [[1, 1, 1]]
        hyperparameters['window_size'] = [0.01]
        hyperparameters['window_stride'] = [0.002]
    else:
        hyperparameters['sample_balance'] = ['same']
        hyperparameters['window_size'] = [0.05]
        hyperparameters['window_stride'] = [0.005]

    main(expt_conf, hyperparameters, typical_train)

    if expt_conf['expt_id'] == 'debug':
        import shutil
        shutil.rmtree('./mlruns')
    else:
        cfg = dict(
            body=f"Finished experiment {expt_conf['expt_id']}: \n" +
                 "Notion ticket: https://www.notion.so/c2fad3ade9d941588335cb56eafaf27a",
            webhook_url='https://hooks.slack.com/services/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'
        )
        notify_slack(cfg)
