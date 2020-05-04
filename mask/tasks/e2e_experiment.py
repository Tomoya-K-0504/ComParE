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
import torch
from joblib import Parallel, delayed
from librosa.core import load
from ml.preprocess.preprocessor import Preprocessor
from ml.src.dataloader import set_dataloader, set_ml_dataloader
from ml.src.dataset import ManifestWaveDataSet
from ml.tasks.base_experiment import BaseExperimentor, typical_train, base_expt_args, get_metric_list
from ml.utils.notify_slack import notify_slack
from tqdm import tqdm

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
    expt_parser.add_argument('--n-parallel', default=1, type=int)
    expt_parser.add_argument('--only-test', action='store_true')

    return parser


def label_func(row):
    return LABEL2INT[row[1]]


def set_load_func(data_dir, sr, n_waves, rnn=False):
    const_length = sr * 1
    n_features = 500 if rnn else 1

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
        return waves.reshape((n_features, -1))

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


class LoadDataSet(ManifestWaveDataSet):
    def __init__(self, manifest_path, data_conf, phase='train', load_func=None, transform=None, label_func=None):
        super(LoadDataSet, self).__init__(manifest_path, data_conf, phase, load_func, transform, label_func)

    def __getitem__(self, idx):
        try:
            x = torch.load(self.path_df.iloc[idx, 0].replace('.wav', '.pt'))
        except FileNotFoundError as e:
            print(e)
            return super().__getitem__(idx)
        # print(x.size())
        label = self.labels[idx]

        return x, label


def parallel_logmel(expt_conf, load_func, label_func):
    def parallel_preprocess(dataset, idx):
        processed, _ = dataset[idx]
        path = dataset.path_df.iloc[idx, 0]
        torch.save(processed.to('cpu'), str(Path('input/processed') / path.replace('.wav', '.pt')))

    for phase in tqdm(['train', 'infer']):
        process_func = Preprocessor(expt_conf, phase).preprocess
        dataset = ManifestWaveDataSet(expt_conf[f'{phase}_path'], expt_conf, phase, load_func, process_func,
                                      label_func)
        Parallel(n_jobs=8, verbose=0)(
            [delayed(parallel_preprocess)(dataset, idx) for idx in range(len(dataset))])
        print(f'{phase} done')


def main(expt_conf, hyperparameters, typical_train_func):
    if expt_conf['expt_id'] == 'timestamp':
        expt_conf['expt_id'] = dt.today().strftime('%Y-%m-%d_%H:%M')

    expt_dir = (Path(__file__).resolve().parents[1] / 'output' / f"{expt_conf['expt_id']}")
    expt_dir.mkdir(exist_ok=True)

    logging.basicConfig(level=logging.DEBUG, format="[%(name)s] [%(levelname)s] %(message)s",
                        filename=expt_dir / 'expt.log')
    expt_conf['log_dir'] = str(expt_dir / 'tensorboard')

    expt_conf['class_names'] = [0, 1]
    metrics_names = {'train': ['loss', 'uar'], 'val': ['loss', 'uar'], 'infer': []}

    expt_conf['sample_rate'] = 16000

    manifest_df = pd.read_csv(expt_conf['manifest_path'])
    expt_conf = set_data_paths(expt_conf, phases=['train', 'val', 'infer'])
    dataset_cls = ManifestWaveDataSet

    patterns = list(itertools.product(*hyperparameters.values()))
    val_results = pd.DataFrame(np.zeros((len(patterns), len(hyperparameters) + len(metrics_names['val']))),
                               columns=list(hyperparameters.keys()) + metrics_names['val'])

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(hyperparameters)

    groups = None

    def experiment(pattern, expt_conf):
        for i, param in enumerate(hyperparameters.keys()):
            expt_conf[param] = pattern[i]

        expt_conf['model_path'] = str(expt_dir / f"{'_'.join([str(p).replace('/', '-') for p in pattern])}.pth")
        expt_conf['log_id'] = f"{'_'.join([str(p).replace('/', '-') for p in pattern])}"
        wav_path = Path(expt_conf['manifest_path']).resolve().parents[1] / 'wav'
        load_func = set_load_func(wav_path, expt_conf['sample_rate'], expt_conf['n_waves'], expt_conf['model_type'] == 'rnn')

        with mlflow.start_run():
            mlflow.set_tag('target', expt_conf['target'])
            result_series, val_pred, _ = typical_train_func(expt_conf, load_func, label_func, process_func=None,
                                                            dataset_cls=dataset_cls, groups=groups)

            mlflow.log_params({hyperparameter: value for hyperparameter, value in zip(hyperparameters.keys(), pattern)})
            mlflow.log_artifacts(expt_dir)

        return result_series, val_pred

    if expt_conf['only_test']:
        val_results = pd.read_csv(expt_dir / 'val_results.csv')
    else:
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

    best_trial_idx = val_results['uar'].argmax()
    best_pattern = patterns[best_trial_idx]
    for i, param in enumerate(hyperparameters.keys()):
        expt_conf[param] = best_pattern[i]
    dump_dict(expt_dir / 'best_parameters.txt', {p: v for p, v in zip(hyperparameters.keys(), best_pattern)})

    # Train with train + devel dataset
    phases = ['train', 'infer']
    if expt_conf['infer']:
        expt_conf['model_path'] = str(expt_dir / f"{'_'.join([str(p).replace('/', '-') for p in best_pattern])}.pth")
        expt_conf = set_data_paths(expt_conf, phases=['train', 'infer'])
        wav_path = Path(expt_conf['manifest_path']).resolve().parents[1] / 'wav'
        load_func = set_load_func(wav_path, expt_conf['sample_rate'], expt_conf['n_waves'], expt_conf['model_type'] == 'rnn')

        # parallel_logmel(expt_conf, load_func, label_func)

        dataset_cls = ManifestWaveDataSet

        experimentor = BaseExperimentor(expt_conf, load_func, label_func, process_func=None, dataset_cls=dataset_cls)

        metrics = {p: get_metric_list(metrics_names[p]) for p in phases}
        _, pred = experimentor.experiment_without_validation(metrics, seed_average=expt_conf['n_seed_average'])
        expt_conf['model_path'] = str(expt_dir / f"{'_'.join([str(p).replace('/', '-') for p in best_pattern])}_all.pth")
        experimentor.train_manager.model_manager.save_model()

        sub_name = f"sub_{'_'.join([str(p).replace('/', '-') for p in best_pattern])}.csv"
        if (expt_dir / sub_name).is_file():
            sub_df = pd.read_csv(expt_dir / sub_name)
        else:
            sub_df = manifest_df[manifest_df['file_name'].str.startswith('test')][['file_name']].reset_index(drop=True)
        sub_df['prediction'] = pd.Series(pred['infer']).apply(lambda x: list(LABEL2INT.keys())[x])
        sub_df.to_csv(expt_dir / sub_name, index=False)
        print(f"Submission file is saved in {expt_dir / sub_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mask experiment arguments')
    expt_conf = vars(mask_expt_args(parser).parse_args())

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("[%(name)s] [%(levelname)s] %(message)s"))
    console.setLevel(logging.INFO)
    logging.getLogger("ml").addHandler(console)

    if 'debug' in expt_conf['expt_id']:
        hyperparameters = {
            'lr': [1e-3],
            'batch_size': [2],
            'model_type': ['logmel_cnn'],
            'transform': ['logmel'],
            # 'checkpoint_path': ['../cnn14.pth'],
            'window_size': [0.101],
            'window_stride': [0.02],
            'n_waves': [1],
            'epoch_rate': [0.05],
            'mixup_alpha': [0.0],
            'sample_balance': [[1.0, 1.0, 1.0]],
            'time_drop_rate': [0.0],
            'freq_drop_rate': [0.0],
        }
    elif expt_conf['model_type'] == 'cnn':
        hyperparameters = {
            'model_type': ['cnn'],
            'transform': [None],
            'cnn_channel_list': [[4, 8, 16, 32]],
            'cnn_kernel_sizes': [[[4]] * 4],
            'cnn_stride_sizes': [[[4]] * 4],
            'cnn_padding_sizes': [[[0]] * 4],
            'lr': [1e-4, 1e-5],
        }
    elif expt_conf['model_type'] == 'cnn_rnn':
        hyperparameters = {
            'lr': [1e-4, 1e-5],
            'transform': [None],
            'cnn_channel_list': [[4, 8, 16, 32]],
            'cnn_kernel_sizes': [[[4]] * 4],
            'cnn_stride_sizes': [[[2]] * 4],
            'cnn_padding_sizes': [[[1]] * 4],
            'rnn_type': [expt_conf['rnn_type']],
            'bidirectional': [True],
            'rnn_n_layers': [2],
            'rnn_hidden_size': [10, 50],
        }
    elif expt_conf['model_type'] == 'rnn':
        hyperparameters = {
            'bidirectional': [True, False],
            'rnn_type': ['lstm', 'gru'],
            'rnn_n_layers': [2],
            'rnn_hidden_size': [10, 50],
            'transform': [None],
            'lr': [1e-4, 1e-5],
        }
    else:
        hyperparameters = {
            'lr': [1e-3, 1e-4],
            'batch_size': [256],
            'model_type': ['cnn'],
            'transform': [None],
            'kl_penalty': [0.0],
            'entropy_penalty': [0.0],
            'loss_func': ['ce'],
            # 'checkpoint_path': ['../cnn14.pth'],
            'window_size': [0.05],
            'window_stride': [0.002],
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
    else:
        cfg = dict(
            body=f"Finished experiment {expt_conf['expt_id']}: \n" +
                 "Notion ticket: https://www.notion.so/mask-23037943c2f24e4795a48b6daa573206",
            webhook_url='https://hooks.slack.com/services/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'
        )
        notify_slack(cfg)
