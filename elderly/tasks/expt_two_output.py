import argparse
import itertools
import json
import logging
import pprint
from copy import deepcopy
from datetime import datetime as dt
from pathlib import Path
from typing import List

import mlflow
import numpy as np
import pandas as pd
from experiment import LABELS2INT, set_load_func, set_data_paths, get_cv_groups
from joblib import Parallel, delayed
from ml.src.dataset import ManifestWaveDataSet
from ml.tasks.base_experiment import BaseExperimentor, typical_train, base_expt_args


def type_float_list(args) -> List[str]:
    return args.split(',')


def elderly_expt_args(parser):
    parser = base_expt_args(parser)
    expt_parser = parser.add_argument_group("Elderly Experiment arguments")
    expt_parser.add_argument('--target', help='Valence or arousal', default='valence,arousal', type=type_float_list)

    return parser


def set_label_func(targets):
    col_index = {'valence': 5, 'arousal': 6, 'valence_mean': 8, 'valence_dev': 9, 'arousal_mean': 10, 'arousal_dev': 11}
    assert targets[0] in set(col_index.keys()) and targets[1] in set(col_index.keys())

    def label_func(row):
        return row[col_index[targets[0]]], row[col_index[targets[1]]]

    return label_func


def main(expt_conf, hyperparameters):
    if expt_conf['expt_id'] == 'timestamp':
        expt_conf['expt_id'] = dt.today().strftime('%Y-%m-%d_%H:%M')

    expt_dir = (Path(__file__).resolve().parents[1] / 'output' / f"{expt_conf['expt_id']}")
    expt_dir.mkdir(exist_ok=True)

    logging.basicConfig(level=logging.DEBUG, format="[%(name)s] [%(levelname)s] %(message)s",
                        filename=expt_dir / 'expt.log')
    expt_conf['log_dir'] = str(expt_dir / 'tensorboard')

    if expt_conf['task_type'] == 'classify':
        expt_conf['class_names'] = [0, 1, 2]
        val_metrics = ['loss', 'uar', 'loss', 'uar']
        val_metrics_columns = ['v_loss', 'v_uar', 'a_loss', 'a_uar']
    else:
        expt_conf['class_names'] = [0]
        val_metrics = ['loss', 'loss']
        val_metrics_columns = ['target_1_loss', 'target_2_loss']
    expt_conf['sample_rate'] = 16000

    dataset_cls = ManifestWaveDataSet
    label_func = set_label_func(expt_conf['target'])

    manifest_df = pd.read_csv(expt_conf['manifest_path'])
    expt_conf = set_data_paths(expt_conf, phases=['train', 'val', 'infer'])

    patterns = list(itertools.product(*hyperparameters.values()))
    # patterns = [(w_size, w_stride) for w_size, w_stride in patterns if w_size > w_stride]
    val_results = pd.DataFrame(np.zeros((len(patterns), len(hyperparameters) + len(val_metrics_columns))),
                               columns=list(hyperparameters.keys()) + val_metrics_columns)

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
            result_series, val_pred = typical_train(expt_conf, load_func, label_func, dataset_cls, groups, val_metrics)

            mlflow.log_metrics({metric_name: value for metric_name, value in zip(val_metrics_columns, result_series)})
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
        with open(expt_dir / f'{pattern_name}.txt', 'w') as f:
            json.dump(expt_conf, f, indent=4)

    # Train with train + devel dataset
    if expt_conf['train_with_all']:
        if expt_conf['task_type'] == 'classify':
            best_trial_idx = (val_results['v_uar'] + val_results['a_uar']).argmax()
        else:
            best_trial_idx = (val_results['target_1_loss'] + val_results['target_2_loss']).argmin()

        best_pattern = patterns[best_trial_idx]
        for i, param in enumerate(hyperparameters.keys()):
            expt_conf[param] = best_pattern[i]

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
            sub_df = manifest_df[manifest_df['partition'] == 'test'][['filename_text']]

        for i, target in enumerate(expt_conf['target']):
            if expt_conf['task_type'] == 'classify':
                sub_df[target] = pd.Series(pred[:, i]).apply(lambda x: list(LABELS2INT.keys())[x])
            else:
                sub_df[target] = pred[:, i]
        sub_df.to_csv(expt_dir / sub_name, index=False)
        print(f"Submission file is saved in {expt_dir / sub_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train arguments')
    expt_conf = vars(elderly_expt_args(parser).parse_args())

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("[%(name)s] [%(levelname)s] %(message)s"))
    console.setLevel(logging.DEBUG)
    logging.getLogger("ml").addHandler(console)

    expt_conf['n_tasks'] = len(expt_conf['target'])

    if 'valence' not in expt_conf['target'] or 'arousal' not in expt_conf['target']:
        expt_conf['task_type'] = 'regress'

    if expt_conf['expt_id'] == 'debug':
        hyperparameters = {
            'model_type': ['multitask_panns'],
            'batch_size': [1],
            'checkpoint_path': ['../cnn14.pth'],
            'window_size': [0.02],
            'window_stride': [0.01],
            'n_waves': [1],
            'epoch_rate': [0.05],
        }
    else:
        hyperparameters = {
            'model_type': ['multitask_panns'],
            'batch_size': [8],
            'checkpoint_path': ['../cnn14.pth'],
            'window_size': [0.08],
            'window_stride': [0.02],
            'n_waves': [1],
            'epoch_rate': [1.0],
            'mixup_alpha': [0.0, 0.2, 0.4],
        }

    main(expt_conf, hyperparameters)

    if expt_conf['expt_id'] == 'debug':
        import shutil
        shutil.rmtree('./mlruns')