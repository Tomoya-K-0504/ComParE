import argparse
import argparse
import itertools
import logging
import pprint
from copy import deepcopy
from datetime import datetime as dt
from pathlib import Path
from typing import Dict

import mlflow
import numpy as np
import pandas as pd
from experiment import LABELS2INT, dump_dict, AVAILABLE_TARGET
from joblib import Parallel, delayed
from ml.src.dataset import CSVDataSet
from ml.tasks.base_experiment import BaseExperimentor, typical_train, base_expt_args, get_metric_list

TARGET2COL = {'valence': 'V_cat_no', 'arousal': 'A_cat_no'}


def expt_args(parser):
    parser = base_expt_args(parser)
    expt_parser = parser.add_argument_group("Elderly Experiment arguments")
    expt_parser.add_argument('--target', help='Valence or arousal', default='valence', choices=AVAILABLE_TARGET)
    expt_parser.add_argument('--n-waves', help='Number of wave files to make one instance', type=int, default=1)
    expt_parser.add_argument('--shuffle-order', action='store_true', default=False,
                             help='Shuffle wave orders on multiple waves or not')
    expt_parser.add_argument('--n-parallel', default=1, type=int)

    return parser


def load_func(path):
    part = path.split('.')[-2]
    df = pd.read_csv(path)
    if part == 'test':
        return df.iloc[:, 1:].values, np.zeros((df.shape[0],))

    labels = pd.read_csv(Path(path).resolve().parents[1] / 'lab' / 'labels.csv')
    target_col = path.split('.')[-3]
    labels = labels.drop_duplicates(subset='filename_text').set_index('filename_text')
    return df.iloc[:, 1:].values, labels.loc[df['filename'], target_col].values.astype(int)


def process_func(x):
    # TODO
    # return (x - mean) / std
    return x


def set_data_paths(expt_dir, expt_conf, phases) -> Dict:
    data_dir = expt_dir.parents[1] / 'features'
    file_prefix = 'ComParE2020_USOMS-e.frozen-bert-gmax'
    target = expt_conf['target']
    if phases == ['train', 'val', 'infer']:
        for phase, part in zip(['train', 'val', 'infer'], ['train', 'devel', 'test']):
            expt_conf[f'{phase}_path'] = str(data_dir / f'{file_prefix}.{TARGET2COL[target]}.{part}.csv')

    elif phases == ['train', 'infer']:
        train_df = pd.concat([pd.read_csv(expt_conf[f'{phase}_path']) for phase in ['train', 'val']])
        train_df.to_csv(data_dir / f'{file_prefix}.{TARGET2COL[target]}.train_val.csv', index=False)
        expt_conf['train_path'] = str(data_dir / f'{file_prefix}.{TARGET2COL[target]}.train_val.csv')
        expt_conf['infer_path'] = str(data_dir / f'{file_prefix}.{TARGET2COL[target]}.test.csv')

    return expt_conf


def main(expt_conf, hyperparameters, typical_train_func):
    if expt_conf['expt_id'] == 'timestamp':
        expt_conf['expt_id'] = dt.today().strftime('%Y-%m-%d_%H:%M')

    expt_dir = (Path(__file__).resolve().parents[1] / 'output' / f"{expt_conf['expt_id']}")
    expt_dir.mkdir(exist_ok=True, parents=True)

    logging.basicConfig(level=logging.DEBUG, format="[%(name)s] [%(levelname)s] %(message)s",
                        filename=expt_dir / 'expt.log')

    expt_conf['class_names'] = [0, 1, 2]
    metrics_names = {'train': ['uar'], 'val': ['uar'], 'infer': []}

    dataset_cls = CSVDataSet
    expt_conf = set_data_paths(expt_dir, expt_conf, phases=['train', 'val', 'infer'])
    label_func = None

    patterns = list(itertools.product(*hyperparameters.values()))
    # patterns = [(w_size, w_stride) for w_size, w_stride in patterns if w_size > w_stride]
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

        with mlflow.start_run():
            result_series, val_pred = typical_train_func(expt_conf, load_func, label_func, process_func, dataset_cls,
                                                         groups, metrics_names)

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
    phases = ['train', 'infer']
    if expt_conf['infer']:
        best_trial_idx = val_results['uar'].argmax()

        best_pattern = patterns[best_trial_idx]
        for i, param in enumerate(hyperparameters.keys()):
            expt_conf[param] = best_pattern[i]
        dump_dict(expt_dir / 'best_parameters.txt', {p: v for p, v in zip(hyperparameters.keys(), best_pattern)})

        expt_conf['model_path'] = str(expt_dir / f"{'_'.join([str(p).replace('/', '-') for p in best_pattern])}.pth")
        expt_conf = set_data_paths(expt_dir, expt_conf, phases=phases)
        experimentor = BaseExperimentor(expt_conf, load_func, label_func, process_func, dataset_cls)

        metrics = {p: get_metric_list(metrics_names[p]) for p in phases}
        pred = experimentor.experiment_without_validation(metrics, seed_average=expt_conf['n_seed_average'])['infer']

        sub_name = f"sub_{'_'.join([str(p).replace('/', '-') for p in best_pattern])}.csv"
        if (expt_dir / sub_name).is_file():
            sub_df = pd.read_csv(expt_dir / sub_name)
        else:
            sub_df = pd.read_csv(expt_conf['train_path'])[['filename']]
        sub_df[expt_conf['target']] = pd.Series(pred).apply(lambda x: list(LABELS2INT.keys())[x])
        pd.DataFrame(pred).to_csv(expt_dir / sub_name, index=False)
        print(f"Submission file is saved in {expt_dir / sub_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train arguments')
    expt_conf = vars(expt_args(parser).parse_args())

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("[%(name)s] [%(levelname)s] %(message)s"))
    console.setLevel(logging.DEBUG)
    logging.getLogger("ml").addHandler(console)

    if 'debug' in expt_conf['expt_id']:
        hyperparameters = {
            'C': [0.0001, 0.001, 0.01, 0.1, 1.0],
            'sample_balance': [None],
        }
    else:
        hyperparameters = {
            'C': [0.0001, 0.001, 0.01, 0.1, 1.0],
            'sample_balance': [None],
        }

    main(expt_conf, hyperparameters, typical_train)

    # if expt_conf['expt_id'] == 'debug':
    import shutil
    shutil.rmtree('./mlruns')
