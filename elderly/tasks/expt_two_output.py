import argparse
import itertools
import logging
import pprint
from copy import deepcopy
from datetime import datetime as dt
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from experiment import LABELS2INT, elderly_expt_args, set_load_func, set_data_paths, get_cv_groups
from joblib import Parallel, delayed
from ml.src.dataset import ManifestWaveDataSet
from ml.tasks.base_experiment import BaseExperimentor, typical_experiment


def label_func(row):
    return row[5], row[6]


def main(expt_conf):
    if expt_conf['expt_id'] == 'timestamp':
        expt_conf['expt_id'] = dt.today().strftime('%Y-%m-%d_%H:%M')

    expt_dir = (Path(__file__).resolve().parents[1] / 'output' / f"{expt_conf['expt_id']}")
    expt_dir.mkdir(exist_ok=True)

    logging.basicConfig(level=logging.DEBUG, format="[%(name)s] [%(levelname)s] %(message)s",
                        filename=expt_dir / 'expt.log')

    hyperparameters = {
        'model_type': ['multitask_panns'],
        'target': [['valence', 'arousal']],
        'checkpoint_path': ['../cnn14.pth'],
        'window_size': [0.08],
        'window_stride': [0.01, 0.02]
    }

    expt_conf['class_names'] = [0, 1, 2]
    expt_conf['sample_rate'] = 16000
    expt_conf['n_tasks'] = 2

    load_func = set_load_func(Path(expt_conf['manifest_path']).resolve().parents[1] / 'wav', expt_conf['sample_rate'])
    dataset_cls = ManifestWaveDataSet
    val_metrics = ['loss', 'uar', 'loss', 'uar']
    val_metrics_columns = ['v_loss', 'v_uar', 'a_loss', 'a_uar']

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

        with mlflow.start_run():
            mlflow.set_tag('target', expt_conf['target'])
            result_series, pred = typical_experiment(expt_conf, load_func, label_func, dataset_cls, groups, val_metrics)

            mlflow.log_metrics({metric_name: value for metric_name, value in zip(val_metrics_columns, result_series)})
            mlflow.log_params({hyperparameter: value for hyperparameter, value in zip(hyperparameters.keys(), pattern)})
            mlflow.log_artifacts(expt_dir)

        return result_series, pred

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
    pred_list = np.array([pred for result, pred in result_pred_list])
    val_results.iloc[:, len(hyperparameters):] = result_list
    pp.pprint(val_results)

    pp.pprint(val_results.iloc[:, len(hyperparameters):].describe())
    val_results.to_csv(expt_dir / 'val_results.csv', index=False)
    print(f"Devel results saved into {expt_dir / 'val_results.csv'}")

    # Train with train + devel dataset
    if expt_conf['train_with_all']:
        best_trial_idx = (val_results['v_uar'] + val_results['a_uar']).argmax()
        best_pattern = patterns[best_trial_idx]

        for i, param in enumerate(hyperparameters.keys()):
            expt_conf[param] = best_pattern[i]

        expt_conf['model_path'] = str(expt_dir / f"{'_'.join([str(p).replace('/', '-') for p in best_pattern])}.pth")
        expt_conf = set_data_paths(expt_conf, phases=['train', 'infer'])
        experimentor = BaseExperimentor(expt_conf, load_func, label_func, dataset_cls)

        pred = experimentor.experiment_without_validation(seed_average=expt_conf['n_seed_average'])

        sub_name = f"sub_{'_'.join([str(p).replace('/', '-') for p in best_pattern])}.csv"
        if (expt_dir / sub_name).is_file():
            sub_df = pd.read_csv(expt_dir / sub_name)
        else:
            sub_df = manifest_df[manifest_df['partition'] == 'test'][['filename_text']]
        for i, target in enumerate(expt_conf['target']):
            sub_df[target] = pd.Series(pred[:, i]).apply(lambda x: list(LABELS2INT.keys())[x])
        sub_df.to_csv(expt_dir / sub_name, index=False)
        print(f"Submission file is saved in {expt_dir / sub_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train arguments')
    expt_conf = vars(elderly_expt_args(parser).parse_args())
    assert expt_conf['train_path'] != '' or expt_conf['val_path'] != '', \
        'You need to select training, validation data file to training, validation in --train-path, --val-path argments'

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("[%(name)s] [%(levelname)s] %(message)s"))
    console.setLevel(logging.INFO)
    logging.getLogger("ml").addHandler(console)

    main(expt_conf)
