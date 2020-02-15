import argparse
import itertools
import pprint
from copy import deepcopy
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from experiment import LABELS2INT, elderly_expt_args, set_load_func
from joblib import Parallel, delayed
from ml.tasks.base_experiment import BaseExperimentor, CrossValidator


def label_func(row):
    return row[5], row[6]


def main(expt_conf):
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
    val_metrics = ['loss', 'uar', 'loss', 'uar']
    val_metrics_columns = ['v_loss', 'v_uar', 'a_loss', 'a_uar']

    manifest_df = pd.read_csv(expt_conf['manifest_path'])
    for phase, part in zip(['train', 'val', 'infer'], ['train', 'devel', 'test']):
        phase_df = manifest_df[manifest_df[f'partition'].str.startswith(part)]
        phase_df.to_csv(Path(expt_conf['manifest_path']).parent / f'{phase}_manifest.csv', index=False, header=None)
        expt_conf[f'{phase}_path'] = str(Path(expt_conf['manifest_path']).parent / f'{phase}_manifest.csv')

    patterns = list(itertools.product(*hyperparameters.values()))
    # patterns = [(w_size, w_stride) for w_size, w_stride in patterns if w_size > w_stride]
    val_results = pd.DataFrame(np.zeros((len(patterns), len(hyperparameters) + len(val_metrics_columns))),
                               columns=list(hyperparameters.keys()) + val_metrics_columns)

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(hyperparameters)

    groups = None
    if expt_conf['cv_name'] == 'group':
        train_val_manifest = manifest_df[manifest_df['partition'] != 'test']
        subjects = pd.DataFrame(train_val_manifest['filename_text'].str.slice(0, -4).unique())
        subjects['group'] = [j % expt_conf['n_splits'] for j in range(len(subjects))]
        subjects = subjects.set_index(0)
        groups = train_val_manifest['filename_text'].str.slice(0, -4).apply(lambda x: subjects.loc[x, 'group'])

    def experiment(i, pattern, expt_conf):
        for idx, param in enumerate(hyperparameters.keys()):
            expt_conf[param] = pattern[idx]

        if expt_conf['cv_name']:
            experimentor = CrossValidator(expt_conf, load_func, label_func, expt_conf['cv_name'], expt_conf['n_splits'],
                                          groups)
        else:
            experimentor = BaseExperimentor(expt_conf, load_func, label_func)

        with mlflow.start_run():
            mlflow.set_tag('target', expt_conf['target'])
            result_series, pred = experimentor.experiment_with_validation(val_metrics)

            # mlflow.log_params(expt_conf)
            mlflow.log_params({hyperparameter: value for hyperparameter, value in zip(hyperparameters.keys(), pattern)})
            mlflow.log_metrics({metric_name: value for metric_name, value in zip(val_metrics, result_series)})

        return result_series, pred

    # For debugging
    if expt_conf['n_jobs'] == 1:
        result_pred_list = [experiment(i, pattern, deepcopy(expt_conf)) for i, pattern in enumerate(patterns)]
    else:
        result_pred_list = Parallel(n_jobs=expt_conf['n_jobs'], verbose=0)(
            [delayed(experiment)(i, pattern, deepcopy(expt_conf)) for i, pattern in enumerate(patterns)])

    val_results.iloc[:, :len(hyperparameters)] = patterns
    result_list = [result for result, pred in result_pred_list]
    pred_list = np.array([pred for result, pred in result_pred_list])
    val_results.iloc[:, len(hyperparameters):] = result_list
    pp.pprint(val_results)

    (Path(__file__).resolve().parents[1] / 'output' / 'metrics').mkdir(exist_ok=True)
    expt_path = Path(__file__).resolve().parents[1] / 'output' / 'metrics' / f"{expt_conf['expt_id']}.csv"
    print(val_results)
    print(val_results.iloc[:, len(hyperparameters):].describe())
    val_results.to_csv(expt_path, index=False)
    print(f'Devel results saved into {expt_path}')

    # Train with train + devel dataset
    if expt_conf['train_with_all']:
        best_trial_idx = (val_results['v_uar'] + val_results['a_uar']).argmax()
        best_pattern = patterns[best_trial_idx]

        for i, param in enumerate(hyperparameters.keys()):
            expt_conf[param] = best_pattern[i]

        infer_df = manifest_df[manifest_df['partition'] == 'test']
        train_devel_df = manifest_df[~manifest_df.index.isin(infer_df.index)]
        infer_df.to_csv(Path(expt_conf['manifest_path']).parent / f'infer_manifest.csv', index=False, header=None)
        train_devel_df.to_csv(Path(expt_conf['manifest_path']).parent / f'train_manifest.csv', index=False, header=None)

        expt_conf[f'infer_path'] = str(Path(expt_conf['manifest_path']).parent / f'infer_manifest.csv')
        expt_conf[f'train_path'] = str(Path(expt_conf['manifest_path']).parent / f'train_manifest.csv')

        experimentor = BaseExperimentor(expt_conf, load_func, label_func)

        pred = experimentor.experiment_without_validation(seed_average=expt_conf['n_seed_average'])

        sub_name = f"{expt_conf['expt_id']}_{'_'.join([str(p).replace('/', '-') for p in best_pattern])}.csv"
        if (Path(__file__).resolve().parents[1] / 'output' / 'sub' / sub_name).is_file():
            sub_df = pd.read_csv(Path(__file__).resolve().parents[1] / 'output' / 'sub' / sub_name)
        else:
            sub_df = manifest_df[manifest_df['partition'] == 'test'][['filename_text']]
            (Path(__file__).resolve().parents[1] / 'output' / 'sub').mkdir(exist_ok=True)
        for i, target in enumerate(expt_conf['target']):
            sub_df[target] = pd.Series(pred[:, i]).apply(lambda x: list(LABELS2INT.keys())[x])
        sub_df.to_csv(Path(__file__).resolve().parents[1] / 'output' / 'sub' / sub_name, index=False)
        print(f"Submission file is saved in {Path(__file__).resolve().parents[1] / 'output' / 'sub' / sub_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train arguments')
    expt_conf = vars(elderly_expt_args(parser).parse_args())
    assert expt_conf['train_path'] != '' or expt_conf['val_path'] != '', \
        'You need to select training, validation data file to training, validation in --train-path, --val-path argments'

    main(expt_conf)
