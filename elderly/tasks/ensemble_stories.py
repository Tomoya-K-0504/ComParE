import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd
from experiment import AVAILABLE_TARGET
from sklearn.metrics import balanced_accuracy_score


def type_float_list(args) -> List[str]:
    return args.split(',')


def ensemble_args(parser):
    ensemble_parser = parser.add_argument_group('aggregate arguments')
    ensemble_parser.add_argument('--expt-id')
    ensemble_parser.add_argument('--target', choices=AVAILABLE_TARGET)
    ensemble_parser.add_argument('--manifest-path', default='lab/labels_mean_diff.csv')

    return parser


def ensemble_stories(cfg):
    expt_id = cfg['expt_id'] + '_' + cfg['target']
    expt_out_dir = Path(__file__).resolve().parents[1] / 'output' / expt_id
    
    target2col = {'valence': 'V_cat_no', 'arousal': 'A_cat_no'}

    with open(expt_out_dir / 'best_parameters.txt', 'r') as f:
        best_pattern = json.load(f)
    val_pred_path = expt_out_dir / f"{'_'.join([str(p).replace('/', '-') for p in best_pattern.values()])}_val_pred.csv"
    val_pred = pd.read_csv(val_pred_path)
    print(val_pred.describe())
    label_df = pd.read_csv(Path(__file__).resolve().parents[1] / 'lab/labels.csv')
    val_labels = label_df.loc[label_df['partition'] == 'devel', target2col[cfg['target']]].astype(int)
    print(val_labels.describe())
    print(balanced_accuracy_score(val_labels, val_pred))
    a = ''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='aggregate arguments')
    cfg = vars(ensemble_args(parser).parse_args())
    ensemble_stories(cfg)
