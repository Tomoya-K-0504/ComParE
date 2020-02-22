import argparse
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.metrics import balanced_accuracy_score


def type_float_list(args) -> List[str]:
    return args.split(',')


def postprocess_args(parser):
    postprocess_parser = parser.add_argument_group('aggregate arguments')
    postprocess_parser.add_argument('--expt-id')
    postprocess_parser.add_argument('--target', type=type_float_list)
    postprocess_parser.add_argument('--manifest-path', default='lab/labels_mean_diff.csv')

    return parser


def postprocess(cfg):
    expt_id = cfg['expt_id'] + '_' + ','.join(cfg['target'])
    expt_out_dir = Path(__file__).resolve().parents[1] / 'output' / expt_id

    target2col = {'valence': 'V_cat_no', 'arousal': 'A_cat_no'}

    target = cfg['target'][0].replace('_mean', '').replace('_dev', '')

    with open(expt_out_dir / 'best_parameters.txt', 'r') as f:
        best_pattern = json.load(f)
    val_pred_path = expt_out_dir / f"{'_'.join([str(p).replace('/', '-') for p in best_pattern.values()])}_val_pred.csv"
    val_pred = pd.read_csv(val_pred_path).sum(axis=1)
    val_pred = val_pred.apply(lambda x: int(max(0, min(2, Decimal(str(x)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)))))

    manifest_df = pd.read_csv(cfg['manifest_path'])
    val_ans = manifest_df.loc[manifest_df['partition'] == 'devel', target2col[target]]
    uar = balanced_accuracy_score(val_ans, val_pred)
    print(uar)
    with open(expt_out_dir / f'uar_{uar}.txt', 'w') as f:
        f.write('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='aggregate arguments')
    cfg = vars(postprocess_args(parser).parse_args())
    postprocess(cfg)
