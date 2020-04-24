import argparse
import logging
from pathlib import Path

import pandas as pd
from experiment import label_func
from sklearn.metrics import balanced_accuracy_score


def mask_repr_args(parser):
    repr_parser = parser.add_argument_group("Mask Experiment arguments")
    repr_parser.add_argument('--expt-id', default='sth')

    return parser


def main(repr_conf):
    expt_dir = Path(__file__).parents[1] / 'output' / repr_conf['expt_id']
    phase = 'val'

    repr_conf[f'{phase}_path'] = Path(__file__).parents[1] / 'lab' / f'{phase}_manifest.csv'

    val_results = pd.read_csv(expt_dir / 'val_results.csv')
    best_expt_name = '_'.join(val_results.iloc[val_results['uar'].argmax(), :-2].astype(str))
    repr_conf['model_path'] = expt_dir / f'{best_expt_name}.pth'

    # load_func = set_load_func(wav_path, expt_conf['sample_rate'], expt_conf['n_waves'])
    # reproducer = Reproducer(repr_conf)
    # val_result, val_pred = reproducer.reproduce(phase='val')

    val_pred = pd.read_csv(expt_dir / f'{best_expt_name}_val_pred.csv')
    val_ans = pd.read_csv(repr_conf['val_path'], header=None).apply(label_func, axis=1)
    uar = balanced_accuracy_score(val_ans, val_pred)
    print(uar)
    # test_pred = reproducer.reproduce(phase='infer')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mask experiment arguments')
    repr_conf = vars(mask_repr_args(parser).parse_args())

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("[%(name)s] [%(levelname)s] %(message)s"))
    console.setLevel(logging.DEBUG)
    logging.getLogger("ml").addHandler(console)

    main(repr_conf)
