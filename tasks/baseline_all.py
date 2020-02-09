#!/usr/bin/python
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import recall_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler


def baseline_args(parser):
    baseline_parser = parser.add_argument_group("Experiment arguments")
    baseline_parser.add_argument('--sub-task', help='Name of the sub challenge', choices=['mask', 'elderly', 'breathing'],
                                 required=True)

    return parser


def main(conf_key, sub_task):
    # Task
    task_converter = {'mask': 'Mask', 'elderly': 'Elderly', 'breathing': 'Breathing'}
    task_name = f"ComParE2020_{task_converter[sub_task]}"
    classes    = ['clear', 'mask']

    # Enter your team name HERE
    team_name = 'baseline'

    # Enter your submission number HERE
    submission_index = 1

    # Option
    show_confusion = True   # Display confusion matrix on devel

    # Configuration
    feature_set = conf_key  # For all available options, see the dictionary feat_conf
    complexities = [1e-3,1e-2,1e-1,1e0, 1e1]  # SVM complexities (linear kernel)

    # Mapping each available feature set to tuple (number of features, offset/index of first feature, separator, header option)
    feat_conf = {'ComParE':      (6373, 1, ';', 'infer'),
                 'BoAW-125':     ( 250, 1, ';',  None),
                 'BoAW-250':     ( 500, 1, ';',  None),
                 'BoAW-500':     (1000, 1, ';',  None),
                 'BoAW-1000':    (2000, 1, ';',  None),
                 'BoAW-2000':    (4000, 1, ';',  None),
                 'auDeep-30':    (1024, 2, ',', 'infer'),
                 'auDeep-45':    (1024, 2, ',', 'infer'),
                 'auDeep-60':    (1024, 2, ',', 'infer'),
                 'auDeep-75':    (1024, 2, ',', 'infer'),
                 'auDeep-fused': (4096, 2, ',', 'infer'),
                 'DeepSpectrum_resnet50': (2048, 1, ',', 'infer')}
    num_feat = feat_conf[feature_set][0]
    ind_off  = feat_conf[feature_set][1]
    sep      = feat_conf[feature_set][2]
    header   = feat_conf[feature_set][3]

    # Path of the features and labels
    features_path = f'{sub_task}/features/'
    label_file    = f'{sub_task}/lab/labels.csv'

    # Start
    print('\nRunning ' + task_name + ' ' + feature_set + ' baseline ... (this might take a while) \n')

    # Load features and labels
    X_train = pd.read_csv(features_path + task_name + '.' + feature_set + '.train.csv', sep=sep, header=header, usecols=range(ind_off,num_feat+ind_off), dtype=np.float32).values
    X_devel = pd.read_csv(features_path + task_name + '.' + feature_set + '.devel.csv', sep=sep, header=header, usecols=range(ind_off,num_feat+ind_off), dtype=np.float32).values
    X_test  = pd.read_csv(features_path + task_name + '.' + feature_set + '.test.csv',  sep=sep, header=header, usecols=range(ind_off,num_feat+ind_off), dtype=np.float32).values

    df_labels = pd.read_csv(label_file)
    y_train = df_labels['label'][df_labels['file_name'].str.startswith('train')].values
    y_devel = df_labels['label'][df_labels['file_name'].str.startswith('devel')].values

    # Concatenate training and development for final training
    X_traindevel = np.concatenate((X_train, X_devel))
    y_traindevel = np.concatenate((y_train, y_devel))

    # Feature normalisation
    scaler       = MinMaxScaler()
    X_train      = scaler.fit_transform(X_train)
    X_devel      = scaler.transform(X_devel)
    X_traindevel = scaler.fit_transform(X_traindevel)
    X_test       = scaler.transform(X_test)

    # Train SVM model with different complexities and evaluate
    uar_scores = []
    for comp in complexities:
        print('\nComplexity {0:.6f}'.format(comp))
        clf = svm.LinearSVC(C=comp, random_state=0, max_iter=100000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_devel)
        uar_scores.append( recall_score(y_devel, y_pred, labels=classes, average='macro') )
        print('UAR on Devel {0:.1f}'.format(uar_scores[-1]*100))
        if show_confusion:
            print('Confusion matrix (Devel):')
            print(classes)
            print(confusion_matrix(y_devel, y_pred, labels=classes))

    # Train SVM model on the whole training data with optimum complexity and get predictions on test data
    optimum_complexity = complexities[np.argmax(uar_scores)]
    print('\nOptimum complexity: {0:.6f}, maximum UAR on Devel {1:.1f}\n'.format(optimum_complexity, np.max(uar_scores)*100))

    clf = svm.LinearSVC(C=optimum_complexity, random_state=0)
    clf.fit(X_traindevel, y_traindevel)
    y_pred = clf.predict(X_test)

    # Write out predictions to csv file (official submission format)
    pred_file_name = task_name + '.' + feature_set +'.test.' + team_name + '_' + str(submission_index) + '.csv'
    print('Writing file ' + pred_file_name + '\n')
    df = pd.DataFrame(data={'file_name': df_labels['file_name'][df_labels['file_name'].str.startswith('test')].values,
                            'prediction': y_pred.flatten()},
                      columns=['file_name','prediction'])
    Path(f'{sub_task}/output').mkdir(exist_ok=True)
    df.to_csv(f'{sub_task}/output/{pred_file_name}', index=False)

    print('Done.\n')

    return uar_scores[np.argmax(uar_scores)], complexities[np.argmax(uar_scores)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train arguments')
    baseline_conf = vars(baseline_args(parser).parse_args())

    uar_list = []
    feat_conf = {
                 'ComParE': (6373, 1, ';', 'infer'),
                 'BoAW-125': (250, 1, ';', None),
                 'BoAW-250': (500, 1, ';', None),
                 'BoAW-500': (1000, 1, ';', None),
                 'BoAW-1000': (2000, 1, ';', None),
                 'BoAW-2000': (4000, 1, ';', None),
                 'auDeep-30': (1024, 2, ',', 'infer'),
                 'auDeep-45': (1024, 2, ',', 'infer'),
                 'auDeep-60': (1024, 2, ',', 'infer'),
                 'auDeep-75': (1024, 2, ',', 'infer'),
                 'auDeep-fused': (4096, 2, ',', 'infer'),
                 'DeepSpectrum_resnet50': (2048, 1, ',', 'infer')
    }
    for conf_key in feat_conf.keys():
        devel_max_uar, complexity = main(conf_key, baseline_conf['sub_task'])
        uar_list.append((devel_max_uar * 100, complexity))

    pd.DataFrame(uar_list, columns=['devel_max_uar', 'complexity']).to_csv(
        f"{baseline_conf['sub_task']}/output/baseline.csv", index=False)
