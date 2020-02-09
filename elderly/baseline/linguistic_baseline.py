#!/usr/bin/python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.metrics import recall_score, confusion_matrix
import os.path
import sys

classes = ['L', 'M', 'H']

# Task
task_name = 'ComParE2020_USOMS-e'
# Enter your team name HERE
team_name = 'baseline'
# Enter your submission number HERE
submission_index = 1
# Option
show_confusion = True  # Display confusion matrix on devel
majority_vote_story_id = True
# Configuration
feature_set = 'frozen-bert-gmax'  # For all available options, see the dictionary feat_conf

label_options = ['V_cat', 'A_cat']

complexities = [1e-5,1e-4,1e-3,1e-2,1e-1,1e0]

# Mapping each available feature set to tuple (number of features, offset/index of first feature, separator, header option)
feat_conf = {'frozen-bert-gmax': (512, 1, ',', 'infer'),
             'frozen-bert-rnnatt': (512, 1, ',', 'infer'),
             'frozen-bert-pos-fuse-rnnatt': (512, 1, ',', 'infer'),
             'fused': (1536, 1, ',', 'infer')
             }


num_feat = feat_conf[feature_set][0]
ind_off = feat_conf[feature_set][1]
sep = feat_conf[feature_set][2]
header = feat_conf[feature_set][3]

# Path of the features and labels
features_path = '../features/'
label_file = '../lab/labels.csv'

# Start
for current_label in label_options:

    print('\nRunning ' + task_name + ' ' + feature_set + ' ' + current_label + ' baseline ... (this might take a while) \n')      

    # Load features and labels
    print('\nLoading Features and Labels')
    X_train = pd.read_csv(features_path + task_name + '.' + feature_set + '.' + current_label  + '_no.train.csv', sep=sep, header=header,
                          usecols=range(ind_off, num_feat + ind_off), dtype=np.float32).values
    X_devel = pd.read_csv(features_path + task_name + '.' + feature_set + '.' + current_label  + '_no.devel.csv', sep=sep, header=header,
                          usecols=range(ind_off, num_feat + ind_off), dtype=np.float32).values
    X_test = pd.read_csv(features_path + task_name + '.' + feature_set + '.' + current_label  + '_no.test.csv', sep=sep, header=header,
                         usecols=range(ind_off, num_feat + ind_off), dtype=np.float32).values

    df_labels = pd.read_csv(label_file)   
    df_agg = df_labels.groupby(['filename_text','partition'])[current_label].agg(
        lambda x: x.value_counts().index[0]).reset_index()

    print('currently running on dev for ' + current_label)
    y_train = df_agg[current_label][df_agg['partition'] == 'train'].values
    y_devel = df_agg[current_label][df_agg['partition'] == 'devel'].values

    # Concatenate training and development for final training
    X_traindevel = np.concatenate((X_train, X_devel))
    y_traindevel = np.concatenate((y_train, y_devel))

    # Upsampling / Balancing
    print('Upsampling ... ')
    num_samples_train = []
    num_samples_traindevel = []
    for label in classes:
        num_samples_train.append(len(y_train[y_train == label]))
        num_samples_traindevel.append(len(y_traindevel[y_traindevel == label]))

    for label, ns_tr, ns_trd in zip(classes, num_samples_train, num_samples_traindevel):
        factor_tr = np.max(num_samples_train) // ns_tr
        X_train = np.concatenate((X_train, np.tile(X_train[y_train == label], (factor_tr - 1, 1))))
        y_train = np.concatenate((y_train, np.tile(y_train[y_train == label], (factor_tr - 1))))
        factor_trd = np.max(num_samples_traindevel) // ns_trd
        X_traindevel = np.concatenate((X_traindevel, np.tile(X_traindevel[y_traindevel == label], (factor_trd - 1, 1))))
        y_traindevel = np.concatenate((y_traindevel, np.tile(y_traindevel[y_traindevel == label], (factor_trd - 1))))

    # Feature normalisation
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_devel = scaler.transform(X_devel)
    X_traindevel = scaler.fit_transform(X_traindevel)
    X_test = scaler.transform(X_test)

    # Train SVM model with different complexities and evaluate
    uar_scores = []
    for comp in complexities:
        print('\nComplexity {0:.6f}'.format(comp))
        clf = svm.LinearSVC(C=comp, random_state=0, max_iter=10000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_devel)
        uar_scores.append(recall_score(y_devel, y_pred, labels=classes, average='macro'))
        print('UAR on Devel {0:.1f}'.format(uar_scores[-1] * 100))
        if show_confusion:
            print('Confusion matrix (Devel):')
            print(classes)
            print(confusion_matrix(y_devel, y_pred, labels=classes))

    # Train SVM model on the whole training data with optimum complexity and get predictions on test data
    optimum_complexity = complexities[np.argmax(uar_scores)]
    print('\nOptimum complexity: {0:.6f}, maximum UAR on Devel {1:.1f}\n'.format(
                              optimum_complexity, np.max(uar_scores) * 100))

    clf = svm.LinearSVC(C=optimum_complexity, random_state=0)
    clf.fit(X_traindevel, y_traindevel)
    y_pred = clf.predict(X_test)


    # Write out predictions to csv file (official submission format)
    pred_file_name = task_name + '_' + feature_set + '_' + current_label + '.test.' + team_name + '_' + str(submission_index) + '.csv'

    print('Writing file ' + pred_file_name + '\n')

    df = pd.DataFrame(data={'filename': df_agg['filename_text'][df_agg['partition'] == 'test'].values,
                            'prediction': y_pred.flatten()},
                      columns=['filename', 'prediction'])

    df.to_csv(pred_file_name, index=False)

    print('Done.\n')
