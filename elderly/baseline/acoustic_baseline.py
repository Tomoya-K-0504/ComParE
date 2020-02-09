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
majority_vote_story_id = True # Perform a majority vote over all audio file prediction, based on SpeakerID + Story
# Configuration
feature_set = 'ComParE'  # For all available options, see the dictionary feat_conf

complexities = [1e-5,1e-4,1e-3,1e-2,1e-1,1e0]


# Mapping each available feature set to tuple (number of features, offset/index of first feature, separator, header option)
feat_conf = {'ComParE': (6373, 1, ';', 'infer'),
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
             'DeepSpectrum_resnet50': (2048, 1, ',', 'infer')}

num_feat = feat_conf[feature_set][0]
ind_off = feat_conf[feature_set][1]
sep = feat_conf[feature_set][2]
header = feat_conf[feature_set][3]

# Path of the features and labels
features_path = '../features/'
label_file = '../lab/labels.csv'

# Labels
label_options = ['V_cat', 'A_cat']

for current_label in label_options:

    print('\nRunning ' + task_name + ' ' + feature_set + ' baseline ... (this might take a while) \n')

    # Load features and labels
    print('\nLoading Features and Labels')
    X_train = pd.read_csv(features_path + task_name + '.' + feature_set + '.train.csv', sep=sep, header=header,
                          usecols=range(ind_off, num_feat + ind_off), dtype=np.float32).values
    X_devel = pd.read_csv(features_path + task_name + '.' + feature_set + '.devel.csv', sep=sep, header=header,
                          usecols=range(ind_off, num_feat + ind_off), dtype=np.float32).values
    X_test = pd.read_csv(features_path + task_name + '.' + feature_set + '.test.csv', sep=sep, header=header,
                         usecols=range(ind_off, num_feat + ind_off), dtype=np.float32).values

    df_labels = pd.read_csv(label_file)

    print('currently running on dev for ' + current_label)
    y_train = df_labels[current_label][df_labels['filename_audio'].str.startswith('train')].values
    y_devel = df_labels[current_label][df_labels['filename_audio'].str.startswith('devel')].values

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

    # Perform a majority vote for the developement predictions, based on Speaker_ID + Story (1-3)
    if majority_vote_story_id:
        print('performing majority vote ...')
        y_pred = clf.predict(X_devel)
        df_with_stories_devel = pd.DataFrame(
	      data={'filename_audio': df_labels['filename_audio'][df_labels['filename_audio'].str.startswith('devel')].values,
              'filename_text': df_labels['filename_text'][df_labels['filename_audio'].str.startswith('devel')].values,  # filename_text == ID_Story
              'prediction': y_pred.flatten(),
              'true': y_devel.flatten()},columns=['filename_audio', 'filename_text', 'prediction', 'true'])
        df_maj_devel = df_with_stories_devel.groupby(['filename_text'])['prediction', 'true'].agg(
            lambda x: x.value_counts().sort_index().sort_values(ascending=False, kind='mergesort').index[0])
        uar_maj = recall_score(df_maj_devel['true'].values, df_maj_devel['prediction'].values, labels=classes,average='macro')
        if show_confusion:
            print('Confusion matrix with majority voting (Devel):')
            print(classes)
            print(confusion_matrix(df_maj_devel['true'].values, df_maj_devel['prediction'].values, labels=classes))




    # Train SVM model on the whole training data with optimum complexity and get predictions on test data
        optimum_complexity = complexities[np.argmax(uar_scores)]
        print('\nOptimum complexity: {0:.6f}, maximum UAR on Devel {1:.1f}, with majority voting {2:.1f}\n'.format(
            optimum_complexity, np.max(uar_scores) * 100, uar_maj * 100))

        clf = svm.LinearSVC(C=optimum_complexity, random_state=0)
        clf.fit(X_traindevel, y_traindevel)
        y_pred = clf.predict(X_test)

    else:
        optimum_complexity = complexities[np.argmax(uar_scores)]
        print('\nOptimum complexity: {0:.6f}, maximum UAR on Devel {1:.1f}, with majority voting {2:.1f}\n'.format(
            optimum_complexity, np.max(uar_scores) * 100, uar_maj * 100))


    # Write out predictions to csv file (official submission format)
    pred_file_name = task_name + '_' + feature_set + '_' + current_label + '.test.' + team_name + '_' + str(
        submission_index) + '.csv'
    pred_file_name_maj = task_name + '_' + feature_set + '_' + current_label + '.test.' + team_name + '_' + str(
        submission_index) + '_maj.csv'

    print('Writing file ' + pred_file_name + '\n')
    df = pd.DataFrame(data={'filename': df_labels['filename_audio'][df_labels['filename_audio'].str.startswith('test')].values,
                            'prediction': y_pred.flatten()},
                      columns=['filename', 'prediction'])

    # Perform a majority vote for the test predictions, based on Speaker_ID + Story (1-3)
    df_with_stories = pd.DataFrame(
        data={'filename_audio': df_labels['filename_audio'][df_labels['filename_audio'].str.startswith('test')].values,
              'filename': df_labels['filename_text'][df_labels['filename_audio'].str.startswith('test')].values,
              'prediction': y_pred.flatten()},
        columns=['filename_audio', 'filename', 'prediction'])

    df_maj = df_with_stories.groupby(['filename'])['prediction'].agg(
        lambda x: x.value_counts().sort_index().sort_values(ascending=False, kind='mergesort').index[0]).reset_index()
    df.to_csv(pred_file_name, index=False)
    df_maj.to_csv(pred_file_name_maj, index=False)


    print('Done.\n')
