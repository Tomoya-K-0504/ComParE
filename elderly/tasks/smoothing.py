import pandas as pd


def smoothing(sub_df, target, method='majority'):
    if method == 'majority':
        return sub_df.groupby('filename_text')[target].agg(pd.Series.mode).to_frame().reset_index()
    else:
        raise NotImplementedError


if __name__ == '__main__':
    path = '/home/tomoya/workspace/research/compare/elderly/output/debug/sub_0.0001_1_logmel_cnn_logmel_0.01_0.002_1_0.05_0.0_[1, 1, 1]_0.0_0.0.csv'
    sub_df = pd.read_csv(path)
    smoothing(sub_df, 'valence')