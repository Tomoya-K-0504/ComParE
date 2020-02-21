import itertools

import pandas as pd
from ml.src.dataset import ManifestWaveDataSet


class ManifestMultiWaveDataSet(ManifestWaveDataSet):
    def __init__(self, manifest_path, data_conf, load_func=None, process_func=None, label_func=None, phase='train'):
        super(ManifestMultiWaveDataSet, self).__init__(manifest_path, data_conf, load_func, process_func, label_func, phase)
        self.n_waves = data_conf['n_waves']
        self.shuffle_order = data_conf['shuffle_order']
        self._concat_paths()

    def _concat_paths(self) -> None:
        if self.n_waves == 1:
            return

        master_list = []

        if self.phase != 'train':
            unique_scripts = self.path_df.iloc[:, 1].unique()
            for script in unique_scripts:
                one_script_df = self.path_df[self.path_df.iloc[:, 1] == script]
                wav_df = one_script_df.iloc[:, [0]]
                for i_shift in range(1, self.n_waves):
                    shifted = one_script_df.iloc[:, 0].shift(-1 * i_shift)
                    wav_df[f'shift_{i_shift}'] = shifted
                master_list.append(wav_df.fillna(method='ffill', axis=1))

            master_df = pd.concat(master_list, axis=0)
            self.path_df.iloc[:, 0] = master_df.apply(lambda x: ','.join(x), axis=1)

        else:
            label_list = []
            unique_scripts = self.path_df.iloc[:, 1].unique()
            for script in unique_scripts:
                one_script_df = self.path_df[self.path_df.iloc[:, 1] == script]
                wav_df = one_script_df.iloc[:, [0]]
                for i_shift in range(1, self.n_waves):
                    shifted = one_script_df.iloc[:, 0].shift(-1 * i_shift)
                    wav_df[i_shift] = shifted
                wav_df = wav_df.dropna(how='any', axis=0)
                if self.shuffle_order:
                    for combination in itertools.permutations(list(range(self.n_waves))):
                        wav_df = wav_df.loc[:, list(combination)].rename(columns={c: i for i, c in enumerate(combination)})
                        master_list.append(wav_df)
                        label_list.extend(list(one_script_df.iloc[self.n_waves - 1:, :].apply(self.label_func, axis=1)))
                else:
                    master_list.append(wav_df)
                    label_list.extend(list(one_script_df.iloc[self.n_waves - 1:, :].apply(self.label_func, axis=1)))
            master_df = pd.concat(master_list, axis=0).reset_index(drop=True)
            assert master_df.shape[0] == len(label_list), f'{master_df.shape[0]}, {len(label_list)}'
            master_df = master_df.apply(lambda x: ','.join(x), axis=1)
            self.path_df = pd.concat([master_df, pd.Series(label_list)], axis=1)
            self.labels = label_list

            assert self.path_df.shape[0] == len(label_list), f'{self.path_df.shape[0]}, {len(label_list)}'
