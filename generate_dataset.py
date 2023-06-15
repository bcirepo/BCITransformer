
import numpy as np
from moabb.datasets import Lee2019_MI
from moabb.paradigms import MotorImagery
import scipy.io
from scipy.signal import resample


class LeeMI:
    def __init__(self):
        self._dataset = Lee2019_MI(test_run=False)
        self._subjects = list(np.linspace(1, 54, 54, dtype=int))
        self._parent_dir = 'dataset'

    def get_dataset(self):
        for subject in self._subjects:
            self._dataset.download(subject_list=[subject])
            self.get_subject_data(subject)

    def get_subject_data(self, subject):
        self.access_subject_file(subject)
        np.save(f'{self._parent_dir}/subj_{subject}_data', self._data)
        np.save(f'{self._parent_dir}/subj_{subject}_label', self._label)

    def access_subject_file(self, subject):
        mat_dir = self._dataset.data_path(subject)
        ses1_data = scipy.io.loadmat(mat_dir[0])
        ses1_data, ses1_label = self.load_session_data(ses1_data)

        ses2_data = scipy.io.loadmat(mat_dir[1])
        ses2_data, ses2_label = self.load_session_data(ses2_data)

        self._data = np.concatenate((ses1_data, ses2_data))
        self._label = np.concatenate((ses1_label, ses2_label))-1

    def load_session_data(self, data):
        train_dataset = data['EEG_MI_train']
        test_dataset = data['EEG_MI_test']

        train_data = train_dataset['smt'].item()
        train_label = train_dataset['y_dec'].item()

        test_data = test_dataset['smt'].item()
        test_label = test_dataset['y_dec'].item()

        data = np.concatenate((train_data, test_data), axis=1).transpose((1, 2, 0))
        data = resample(data, 1000, axis=2)
        for i in range(data.shape[0]):
            target_mean = np.mean(data[i])
            target_std = np.std(data[i])
            data[i] = (data[i] - target_mean) / target_std

        label = np.concatenate((train_label, test_label), axis=1)
        return data.transpose((0, 2, 1)), label.transpose((1,0))


class Dataset:
    def __init__(self):
        self.paradigm = LeeMI()

    def get_dataset(self):
        self.paradigm.get_dataset()


if __name__ == '__main__':
    Dataset().get_dataset()