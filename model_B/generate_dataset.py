
import numpy as np
from moabb.datasets import BNCI2014001, PhysionetMI, Cho2017, Lee2019_MI
from moabb.paradigms import MotorImagery
import scipy.io
from scipy.signal import resample
from tqdm import tqdm


class LeeMI:
    def __init__(self):
        self._dataset = Lee2019_MI(test_run=False)
        self._subjects = list(np.linspace(start=1, stop=54, num=54, dtype=int))
        self._parent_dir = 'raw_dataset'

    def get_dataset(self):
        for subject in tqdm(self._subjects):
            self._dataset.download(subject_list=[subject])
            self.get_subject_data(subject)

    def get_subject_data(self, subject):
        self.access_subject_file(subject)
        for i in range(400):
            ids = (subject-1)*400
            data = self._data[i][np.newaxis, :, :]
            label = self._label[i][np.newaxis, :]
            sid = np.asarray([subject-1])[np.newaxis, :]

            l1 = np.where(self._label == 0)[0]
            l2 = np.where(self._label == 1)[0]
            ref = self._data[[l1[-1], l2[-1]]][np.newaxis, :, :, :]

            np.save(f'{self._parent_dir}/trial/T{ids+i}_D', data)
            np.save(f'{self._parent_dir}/label/T{ids+i}_L', label)
            np.save(f'{self._parent_dir}/sid/T{ids + i}_S', sid)
            np.save(f'{self._parent_dir}/reference/T{ids + i}_R', ref)

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
        # for i in range(data.shape[0]):
        #     target_mean = np.mean(data[i])
        #     target_std = np.std(data[i])
        #     data[i] = (data[i] - target_mean) / target_std

        label = np.concatenate((train_label, test_label), axis=1)
        # return data.transpose((0, 2, 1)), label.transpose((1,0))

        return data, label.transpose((1, 0))


class Dataset:
    def __init__(self, dataset):
        if dataset == 'Lee':
            self.paradigm = LeeMI()
        else:
            raise "Dataset is INVALID"

    def get_dataset(self):
        self.paradigm.get_dataset()


if __name__ == '__main__':
    Dataset(dataset='Lee').get_dataset()