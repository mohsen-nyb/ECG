import tensorflow as tf
from torch.utils.data import Dataset
import torch
import numpy as np

def decode_and_normalize(tensor):
    signal = tf.cast(tf.io.parse_tensor(tensor, out_type=tf.int32)[:5000], tf.float32) / 4247.0
    return signal - tf.reduce_mean(signal)


def parse_tfr_element(element):
    data = {
        'ECG_LEAD_I': tf.io.FixedLenFeature([], tf.string),
        'ECG_LEAD_II': tf.io.FixedLenFeature([], tf.string),
        'ECG_LEAD_III': tf.io.FixedLenFeature([], tf.string),
        'ECG_LEAD_AVR': tf.io.FixedLenFeature([], tf.string),
        'ECG_LEAD_AVL': tf.io.FixedLenFeature([], tf.string),
        'ECG_LEAD_AVF': tf.io.FixedLenFeature([], tf.string),
        'ECG_LEAD_V1': tf.io.FixedLenFeature([], tf.string),
        'ECG_LEAD_V2': tf.io.FixedLenFeature([], tf.string),
        'ECG_LEAD_V3': tf.io.FixedLenFeature([], tf.string),
        'ECG_LEAD_V4': tf.io.FixedLenFeature([], tf.string),
        'ECG_LEAD_V5': tf.io.FixedLenFeature([], tf.string),
        'ECG_LEAD_V6': tf.io.FixedLenFeature([], tf.string),
        'abnormal': tf.io.FixedLenFeature([], tf.float32)
    }

    content = tf.io.parse_single_example(element, data)


    ecg_leads = [decode_and_normalize(content[key]) for key in sorted(data.keys()) if key != 'abnormal']
    signal = tf.stack(ecg_leads, axis=0)
    return signal.numpy(), content['abnormal'].numpy()



class ECGDataset(Dataset):
    def __init__(self, tfrecord_files):
        self.signals = []
        self.labels = []

        raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
        for raw_record in raw_dataset:
            signal, label = parse_tfr_element(raw_record)
            self.signals.append(signal)
            self.labels.append(label)


        self.labels = np.array(self.labels)
        print(f'number of all samples: {len(self.labels)}')
        print(f'number samples with abnormal echo: {len(self.labels[self.labels==1])}')
        print(f'percentage of samples with abnormal echo: {len(self.labels[self.labels==1]) / len(self.labels)}')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        signal = torch.tensor(signal, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return signal, label