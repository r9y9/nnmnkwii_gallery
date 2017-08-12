"""Prepare acoustic/linguistic/duration features.

usage:
    prepare_features.py [options] <DATA_ROOT>

options:
    --overwrite          Overwrite files
    --max_num_files=<N>  Max num files to be collected. [default: -1]
    -h, --help           show this help message and exit
"""
from __future__ import division, print_function, absolute_import

from docopt import docopt
import numpy as np

from nnmnkwii.datasets import FileDataSource
from nnmnkwii.datasets import FileSourceDataset
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.preprocessing.f0 import interp1d
from nnmnkwii.util import apply_delta_windows
from nnmnkwii.io import hts
from os.path import join
from glob import glob
import pysptk
import pyworld
from scipy.io import wavfile
from tqdm import tqdm
from os.path import basename, splitext, exists
import os
import sys


global DATA_ROOT

fs = 16000
order = 59
frame_period = 5
fftlen = pyworld.get_cheaptrick_fft_size(fs)
alpha = pysptk.util.mcepalpha(fs)
windows = [
    (0, 0, np.array([1.0])),
    (1, 1, np.array([-0.5, 0.0, 0.5])),
    (1, 1, np.array([1.0, -2.0, 1.0])),
]

global max_num_files
max_num_files = -1


class LinguisticSource(FileDataSource):
    def __init__(self, add_frame_features=False, subphone_features=None):
        self.add_frame_features = add_frame_features
        self.subphone_features = subphone_features
        self.test_paths = None
        self.binary_dict, self.continuous_dict = hts.load_question_set(
            join(DATA_ROOT, "questions-radio_dnn_416.hed"))

    def collect_files(self):
        files = sorted(glob(join(DATA_ROOT, "label_state_align", "*.lab")))
        if max_num_files is not None and max_num_files > 0:
            return files[:max_num_files]
        else:
            return files

    def collect_features(self, path):
        labels = hts.load(path)
        features = fe.linguistic_features(
            labels, self.binary_dict, self.continuous_dict,
            add_frame_features=self.add_frame_features,
            subphone_features=self.subphone_features)
        if self.add_frame_features:
            indices = labels.silence_frame_indices().astype(np.int)
        else:
            indices = labels.silence_phone_indices()
        features = np.delete(features, indices, axis=0)

        return features.astype(np.float32)


class DurationFeatureSource(FileDataSource):
    def collect_files(self):
        files = sorted(glob(join(DATA_ROOT, "label_state_align", "*.lab")))
        if max_num_files is not None and max_num_files > 0:
            return files[:max_num_files]
        else:
            return files

    def collect_features(self, path):
        labels = hts.load(path)
        features = fe.duration_features(labels)
        indices = labels.silence_phone_indices()
        features = np.delete(features, indices, axis=0)
        return features.astype(np.float32)


class AcousticSource(FileDataSource):
    def collect_files(self):
        wav_paths = sorted(glob(join(DATA_ROOT, "wav", "*.wav")))
        label_paths = sorted(
            glob(join(DATA_ROOT, "label_state_align", "*.lab")))
        if max_num_files is not None and max_num_files > 0:
            return wav_paths[:max_num_files], label_paths[:max_num_files]
        else:
            return wav_paths, label_paths

    def collect_features(self, wav_path, label_path):
        fs, x = wavfile.read(wav_path)
        x = x.astype(np.float64)
        f0, timeaxis = pyworld.dio(x, fs, frame_period=frame_period)
        f0 = pyworld.stonemask(x, f0, timeaxis, fs)
        spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
        aperiodicity = pyworld.d4c(x, f0, timeaxis, fs)

        bap = pyworld.code_aperiodicity(aperiodicity, fs)
        mgc = pysptk.sp2mc(spectrogram, order=order, alpha=alpha)
        f0 = f0[:, None]
        lf0 = f0.copy()
        nonzero_indices = np.nonzero(f0)
        lf0[nonzero_indices] = np.log(f0[nonzero_indices])
        vuv = (lf0 != 0).astype(np.float32)
        lf0 = interp1d(lf0, kind="slinear")

        mgc = apply_delta_windows(mgc, windows)
        lf0 = apply_delta_windows(lf0, windows)
        bap = apply_delta_windows(bap, windows)

        features = np.hstack((mgc, lf0, vuv, bap))

        # Cut silence frames by HTS alignment
        labels = hts.load(label_path)
        features = features[:labels.num_frames()]
        indices = labels.silence_frame_indices()
        features = np.delete(features, indices, axis=0)

        return features.astype(np.float32)


if __name__ == "__main__":
    args = docopt(__doc__)
    DATA_ROOT = args["<DATA_ROOT>"]
    DST_ROOT = DATA_ROOT
    max_num_files = int(args["--max_num_files"])
    overwrite = args["--overwrite"]

    # Features required to train duration model
    # X -> Y
    # X: linguistic
    # Y: duration
    X_duration_source = LinguisticSource(
        add_frame_features=False, subphone_features=None)
    Y_duration_source = DurationFeatureSource()

    X_duration = FileSourceDataset(X_duration_source)
    Y_duration = FileSourceDataset(Y_duration_source)

    # Features required to train acoustic model
    # X -> Y
    # X: linguistic
    # Y: acoustic
    X_acoustic_source = LinguisticSource(
        add_frame_features=True, subphone_features="full")
    Y_acoustic_source = AcousticSource()
    X_acoustic = FileSourceDataset(X_acoustic_source)
    Y_acoustic = FileSourceDataset(Y_acoustic_source)

    # Save as files
    X_duration_root = join(DST_ROOT, "X_duration")
    Y_duration_root = join(DST_ROOT, "Y_duration")
    X_acoustic_root = join(DST_ROOT, "X_acoustic")
    Y_acoustic_root = join(DST_ROOT, "Y_acoustic")

    skip_duration_feature_extraction = exists(
        X_duration_root) and exists(Y_duration_root)
    skip_acoustic_feature_extraction = exists(
        X_acoustic_root) and exists(Y_acoustic_root)

    if overwrite:
        skip_acoustic_feature_extraction = False
        skip_duration_feature_extraction = False

    for d in [X_duration_root, Y_duration_root, X_acoustic_root, Y_acoustic_root]:
        if not os.path.exists(d):
            print("mkdirs: {}".format(d))
            os.makedirs(d)

    # Save features for duration model
    if not skip_duration_feature_extraction:
        for idx, (x, y) in tqdm(enumerate(zip(X_duration, Y_duration))):
            name = splitext(basename(X_duration.collected_files[idx][0]))[0]
            xpath = join(X_duration_root, name + ".bin")
            ypath = join(Y_duration_root, name + ".bin")
            x.tofile(xpath)
            y.tofile(ypath)
    else:
        print("Features for duration model training found, skipping feature extraction.")

    # Save features for acoustic model
    if not skip_acoustic_feature_extraction:
        for idx, (x, y) in tqdm(enumerate(zip(X_acoustic, Y_acoustic))):
            name = splitext(basename(X_acoustic.collected_files[idx][0]))[0]
            xpath = join(X_acoustic_root, name + ".bin")
            ypath = join(Y_acoustic_root, name + ".bin")
            x.tofile(xpath)
            y.tofile(ypath)
    else:
        print("Features for acousic model training found, skipping feature extraction.")

    sys.exit(0)
