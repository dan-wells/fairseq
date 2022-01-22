# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import random
from typing import List, Tuple

import numpy as np

from examples.textless_nlp.gslm.speech2unit.pretrained.utils import (
    read_manifest,
)


def get_audio_files(manifest_path: str) -> Tuple[str, List[str], List[int]]:
    fnames, sizes = [], []
    with open(manifest_path, "r") as f:
        root_dir = f.readline().strip()
        for line in f:
            items = line.strip().split("\t")
            assert (
                len(items) == 2
            ), f"File must have two columns separated by tab. Got {line}"
            fnames.append(items[0])
            sizes.append(int(items[1]))
    return root_dir, fnames, sizes


def load_features(in_features_path, flatten=False, per_utt=False, manifest_path=None,
                  sample_pct=1.0):
    if per_utt:
        # match order of data loaded from flat array using same manifest
        if manifest_path is not None:
            file_path_list = read_manifest(manifest_path)
            if sample_pct < 1.0:
                file_path_list = random.sample(
                    file_path_list, int(sample_pct * len(file_path_list))
                )
            features_files = []
            for f in file_path_list:
                f = os.path.basename(f)
                features_file = os.path.splitext(f)[0] + ".npy"
                features_file = os.path.join(in_features_path, features_file)
                features_files.append(features_file)
        else:
            features_files = glob.glob(os.path.join(in_features_path, "*.npy"))
        features_batch = []
        for features_file in features_files:
            features = np.load(features_file)
            features_batch.append(features)
        features_batch = np.asarray(features_batch, dtype=object)
        if flatten:
            # 1-d array of individual frames
            features_batch = np.concatenate(features_batch)
    else:
        features_batch = np.load(in_features_path, allow_pickle=True)
    return features_batch


def load_features_batched(in_features_path, batch_size, manifest_path=None):
    if manifest_path is not None:
        file_path_list = read_manifest(manifest_path)
        features_files = []
        for f in file_path_list:
            f = os.path.basename(f)
            features_file = os.path.splitext(f)[0] + ".npy"
            features_file = os.path.join(in_features_path, features_file)
            features_files.append(features_file)
    else:
        features_files = glob.glob(os.path.join(in_features_path, "*.npy"))
    n_feats = len(features_files)
    for start in range(0, n_feats, batch_size):
        end = min(start + batch_size, n_feats)
        yield [np.load(f) for f in features_files[start:end]]

