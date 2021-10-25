# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gc
import os
import random
import shutil
import numpy as np

import torch
import tqdm
from examples.textless_nlp.gslm.speech2unit.pretrained.cpc_feature_reader import (
    CpcFeatureReader,
)
from examples.textless_nlp.gslm.speech2unit.pretrained.hubert_feature_reader import (
    HubertFeatureReader,
)
from examples.textless_nlp.gslm.speech2unit.pretrained.logmel_feature_reader import (
    LogMelFeatureReader,
)
from examples.textless_nlp.gslm.speech2unit.pretrained.w2v2_feature_reader import (
    Wav2VecFeatureReader,
)


def get_feature_reader(feature_type):
    if feature_type == "logmel":
        return LogMelFeatureReader
    elif feature_type == "hubert":
        return HubertFeatureReader
    elif feature_type == "w2v2":
        return Wav2VecFeatureReader
    elif feature_type == "cpc":
        return CpcFeatureReader
    else:
        raise NotImplementedError(f"{feature_type} is not supported.")


def get_feature_iterator(
    feature_type, checkpoint_path, layer, manifest_path, sample_pct
):
    file_path_list = read_manifest(manifest_path)
    if sample_pct < 1.0:
        file_path_list = random.sample(
            file_path_list, int(sample_pct * len(file_path_list))
        )

    feature_reader_cls = get_feature_reader(feature_type)
    reader = feature_reader_cls(
        checkpoint_path=checkpoint_path, layer=layer
    )

    def iterate():
        for file_path in file_path_list:
            feats = reader.get_feats(file_path)
            yield feats.cpu().numpy()

    return iterate, file_path_list


def get_features(
    feature_type, checkpoint_path, layer, manifest_path, sample_pct, flatten
):
    generator, file_path_list = get_feature_iterator(
        feature_type=feature_type,
        checkpoint_path=checkpoint_path,
        layer=layer,
        manifest_path=manifest_path,
        sample_pct=sample_pct,
    )
    iterator = generator()

    features_list = []
    for features in tqdm.tqdm(iterator, total=len(file_path_list)):
        features_list.append(features)

    # Explicit clean up
    del iterator
    del generator
    gc.collect()
    torch.cuda.empty_cache()

    features_list = np.asarray(features_list)
    if flatten:
        return np.concatenate(features_list)

    return features_list


def get_and_dump_features(
    feature_type,
    checkpoint_path,
    layer,
    manifest_path,
    sample_pct,
    flatten,
    out_features_path,
):
    # Feature extraction
    features_batch = get_features(
        feature_type=feature_type,
        checkpoint_path=checkpoint_path,
        layer=layer,
        manifest_path=manifest_path,
        sample_pct=sample_pct,
        flatten=flatten,
    )

    # Save features
    out_dir_path = os.path.dirname(out_features_path)
    os.makedirs(out_dir_path, exist_ok=True)
    shutil.copyfile(
        manifest_path,
        os.path.join(out_dir_path, os.path.basename(manifest_path)),
    )
    np.save(out_features_path, features_batch)

    return features_batch


def get_and_dump_features_per_utt(
    feature_type,
    checkpoint_path,
    layer,
    manifest_path,
    sample_pct,
    out_features_path,
):
    os.makedirs(out_features_path, exist_ok=True)
    shutil.copyfile(
        manifest_path,
        os.path.join(out_features_path, os.path.basename(manifest_path)),
    )

    generator, file_path_list = get_feature_iterator(
        feature_type=feature_type,
        checkpoint_path=checkpoint_path,
        layer=layer,
        manifest_path=manifest_path,
        sample_pct=sample_pct,
    )
    iterator = generator()

    for file_path, features in tqdm.tqdm(zip(file_path_list, iterator), total=len(file_path_list)):
        features = np.asarray(features)
        file_path = os.path.basename(file_path)
        out_features_file = os.path.splitext(file_path)[0] + '.npy'
        out_features_file = os.path.join(out_features_path, out_features_file)
        np.save(out_features_file, features)


def read_manifest(manifest_path):
    with open(manifest_path, "r") as fp:
        lines = fp.read().split("\n")
        root = lines.pop(0).strip()
        file_path_list = [
            os.path.join(root, line.split("\t")[0])
            for line in lines
            if len(line) > 0
        ]
    return file_path_list
