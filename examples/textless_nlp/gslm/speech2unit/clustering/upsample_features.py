#!/usr/bin/env python

import argparse
import glob
import os

import numpy as np
import tqdm


def interpolate(feats):
    feats_interp = np.asarray([np.mean(i, axis=0) for i in zip(feats, feats[1:])])
    # len = (2 * feats.shape[0]) - 1
    feats_up = np.empty((len(feats) + len(feats_interp), feats.shape[1]), dtype=feats.dtype)
    feats_up[0::2] = feats
    feats_up[1::2] = feats_interp
    return feats_up


def double(feats):
    feats_up = np.empty((len(feats) * 2, feats.shape[1]), dtype=feats.dtype)
    feats_up[0::2] = feats
    feats_up[1::2] = feats
    return feats_up


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feats_dir', type=str, help='Directory with feature archives per utterance')
    parser.add_argument('--feats_out', type=str, help='Output directory for upsampled features')
    parser.add_argument('--method', type=str, choices=['interpolate', 'double'], default='interpolate',
        help='Upsampling method, either linear interpolation (default) or doubling frames')
    args = parser.parse_args()

    os.makedirs(args.feats_out, exist_ok=True)

    feats_files = glob.glob(os.path.join(args.feats_dir, '*.npy'))
    for feats_f in tqdm.tqdm(feats_files, 'Upsampling 2x'):
        feats = np.load(feats_f)
        if args.method == 'interpolate':
            feats_up = interpolate(feats)
        else:
            feats_up = double(feats)
        np.save(os.path.join(args.feats_out, os.path.basename(feats_f)), feats_up)
