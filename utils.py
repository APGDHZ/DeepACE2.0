# -*- coding: utf-8 -*-
"""
DeepACE
===================================================================================
Copyright (c) 2021, Deutsches HörZentrum Hannover, Medizinische Hochschule Hannover
Author: Tom Gajecki (gajecki.tomas@mh-hannover.de)

*** Optimized DeepACE: DeepACE_mask, by Tom Gajecki & Yichi Zhang ***
*** new model implemented, MSE (mean squared error) and BCE (binary cross-entropy) as loss functions ***

Reference paper:
Tom Gajecki and Waldo Nogueira. An end-to-end deep learning speech coding and denoising
strategy for cochlear implants. In ICASSP 2022-2022 IEEE International Conference on
Acoustics, Speech and Signal Processing (ICASSP), pages 3109–3113. IEEE, 2022.

All rights reserved.
===================================================================================
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from scipy.io import savemat


def setup():
    parser = argparse.ArgumentParser(description='Main configuration')

    parser.add_argument('-top', '--topology', type=str, default="DeepACE-mse15_cr1")

    parser.add_argument('-mo', '--mode', type=str, default='train')

    parser.add_argument('-gpu', '--GPU', type=bool, default=True)

    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-ld', '--model_dir', type=str, default='./model')
    parser.add_argument('-dd', '--data_dir', type=str, default='./data')
    parser.add_argument('-sr', '--sample_rate', type=int, default=16e3)
    parser.add_argument('-c', '--causal', type=bool, default=True)
    parser.add_argument('-me', '--max_epoch', type=int, default=100)
    parser.add_argument('-bs', '--batch_size', type=int, default=2)
    parser.add_argument('-k', '--skip', type=bool, default=True)
    parser.add_argument('-d', '--duration', type=int, default=4)

    parser.add_argument('-N', '-N', type=int, default=64)
    parser.add_argument('-L', '-L', type=int, default=32)
    parser.add_argument('-B', '-B', type=int, default=64)
    parser.add_argument('-H', '-H', type=int, default=128)
    parser.add_argument('-S', '-S', type=int, default=32)
    parser.add_argument('-P', '-P', type=int, default=3)
    parser.add_argument('-X', '-X', type=int, default=8)
    parser.add_argument('-R', '-R', type=int, default=3)

    parser.add_argument('-csr', '--channel_stim_rate', type=int, default=1000)
    parser.add_argument('-bl', '--block_length', type=int, default=128)
    parser.add_argument('-M', '--n_electrodes', type=int, default=22)

    args = parser.parse_args()

    return args


def write_to_lgf(model, ds, args, path):
    c = 0
    for inp in ds:
        c += 1
    j = 1
    i = 0

    print("\nWriting predictions to LGF...\n")

    for inp in tqdm(ds, total=c, position=0, leave=True):

        prediction = pad(inp, args, model)
        lgf = prediction.numpy()
        fname = os.path.join(path, '{}_test-output.mat'.format(i))
        savemat(fname, {'lgf': lgf})
        j += 1
        i += 1
        if j == c - 1:
            print("\nDone!")


def pad(inp, args, model):
    global prediction_final
    original_length = inp[0].shape[-1]
    slices = (original_length - tf.math.floormod(original_length,
                                                 args.duration * int(args.sample_rate)).numpy()) // (
                     args.duration * int(args.sample_rate)) + 1
    inp_pad = np.zeros((1, slices * int(args.sample_rate) * args.duration), dtype=np.float32)
    inp_pad[0][0:original_length] = inp[0]

    for i in range(slices):
        x = np.expand_dims(
            inp_pad[0][i * int(args.sample_rate) * args.duration:(i + 1) * int(args.sample_rate) * args.duration],
            axis=0)
        prediction = model(x)
        prediction = tf.squeeze(input=prediction)
        if i == 0:
            prediction_final = prediction
        else:
            prediction_final = tf.concat([prediction_final, prediction], axis=0)
    length_out = original_length // 16

    return prediction_final[0:length_out, :]
