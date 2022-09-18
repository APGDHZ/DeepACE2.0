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

import tensorflow as tf



class _ChannelNorm(tf.keras.layers.Layer):
    def __init__(self, encoded_len, channel_size, **kwargs):
        super(_ChannelNorm, self).__init__(**kwargs)
        self.encoded_len = encoded_len
        self.dt = tf.float32
        self.channel_size = channel_size
        w_init = tf.keras.initializers.GlorotNormal()
        self.gamma = tf.Variable(initial_value=w_init(shape=(1, 1,
                                                             self.channel_size), dtype=self.dt),
                                 trainable=True, name="CN_gamma")
        self.beta = tf.Variable(initial_value=w_init(shape=(1, 1,
                                                            self.channel_size), dtype=self.dt),
                                trainable=True, name="CN_beta")

    def get_config(self):
        config = super(_ChannelNorm, self).get_config().copy()
        config.update({
            'encoded_len': self.encoded_len,
            'channel_size': self.channel_size,
            'dt': self.dt})
        return config

    def call(self, inputs, *args, **kwargs):
        E = tf.reshape(
            tf.reduce_mean(inputs, axis=[2], name="E_CN_rm"), [-1, self.encoded_len, 1], name="E_CN_reshape")
        Var = tf.reshape(tf.reduce_mean((inputs - E) ** 2,
                                        axis=[2], name="Var_CN_rm"), [-1, self.encoded_len, 1], name="Var_CN_reshape")
        return ((inputs - E) / tf.math.sqrt(Var + tf.keras.backend.epsilon())) * self.gamma + self.beta


class _GlobalNorm(tf.keras.layers.Layer):
    def __init__(self, channel_size, **kwargs):
        super(_GlobalNorm, self).__init__(**kwargs)
        self.dt = tf.float32
        self.channel_size = channel_size
        w_init = tf.keras.initializers.GlorotNormal()
        self.gamma = tf.Variable(initial_value=w_init(shape=(1, 1,
                                                             self.channel_size), dtype=self.dt),
                                 trainable=True, name="GN_gamma")
        self.beta = tf.Variable(initial_value=w_init(shape=(1, 1,
                                                            self.channel_size), dtype=self.dt),
                                trainable=True, name="GN_beta")

    def get_config(self):
        config = super(_GlobalNorm, self).get_config().copy()
        config.update({
            'channel_size': self.channel_size,
            'dt': self.dt})
        return config

    def call(self, inputs, *args, **kwargs):
        E = tf.reshape(tf.reduce_mean(inputs, axis=[1, 2], name="E_GN_rm"), [-1, 1, 1], name="E_GN_reshape")
        Var = tf.reshape(
            tf.reduce_mean((inputs - E) ** 2, axis=[1, 2], name="Var_GN_rm"), [-1, 1, 1], name="Var_GN_reshape")
        return ((inputs - E) / tf.math.sqrt(Var + tf.keras.backend.epsilon())) * self.gamma + self.beta


class Rectifier(tf.keras.layers.Layer):
    def __init__(self, initializer="he_normal", **kwargs):
        super(Rectifier, self).__init__(**kwargs)
        self.initializer = tf.keras.initializers.get(initializer)

    def build(self, input_shape):
        output_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(output_dim * 2, output_dim),
            initializer=self.initializer,
            name="kernel",
            trainable=True)

    def get_config(self):
        config = super(Rectifier, self).get_config()
        config.update({"initializer": tf.keras.initializers.serialize(self.initializer)})
        return config

    def call(self, inputs, *args, **kwargs):
        inputs -= tf.reduce_mean(inputs, axis=-1, keepdims=True)
        pos = tf.keras.activations.relu(inputs)
        neg = tf.keras.activations.relu(-inputs)
        concatenated = tf.concat([pos, neg], axis=-1)
        mixed = tf.matmul(concatenated, tf.math.abs(self.kernel))
        return mixed


class Encoder(tf.keras.layers.Layer):
    def __init__(self, N, L, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.N = N
        self.L = L

    def build(self, inputs):
        self.encoder = tf.keras.layers.Conv1D(
            filters=self.N,
            kernel_size=self.L,
            strides=self.L // 2,
            activation=Rectifier(),
            padding="same",
            name="encode_conv1d")

    def get_config(self):
        config = super(Encoder, self).get_config().copy()
        config.update({
            'layers': self.encoder,
            'n': self.N,
            'l': self.L})
        return config

    def call(self, inputs, **kwargs):
        encoded_input = self.encoder(inputs=tf.expand_dims(inputs, -1))
        return encoded_input


class TCN(tf.keras.layers.Layer):
    def __init__(self, N, L, B, H, S, P, X, R, causal, skip, duration, sample_rate, block_shift, GPU, **kwargs):
        super(TCN, self).__init__(**kwargs)
        self.N = N
        self.L = L
        self.B = B
        self.H = H
        self.S = S
        self.P = P
        self.X = X
        self.R = R
        self.GPU = GPU
        self.skip = skip
        self.causal = causal
        self.duration = duration
        self.block_shift = block_shift
        self.sample_rate = sample_rate

        self.encoded_len = int(tf.math.ceil(self.duration * self.sample_rate / self.block_shift))

        if self.causal:
            self.padding = "causal"
        else:
            self.padding = "same"

    def build(self, inputs):

        if self.causal:
            inp_norm = _ChannelNorm(self.encoded_len, self.N, name="Input_Channel_Norm")
            frst_norm = _ChannelNorm(self.encoded_len, self.H, name="First_Channel_Norm")
            scnd_norm = _ChannelNorm(self.encoded_len, self.H, name="Second_Channel_Norm")
        else:
            inp_norm = _GlobalNorm(self.N, name="Input_Global_Norm")
            frst_norm = _GlobalNorm(self.H, name="First_Global_Norm")
            scnd_norm = _GlobalNorm(self.H, name="Second_Global_Norm")

        layers = {"input_norm": inp_norm,
                  "bottleneck": tf.keras.layers.Conv1D(self.B, 1, 1, name="bottleneck")}

        for r in range(self.R):
            for x in range(self.X):
                this_block = "block_{}_{}_".format(r, x)

                layers[this_block + "first_1x1_conv"] = tf.keras.layers.Conv1D(
                    filters=self.H, kernel_size=1, name="first_1x1_conv")

                layers[this_block + "first_PReLU"] = tf.keras.layers.PReLU(
                    shared_axes=[1], name="first_PReLU")

                layers[this_block + "first_norm"] = frst_norm

                if self.GPU:
                    layers[this_block + "dw_conv"] = tf.keras.layers.Conv1D(self.H, self.P,
                                                                            padding=self.padding,
                                                                            dilation_rate=2 ** x, groups=self.H)

                layers[this_block + "second_PReLU"] = tf.keras.layers.PReLU(
                    shared_axes=[1], name="second_PReLU")

                layers[this_block + "second_norm"] = scnd_norm

                layers[this_block + "res_1x1_conv"] = tf.keras.layers.Conv1D(
                    filters=self.B, kernel_size=1, name="res_1x1_conv")

                layers[this_block + "skip_1x1_conv"] = tf.keras.layers.Conv1D(
                    filters=self.S, kernel_size=1, name="skip_1x1_conv")
        self.lrs = layers

    def get_config(self):
        config = super(TCN, self).get_config().copy()
        config.update({
            'layers': self.lrs,
            'n': self.N,
            'l': self.L,
            'b': self.B,
            'h': self.H,
            's': self.S,
            'p': self.P,
            'x': self.X,
            'r': self.R,
            'gpu': self.GPU,
            'c': self.causal,
            'skip': self.skip,
            'sr': self.sample_rate,
            'duration': self.duration,
            'enc_len': self.encoded_len})
        return config

    def call(self, encoded_input, **kwargs):

        self.encoded_len = encoded_input.shape[1]

        norm_input = self.lrs["input_norm"](encoded_input)

        block_input = self.lrs["bottleneck"](norm_input)

        skip_connections = 0

        for r in range(self.R):

            for x in range(self.X):

                now_block = "block_{}_{}_".format(r, x)

                block_output = self.lrs[now_block + "first_1x1_conv"](block_input)

                block_output = self.lrs[now_block + "first_PReLU"](block_output)

                block_output = self.lrs[now_block + "first_norm"](block_output)

                if self.GPU:
                    block_output = self.lrs[now_block + "dw_conv"](block_output)

                block_output = self.lrs[now_block + "second_PReLU"](block_output)

                block_output = self.lrs[now_block + "second_norm"](block_output)

                residual = self.lrs[now_block + "res_1x1_conv"](block_output)

                skip = self.lrs[now_block + "skip_1x1_conv"](block_output)

                block_input = residual + block_input

                skip_connections = skip_connections + skip

        if self.skip:
            return skip_connections
        else:
            return block_input


class Masker(tf.keras.layers.Layer):
    def __init__(self, n_electrodes, **kwargs):
        super(Masker, self).__init__(**kwargs)
        self.n_electrodes = n_electrodes

    def build(self, inputs):
        self.prelu = tf.keras.layers.PReLU(shared_axes=[1], name="decode_PReLU")
        self.decode = tf.keras.layers.Conv1D(self.n_electrodes, 1, 1, name="1x1_conv_decoder")
        self.activation = tf.keras.activations.softmax
        self.envdet_c1 = tf.keras.layers.Conv1D(64, 1, 1,
                                                      name="1d_conv_envdet_c1")
        self.prelu_c1 = tf.keras.layers.PReLU(shared_axes=[1], name="PReLU_c1")
        self.envdet_c2 = tf.keras.layers.Conv1D(64, 3, 1, padding='same',
                                                name="1d_conv_envdet_c2")
        self.prelu_c2 = tf.keras.layers.PReLU(shared_axes=[1], name="PReLU_c2")
        self.envdet_c3 = tf.keras.layers.Conv1D(self.n_electrodes, 3, 1, padding='same',
                                               name="1d_conv_envdet_c3")
        self.prelu_c3 = tf.keras.layers.PReLU(shared_axes=[1], name="PReLU_c3")


    def get_config(self):
        config = super(Masker, self).get_config().copy()
        config.update({
            'prelu': self.prelu,
            'decode': self.decode,
            'activation': self.activation,
            'envdet_c1': self.envdet_c1,
            'prelu_c1': self.prelu_c1,
            'envdet_c2': self.envdet_c2,
            'prelu_c2': self.prelu_c2,
            'envdet_c3': self.envdet_c3,
            'prelu_c3': self.prelu_c3
        })
        return config

    def call(self, inputs, encoded_input=None, **kwargs):
        inputs = self.prelu(inputs)
        sep_output = self.decode(inputs)
        mask = self.activation(sep_output)
        encoded_input = self.envdet_c1(encoded_input)
        encoded_input = self.prelu_c1(encoded_input)
        encoded_input = self.envdet_c2(encoded_input)
        encoded_input = self.prelu_c2(encoded_input)
        encoded_input = self.envdet_c3(encoded_input)
        encoded_input = self.prelu_c3(encoded_input)
        return mask * encoded_input, mask


class Decoder(tf.keras.layers.Layer):
    def __init__(self, n_electrodes, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.n_electrodes = n_electrodes

    def build(self, inputs):
        self.deconv = tf.keras.layers.Conv1DTranspose(self.n_electrodes, 1,
                                                      activation=tf.keras.layers.ReLU(max_value=1.0),
                                                      name="1d_deconv")

    def get_config(self):
        config = super(Decoder, self).get_config().copy()
        config.update({
            'deconv': self.deconv})
        return config

    def call(self, inputs, **kwargs):
        output = self.deconv(inputs)
        return output


class DeepACE():
    def __init__(self, args):
        self.N = args.N
        self.L = args.L
        self.B = args.B
        self.H = args.H
        self.S = args.S
        self.P = args.P
        self.X = args.X
        self.R = args.R
        self.GPU = args.GPU
        self.skip = args.skip
        self.top = args.topology
        self.causal = args.causal
        self.duration = args.duration
        self.sample_rate = args.sample_rate
        self.n_electrodes = args.n_electrodes

        self.block_shift = int(tf.math.ceil(self.sample_rate / args.channel_stim_rate))

        self.model_name = self.top

        self.encoder = Encoder(self.N, self.L, name="Encoder")

        self.TCN = TCN(self.N, self.L, self.B, self.H, self.S, self.P,
                       self.X, self.R,
                       self.causal, self.skip, self.duration,
                       self.sample_rate, self.block_shift,
                       self.GPU, name="TCN")

        self.masker = Masker(self.n_electrodes, name="Masker")

        self.decoder = Decoder(self.n_electrodes, name="Decoder")

    def call(self):

        inp = tf.keras.Input(shape=(None,), name="Input1")

        enc_inp = self.encoder(inp)

        skp = self.TCN(enc_inp)

        masked, mask = self.masker(skp, enc_inp)

        out = self.decoder(masked)

        model = tf.keras.Model(inputs=inp, outputs=[out, mask], name=self.model_name)

        for i, w in enumerate(model.weights):
            split_name = w.name.split('/')
            if len(split_name) > 1:
                new_name = split_name[0] + '_' + str(i) + '/' + split_name[1] + '_' + str(i) + '/' + split_name[-1]
                model.weights[i]._handle_name = new_name
            else:
                new_name = "new" + '_' + str(i) + '/' + "layer_name" + '_' + str(i) + '/' + split_name[-1]

        return model

