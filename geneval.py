import json
import os
from re import S
from statistics import mean
from tensorflow.keras import backend as K
import glob
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Dense,
    Lambda,
    LSTM,
    Input,
    RepeatVector,
    TimeDistributed,
)
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import copy
from sklearn import preprocessing
import argparse as ap
import pandas as pd
import random
import keras_tuner


class HyperVAE(keras_tuner.HyperModel):
    def __init__(self, sequence_length, input_dim, alpha, learning_rate):
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.alpha = alpha
        self.learning_rate = learning_rate
        """
        Constructor for the class
        """

    def build(self, hp):
        layers = hp.Int("n_layers", 1, 4)
        activation = hp.Choice("activation", values=["tanh", "relu"], default="tanh")
        dense_layer_units = hp.Int(
            "dense_layer_units", min_value=32, max_value=64, step=16
        )
        latent_space_dim = hp.Int("latent_space_dim", min_value=4, max_value=16, step=4)
        first_layer_units = hp.Int(
            "first_layer_units", min_value=32, max_value=512, step=32
        )
        last_layer_units = hp.Int(
            "last_layer_units", min_value=32, max_value=512, step=32
        )

        lstm_layers = list()
        for i in range(layers):
            lstm_layers.append(
                hp.Int(f"lstm_{i}_units", min_value=32, max_value=512, step=32)
            )

        model_input = Input(
            shape=(self.sequence_length, self.input_dim), name="encoder_input"
        )
        enc = LSTM(
            first_layer_units,
            name="encoder_lstm_first",
            return_sequences=True,
            activation=activation,
        )(model_input)
        for i in range(layers):
            enc = LSTM(lstm_layers[i], return_sequences=True, activation=activation)(
                enc
            )
        enc = LSTM(last_layer_units, activation=activation)(enc)
        enc = Dense(dense_layer_units, activation="relu")(enc)
        mu = Dense(latent_space_dim, name="mu")(enc)
        log_var = Dense(latent_space_dim, name="log_variance")(enc)
        # encoder_first = Model(model_input, [mu, log_var], name="encoder_first")
        enc_output = Lambda(
            sample_point_from_normal_distribution, name="encoder_output"
        )([mu, log_var])
        encoder = Model(model_input, enc_output, name="encoder")

        dec_input = Input(shape=(hp.Int("latent_space_dim"),))
        dec = RepeatVector(self.sequence_length)(dec_input)
        dec = Dense(dense_layer_units, activation="relu")(dec)
        dec = LSTM(last_layer_units, return_sequences=True, activation=activation)(dec)
        for i in range(layers):
            dec = LSTM(
                lstm_layers[layers - 1 - i],
                return_sequences=True,
                activation=activation,
            )(dec)
        dec = LSTM(first_layer_units, return_sequences=True, activation=activation)(dec)
        dec_output = TimeDistributed(Dense(self.input_dim))(dec)
        decoder = Model(dec_input, dec_output, name="decoder")

        model_output = decoder(encoder(model_input))
        vae = Model(model_input, model_output, name="autoencoder")
        vae.add_loss(
            vae_loss(
                model_input, model_output, log_var, mu, self.sequence_length, self.alpha
            )
        )

        optimizer = Adam(learning_rate=self.learning_rate)

        vae.compile(loss=None, optimizer=optimizer, metrics=["mse"])

        return vae

    """Builds a model with the hyperparameters provided by the tuner.
    Args:
        self: The object itself.
        data: The data to be added to.

    Returns: The data with the new model added.
    """

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            **kwargs,
        )

    """Fits the model with the data provided. This is the function that is called by the tuner.
    Args:
        self: The object itself.
        data: The data to be added to.
        model: The model to be fitted.
        *args: The arguments to be passed to the model.
        **kwargs: The keyword arguments to be passed to the model.

  """


def add_percentage_anomaly(
    arr,
    anomaly_normal_ratio=0.01,
    noise=0.05,
    wave=False,
    noise_direction="p",
    wave_length=0,
):
    anomaly_count = int(arr.shape[0] * anomaly_normal_ratio)
    print(anomaly_count, "ANOMALIES")

    anomaly_points = list()
    if wave and wave_length != 0:
        anomaly_waves = list()
        for _ in range(anomaly_count):
            max_starting_point = arr.shape[0] - wave_length
            starting_point = random.randrange(0, max_starting_point)
            anomaly_points = list(range(starting_point, starting_point + wave_length))
            anomaly_waves.append(anomaly_points)
    else:
        anomaly_points = random.sample(range(arr.shape[0]), anomaly_count)

    ano_arr = copy.deepcopy(arr)
    if wave and wave_length != 0:
        for wave in anomaly_waves:
            for j in range(arr.shape[1]):
                for i in range(arr.shape[0]):
                    if i in wave:
                        if noise_direction == "p":
                            ano_arr[i][j] += noise * arr[i][j]
                        elif noise_direction == "n":
                            ano_arr[i][j] -= noise * arr[i][j]
    elif not wave:
        for j in range(arr.shape[1]):
            for i in range(arr.shape[0]):
                if i in anomaly_points:
                    if noise_direction == "p":
                        ano_arr[i][j] += noise * arr[i][j]
                    elif noise_direction == "n":
                        ano_arr[i][j] -= noise * arr[i][j]

    # print(ano_arr)
    return ano_arr, anomaly_points


""" Adds anomalies to the data.
    Args:
        arr: Wanted values at the given columns.
        anomaly_normal_ratio: The ratio of the number of anomalies to the number of normal values. ,
        noise: Noise to be added to the values.,
        wave: True if you want to add waves of anomalies. False otherwise. ,
        noise_direction:'p' for positive noise and 'n' for negative noise. ,
        wave_length: Wave length of the anomaly wave. ,

    Returns: Anomaly array and the anomaly points.
    """


def inverse_transform(arr, col_count, means, stds):
    arr = copy.deepcopy(arr)
    for i in range(col_count):
        arr[:, i] = (arr[:, i] * stds[i]) + means[i]

    return arr


""" Invertly transforms the values of the given array.
    Args:
        arr: Anomaly array.,
        col_count: Number of columns in the array.,
        means: Mean of the columns.,
        stds: Standard deviation of the columns.,

    Returns: Inverse transformed array.
    """


def transform(arr, col_count, means, stds):
    arr = copy.deepcopy(arr)
    for i in range(col_count):
        arr[:, i] = (arr[:, i] - means[i]) / stds[i]

    return arr


""" Transforms the array to the given means and standard deviations.
    Args:
        arr: Anomaly array.,
        col_count: Number of columns in the array.,
        means: Mean of the columns.,
        stds: Standard deviation of the columns.,

    Returns: Transformed array.
    """


def gen_seq(id_df, seq_length, seq_cols):

    data_matrix = id_df[seq_cols]
    num_elements = data_matrix.shape[0]

    for start, stop in zip(
        range(0, num_elements - seq_length, 1), range(seq_length, num_elements, 1)
    ):

        yield data_matrix[stop - seq_length : stop].values.reshape((-1, len(seq_cols)))


""" 
    Generates sequences of the given length from the given dataframe.
    Args:
        id_df: Dataframe with the id and the values.,
        seq_length: Length of the sequence.,
        seq_cols: Columns of the sequence.,

    
    """


def prepare_data_non_split(df, columns, seq_len):
    means = list()
    stds = list()
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        means.append(mean)
        stds.append(std)

    sub_train = list()
    for _, train_seq in enumerate(gen_seq(df, seq_len, columns)):
        seq = transform(train_seq, len(columns), means, stds)
        sub_train.append(copy.deepcopy(seq))

    return np.asarray(sub_train), means, stds


"""
    Prepares non-split data for the given dataframe. 
    Args:
        df: Dataframe with the id and the values.,
        columns: Columns of the sequence.,
        seq_len: Length of the sequence.
    Returns: returns the sub_train as an array and the means & standard deviations of the columns. 
    
"""


def prepare_data(df, columns, seq_len, test_dates):
    test_dfs = list()

    means = list()
    stds = list()
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        means.append(mean)
        stds.append(std)

    train_input = list()
    test_input = list()

    test_dfs = list()
    train_dfs = list()
    for date1, date2 in test_dates:
        test_mask = (df.index > date1) & (df.index <= date2)
        train_mask = df.index <= date1
        final_mask = df.index > date2
        test_df = copy.deepcopy(df.loc[test_mask])
        train_df = copy.deepcopy(df.loc[train_mask])
        test_dfs.append(test_df)
        train_dfs.append(train_df)
        df = df.loc[final_mask]

    for train_df in train_dfs:
        sub_train = list()
        for _, train_seq in enumerate(gen_seq(train_df, seq_len, columns)):
            seq = transform(train_seq, len(columns), means, stds)
            sub_train.append(copy.deepcopy(seq))

        train_input.append(np.asarray(sub_train))

    for test_df in test_dfs:
        sub_test = list()
        for _, test_seq in enumerate(gen_seq(test_df, seq_len, columns)):
            t_seq = transform(test_seq, len(columns), means, stds)
            sub_test.append(t_seq)
        test_input.append(np.asarray(sub_test))
        # df = pd.concat([df, test_df, test_df, train_df, train_df]).drop_duplicates(keep=False)
    return train_input, test_input, means, stds


"""
 
    Prepares data for training and testing. 
    Args:
        df: Dataframe with the id and the values.,
        columns: Columns of the sequence.,
        seq_len: Length of the sequence.
        test_dates: Test dates.
    Returns: training and testing data & the means and standard deviations of the columns.
    
"""


def read_csv_single(
    path, index=None, delimiter=None, header=None, parse_dates=True, nrows=None
):
    if not parse_dates:
        # df = pd.read_csv(path, delimiter=delimiter, header=header, index_col=index, parse_dates=parse_dates)
        df = pd.read_csv(
            path,
            delimiter=delimiter,
            header=header,
            parse_dates=parse_dates,
            nrows=nrows,
        )
    elif parse_dates:
        df = pd.read_csv(
            path,
            delimiter=delimiter,
            header=header,
            parse_dates=parse_dates,
            nrows=nrows,
        )
        df = df.set_index(index)
    return df


"""

    Reads the csv file and converts it to a dataframe.
    Args:
        path: Path of the csv file.,
        index: Index column.,
        delimiter: Delimiter of the csv file.,
        header: Header of the csv file.,
        parse_dates: Parse dates of the csv file.,
        nrows: Number of rows to read.,
    Returns: Dataframe of the csv file.
    
"""


def read_csv_multi(
    path, index=None, delimiter=None, parse_dates=False, header=True, nrows=None
):
    all_files = glob.glob(path + "*.csv")
    df_list = []
    for f in all_files:
        df = pd.read_csv(
            f,
            delimiter=delimiter,
            index_col=index,
            parse_dates=parse_dates,
            nrows=nrows,
        )
        df.sort_index(axis=1, inplace=True)
        df_list.append(df)

    combined = pd.concat((f for f in df_list), axis=1)
    combined.sort_index(axis=1, inplace=True)
    return combined


"""
 
    Reads multiple csv files and adds into a list of dataframes.
    Args:
        path: Path of the csv file.,
        index: Index column.,
        delimiter: Delimiter of the csv file.,
        header: Header of the csv file.,
        parse_dates: Parse dates of the csv file.,
        nrows: Number of rows to read.,
    Returns: Dataframe of the all csv files
    
"""


def sample_point_from_normal_distribution(args):
    mu, log_variance = args
    epsilon = K.random_normal(shape=K.shape(mu), mean=0.0, stddev=1.0)
    sampled_point = mu + K.exp(log_variance / 2) * epsilon
    return sampled_point


"""

    Samples a point from a normal distribution.  
        mu: Mean of the normal distribution.,
        log_variance: Log variance of the normal distribution.,

    
    Returns: Sampled point from the normal distribution.

    
"""


def sample_z(args, ano=False):
    mu, log_sigma = args
    eps = K.random_normal(shape=(len(mu),), mean=0.0, stddev=1.0)
    if not ano:
        return mu + K.exp(log_sigma / 2) * eps
    else:
        return mu + K.exp(log_sigma / 2) * eps + 1.09 * log_sigma


"""

    Args:
        mu: Mean of the normal distribution.,
        log_sigma: Log variance of the normal distribution.,
        ano: If true, the normal distribution is anological. If false it is not.,


    
    Returns: if ano is false, makes a sampling from a normal distribution.
             if ano is true, makes a sampling from defined distribution's outliers.
             And deconstructor creates anomalies from these outliers. 
    
    
"""


def vae_loss(original, out, z_log_sigma, z_mean, sequence_length, alpha):

    reconstruction = K.mean(K.square(original - out)) * sequence_length
    kl = -0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))

    return alpha * reconstruction + kl


"""

    Calculates the loss of the VAE.
    Args:
        original: Original sequence.,
        out: Output sequence.,
        z_log_sigma: Log variance of the normal distribution.,
        z_mean: Mean of the normal distribution.,
        sequence_length: Length of the sequence.,
        alpha: Weight of the reconstruction loss.,


    
    Returns: Loss of the VAE.

"""


def autoencoder_loss(original, out):

    reconstruction = K.mean(K.square(original - out))

    return reconstruction


"""
    Calculates the loss of the autoencoder.
    Args:
        original: Original sequence.,
        out: Output sequence.,


    
    Returns: Reconstruction loss.

"""


def create_vae(
    sequence_length,
    input_dim,
    alpha,
    latent_space_dim,
    learning_rate,
    batch_size,
    epochs,
    training_data,
):
    model_input = Input(shape=(sequence_length, input_dim), name="encoder_input")
    enc = LSTM(64, name="encoder_lstm_1", return_sequences=True, activation="tanh")(
        model_input
    )
    enc = LSTM(32, name="encoder_lstm_2", activation="tanh")(enc)
    enc = Dense(32, activation="relu")(enc)
    mu = Dense(latent_space_dim, name="mu")(enc)
    log_var = Dense(latent_space_dim, name="log_variance")(enc)
    encoder_first = Model(model_input, [mu, log_var], name="encoder_first")
    enc_output = Lambda(sample_point_from_normal_distribution, name="encoder_output")(
        [mu, log_var]
    )
    encoder = Model(model_input, enc_output, name="encoder")

    dec_input = Input(shape=(latent_space_dim,))
    dec = RepeatVector(sequence_length)(dec_input)
    dec = Dense(32, activation="relu")(dec)
    dec = LSTM(32, return_sequences=True, activation="tanh")(dec)
    dec = LSTM(64, return_sequences=True, activation="tanh")(dec)
    dec_output = TimeDistributed(Dense(input_dim))(dec)
    decoder = Model(dec_input, dec_output, name="decoder")

    model_output = decoder(encoder(model_input))
    vae = Model(model_input, model_output, name="autoencoder")
    vae.add_loss(
        vae_loss(model_input, model_output, log_var, mu, sequence_length, alpha)
    )

    optimizer = Adam(learning_rate=learning_rate)

    vae.compile(loss=None, optimizer=optimizer)

    vae.fit(
        training_data, batch_size=batch_size, epochs=epochs, shuffle=False, verbose=1
    )

    return encoder, decoder, encoder_first, vae


"""

    Creates the VAE model.
    Args:
        sequence_length: Length of the sequence.,
        input_dim: Dimension of the input.,
        alpha: Weight of the reconstruction loss.,
        latent_space_dim: Dimension of the latent space.,
        learning_rate: Learning rate of the optimizer.,
        batch_size: Batch size of the training data.,
        epochs: Number of epochs.,
        training_data: Training data.,


    
    Returns: Encoder, decoder, encoder_first, vae.

"""


def temporal_distance(target, candidate):
    ttc = 0
    for i, tp in enumerate(target):
        if tp == 1:
            closest = len(target)
            for j, cp in enumerate(candidate):
                if cp == 1:
                    if abs(i - j) < closest:
                        closest = abs(i - j)
            ttc += closest

    ctt = 0
    for j, cp in enumerate(candidate):
        if cp == 1:
            closest = len(candidate)
            for i, tp in enumerate(target):
                if tp == 1:
                    if abs(i - j) < closest:
                        closest = abs(i - j)
            ctt += closest

    return ttc + ctt


"""

    Calculates the temporal distance.
    Args:
        target: Target sequence.,
        candidate: Candidate sequence.,
    


    
    Returns: sum of the target and candidate temporal distance.

"""


def counting_method(target, candidate, range):
    exact_match = 0
    detected_anomaly = 0
    missed_anomaly = 0
    false_anomaly = 0

    for i, tp in enumerate(target):
        if tp == 1:
            if candidate[i] == 1:
                exact_match += 1
            else:
                range_low = max(0, i - range)
                range_high = i + range + 1
                range_window = candidate[range_low:range_high]
                if 1 in range_window:
                    detected_anomaly += 1
                else:
                    missed_anomaly += 1
        else:
            if candidate[i] == 1:
                false_anomaly += 1

    tdir = (exact_match + detected_anomaly) / (
        exact_match + detected_anomaly + missed_anomaly
    )
    dair = (exact_match + detected_anomaly) / (
        exact_match + detected_anomaly + false_anomaly
    )

    return tdir, dair, exact_match, detected_anomaly, missed_anomaly, false_anomaly


"""
 
    -----
    tdir: Temporal distance ratio.
    dair: Detection accuracy ratio.
    exact_match: Number of exact matches.
    detected_anomaly: Number of detected anomalies.
    missed_anomaly: Number of missed anomalies.
    false_anomaly: Number of false anomalies.
    
    Args:
        target: Target sequence.,
        candidate: Candidate sequence.,
        range: Range of the temporal distance.,



"""


def test_anomaly(test_data, stats_generator, decoder):
    length = len(test_data)
    ano_points = random.sample(range(length), length // 20)
    stats = stats_generator.predict(test_data)
    recon = list()
    for i, mean in enumerate(stats[0]):
        std = stats[1][i]
        z = sample_z((mean, std), (i in ano_points))

        rec = decoder.predict(np.array(z.numpy()).reshape(1, len(std)))
        recon.append(rec)

    return np.asarray(recon), ano_points


"""
 
    Tests the anomaly detection.
        
     
    Args:
        test_data: Test data.,
        stats_generator: Stats generator model.,
        decoder: Decoder model.,
    Returns: Reconstructed data, anomaly points.


"""


def get_final_recon(recon):
    final_recon = list()
    for i, window in enumerate(recon):
        if i == 0:
            for point in window:
                final_recon.append(point)
        else:
            final_recon.append(window[-1])

    return np.asarray(final_recon)


"""
 
    If the window is the first one, it adds the point to the final recon.
    If the window is not the first one, it adds the last point of the previous window to the final recon.

    Args: 
        recon: Reconstructed data.,
    Returns: Reconstructed data as a numpy array.


"""


def plot_recon(final_recon, original, col_count, dir):

    ano_dir = dir + "/ano/"
    clean_dir = dir + "/clean/"

    if not os.path.isdir(ano_dir):
        os.makedirs(ano_dir)

    if not os.path.isdir(clean_dir):
        os.makedirs(clean_dir)

    for i in range(col_count):
        plt.figure(figsize=(12, 8))
        plt.plot(original[:, i])
        plt.plot(final_recon[:, i])

        plt.savefig(ano_dir + str(i) + ".png")
        plt.close()

        plt.figure(figsize=(12, 8))
        plt.plot(original[:, i])
        plt.savefig(clean_dir + str(i) + ".png")
        plt.close()


"""
 
    Plots the reconstructed data. It also saves the images. 

    Args: 
        final_recon: Reconstructed data.,
        original: Original data.,
        col_count: Number of columns.,
        dir: Directory to save the images.,
    


"""


def create_autoencoder(sequence_length, input_dim, latent_space_dim, learning_rate):
    model_input = Input(shape=(sequence_length, input_dim), name="encoder_input")
    enc = LSTM(128, name="encoder_lstm_1", return_sequences=True, activation="tanh")(
        model_input
    )
    enc = LSTM(64, name="encoder_lstm_2", return_sequences=True, activation="tanh")(enc)
    enc = LSTM(32, name="encoder_lstm_3", return_sequences=False, activation="tanh")(
        enc
    )
    # enc = Dense(latent_space_dim, name="encoder_l_4", activation='tanh')(enc)
    enc_output = Dense(latent_space_dim, name="enc_output", activation="tanh")(enc)

    encoder = Model(model_input, enc_output, name="encoder")

    dec_input = Input(shape=(latent_space_dim,))
    dec = RepeatVector(sequence_length)(dec_input)
    dec = LSTM(32, name="decoder_lstm_1", return_sequences=True, activation="tanh")(dec)
    dec = LSTM(64, name="decoder_lstm_2", return_sequences=True, activation="tanh")(dec)
    dec = LSTM(128, name="decoder_lstm_3", return_sequences=True, activation="tanh")(
        dec
    )
    # dec = LSTM(latent_space_dim, name="decoder_lstm_1", activation='tanh', return_sequences=True)(dec)
    dec_output = TimeDistributed(Dense(input_dim, activation="tanh"))(dec)
    decoder = Model(dec_input, dec_output, name="decoder")

    model_output = decoder(encoder(model_input))
    autoencoder = Model(model_input, model_output, name="autoencoder")
    autoencoder.add_loss(autoencoder_loss(model_input, model_output))

    optimizer = Adam(learning_rate=learning_rate)

    autoencoder.compile(loss=None, optimizer=optimizer)

    # vae.fit(training_data,
    #     batch_size=batch_size, epochs=epochs, shuffle=False, verbose=1)

    return autoencoder, encoder, decoder


"""
 
    Creates the autoencoder model. LSTM is used for the encoder and decoder.
    Args: 
        sequence_length: Sequence length.,
        input_dim: Input dimension.,
        latent_space_dim: Latent space dimension.,
        learning_rate: Learning rate.,
    Returns: 
        autoencoder: Autoencoder model.,
        encoder: Encoder model.,
        decoder: Decoder model.


"""


def generate_non_vae_anomalies(noise_params, imdir, outdir):
    path = noise_params["path"]
    if noise_params["csv_mode"] == "single":
        df = read_csv_single(
            path, header=0, index=train_params["key"], nrows=noise_params["nrows"]
        )
    else:
        df = read_csv_multi(
            path, index=[noise_params["key"]], header=True, nrows=noise_params["nrows"]
        )
    test_dfs = list()

    test_dfs = list()
    train_dfs = list()

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    for date1, date2 in noise_params["dates"]:
        test_mask = (df.index > date1) & (df.index <= date2)
        train_mask = df.index <= date1
        final_mask = df.index > date2
        test_df = copy.deepcopy(df.loc[test_mask])
        train_df = copy.deepcopy(df.loc[train_mask])
        train_df[noise_params["columns"]].to_csv(outdir + str(9) + "trainData.csv")
        test_dfs.append(test_df)
        train_dfs.append(train_df)
        # df = df.loc[final_mask]

    final_ano = list()
    final_ano_points = list()
    for i, df in enumerate(test_dfs):
        random_ano, random_points = add_percentage_anomaly(
            df[noise_params["columns"]].values,
            noise_params["anomaly_normal_ratio"],
            noise_params["noise"],
            noise_params["wave"],
            noise_params["noise_direction"],
            noise_params["wave_length"],
        )

        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        header = ""
        for x, col in enumerate(noise_params["columns"]):
            if x != len(noise_params) - 1:
                header += col + ","
            else:
                header += col
        np.savetxt(
            outdir + str(i) + "randomNoise.csv",
            random_ano,
            delimiter=",",
            header=header,
            comments="",
        )
        with open(outdir + str(i) + "randomNoiseIndex.json", "w") as f:
            json.dump({"anoindex": random_points}, f)
        final_ano.append(random_ano)
        final_ano_points.append(random_points)
    if imdir != "":
        ano_dir = imdir + "/ano/"

        if not os.path.isdir(ano_dir):
            os.makedirs(ano_dir)

        for i in range(len(noise_params["columns"])):
            plt.figure(figsize=(12, 8))
            plt.plot(test_dfs[0][noise_params["columns"][i]].values)
            plt.plot(random_ano[:, i])

            plt.savefig(ano_dir + str(i) + "randomNoise.png")

        for i in range(len(noise_params["columns"])):
            plt.figure(figsize=(12, 8))
            plt.plot(train_dfs[0][noise_params["columns"][i]].values)

            plt.savefig(ano_dir + str(i) + "randomNoiseClean.png")

    # return final_ano, final_ano_points


"""
 
    
    Args: 
        noise_params: Noise parameters.,
        imdir: Image directory.,
        outdir: Output directory.,
    


"""


def generate_anomalies(train_params, imdir):
    path = train_params["path"]
    df1 = read_csv_single(path, header=0, index=train_params["key"])
    sub_df = copy.deepcopy(df1)
    sub_df = sub_df.loc[:, (sub_df != sub_df.iloc[0]).any()]
    used_cols = train_params["columns"]
    if len(used_cols) == 0:
        used_cols = sub_df.columns

    dates = train_params["dates"]
    test_mask = (sub_df.index > dates[0][0]) & (sub_df.index <= dates[0][1])
    test_df = copy.deepcopy(sub_df.loc[test_mask])
    sub_test = list()
    for _, test_seq in enumerate(gen_seq(test_df, 10, used_cols)):
        sub_test.append(test_seq)

    train_params["input_dim"] = len(used_cols)

    training_data, testing_data, means, stds = prepare_data(
        sub_df, used_cols, train_params["sequence_length"], dates
    )
    training_data = np.asarray(training_data)
    testing_data = np.asarray(testing_data)
    _encoder, decoder, stats_generator, _vae = create_vae(
        train_params["sequence_length"],
        train_params["input_dim"],
        100,
        train_params["latent_space_dim"],
        train_params["learning_rate"],
        train_params["batch_size"],
        train_params["epochs"],
        np.asarray(training_data[0]).astype("float32"),
    )

    ano_recon, ano_points = test_anomaly(testing_data[0], stats_generator, decoder)
    ano_recon = ano_recon.reshape(
        ano_recon.shape[0], ano_recon.shape[2], ano_recon.shape[3]
    )

    final_recon = get_final_recon(ano_recon)
    final_recon = inverse_transform(final_recon, len(used_cols), means, stds)
    final_org = get_final_recon(sub_test)

    if imdir != "":
        plot_recon(final_recon, final_org, len(used_cols), imdir)

    ano_df = pd.DataFrame(final_recon, columns=used_cols)

    return ano_df, ano_points


"""
    
      
    Args: 
        train_params: Training parameters.,
        imdir: Image directory.,
    Returns:
        ano_df: Anomaly dataframe.,
        ano_points: Anomaly points.,

    


"""


def evaluate_model(input):
    target_windows = input["targets"]
    candidate_windows = input["candidates"]

    tdir_arr = list()
    dair_arr = list()
    em_arr = list()
    da_arr = list()
    ma_arr = list()
    fa_arr = list()
    td_arr = list()

    for i, target in enumerate(target_windows):
        tdir, dair, em, da, ma, fa = counting_method(
            target, candidate_windows[i], input["range"]
        )
        td_metric = temporal_distance(target, candidate_windows[i])
        tdir_arr.append(tdir)
        dair_arr.append(dair)
        em_arr.append(em)
        da_arr.append(da)
        ma_arr.append(ma)
        fa_arr.append(fa)
        td_arr.append(td_metric)

    final_pkg = {
        "td": td_arr,
        "tdir": tdir_arr,
        "dair": dair_arr,
        "exact_match": em_arr,
        "detected_anomaly": da_arr,
        "missed_anomaly": ma_arr,
        "false_anomaly": fa_arr,
        "average_td": mean(td_arr),
        "average_tdir": mean(tdir_arr),
        "average_dair": mean(dair_arr),
        "average_em": mean(em_arr),
        "average_da": mean(da_arr),
        "average_ma": mean(ma_arr),
        "average_fa": mean(fa_arr),
    }

    return final_pkg


"""
    If task flag is set to "eval", this function is called to evaluate the model.
    target_windows: Target windows.,
    candidate_windows: Candidate windows.,
    dair_arr: Detection accuracy array.,
    em_arr: Exact match array.,
    da_arr: Detected anomaly array.,
    ma_arr: Missed anomaly array.,
    fa_arr: False anomaly array.,
    td_arr: Temporal distance array.,
    Args:
        input: Input parameters.,
        
    Returns: 
        final_pkg: Constructed package that contains the evaluation metrics.,


"""

if __name__ == "__main__":
    parser = ap.ArgumentParser(
        description="Generate anomalous data or evaluate anomaly detection output"
    )
    parser.add_argument(
        "-t", "--task", help='Task to perform, "ano" or "eval"', type=str, required=True
    )
    parser.add_argument("-f", "--input", help="Input file", type=str, required=True)
    parser.add_argument("-d", "--outdir", help="Directory for output", required=True)

    args = vars(parser.parse_args())

    filedir = args["input"]
    with open(filedir) as f:
        input_file = json.load(f)

    if args["task"] == "ano":
        train_params = input_file["train_params"]
        noise_params = None
        if "noise_params" in input_file.keys():
            noise_params = input_file["noise_params"]
            generate_non_vae_anomalies(
                noise_params, input_file["imdir"], args["outdir"]
            )
            exit()
        ano_df, ano_points = generate_anomalies(
            train_params, input_file["imdir"], args["outdir"]
        )
        if not os.path.isdir(args["outdir"]):
            os.makedirs(args["outdir"])

        ano_df.to_csv(args["outdir"] + "anocsv.csv")
        with open(args["outdir"] + "anoindex.json", "w") as f:
            json.dump({"anoindex": ano_points}, f)
    elif args["task"] == "eval":
        eval_results = evaluate_model(input_file)
        if not os.path.isdir(args["outdir"]):
            os.makedirs(args["outdir"])

        with open(args["outdir"] + "evalresults.json", "w") as f:
            json.dump(eval_results, f)
