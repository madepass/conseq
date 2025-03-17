import numpy as np
from dtaidistance import dtw
import bz2
from random import randrange
import pandas as pd
# Source: https://tech.gorilla.co/how-can-we-quantify-similarity-between-time-series-ed1d0b633ca0


def generate_random_sequence(sequence_length: int) -> np.array:
    random_sequence = np.array([randrange(2) for _ in range(sequence_length)])
    return random_sequence


def simple_matching_coefficient(ts1: list, ts2: list) -> int:
    smc = len(np.where(np.array(ts1) == np.array(ts2))[0])
    return smc


def binary_correlation(ts1: np.array, ts2: np.array) -> float:
    """
    calculate correlation for two binary time series of equal length
    :param ts1: time series 1
    :param ts2: time series 2
    :return: correlation coefficient
    """
    n_matches = len(np.where(ts1 == ts2)[0])
    n_mismatches = len(ts1) - n_matches
    if n_matches == n_mismatches:
        cor = 0
    else:
        cor = (n_matches - n_mismatches) / len(ts1)
    return cor


def binary_cross_correlation(ts1: np.array, ts2: np.array) -> tuple:
    """
    calculate cross correlation between binary time series of equal length
    :param ts1:
    :param ts2:
    :return: the best correlation (and lag associated with it)
    """
    cross_cor = []
    lag = 0
    while lag < len(ts1) // 2:
        cross_cor.append(binary_correlation(ts1[:-lag], ts2[lag:]))
        lag += 1
    best_lag = int(np.where(cross_cor == np.max(cross_cor))[0])
    best_cor = np.max(abs(np.array(cross_cor)))
    # print(cross_cor)
    return best_cor, best_lag


def calc_euclidean(actual, model):
    return np.sqrt(np.sum((actual - model) ** 2))


def calc_mape(actual, model):
    return np.mean(np.abs((actual - model) / actual))


def calc_correlation(actual, model):
    a_diff = actual - np.mean(actual)
    p_diff = model - np.mean(model)
    numerator = np.sum(a_diff * p_diff)
    denominator = np.sqrt(np.sum(a_diff ** 2)) * np.sqrt(np.sum(p_diff ** 2))
    return numerator / denominator


def dynamic_time_warping(actual, model):
    return dtw.distance_fast(actual, model)


class CompressionBasedDissimilarity(object):
    # https://github.com/wannesm/dtaidistance
    def __init__(self, n_letters=7):
        self.bins = None
        self.n_letters = n_letters

    def set_bins(self, bins):
        self.bins = bins

    def sax_bins(self, all_values):
        bins = np.percentile(
            all_values[all_values > 0], np.linspace(0, 100, self.n_letters + 1)
        )
        bins[0] = 0
        bins[-1] = 1e1000
        return bins

    @staticmethod
    def sax_transform(all_values, bins):
        indices = np.digitize(all_values, bins) - 1
        alphabet = np.array([*("abcdefghijklmnopqrstuvwxyz"[:len(bins) - 1])])
        text = "".join(alphabet[indices])
        return str.encode(text)

    def calculate(self, m, n):
        if self.bins is None:
            m_bins = self.sax_bins(m)
            n_bins = self.sax_bins(n)
        else:
            m_bins = n_bins = self.bins
        m = self.sax_transform(m, m_bins)
        n = self.sax_transform(n, n_bins)
        len_m = len(bz2.compress(m))
        len_n = len(bz2.compress(n))
        len_combined = len(bz2.compress(m + n))
        return len_combined / (len_m + len_n)


def compression_based_dissimilarity(actual, model):
    cbd = CompressionBasedDissimilarity()
    return cbd.calculate(actual, model)


def calculate_learning_time(df) -> list:
    """
    Function to calculate learning times for all horizons and subjects
    Learning time defined as episode# after which participant executes correct policy for at least 9/10 subsequent
    trials. Error trials & most difficult trials are excluded
    :param df: dataframe containing all subjects data.
    :return: list of lists. Each sublist contains the H0, H1, and H2 learning times, quantified as the trial # after
    which the participant executes teh correct policy for at least 9/10 subsequent trials (exluding error trials
    and most difficulty trials).
    """
    subject_learning_times = []
    for s in df['subject'].unique():
        subject_learning_time = []
        for h in df['horizon'].unique():
            df_ = df.loc[(df['subject'] == s) & (df['horizon'] == h) & (df['err'] == 0) & (df['diff'] != 0.99) &
                         df['performance'].notna()]
            # perf = [e for e in np.array(df_[['performance']]) if np.isnan(e) == 0]  # extract trial# as well
            perf = df_[['trial', 'performance']].reset_index(drop=True)
            for i, p in perf.iterrows():
                print(p)
                if len(np.where(perf.loc[i:(i + 10), 'performance'] == 1)[0]) >= 9:
                    print('a')
                    learning_time = perf.loc[i, 'trial']
                    break
                if i == (len(perf) - 11):
                    print('b')
                    learning_time = -1
                    break
            subject_learning_time.append(learning_time)
        subject_learning_times.append(subject_learning_time)
    return subject_learning_times


def learning_time(ts):
    for i, t in enumerate(ts):
        if len(np.where(ts[i:(i+10)] == 1)[0]) >= 9:
            learning_time = i
            break
        else:
            learning_time = np.nan  # no learning
    return learning_time


# def learning_time_difference(actual, model):
#
#
#     return abs(compute_learning_time(actual) - compute_learning_time(model))
































