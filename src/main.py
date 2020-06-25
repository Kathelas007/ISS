import os
import sys
from os.path import isfile, join

from os import listdir

import statistics as st
import matplotlib.pyplot as plt

import numpy as np
import soundfile

from scipy.io import wavfile
from scipy.signal import spectrogram
from scipy.signal import get_window

root_dir = os.path.dirname(os.path.dirname(__file__))
sentence_dir = os.path.join(root_dir, 'sentences')
query_dir = os.path.join(root_dir, 'queries')
img_dir = os.path.join(root_dir, 'img')

# frequeryncy
fs = 16000
# number of samples in segment (400 => 512)
N = 512

# sample info
s_lenght = 25e-3 * fs
s_shift = 10e-3 * fs
s_overlap = s_lenght - s_shift


def get_signal(path):
    # load wav
    fss, data = wavfile.read(path)

    # range between (-1, 1)
    data = data / (2 ** 15)

    # sub mean value
    mean_value = np.mean(data)
    data = np.array(list(item - mean_value for item in data))

    return data


def get_framed_signal(signal):
    # segments
    range_end = len(signal) - int(s_lenght)
    range_step = int(s_shift)

    # co hamming??
    segments = np.array(
        list(
            signal[frame_beg: frame_beg + int(s_lenght)] for frame_beg in
            range(0, range_end, range_step)))

    # segments = np.array(
    #     list(np.append(seg, [0] * (N - len(seg))) for seg in segments))

    return segments


def get_density_signal(signal):
    f, t, sgr = spectrogram(signal, fs, get_window('hamming', int(s_lenght)), noverlap=int(s_overlap), nfft=510)

    # f_p = np.array(list(i / N * fs for i in range(0, int(N // 2 - 1))))
    # t_p = np.array(list(i * s_shift / fs for i in range(0, np.math.floor(((len(signal) - s_lenght) / s_shift) - 1))))

    sgr_nn = 10 * np.log10(sgr + 1e-20)

    Gs = np.array(list(10 * np.log10(1.0 / N * np.abs(s) ** 2) for s in sgr_nn))

    return f, t, Gs


def get_features(den_signal):
    # transpose
    switched = den_signal.transpose()

    # reshape  n x 256  => n x 16 x 16
    reshaped = np.reshape(switched, (len(switched), 16, 16))

    # cumpute sums, frames * 16
    features = np.array(list(row.sum() for frame in reshaped for row in frame))

    # reshape frames x 16
    features.resize((len(switched), 16))

    return features


def get_frame_probability(sen_features, query_features):
    Nc = len(sen_features)

    sen_average = np.average(sen_features)
    query_average = np.average(query_features)

    denominator = [None] * Nc
    numerator_s = [None] * Nc
    numerator_q = [None] * Nc

    for i in range(0, Nc):
        denominator[i] = (sen_features[i] - sen_average) * (query_features[i] - query_average)
        numerator_s[i] = (sen_features[i] - sen_average) ** 2
        numerator_q[i] = (query_features[i] - query_average) ** 2

    result = sum(denominator) / (np.sqrt(sum(numerator_s)) * np.sqrt(sum(numerator_q)))

    return result


def get_sentence_probability(sen_features, query_features):
    samples_num = int(len(sen_features) - len(query_features))
    samples_5 = int(samples_num / 5)

    prob_vector = [[0.0 for _ in query_features] for _ in range(0, samples_5)]

    for pp in range(0, samples_5):
        for k in range(0, len(query_features)):
            prob_vector[pp][k] = get_frame_probability(sen_features[5 * pp + k], query_features[k])
        prob_vector[pp] = np.sum(prob_vector[pp]) / len(query_features)

    return prob_vector


def make_signal_im(signal, file):
    t = np.arange(signal.size) / fs

    plt.figure(figsize=(9, 2))
    plt.plot(t, signal)

    plt.gca().set_xlabel('Time [s]')
    plt.gca().set_ylabel('Frequeryncy [Hz]')
    plt.gca().set_title('Signal ' + file.split(".")[0])
    plt.xlim([0, signal.size / fs])

    plt.tight_layout()

    fig1 = plt.gcf()

    img_name = file.split(".")[0] + '_signal.png'
    img_path = os.path.join(img_dir, img_name)
    fig1.savefig(img_path, format='png', bbox_inches='tight', pad_inches=0)

    plt.clf()
    plt.close()


def make_spec_im(Gs, f, t, file):
    plt.figure(figsize=(9, 2))

    plt.pcolormesh(t, f, Gs[0:Gs.size // 2 + 1])
    plt.gca().set_xlabel('Time [s]')
    plt.gca().set_ylabel('Frequeryncy [Hz]')
    plt.gca().set_title('Spectral power density ' + file.split(".")[0])

    plt.tight_layout()

    fig1 = plt.gcf()

    img_name = file.split(".")[0] + '_desity.png'
    img_path = os.path.join(img_dir, img_name)
    fig1.savefig(img_path, format='png', bbox_inches='tight', pad_inches=0)

    plt.clf()
    plt.close()


def make_prob_im(prob, sen_file, query_names, sample_num):
    plt.figure(figsize=(9, 2))
    palette = plt.get_cmap('Set1')

    for i, p in enumerate(prob):
        lst = list((5 * i * s_shift) / fs for i in range(0, len(p)))

        plt.plot(lst, p, color=palette(i), label=query_names[i].split()[0])

    plt.legend(loc='best')

    plt.xlim([0, sample_num])
    plt.ylim([0, 1])
    plt.gca().set_xlabel('Time [s]')
    plt.gca().set_ylabel('Probability')
    plt.gca().set_title('Probability ' + sen_file.split(".")[0])

    plt.tight_layout()

    fig1 = plt.gcf()

    img_name = sen_file.split(".")[0] + '_probability.png'
    img_path = os.path.join(img_dir, img_name)
    fig1.savefig(img_path, format='png', bbox_inches='tight', pad_inches=0)

    plt.clf()
    plt.close()


def main():
    os.makedirs(img_dir, exist_ok=True)

    sen_files = [f for f in os.listdir(sentence_dir) if isfile(join(sentence_dir, f))]
    query_files = [f for f in os.listdir(query_dir) if isfile(join(query_dir, f))]

    sen_features = [None] * len(sen_files)
    query_features = [None] * len(query_files)

    sen_lens = [None] * len(sen_files)

    for i, sen in enumerate(sen_files):
        signal = get_signal(os.path.join(sentence_dir, sen))
        make_signal_im(signal, sen)
        sen_lens[i] = signal.size / fs

        f, t, den_signal = get_density_signal(signal)
        make_spec_im(den_signal, f, t, sen)

        sen_features[i] = get_features(den_signal)

    for i, query in enumerate(query_files):
        signal = get_signal(os.path.join(query_dir, query))
        make_signal_im(signal, query)

        f, t, den_signal = get_density_signal(signal)
        make_spec_im(den_signal, f, t, query)

        query_features[i] = get_features(den_signal)

    probabilities = [[None] * len(query_files)] * len(sen_features)
    for i, sen in enumerate(sen_features):
        for j, query in enumerate(query_features):
            probabilities[i][j] = get_sentence_probability(sen, query)
        make_prob_im(probabilities[i], sen_files[i], query_files, sen_lens[i])

    # for si, s in enumerate(probabilities):
    #     for qi, q in enumerate(s):
    #         if 'ghost' in query_files[qi]:
    #             print(sen_files[si], max(q))
    #         for p in q:
    #             if p >= 0.88:
    #                 if ('233' in sen_files[si] and 'high' in query_files[qi]) or (
    #                         '1943' in sen_files[si] and 'gho' in query_files[qi]):
    #                     print('jooo', sen_files[si], query_files[qi])
    #                 else:
    #                     print('neee', sen_files[si], query_files[qi])
    #             pass


if __name__ == '__main__':
    main()
