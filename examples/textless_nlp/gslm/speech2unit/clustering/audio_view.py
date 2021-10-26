#!/usr/bin/env python

import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tgt


#def plot_audio(wav, sr=16000, n_fft=1024, hop_length=256, win_length=1024, n_mels=80):
#def plot_audio(wav, sr=16000, n_fft=2048, hop_length=512, win_length=2048, n_mels=128,
def plot_audio(wav, sr=16000, n_fft=512, hop_length=128, win_length=512, n_mels=128,
               textgrids_dir=None, rle_units=None):
    y, sr = librosa.load(wav, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                       win_length=win_length, n_mels=n_mels)
    #S = librosa.feature.melspectrogram(y=y, sr=sr)
    #fig, (trans_ax, spec_ax) = plt.subplots(nrows=2, ncols=1, sharex=True, tight_layout=True)
    fig = plt.figure(constrained_layout=False)
    gs = fig.add_gridspec(nrows=6, ncols=1, hspace=0)

    spec_ax = fig.add_subplot(gs[3:, :])
    word_ax = fig.add_subplot(gs[0, :], ylabel='Words', sharex=spec_ax)
    phone_ax = fig.add_subplot(gs[1, :], ylabel='Phones', sharex=spec_ax)
    unit_ax = fig.add_subplot(gs[2, :], ylabel='Units', sharex=spec_ax)
    for ax in [word_ax, phone_ax, unit_ax]:
        ax.axes.xaxis.set_visible(False)
        #ax.axes.yaxis.set_visible(False)
        ax.axes.yaxis.set_ticks([])
        ax.axes.yaxis.set_ticklabels([])
        ax.set_ylabel(ax.get_ylabel(), rotation='horizontal', ha='right', va='center')
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
    word_ax.spines['top'].set_visible(False)
    #spec_ax.spines['top'].set_visible(False)

    S_db = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear',
                                   sr=sr, hop_length=hop_length,
                                   ax=spec_ax, cmap='viridis') # viridis, magma, gray_r
    spec_ax.set_ylabel('Frequency (Hz)')
    spec_ax.set_xlabel('Time (s)')
    #spec_ax.set_yticks(ticks=range(0, 8001, 1000))
    #spec_ax.set_xticks(ticks=np.arange(0, 6, 0.5))
    #plt.xlim([0, 1]) # limit time view
    plt.xlim([0.45, 0.75])
    #plt.ylim([0, 5000]) # doesn't work to limit freq view?
    #fig.colorbar(img, ax=ax, format='%+2.0f dB')
    #ax.set(title='Mel-frequency spectrogram')

    if textgrids_dir is not None:
        utt_id = os.path.splitext(os.path.basename(wav))[0]
        tgf = os.path.join(textgrids_dir, f'{utt_id}.TextGrid')
        textgrid = tgt.io.read_textgrid(tgf, include_empty_intervals=True)
        
        tier = textgrid.get_tier_by_name('words')
        words, durs, start, end = parse_textgrid(tier, sr, hop_length)
        #print(words, durs, sum(durs), start, end)
        total_dur = 0
        for i, (word, dur) in enumerate(zip(words, durs)):
            word_ax.axvline(total_dur, ls='--')
            #word_ax.annotate(word, xy=(total_dur - 0.05, abs(0.3 - (i % 2))), ha='right', va='center')
            word_ax.annotate(word, xy=(total_dur + 0.002, 0.1), ha='left', rotation='vertical')
            total_dur += dur

        tier = textgrid.get_tier_by_name('phones')
        phones, durs, start, end = parse_textgrid(tier, sr, hop_length)
        #print(words, durs, sum(durs), start, end)
        total_dur = 0
        for i, (phone, dur) in enumerate(zip(phones, durs)):
            phone_ax.axvline(total_dur, ls='--', color='green')
            #phone_ax.annotate(phone, xy=(total_dur - 0.05, abs(0.3 - (i % 2))), ha='right', va='center')
            phone_ax.annotate(phone, xy=(total_dur + 0.002, 0.5), va='center')
            total_dur += dur

    # run-length encoded units
    total_dur = 0
    for i, (unit, run) in enumerate(rle_units):
        unit_ax.axvline(total_dur, ls='--', color='orange')
        #dur = (1 / sr) * run
        dur = 0.02 * run # hubert samples at 50 Hz
        unit_ax.annotate(str(unit), xy=(total_dur + 0.002, abs(0.25 - (i % 2))), va='center')
        total_dur += dur

    #plt.show()
    plt.savefig('spec.pdf')
    plt.close()


def parse_textgrid(tier, sampling_rate, hop_length):
    sil_tokens = ["sil", "sp", "spn", ""]
    start_time = tier[0].start_time
    end_time = tier[-1].end_time
    tokens = []
    durations = []
    for i, t in enumerate(tier._objects):
        s, e, p = t.start_time, t.end_time, t.text
        if p not in sil_tokens:
            tokens.append(p)
        else:
            if (i == 0) or (i == len(tier) - 1):
                # leading or trailing silence
                tokens.append("sil")
            else:
                # short pause between words
                tokens.append("sp")
        #r = sampling_rate / hop_length
        #durations.append(int(np.ceil(e * r) - np.ceil(s * r)))
        durations.append(e - s)
    return tokens, durations, start_time, end_time

