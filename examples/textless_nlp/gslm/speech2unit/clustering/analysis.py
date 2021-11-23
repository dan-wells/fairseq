#!/usr/bin/env python

import os
import random
from collections import Counter, defaultdict
from itertools import zip_longest

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tgt
import umap
import umap.plot
from scipy.spatial import Voronoi, voronoi_plot_2d
from tabulate import tabulate
from tqdm import tqdm

from examples.hubert.measure_teacher_quality import (
    comp_avg_seg_dur, comp_joint_prob, comp_norm_mutual_info, comp_purity
)

class QuantizedUtterances():
    def __init__(self, quantized_utts_file, wavs_dir=None, textgrids_dir=None,
                 sr=16000, n_fft=512, win_length=512, hop_length=320,
                 n_mels=80, kmeans_model=None, verbose=False):
        self.quantized_utts = self.load_quantized_utts(quantized_utts_file)
        self.wavs_dir = wavs_dir
        self.textgrids_dir = textgrids_dir
        self.sr = sr
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.verbose = verbose

        if self.textgrids_dir is not None:
            self.phone_alignments = self.textgrids_to_durs("phones")
        self.word_alignments = None

        if kmeans_model is not None:
            with open(kmeans_model, "rb") as kmeans_model_file:
                self.kmeans = joblib.load(kmeans_model_file)
                #self.kmeans.cluster_centers_.shape (n_clusters, dim)

    def load_quantized_utts(self, filename):
        """Load quantized utterance metadata file.

        Args:
          filename: Path to metadata file.

        Returns:
          Dictionary mapping utterance IDs to run-length encoded quantized
          unit sequences.
        """
        quantized_utts = {}
        with open(filename) as inf:
            for line in inf:
                utt_id, unit_seq = line.strip().split("|")
                unit_seq = [int(i) for i in unit_seq.split()]
                quantized_utts[utt_id] = self.run_length_encode(unit_seq)
        return quantized_utts

    def run_length_encode(self, tokens):
        """Encode a sequence of repeating tokens as (token, run length) pairs.

        Args:
          tokens: A sequence of tokens characterizing a single utterance.

        Returns:
          A list of (token, run length) tuples providing durations for each
          token in the input sequence.
        """
        run_lengths = []
        dur = 1
        for u0, u1 in zip(tokens, tokens[1:]):
            if u1 == u0:
                dur += 1
            else:
                run_lengths.append((u0, dur))
                dur = 1
        run_lengths.append((u1, dur))
        return run_lengths

    def run_length_decode(self, rle_tokens):
        """Expand (token, run_length) pairs to a sequence of repeating tokens."""
        tokens = []
        for token, run_length in rle_tokens:
            tokens.extend([token] * run_length)
        return tokens

    def textgrids_to_durs(self, tier_label="phones", durs_in_frames=True):
        """Extract token-duration alignments from utterance TextGrid files

        Args:
          tier_label: Name of TextGrid interval tier to read, typically 'phones'
            (default) or 'words'
          durs_in_frames: Boolean to return durations in frames or seconds

        Returns:
          Dictionary mapping utterance IDs to sequences of (token, duration)
          pairs, for all quantized utterances with matching TextGrids
        """
        tg_alignments = {}
        for utt in tqdm(self.quantized_utts, "Loading TextGrids"):
            tokens, durations, _, _ = self.load_textgrid(utt, tier_label, durs_in_frames)
            if tokens is not None:
                tg_alignments[utt] = list(zip(tokens, durations))
        return tg_alignments

    def load_textgrid(self, utt_id, tier_label="phones", durs_in_frames=True):
        """Load and parse selected tier from TextGrid file given an utterance ID.

        Args:
          utt_id: Utterance ID corresponding to TextGrid file to load. Expected
            filepath is {self.textgrids_dir}/{utt_id}.TextGrid
          tier_label: Name of TextGrid interval tier to read, typically 'phones'
            (default) or 'words'
          durs_in_frames: Boolean to return durations in frames or seconds

        Returns:
          tokens: List of tokens in transcription tier
          durations: List of token durations in frames (default) or seconds
          start_time: Starting time index of transcription tier
          end_time: Ending time index of transcription tier, total duration of
            transcribed utterance
        """
        tokens, durations, start_time, end_time = None, None, None, None
        tgf = os.path.join(self.textgrids_dir, "{}.TextGrid".format(utt_id))
        try:
            textgrid = tgt.io.read_textgrid(tgf, include_empty_intervals=True)
            tier = textgrid.get_tier_by_name(tier_label)
            assert isinstance(tier, tgt.core.IntervalTier)
            tokens, durations, start_time, end_time = \
                self.parse_textgrid_tier(tier, durs_in_frames)
        except FileNotFoundError:
            if self.verbose:
                print("No TextGrid found for utterance {}, "
                      "omitting from analysis.".format(utt_id))
        except ValueError:
            if self.verbose:
                print("No tier labelled '{}' found in TextGrid file {}, omitting "
                      "from analysis.".format(tier_label, tgf))
        except AssertionError:
            if self.verbose:
                print("Selected tier '{}' in TextGrid file {} is not an interval "
                      "tier, omitting from analysis.".format(tier_label, tgf))
        return tokens, durations, start_time, end_time

    def parse_textgrid_tier(self, tier, durs_in_frames=True):
        """Parse a single TextGrid interval tier to extract tokens and durations.

        Args:
          tier: tgt.core.IntervalTier with transcription in chosen units
          durs_in_frames: Boolean to return durations in frames or seconds

        Returns:
          tokens: List of tokens in transcription tier
          durations: List of token durations in frames (default) or seconds
          start_time: Starting time index of transcription tier
          end_time: Ending time index of transcription tier, total duration of
            transcribed utterance
        """
        sil_tokens = ["sil", "sp", "spn", ""]
        start_time = tier[0].start_time
        end_time = tier[-1].end_time
        tokens = []
        durations = []
        for i, interval in enumerate(tier.intervals):
            start = interval.start_time
            end = interval.end_time
            token = interval.text
            if token not in sil_tokens:
                tokens.append(token)
            else:
                if (i == 0) or (i == len(tier) - 1):
                    # leading or trailing silence
                    tokens.append("sil")
                else:
                    # short pause between words
                    tokens.append("sp")
            if durs_in_frames:
                fps = self.sr / self.hop_length
                durations.append(int(np.ceil(end * fps) - np.ceil(start * fps)))
            else:
                # durations in seconds
                durations.append(end - start)
        return tokens, durations, start_time, end_time

    def unit_phone_alignment(self, utt, trim=False):
        """Align frame-level sequences of quantized units and phones."""
        units = self.run_length_decode(self.quantized_utts[utt])
        phones = self.run_length_decode(self.phone_alignments[utt])
        if trim:
            # unit sequences tend to be 1 frame short, truncating any final
            # frames < 20 ms in duration
            return list(zip(units, phones))
        else:
            return list(zip_longest(units, phones, fillvalue=""))

    def print_unit_phone_alignment(self, utt):
        """Print frame-level alignments of quantized units and phones."""
        for unit, phone in self.unit_phone_alignment(utt):
            print("{}\t{}".format(unit, phone))

    def word_unit_pronunciations(self, utt, squash_runs=True, sort_words=False):
        """Extract unit pronunciations for each word in an utterance.

        Args:
          utt: ID of utterance to process
          squash_runs: Boolean to squash repeated units to a single token
            (default), else return full repeating strings.
          sort_words: Sort pronunciations alphabetically by word if True
            (better for checking variation per word), else return in the same
            order they appear in the utterance (better for checking original
            context of each word). Default False.

        Returns:
          prons: List of tuples like (word, pronunciation), where pronunciation
            is a tuple of integer unit IDs.
        """
        try:
            words = self.run_length_decode(self.word_alignments[utt])
        except TypeError:  # self.word_alignments is None
            print("Load word alignments from TextGrids first: "
                  "QuantizedUtterances.textgrids_to_durs('words')")
            return None
        units = self.run_length_decode(self.quantized_utts[utt])
        
        prons = []
        w_pron = [units[0]]
        prev_unit = units[0]
        # bit of tricky indexing to account for unit sequences often
        # being one frame shorter than alignments
        for i, (w0, w1) in enumerate(zip(words, words[1:-1]), 1):
            try:
                curr_unit = units[i]  # w1 unit
            except IndexError:  # some are two frames short!
                continue
            if w1 == w0:
                if squash_runs and curr_unit == prev_unit:
                    continue
                w_pron.append(curr_unit)
            elif w1 != w0:
                prons.append((w0, tuple(w_pron)))  # needs to be hashable
                w_pron = [curr_unit]
            prev_unit = curr_unit
        # end of utterance
        prons.append((w1, tuple(w_pron)))

        if sort_words:
            # sort alphabetically by key word
            prons = sorted(prons, key=lambda x: x[0].lower())
        return prons

    def write_word_unit_lexicon(self, lex_file, squash_runs=True, uniq=True,
                                top_n=None, write_counts=False):
        """Write lexicon file with unit pronunciations of words.

        Entries will be written in descending frequency order.

        Args:
          lex_file: Output filename
          squash_runs: Boolean to squash repeated units to a single token
            (default), else write full repeating strings.
          uniq: Boolean, write a single entry per unique pron if True
            (default), else one entry for every instance of each word.
          top_n: Integer to write only the N most frequent pronunciations
            per word, or None to write all (default).
          write_counts: Boolean to write pronunciation counts to lexicon
            file in a third column (default False).
        """
        if self.word_alignments is None:
            self.word_alignments = self.textgrids_to_durs("words")
        prons = defaultdict(list)
        for utt in self.phone_alignments:
            utt_prons = self.word_unit_pronunciations(utt, squash_runs)
            for word, pron in utt_prons:
                prons[word].append(pron)
        with open(lex_file, "w") as outf:
            for word, w_prons in sorted(prons.items()):
                pron_counts = Counter(w_prons)
                for pron, count in pron_counts.most_common(top_n):
                    if write_counts:
                        lex_line = "{}\t{}\t{}\n".format(
                            word, " ".join(str(i) for i in pron), count)
                    else:
                        lex_line = "{}\t{}\n".format(
                            word, " ".join(str(i) for i in pron))
                    if uniq:
                        count = 1
                    for _ in range(count):
                        outf.write(lex_line)

    def mean_token_durations(self, token_type="unit", durs_in_frames=True,
                             exclude_tokens=None, exclude_silence=True):
        """Calculate mean token durations across utterances.

        Args:
          token_type: Either 'unit' (default) or 'phone'. Calculate durations
            for quantized units or phones from TextGrids
          durs_in_frames: Boolean to return durations in frames or seconds
          exclude_tokens: Tokens to exclude from mean duration calculation
          exclude_silence: Boolean to exclude silence symbols when calculating
            phone durations (defaults True)

        Returns:
          mean_duration: Mean duration over all non-excluded tokens
          token_durations: Dictionary mapping tokens to individual mean durations
          excluded_durations: Dictionary mapping excluded tokens to mean durations
        """
        if token_type == "unit":
            utts = self.quantized_utts
        elif token_type == "phone":
            utts = self.phone_alignments

        if exclude_tokens is None:
            exclude_tokens = []
        if exclude_silence:
            exclude_tokens.extend(["sp", "sil", "spn"])
        exclude_tokens = set(exclude_tokens)

        mean_duration = 0
        total_count = 0
        token_durations = defaultdict(int)
        token_counts = defaultdict(int)
        excluded_durations = defaultdict(int)
        excluded_counts = defaultdict(int)
        for utt_id, utt in utts.items():
            for token, dur in utt:
                if not durs_in_frames:
                    # duration in ms
                    dur = dur / (self.sr / self.hop_length) * 1000
                if token in exclude_tokens:
                    excluded_durations[token] += dur
                    excluded_counts[token] += 1
                else:
                    token_durations[token] += dur
                    token_counts[token] += 1
                    mean_duration += dur
                    total_count += 1
        mean_duration /= total_count
        token_durations = {
            k: token_durations[k] / token_counts[k] for k in token_counts}
        excluded_durations = {
            k: excluded_durations[k] / excluded_counts[k] for k in excluded_counts}
        return mean_duration, token_durations, excluded_durations

    def print_mean_token_durations(self, token_type="unit", durs_in_frames=True,
                                   exclude_tokens=None, exclude_silence=True):
        """Print a report on mean token durations across utterances."""
        mean_duration, token_durations, excluded_durations = \
            self.mean_token_durations(token_type, durs_in_frames,
                                      exclude_tokens, exclude_silence)
        header = "Average {} duration ({}): {:.2f}".format(
            token_type, "frames" if durs_in_frames else "ms", mean_duration)
        hrule = "-" * 16
        print(header)

        print("\nPer token:")
        print(hrule)
        row_template = "{:>4}    {:>6.1f}"
        token_durations = sorted(token_durations.items(),
                                 key=lambda x: x[1], reverse=True)
        for token, dur in token_durations:
            print(row_template.format(token, dur))

        if excluded_durations:
            print("\nExcluded tokens:")
            print(hrule)
            excluded_durations = sorted(excluded_durations.items(),
                                        key=lambda x: x[1], reverse=True)
            for token, dur in excluded_durations:
                print(row_template.format(token, dur))

    def label_purity(self, label_type="unit"):
        """Calculate conditional probabilities of tokens aligned to labels.

        Args:
          label_type: Either 'unit' (default) or 'phone'. Given one, calculate
            conditional probability distribution over the other.
        
        Returns:
          tokens_per_label: Dictionary mapping labels (unit IDs or phones) to
            conditional probability distribution of tokens (vice versa) aligned
            to each label, stored as a list of tuples like (token, prob).
        """
        tokens_per_label = defaultdict(lambda: defaultdict(int))
        for utt in self.phone_alignments:
            frame_alignment = self.unit_phone_alignment(utt, trim=True)
            for unit, phone in frame_alignment:
                if label_type == "unit":
                    # phone purity: how many phones per unit
                    tokens_per_label[unit][phone] += 1
                elif label_type == "phone":
                    # cluster purity: how many units per phone
                    tokens_per_label[phone][unit] += 1
        for label, tokens in tokens_per_label.items():
            total_count = sum(tokens.values())
            tokens = {i: n / total_count for i, n in tokens.items()}
            tokens_per_label[label] = sorted(
                    tokens.items(), key=lambda x: x[1], reverse=True)
        return dict(tokens_per_label)

    def print_label_purity(self, label_type="unit", n_rows=5, delineate=True,
                           show_more=False):
        """Print a report on label purities, from most to least pure.

        Args:
          label_type: Either 'unit' (default) or 'phone'. Given one, print
            purity stats over the other.
          n_rows: Print top N most probable tokens per label, default 5. Set
            to 0 to print all aligned tokens.
          delineate: Boolean to add line separators between labels, default
            True (might want to disable for n_rows=1).
          show_more: Boolean to show how many other tokens were aligned to
            each label, if n_rows < tokens.
        """
        if label_type == "unit":
            token_type = "phone"
        elif label_type == "phone":
            token_type = "unit"
        hrule = "-" * 24
        print("{}\t{}\tp({}|{})".format(
            label_type.capitalize(), token_type.capitalize(),
            token_type[0], label_type[0]))
        print(hrule)

        tokens_per_label = self.label_purity(label_type)
        for label, tokens in sorted(tokens_per_label.items(),
                                    key=lambda x: x[1][0][1],
                                    reverse=True):
            for i, (token, prob) in enumerate(tokens):
                if i < n_rows or not n_rows:
                    print("{}\t{}\t{:.4f}".format(label if not i else "",
                                                  token, prob))
            if show_more and n_rows:
                print("\t...\t{} more".format(len(tokens) - n_rows))
            if delineate:
                print(hrule)

    def purity_stats(self):
        """Calculate cluster and phone purity stats.

        This method uses code from the HuBERT paper (Hsu et al., 2021) to
        calculate cluster purity (pseudo-labels per phone class), phone purity
        (phones per pseudo-label class) and phone-normalized mutual information
        (PNMI, i.e. portion of uncertainty about phones eliminated when given
        a pseudo-label). We discard some additional stats, see full code in
        examples.hubert.measure_teacher_quality._main for reference (nb. in
        that code refs = phones, hyps = clusters, uid = utterance ID).

        Returns:
          p_xy: numpy.ndarray of shape (# pseudo-labels, # phones) representing
            conditional probability distributions.
          cluster_purity: Average pseudo-label purity within a phone class,
            higher values => fewer labels aligned to each phone on average.
          phone_purity: Average phone purity within a pseudo-class, higher
            values => fewer phones aligned to each label on average.
          mi_norm_by_phone: Phone-normalized mutual information, higher values
            => less uncertainty about phone identity given a pseudo-label.
          unit_dur: Average pseudo-label run length in frames.
          phone_dur: Average phone duration in frames.
        """
        utt2units = {}
        utt2phones = {}
        for utt in self.phone_alignments:
            units, phones = zip(*self.unit_phone_alignment(utt, trim=True))
            utt2units[utt], utt2phones[utt] = units, phones
        # see examples.hubert.measure_teacher_quality._main for reference on
        # full stats (nb. refs = phones, hyps = clusters, uid = utterance ID)
        p_xy, unit2idx, phone2idx, *_ = comp_joint_prob(utt2units, utt2phones)
        cluster_purity_by_phone, cluster_purity = comp_purity(p_xy, axis=0)
        # probability of most probable cluster per phone label
        #for p, i in phone2idx.items():
        #    print("{}\t{}".format(p, cluster_purity_by_phone[i]))
        phone_purity_by_cluster, phone_purity = comp_purity(p_xy, axis=1)
        # probability of most probable phone per cluster label
        #for u, i in unit2idx.items():
        #    print("{}\t{}".format(u, phone_purity_by_cluster[i]))
        (mi, mi_norm_by_unit, mi_norm_by_phone, _, _) = comp_norm_mutual_info(p_xy)
        unit_dur = comp_avg_seg_dur(utt2units.values())
        phone_dur = comp_avg_seg_dur(utt2phones.values())
        return p_xy, cluster_purity, phone_purity, mi_norm_by_phone, unit_dur, phone_dur

    def print_purity_stats(self, as_table=True):
        """Print a report on cluster and phone purity stats.

        Args:
          as_table: Boolean, print stats in a single-line table format (default
            True). If False, print slightly more verbose multi-line report.
        """
        (p_xy, cluster_purity, phone_purity, mi_norm_by_phone, 
            unit_dur, phone_dur) = self.purity_stats()
        if as_table:
            outputs = {
                "Phone": phone_purity,
                "Unit": cluster_purity,
                #"H(unit)": h_ref,
                #"H(phone)": h_hyp,
                #"MI": mi,
                #"MI/H(unit)": mi_norm_by_unit,
                "PNMI": mi_norm_by_phone,
                "Phone dur.": phone_dur,
                "Unit dur.": unit_dur,
                "# units, phones": p_xy.shape,
            }
            print(tabulate([outputs.values()], outputs.keys(), floatfmt=".4f"))
        else:
            hrule = "-" * 24
            print("Purity stats:")
            print(hrule)
            row_template = "{:<8}{:>8.4f}"
            print(row_template.format("Phone", phone_purity))
            print(row_template.format("Unit", cluster_purity))
            print(row_template.format("PNMI", mi_norm_by_phone))

            print("\nAverage token durations:")
            print(hrule)
            row_template = "{:<8}{:>8.2f}"
            print(row_template.format("Phone", phone_dur))
            print(row_template.format("Unit", unit_dur))

            print("\nToken inventory sizes:")
            print(hrule)
            row_template = "{:<8}{:>8}"
            print(row_template.format("Unit", p_xy.shape[0]))
            print(row_template.format("Phone", p_xy.shape[1]))

    def plot_kmeans(self, umap_dims=2, utt_pct=0.01):
        frame_samples = []
        phone_samples = []
        unit_samples = []
        random.seed(1337)
        utt_sample = random.sample(self.phone_alignments.keys(), int(utt_pct * len(self.phone_alignments)))
        for utt in utt_sample:
            feats = np.load("/home/s1462938/tools/fairseq/examples/textless_nlp/gslm/speech2unit/clustering/northandsouth_hubert/feats/{}.npy".format(utt))
            np.random.seed(1337)
            sample_idx = np.random.randint(feats.shape[0], size=int(utt_pct * feats.shape[0]))
            frame_sample = feats[sample_idx, :]
            phone_sample = np.asarray(self.run_length_decode(self.phone_alignments[utt]))[sample_idx]
            unit_sample = np.asarray(self.run_length_decode(self.quantized_utts[utt]))[sample_idx]
            frame_samples.extend(frame_sample)
            phone_samples.extend(phone_sample)
            unit_samples.extend(unit_sample)

        centroids = self.kmeans.cluster_centers_
        #frame_samples.extend(centroids)
        reducer = umap.UMAP(n_components=umap_dims, random_state=1337, transform_seed=1337, verbose=True)
        embedding = reducer.fit_transform(frame_samples)  # n_clusters x umap_dims
        centroid_embeddings = reducer.transform(centroids)
        frame_embeddings = reducer.embedding_

        fig, ax = plt.subplots()
        cmap = matplotlib.cm.get_cmap('rainbow')#, self.kmeans.n_clusters)
        # TODO: alpha as phone purity * distance from centroid, normalised
        # between [0,1] over all points in that cluster
        # np.linalg.norm(centroids[unit_samples[i]] - frame_samples[i])
        phone_purity = self.label_purity('unit')
        purity_alpha = {}
        for unit, phones in phone_purity.items():
            purity_alpha[unit] = {}
            for phone, alpha in phones:
                purity_alpha[unit][phone] = alpha

        for i, (phone, xy) in enumerate(zip(phone_samples, frame_embeddings)):
            ax.annotate(phone, xy,
                        color=cmap(unit_samples[i]/self.kmeans.n_clusters, alpha=purity_alpha[unit_samples[i]][phone]), 
                        ha='center', va='center', zorder=0)
        for i in range(self.kmeans.n_clusters):
            ax.scatter(*centroid_embeddings[i], color=cmap(i/self.kmeans.n_clusters), edgecolors='k', zorder=2)
            ax.annotate(i, centroid_embeddings[i], color='k', zorder=1)
        plt.show()
