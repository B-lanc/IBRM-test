import pickle
import os

import numpy as np
import librosa
from scipy.signal import stft, istft
import museval

from museparation.scripts.get_musdb import get_musdbhq


def main(args):
    musdb = get_musdbhq(args["path"])
    eps = np.finfo(float).eps

    results = dict()
    for nft in args["n_fft"]:
        for hop in args["hop_size"]:
            results[f"{nft}_{hop * nft}"] = {
                "bass": list(),
                "drums": list(),
                "other": list(),
                "vocals": list(),
            }

    for song in musdb["test"]:
        bass = song["bass"]
        drums = song["drums"]
        other = song["other"]
        vocals = song["vocals"]
        mixture = song["mixture"]
        name = mixture.split("/")[-2]

        bass, _ = librosa.load(bass, sr=44100, mono=False)
        drums, _ = librosa.load(drums, sr=44100, mono=False)
        other, _ = librosa.load(other, sr=44100, mono=False)
        vocals, _ = librosa.load(vocals, sr=44100, mono=False)
        mixture, _ = librosa.load(mixture, sr=44100, mono=False)

        for nft in args["n_fft"]:
            for hop in args["hop_size"]:
                noverlap = nft - nft * hop
                bass_stft = stft(
                    bass, nperseg=nft, noverlap=noverlap, boundary=None
                )[2]
                drums_stft = stft(
                    drums, nperseg=nft, noverlap=noverlap, boundary=None
                )[2]
                other_stft = stft(
                    other, nperseg=nft, noverlap=noverlap, boundary=None
                )[2]
                vocals_stft = stft(
                    vocals, nperseg=nft, noverlap=noverlap, boundary=None
                )[2]
                mixture_stft = stft(
                    mixture, nperseg=nft, noverlap=noverlap, boundary=None
                )[2]

                bass_IRM = np.divide(np.abs(bass_stft), np.abs(mixture_stft) + eps)
                drums_IRM = np.divide(np.abs(drums_stft), np.abs(mixture_stft) + eps)
                other_IRM = np.divide(np.abs(other_stft), np.abs(mixture_stft) + eps)
                vocals_IRM = np.divide(np.abs(vocals_stft), np.abs(mixture_stft) + eps)

                bass_preds = istft(
                    bass_IRM * mixture_stft,
                    nperseg=nft,
                    noverlap=noverlap,
                    boundary=False,
                )[1]
                drums_preds = istft(
                    drums_IRM * mixture_stft,
                    nperseg=nft,
                    noverlap=noverlap,
                    boundary=False,
                )[1]
                other_preds = istft(
                    other_IRM * mixture_stft,
                    nperseg=nft,
                    noverlap=noverlap,
                    boundary=False,
                )[1]
                vocals_preds = istft(
                    vocals_IRM * mixture_stft,
                    nperseg=nft,
                    noverlap=noverlap,
                    boundary=False,
                )[1]

                if bass_preds.shape != bass.shape:
                    bass_preds = bass_preds[:, : bass.shape[1]]
                if drums_preds.shape != drums.shape:
                    drums_preds = drums_preds[:, : drums.shape[1]]
                if other_preds.shape != other.shape:
                    other_preds = other_preds[:, : other.shape[1]]
                if vocals_preds.shape != vocals.shape:
                    vocals_preds = vocals_preds[:, : vocals.shape[1]]

                bass_SDR, _, _, _, _ = museval.metrics.bss_eval(
                    bass.T[np.newaxis, :, :], bass_preds.T[np.newaxis, :, :]
                )
                drums_SDR, _, _, _, _ = museval.metrics.bss_eval(
                    drums.T[np.newaxis, :, :], drums_preds.T[np.newaxis, :, :]
                )
                other_SDR, _, _, _, _ = museval.metrics.bss_eval(
                    other.T[np.newaxis, :, :], other_preds.T[np.newaxis, :, :]
                )
                vocals_SDR, _, _, _, _ = museval.metrics.bss_eval(
                    vocals.T[np.newaxis, :, :], vocals_preds.T[np.newaxis, :, :]
                )

                results[f"{nft}_{hop * nft}"]["bass"].append(bass_SDR)
                results[f"{nft}_{hop * nft}"]["drums"].append(drums_SDR)
                results[f"{nft}_{hop * nft}"]["other"].append(other_SDR)
                results[f"{nft}_{hop * nft}"]["vocals"].append(vocals_SDR)

    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    # If need to change something, only change these values
    #####################################################################
    PATH = "/mnt/Data/MachineLearning/Datasets/musdb18hq"
    N_FFT = [64, 128, 256, 512, 1024, 2048, 4096]
    HOP_SIZE = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1]
    #####################################################################
    args = {"path": PATH, "n_fft": N_FFT, "hop_size": HOP_SIZE}
    main(args)
