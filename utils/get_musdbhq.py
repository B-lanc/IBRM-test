import os
import glob
import numpy as np


def load_dataset(path, inst):
    """ """

    tracks = glob.glob(os.path.join(path, "*"))
    samples = list()

    for track_folder in tracks:
        track = dict()
        for i in inst:
            track[i] = os.path.join(track_folder, i + ".wav")
        samples.append(track)

    return samples


def get_musdbhq(path, p_val=0.25, shuffle=False):
    """ """

    instruments = ["bass", "drums", "mixture", "vocals", "other"]

    trainval_set = load_dataset(os.path.join(path, "train"), instruments)
    test_set = load_dataset(os.path.join(path, "test"), instruments)

    # Shuffling the dataset and creating validation set
    _middle = int(len(trainval_set) * p_val // 1)
    if shuffle:
        np.random.shuffle(trainval_set)
    train_set = trainval_set[_middle:]
    val_set = trainval_set[:_middle]

    return {"train": train_set, "val": val_set, "test": test_set}
