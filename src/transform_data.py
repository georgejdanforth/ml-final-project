import os
import h5py
import json
import sqlite3
import numpy as np

from mds_utils import hdf5_getters as getters


with open("tag_genres/genres_10.json") as f:
    genre_variants = json.load(f)

GENRE_MAP = {g: i for i, g in enumerate(sorted(genre_variants.keys()))}

HOME_DIR = os.path.expanduser("~")
MSD_DIR = os.path.join(HOME_DIR, "mount_point", "data")
DATA_DIR = os.path.join(HOME_DIR, "ml-final-project", "data")
SQL_DIR = os.path.join(HOME_DIR, "mount_point", "AdditionalFiles", "artist_term.db")

def write_to_file(X, Y, suf):
    X = np.vstack(X)
    Y = np.vstack(Y)

    X = (X - X.mean(axis=0)) / X.std(axis=0)

    with h5py.File(os.path.join(DATA_DIR, "msd_data_10_" + suf + ".hdf5"), "w") as f:
        f.create_dataset("data", data=X)
        f.create_dataset("labels", data=Y)

def main():
    filepaths = []

    for dirpath, dirnames, filenames in os.walk(os.path.join(MSD_DIR, "A", "A")):
        if len(filenames) > 0:
            for filename in filenames:
                filepaths.append(os.path.join(dirpath, filename))

    num_files = 10
    min_segs = 120
    num_features = 25 * min_segs

    conn = sqlite3.connect("tag_genres/artist_genres_10_mil.db")

    X = []
    Y = []
    num_files = 0

    for filepath in filepaths:

        if len(X) >= 10000:
            write_to_file(X, Y, str(num_files))
            X = []
            Y = []
            num_files += 1

        with getters.open_h5_file_read(filepath) as h5:

            segs = getters.get_segments_start(h5).shape[0]
            if segs < min_segs:
                continue

            artist_id = getters.get_artist_id(h5).decode("UTF-8")
            q = "SELECT genre FROM artist_genres WHERE artist_id='" + artist_id  + "'"
            res = conn.execute(q)
            genre = res.fetchone()
            if genre is None:
                continue

            i = (segs - min_segs) // 2
            t = getters.get_segments_timbre(h5)[i: i + min_segs]
            p = getters.get_segments_pitches(h5)[i: i + min_segs]
            l = (getters.get_segments_loudness_start(h5)[i: i + min_segs])[:,np.newaxis]

        X.append(np.hstack((t, p, l)).flatten())
        y = np.zeros(10, dtype=np.int32)
        y[GENRE_MAP[genre[0]]] += 1
        Y.append(y)

    """
    with h5py.File(os.path.join(DATA_DIR, "msd_data_10_A.hdf5"), "w") as outfile:
        data_dset = outfile.create_dataset("data", (10000, num_features), "f")
        labels_dset = outfile.create_dataset("labels", (10000,), "i")

        i = 0

        for filepath in filepaths:
            with getters.open_h5_file_read(filepath) as h5:

                segs = getters.get_segments_start(h5).shape[0]
                if segs < min_segs:
                    continue

                artist_id = getters.get_artist_id(h5).decode("UTF-8")
                q = "SELECT genre FROM artist_genres WHERE artist_id='" + artist_id  + "'"
                res = conn.execute(q)
                genre = res.fetchone()
                if genre is None:
                    continue

                begin = (segs - min_segs) // 2
                timbre = getters.get_segments_timbre(h5)[begin: begin + min_segs]
                pitches = getters.get_segments_pitches(h5)[begin: begin + min_segs]
                loudness = (getters.get_segments_loudness_start(h5)[begin: begin + min_segs])[:,np.newaxis]

            X = np.hstack((timbre, pitches, loudness)).flatten()

            data_dset[i,:] = X
            labels_dset[i] = GENRE_MAP[genre[0]]

            i += 1

            if i % 100 == 0:
                print(i)
    """

    conn.close()


if __name__ == "__main__":
    main()
