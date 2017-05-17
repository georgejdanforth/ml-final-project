import os
import json
import sqlite3


HOME_DIR = os.path.expanduser("~")
SQL_DIR = os.path.join(HOME_DIR, "mount_point", "AdditionalFiles", "artist_term.db")

GENRES = {
    "rock": ["rock"],
    "punk": ["punk"],
    "folk": ["folk"],
    "pop": ["pop"],
    "electronic": [
        "electro",
        "dance",
        "electronic",
        "house",
        "bass",
        "jungle",
        "techno",
        "dnb"
    ],
    "metal": ["metal"],
    "jazz": [
        "jazz",
        "blues"
    ],
    "classical": [
        "classical",
        "romance",
        "baroque",
        "hymn"
    ],
    "hip-hop": [
        "hip-hop",
        "hip hop",
        "grime",
        "dub",
        "rap"
    ],
    "reggae": ["reggae"]
}


def main():

    GENRE_VARIANTS = {genre: set() for genre in GENRES.keys()}

    with sqlite3.connect(SQL_DIR) as conn:
        for genre in GENRES.keys():
            for variant in GENRES[genre]:
                q = "SELECT mbtag FROM mbtags WHERE mbtag LIKE '%" + variant + "%'"
                res = conn.execute(q)
                mbtags = map(lambda x: x[0], res.fetchall())
                for mbtag in mbtags:
                    GENRE_VARIANTS[genre].add(mbtag)

            GENRE_VARIANTS[genre] = list(GENRE_VARIANTS[genre])

    with open("genres_10_mil.json", "w") as f:
        json.dump(GENRE_VARIANTS, f, indent=4)

if __name__ == "__main__":
    main()
