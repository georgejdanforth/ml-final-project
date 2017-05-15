import os
import json
import sqlite3


HOME_DIR = os.path.expanduser("~")
SQL_DIR = os.path.join(HOME_DIR, "mount_point", "AdditionalFiles", "artist_term.db")


def main():

    genres = None
    with open("genres_10.json") as f:
        genres = json.load(f)

    with sqlite3.connect(SQL_DIR) as conn:

        artists_query = "SELECT artist_id FROM artists"
        artists_res = conn.execute(artists_query)
        artist_ids = map(lambda x: x[0], artists_res.fetchall())

        for artist_id in artist_ids:

            tags_query = "SELECT mbtag FROM artist_mbtag WHERE artist_id='" + artist_id + "'"
            tags_res = conn.execute(tags_query)
            mbtags = set(map(lambda x: x[0], tags_res.fetchall()))

            genre_matches = {}
            
            for genre in genres.keys():
                intersect = set(genres[genre]).intersection(mbtags)
                if len(intersect) <= 0:
                    continue

                genre_matches[genre] = len(intersect)

            if len(genre_matches) > 0:
                sorted_genres = list(genre_matches.keys())
                sorted_genres.sort(key=lambda x: genre_matches[x])
                best_genre = sorted_genres[0]
                print('"' + artist_id + '","' +  best_genre + '"')


if __name__ == "__main__":
    main()
