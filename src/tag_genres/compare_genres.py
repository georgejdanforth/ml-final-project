import json
import itertools as it


def main():
    with open("genres_10.json") as f:
        genres = json.load(f)
        for g1, g2 in it.combinations(genres.keys(), 2):
            intersect = set(genres[g1]).intersection(genres[g2])
            if len(intersect) > 0:
                print(g1, g2)
                print(intersect)


if __name__ == "__main__":
    main()
