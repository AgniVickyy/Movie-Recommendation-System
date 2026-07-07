"""
Movie Recommendation System - Training Pipeline
=================================================
Content-based recommender over the IMDb Top 250 dataset (OMDb export).

Approach
--------
Completes the pipeline started in recommendation.ipynb (plot cleaning +
tokenization) and extends it to a production model:

1. Clean text fields (Plot, Genre, Director, Actors) - lowercase,
   strip non-alphabetic chars, collapse whitespace.
2. Build two feature spaces:
   - Plot semantics: TF-IDF (unigram+bigram, English stopwords) over cleaned plot.
   - Metadata: CountVectorizer over a weighted "soup" of director (x3),
     genre (x2), and top actors (x1). Names are concatenated
     (e.g. "frankdarabont") so tokens are identity-preserving.
     Director outweighs genre: with only ~20 genre tokens across 250
     movies, genre-heavy weighting collapses recommendations into
     "any crime drama" and buries same-director/franchise matches.
3. Blend cosine similarities: 0.45 * plot + 0.55 * metadata.
4. Persist artifacts to models/: similarity matrix, movie metadata,
   title index, and both vectorizers.

Usage
-----
    python train.py [--data recommendation.csv] [--out models/]
"""

import argparse
import logging
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train")

PLOT_WEIGHT = 0.45
META_WEIGHT = 0.55


def clean_text(text: str) -> str:
    """Lowercase, keep letters only, collapse whitespace (same as notebook)."""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def squash_names(field: str, limit: int | None = None) -> str:
    """'Frank Darabont, Stephen King' -> 'frankdarabont stephenking'.

    Concatenating each name into a single token prevents the vectorizer
    from matching unrelated people who share a first name.
    """
    if pd.isna(field) or field == "N/A":
        return ""
    names = [re.sub(r"[^a-zA-Z]", "", n).lower() for n in str(field).split(",")]
    names = [n for n in names if n]
    if limit:
        names = names[:limit]
    return " ".join(names)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df = df.drop_duplicates(subset="Title").reset_index(drop=True)
    required = ["Title", "Year", "Genre", "Director", "Actors", "Plot", "imdbRating", "Poster", "Runtime"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    log.info("Loaded %d movies from %s", len(df), path)
    return df


def build_features(df: pd.DataFrame):
    # --- Plot channel (notebook's cleanPlot, finished) ---
    clean_plot = df["Plot"].apply(clean_text)
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    plot_matrix = tfidf.fit_transform(clean_plot)

    # --- Metadata channel: director x3, genre x2, actors x1 ---
    genre = df["Genre"].apply(lambda g: clean_text(g.replace(",", " ")))
    director = df["Director"].apply(squash_names)
    actors = df["Actors"].apply(lambda a: squash_names(a, limit=3))
    soup = (director + " ") * 3 + (genre + " ") * 2 + actors
    count_vec = CountVectorizer(min_df=1)
    meta_matrix = count_vec.fit_transform(soup)

    sim = PLOT_WEIGHT * cosine_similarity(plot_matrix) + META_WEIGHT * cosine_similarity(meta_matrix)
    np.fill_diagonal(sim, 1.0)
    log.info("Similarity matrix: %s | plot vocab=%d | meta vocab=%d",
             sim.shape, len(tfidf.vocabulary_), len(count_vec.vocabulary_))
    return sim.astype(np.float32), tfidf, count_vec


def build_metadata(df: pd.DataFrame) -> pd.DataFrame:
    meta = df[["Title", "Year", "Rated", "Runtime", "Genre", "Director",
               "Actors", "Plot", "imdbRating", "imdbVotes", "Poster", "imdbID"]].copy()
    meta["imdbRating"] = pd.to_numeric(meta["imdbRating"], errors="coerce")
    return meta


def sanity_check(sim: np.ndarray, meta: pd.DataFrame) -> None:
    """Fail training if recommendations are degenerate."""
    idx = meta.index[meta["Title"] == "The Godfather"][0]
    top = np.argsort(sim[idx])[::-1][1:6]
    titles = meta.loc[top, "Title"].tolist()
    log.info("Sanity - similar to The Godfather: %s", titles)
    assert "The Godfather: Part II" in titles, "Godfather II not in Godfather's top-5; model is degenerate"
    assert sim.min() >= -1e-6 and sim.max() <= 1.0 + 1e-6, "similarity out of range"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="recommendation.csv")
    parser.add_argument("--out", default="models")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_data(Path(args.data))
    sim, tfidf, count_vec = build_features(df)
    meta = build_metadata(df)
    sanity_check(sim, meta)

    title_index = {t.lower(): i for i, t in enumerate(meta["Title"])}

    artifacts = {
        "similarity_matrix.pkl": sim,
        "movies_metadata.pkl": meta,
        "title_index.pkl": title_index,
        "tfidf_vectorizer.pkl": tfidf,
        "count_vectorizer.pkl": count_vec,
    }
    for name, obj in artifacts.items():
        with open(out / name, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("Saved %s", out / name)

    log.info("Training complete. %d movies indexed.", len(meta))


if __name__ == "__main__":
    main()
