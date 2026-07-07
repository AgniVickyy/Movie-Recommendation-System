# CineMatch — Movie Recommendation System

Production-grade content-based movie recommender built on the IMDb Top 250 dataset (OMDb export). Completes and extends the pipeline started in `recommendation.ipynb`.

## How it works

Two similarity channels, blended:

| Channel | Features | Vectorizer | Weight |
|---|---|---|---|
| Plot semantics | cleaned `Plot` text | TF-IDF (1-2 grams, English stopwords) | 0.45 |
| Metadata | Director ×3, Genre ×2, top-3 Actors ×1 (names squashed into single tokens) | CountVectorizer | 0.55 |

`similarity = 0.45 · cos(plot) + 0.55 · cos(metadata)` — precomputed as a 250×250 float32 matrix at train time, so inference is an O(n log n) argsort lookup (<1 ms).

Title resolution is exact → substring → fuzzy (`difflib`, cutoff 0.6), so "godfathr" still resolves.

## Project structure

```
recommendation_system/
├── recommendation.csv       # dataset (250 movies, OMDb schema)
├── recommendation.ipynb     # original exploration notebook
├── train.py                 # training pipeline → models/*.pkl
├── app.py                   # FastAPI service
├── index.html               # web UI (autocomplete, poster cards, similarity bars)
├── models/
│   ├── similarity_matrix.pkl
│   ├── movies_metadata.pkl
│   ├── title_index.pkl
│   ├── tfidf_vectorizer.pkl
│   └── count_vectorizer.pkl
├── tests/test_app.py        # pytest suite
├── requirements.txt
├── Dockerfile
└── .dockerignore
```

## Quickstart (local)

```bash
pip install -r requirements.txt
python train.py                  # regenerates models/ (has built-in sanity check)
uvicorn app:app --port 8000      # http://localhost:8000
```

## Docker

```bash
docker build -t movie-recommender .
docker run -p 8000:8000 movie-recommender
```

Runs as non-root, 2 uvicorn workers, with a `/health` HEALTHCHECK.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/` | Web UI |
| GET | `/health` | Liveness + model status |
| GET | `/api/movies` | All titles (autocomplete source) |
| GET | `/api/movie/{title}` | Single movie details (fuzzy-matched) |
| POST | `/api/recommend` | `{"title": "Heat", "top_n": 10}` → ranked recommendations |

Example:

```bash
curl -s -X POST localhost:8000/api/recommend \
  -H 'Content-Type: application/json' \
  -d '{"title": "The Godfather", "top_n": 5}'
```

Interactive docs at `/docs` (Swagger) and `/redoc`.

## Tests

```bash
pytest tests/ -v
```

Covers: health, catalog, exact/fuzzy/substring title resolution, 404s, input validation (422), score ordering/bounds, no self-recommendation, UI serving.

## Design decisions & trade-offs

- **Precomputed similarity matrix** over on-the-fly vectorization: right call at n=250 (250 KB); switch to approximate NN (e.g. FAISS) if the catalog grows past ~50k.
- **Metadata weighted above plot** (0.55/0.45): plots in this dataset are one-sentence summaries — too sparse to dominate. Director/genre/cast carries more signal.
- **Director ×3 > Genre ×2**: with only ~20 distinct genre tokens across 250 movies, genre-heavy weighting collapses everything into "any crime drama". Director-first weighting surfaces franchise/auteur matches — verified: The Dark Knight → TDK Rises, Batman Begins, The Prestige, Interstellar, Inception; Pulp Fiction → Reservoir Dogs, Django Unchained, Kill Bill.
- **No nltk at inference**: notebook's tokenization is subsumed by sklearn's built-in analyzer — one fewer runtime dependency and no corpus downloads in the container.
- **Cold-start limitation**: content-based only; can't recommend for titles outside the 250. Adding collaborative filtering requires user interaction data you don't have here.
