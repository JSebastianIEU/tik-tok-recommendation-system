import json
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time


# Load the mock data
def load_videos(filepath):
    videos = []
    with open(filepath, 'r') as f:
        for line in f:
            videos.append(json.loads(line))
    return videos


# Combine caption, hashtags, keywords for indexing
def create_text(video):
    caption = video.get('caption', '')
    hashtags = ' '.join(video.get('hashtags', []))
    keywords = ' '.join(video.get('keywords', []))
    return f"{caption} {hashtags} {keywords}"


# BM25 search
def bm25_search(query, videos, top_k=3):
    start = time.time()
    corpus = [create_text(v) for v in videos]
    tokenized = [text.lower().split() for text in corpus]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    elapsed = (time.time() - start) * 1000  # convert to ms

    results = []
    for i in top_indices:
        results.append({
            'video_id': videos[i]['video_id'],
            'caption': videos[i]['caption'],
            'score': scores[i]
        })
    return results, elapsed


# TF-IDF search
def tfidf_search(query, videos, top_k=3):
    start = time.time()
    corpus = [create_text(v) for v in videos]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus + [query])
    query_vec = tfidf_matrix[-1]
    similarities = cosine_similarity(query_vec, tfidf_matrix[:-1])[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    elapsed = (time.time() - start) * 1000  # convert to ms

    results = []
    for i in top_indices:
        results.append({
            'video_id': videos[i]['video_id'],
            'caption': videos[i]['caption'],
            'score': similarities[i]
        })
    return results, elapsed


# Main experiment
if __name__ == "__main__":
    # Load data (adjust path if needed)
    videos = load_videos('../../data/mock/tiktok_posts_mock.jsonl')  # adjust path to your data file

    # Test queries
    queries = [
        "core workout exercises",
        "easy meal recipes",
        "japan travel tips",
        "coding tutorial"
    ]

    print("=" * 80)
    print("RETRIEVAL EXPERIMENT: BM25 vs TF-IDF")
    print("=" * 80)
    print(f"Dataset: {len(videos)} videos\n")

    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 80)

        # BM25 results
        bm25_results, bm25_time = bm25_search(query, videos)
        print(f"\nBM25 Results ({bm25_time:.2f}ms):")
        for i, result in enumerate(bm25_results, 1):
            print(f"  {i}. [{result['video_id']}] {result['caption']} (score: {result['score']:.3f})")

        # TF-IDF results
        tfidf_results, tfidf_time = tfidf_search(query, videos)
        print(f"\nTF-IDF Results ({tfidf_time:.2f}ms):")
        for i, result in enumerate(tfidf_results, 1):
            print(f"  {i}. [{result['video_id']}] {result['caption']} (score: {result['score']:.3f})")

        print(f"\nSpeed comparison: BM25 {bm25_time:.2f}ms vs TF-IDF {tfidf_time:.2f}ms")