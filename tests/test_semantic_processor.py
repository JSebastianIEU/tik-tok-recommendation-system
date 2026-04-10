from __future__ import annotations

from src.recommendation.semantic_processor import process_text


def test_process_text_separates_hashtags_mentions_and_keeps_emoji_signal():
    processed = process_text(
        text="Check this out 😂🔥 #TechReview @Creator https://example.com/demo",
        explicit_hashtags=["#AI"],
    )

    assert processed.hashtags == ["ai", "techreview"]
    assert processed.mentions == ["creator"]
    assert "#TechReview" not in processed.semantic_text
    assert "@Creator" not in processed.semantic_text
    assert "https://example.com" not in processed.semantic_text
    assert "😂" in processed.semantic_text
    assert "🔥" in processed.semantic_text
    assert "face_with_tears_of_joy" in processed.lexical_text
    assert "fire" in processed.lexical_text
    assert "techreview" not in processed.semantic_tokens


def test_process_text_preserves_unicode_words_for_lexical_tokens():
    processed = process_text(text="Mini tutorial fácil para skincare 💖")

    assert "fácil" in processed.lexical_text
    assert "fácil" in processed.lexical_tokens
    assert "sparkling_heart" not in processed.semantic_text
    assert "sparkling_heart" in processed.lexical_text
