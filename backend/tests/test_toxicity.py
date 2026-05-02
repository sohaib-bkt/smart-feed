from app.models.toxicity import is_toxic, score_toxicity

EXPECTED_KEYS = {
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
}


def test_score_toxicity_keys() -> None:
    result = score_toxicity("Hello world, have a nice day!")
    assert "toxicity_score" in result
    assert "toxicity_detail" in result


def test_score_toxicity_range() -> None:
    score = score_toxicity("Hello world, have a nice day!")["toxicity_score"]
    assert 0.0 <= score <= 1.0


def test_toxicity_detail_keys() -> None:
    detail = score_toxicity("Hello world, have a nice day!")["toxicity_detail"]
    assert set(detail.keys()) == EXPECTED_KEYS


def test_clean_text_below_threshold() -> None:
    score = score_toxicity("Hello world, have a nice day!")["toxicity_score"]
    assert score < 0.3


def test_toxic_text_above_threshold() -> None:
    score = score_toxicity("I hate you so much, you are worthless!")["toxicity_score"]
    assert score > 0.3


def test_is_toxic_returns_bool() -> None:
    assert isinstance(is_toxic("Hello world, have a nice day!"), bool)


def test_is_toxic_clean_false() -> None:
    assert is_toxic("Hello world, have a nice day!") is False


def test_is_toxic_toxic_true() -> None:
    assert is_toxic("I hate you so much, you are worthless!") is True


def test_truncation_long_text() -> None:
    long_text = "a" * 600
    result = score_toxicity(long_text)
    assert "toxicity_score" in result
