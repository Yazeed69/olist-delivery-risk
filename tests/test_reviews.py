import pandas as pd

from olist_delivery.cleaning import clean_reviews


def _review(
    review_id: str,
    order_id: str,
    score: int,
    answered_at: str,
) -> dict[str, object]:
    return {
        "review_id": review_id,
        "order_id": order_id,
        "review_score": score,
        "review_creation_date": "2018-01-01",
        "review_answer_timestamp": answered_at,
    }


def test_clean_reviews_resolves_ambiguous_responses_in_the_right_order() -> None:
    reviews = pd.DataFrame(
        [
            _review("shared", "order_a", 3, "2018-01-02"),
            _review("shared", "order_b", 3, "2018-01-03"),
            _review("conflict_1", "order_c", 1, "2018-01-02"),
            _review("conflict_2", "order_c", 5, "2018-01-03"),
            _review("agree_old", "order_d", 4, "2018-01-02"),
            _review("agree_new", "order_d", 4, "2018-01-04"),
            _review("single", "order_e", 5, "2018-01-05"),
        ]
    )

    cleaned = clean_reviews(reviews)

    assert cleaned["order_id"].tolist() == ["order_d", "order_e"]
    assert cleaned["review_id"].tolist() == ["agree_new", "single"]
    assert cleaned["order_id"].is_unique
    assert cleaned["review_id"].is_unique


def test_clean_reviews_does_not_mutate_the_input() -> None:
    reviews = pd.DataFrame(
        [_review("review_1", "order_1", 5, "2018-01-02")]
    )
    original = reviews.copy(deep=True)

    clean_reviews(reviews)

    pd.testing.assert_frame_equal(reviews, original)
