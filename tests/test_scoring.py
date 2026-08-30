import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from olist_delivery.modeling import MODEL_FEATURES
from olist_delivery.scoring import score_dataframe


def test_batch_scoring_preserves_order_ids_and_contract() -> None:
    data = pd.DataFrame({feature: np.arange(4, dtype=float) for feature in MODEL_FEATURES})
    data["customer_state"] = ["SP", "RJ", "SP", "RJ"]
    data["order_id"] = ["a", "b", "c", "d"]
    model = DummyClassifier(strategy="prior").fit(data[MODEL_FEATURES], [0, 0, 1, 0])

    scored = score_dataframe(data, model)

    assert scored["order_id"].tolist() == ["a", "b", "c", "d"]
    assert scored["late_delivery_risk"].tolist() == [0.25] * 4


def test_batch_scoring_rejects_missing_features() -> None:
    with pytest.raises(ValueError, match="missing required features"):
        score_dataframe(pd.DataFrame({"order_id": ["a"]}), object())
