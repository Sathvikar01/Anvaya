from hh_goa_rag.config import stable_fingerprint


def test_fingerprint_is_order_independent() -> None:
    assert stable_fingerprint({"a": 1, "b": 2}) == stable_fingerprint({"b": 2, "a": 1})

