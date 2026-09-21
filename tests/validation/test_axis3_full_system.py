from scripts.validation.axis3_full_system import (
    bounds,
    clean_label_record,
    deterministic_pass,
    passk_bounds,
)


def test_clean_label_record_rejects_both_negative_labels():
    assert clean_label_record({"claims": [{"label": "SUPPORTED"}]}) is True
    assert clean_label_record({"claims": [{"label": "NOT_IN_RESULT"}]}) is False
    assert clean_label_record({"claims": [{"label": "CONTRADICTED"}]}) is False
    assert clean_label_record(None) is None


def test_deterministic_pass_includes_numeric_traceability():
    base = {
        "stage_a_ok": True,
        "stage_b_ok": True,
        "numeric_ok": True,
        "policy_problems": [],
        "error": None,
    }
    assert deterministic_pass(base) is True
    assert deterministic_pass({**base, "numeric_ok": False}) is False
    assert deterministic_pass({**base, "policy_problems": ["bad"]}) is False


def test_bounds_treat_unresolved_as_interval():
    assert bounds([True, False, None, True]) == (0.5, 0.75, 1)


def test_passk_bounds_requires_every_repetition():
    rows = [
        {"case_id": "A", "full_system": True},
        {"case_id": "A", "full_system": True},
        {"case_id": "B", "full_system": True},
        {"case_id": "B", "full_system": None},
        {"case_id": "C", "full_system": True},
        {"case_id": "C", "full_system": False},
    ]
    lo, hi, pending = passk_bounds(rows)
    assert lo == 1 / 3
    assert hi == 2 / 3
    assert pending == 1
