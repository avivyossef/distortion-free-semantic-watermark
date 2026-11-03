from __future__ import annotations

from watermark.generation import NATIVE_KEY_ID, _native_clean_seed, panel_seed_rows

KEY = bytes(range(32))


def _rows(native_key_id: str):
    return panel_seed_rows(prompt_uid="prompt-a", master_key=KEY, mode="candidates", batch_size=2,
                           candidate_count=4, native_key_id=native_key_id)


def test_native_key_id_sets_only_the_clean_seeds():
    default, labelled = _rows(NATIVE_KEY_ID), _rows("key-12")
    changed = [(a["role"], a["batch_slot"]) for a, b in zip(default, labelled) if a["seed"] != b["seed"]]
    assert changed == [("clean", 1), ("clean", 2)]
    clean = [r for r in labelled if r["role"] == "clean" and r["batch_slot"] == 1][0]
    assert clean["seed"] == _native_clean_seed("prompt-a", "key-12")
