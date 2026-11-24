import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_release_model import anonymize  # noqa: E402


def test_anonymize_removes_names_and_unknown_keys():
    p = {"label": "Pack/Secret 01", "x": 0.1, "y": 0.2, "latent": [1.0], "source": "Secret.wav"}
    out = anonymize({"latent_dim": 1, "sample_rate": 44100, "data_dir": "D:/Secret",
                     "points": [p, {**p, "midi": 57.08}]}, "Kick")
    assert "Secret" not in str(out)
    assert [q["label"] for q in out["points"]] == ["Kick 001", "Kick 002 (A3)"]
    assert out["points"][0]["latent"] == [1.0] and out["points"][1]["midi"] == 57.08
