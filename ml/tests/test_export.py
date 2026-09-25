"""Checks the model the web app ships with matches what the app expects."""

from __future__ import annotations

import json

import numpy as np
import pytest

from common import WEB_MODELS, roles_for

ort = pytest.importorskip("onnxruntime")

META = WEB_MODELS / "meta.json"


@pytest.fixture(scope="module")
def meta() -> dict:
    if not META.exists():
        pytest.skip("no exported model; run export_onnx.py")
    return json.loads(META.read_text())


def test_meta_roles_point_at_real_classes(meta: dict) -> None:
    assert {"player", "ball"} <= set(meta["roles"].values())
    for idx in meta["roles"]:
        assert 0 <= int(idx) < len(meta["names"])


def test_onnx_output_is_yolov8_head(meta: dict) -> None:
    sess = ort.InferenceSession(str(WEB_MODELS / meta["file"]))
    size = meta["input"]
    (inp,) = sess.get_inputs()
    assert inp.shape == [1, 3, size, size]
    out = sess.run(None, {inp.name: np.zeros((1, 3, size, size), np.float32)})[0]
    anchors = sum((size // s) ** 2 for s in (8, 16, 32))
    assert out.shape == (1, 4 + len(meta["names"]), anchors)


def test_roles_for_football_names() -> None:
    names = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee", 4: "coach"}
    assert roles_for(names, coco=False) == {"0": "ball", "1": "goalkeeper", "2": "player", "3": "referee"}
    assert roles_for({}, coco=True) == {"0": "player", "32": "ball"}
