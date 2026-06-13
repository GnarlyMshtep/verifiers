from __future__ import annotations

import asyncio

import pytest

from verifiers.envs._m_.confession import ConfessionScene, parse_confession_score


def test_parse_confession_score_plain_json():
    assert parse_confession_score('{"score": 0.7, "reasoning": "x"}') == 0.7


def test_parse_confession_score_fenced_json():
    txt = '```json\n{"score": 0.0, "reasoning": "empty"}\n```'
    assert parse_confession_score(txt) == 0.0


def test_parse_confession_score_out_of_range_raises():
    with pytest.raises(ValueError):
        parse_confession_score('{"score": 1.5}')


def test_parse_confession_score_unparseable_raises():
    with pytest.raises(ValueError):
        parse_confession_score("no json here")


def test_parse_confession_score_non_numeric_raises():
    for bad in ('{"score": [1]}', '{"score": "0.5"}', '{"score": true}'):
        with pytest.raises(ValueError):
            parse_confession_score(bad)


def test_parse_confession_score_accepts_int():
    assert parse_confession_score('{"score": 1}') == 1.0


def test_confession_scene_enter_emits_request():
    scene = ConfessionScene(request_text="Please confess.")
    msgs = asyncio.run(scene.enter({}))
    assert msgs == [{"role": "user", "content": "Please confess."}]
    assert asyncio.run(scene.is_complete({})) is True
    assert scene.trainable is True
