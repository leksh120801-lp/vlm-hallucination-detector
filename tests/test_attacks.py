"""Unit tests for adversarial caption generators."""

from __future__ import annotations

import random

from utils.caption_attack import OBJECT_SWAPS, generate_adversarial_captions, object_swap_attack


def test_object_swap_changes_known_word():
    """An exact-match swap target must be replaced with one of its substitutes."""
    random.seed(0)
    out = object_swap_attack("a dog sitting on grass")
    assert "dog" not in out.split()
    swapped_word = out.split()[1]
    assert swapped_word in OBJECT_SWAPS["dog"]


def test_object_swap_leaves_unknown_word_alone():
    out = object_swap_attack("a hovercraft sitting on grass")
    assert out == "a hovercraft sitting on grass"


def test_object_swap_only_first_match_is_replaced():
    """The current implementation breaks after the first hit by design."""
    random.seed(0)
    out = object_swap_attack("a dog and a cat")
    # 'dog' was first → it must be swapped, 'cat' must stay.
    assert "cat" in out.split()


def test_generate_adversarial_captions_returns_three():
    out = generate_adversarial_captions("a dog sitting on grass")
    assert len(out) == 3
    # The two hard-coded distractors should appear verbatim.
    assert "a spaceship flying in space" in out
    assert "a dinosaur walking in a city" in out
