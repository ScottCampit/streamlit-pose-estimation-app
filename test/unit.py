"""
Unit testing
"""

import pytest

def test_mov_frames(test_mov):
    assert len(test_mov) == 20

def test_mp4_frames(test_mp4):
    assert len(test_mp4) == 20