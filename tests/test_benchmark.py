"""Tests for benchmark utilities."""

import time

from blur_daddy.benchmark import get_memory_usage, timed_section


class TestTimedSection:
    def test_records_duration(self):
        logger = {}
        with timed_section("test_op", logger):
            time.sleep(0.05)
        assert "test_op" in logger
        assert logger["test_op"] >= 0.04

    def test_accumulates_durations(self):
        logger = {}
        with timed_section("test_op", logger):
            time.sleep(0.02)
        with timed_section("test_op", logger):
            time.sleep(0.02)
        assert logger["test_op"] >= 0.03

    def test_multiple_sections(self):
        logger = {}
        with timed_section("op_a", logger):
            pass
        with timed_section("op_b", logger):
            pass
        assert "op_a" in logger
        assert "op_b" in logger


class TestMemoryUsage:
    def test_returns_positive_number(self):
        mem = get_memory_usage()
        assert isinstance(mem, float)
        assert mem > 0
