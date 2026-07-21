"""Shared perceptual hash utilities."""

from __future__ import annotations


def hamming_distance(hex_a: str, hex_b: str) -> int:
    """Compute Hamming distance between two 16-char hex pHash strings (64 bits).

    Args:
        hex_a: First hash as a 16-character hexadecimal string.
        hex_b: Second hash as a 16-character hexadecimal string.

    Returns:
        Number of differing bits (0–64).
    """
    int_a = int(hex_a, 16)
    int_b = int(hex_b, 16)
    return bin(int_a ^ int_b).count("1")
