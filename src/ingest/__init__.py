"""Loader, chunker, and span-to-label mapper.

Reads the official CUAD JSON from ``data/cuad/`` (read-only), splits contracts
into chunks with character offsets, and maps ``answer_start`` spans onto 41
binary labels per chunk.
"""
