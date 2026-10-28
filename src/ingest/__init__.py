"""Loader, chunker, and span-to-label mapper.

Reads the official CUAD JSON from ``data/cuad/`` (read-only), splits contracts
into chunks with character offsets, and maps ``answer_start`` spans onto 41
nullable CUAD category labels per chunk. The transformer selects the 34
categories with positive training examples.
"""
