"""Light and heavier text cleaners.

Both run *after* span-to-label mapping, because cleaning shifts character
positions that ``answer_start`` offsets depend on. The light cleaner feeds the
transformer; the heavier one feeds TF-IDF.
"""
