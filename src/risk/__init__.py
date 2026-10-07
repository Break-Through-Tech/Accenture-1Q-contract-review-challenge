"""Evidence extractor, rule-based scorer, and triage aggregator.

The scorer reads its weights and thresholds from ``configs/risk_rules.yaml``
and stamps every clause with the ``rules_version`` that produced it.
"""
