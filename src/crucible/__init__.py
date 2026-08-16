"""Crucible — a semantic screen for feature-level target leakage.

A column leaks when its value encodes the outcome it would be used to predict.
This package finds such columns by reading what they *mean* against a target
and a stated prediction point, explains each verdict, and measures what the
columns were worth to a model. It never removes anything on its own.

Typical use::

    import pandas as pd
    from crucible import AuditRequest, run_audit, flagged_columns, quantify

    table = pd.read_csv("loans.csv")
    report = await run_audit(AuditRequest(
        table=table,
        target="default",
        prediction_point="at approval, before any repayment history exists",
    ))
    leaks = flagged_columns(report)
    cost = quantify(table, "default", leaks)

Leakage is a property of the triple (column, target, prediction point), not of
a column, which is why the prediction point is a required argument and not an
option with a default. It is the one input nothing can compute.
"""

from .audit import AuditRequest, flagged_columns, run_audit
from .impact import ImpactError, quantify
from .intake import IntakeError, describe_table, load_dictionary, load_table
from .models import CATALOGUE, DEFAULT_MODEL, criterion_for, provider_for
from .providers import Provider, ProviderError, QuotaExhausted, resolve

__version__ = "0.1.0"

__all__ = [
    "AuditRequest", "run_audit", "flagged_columns",
    "quantify", "ImpactError",
    "load_table", "load_dictionary", "describe_table", "IntakeError",
    "CATALOGUE", "DEFAULT_MODEL", "criterion_for", "provider_for",
    "Provider", "ProviderError", "QuotaExhausted", "resolve",
    "__version__",
]
