"""Key-terms structured extractor — public re-exports for the extraction package.

See :mod:`clausewright.extraction` for the full implementation.
"""

from clausewright.extraction import (
    FieldResult,
    KeyTermsExtractor,
    KeyTermsReport,
)

__all__ = ["FieldResult", "KeyTermsExtractor", "KeyTermsReport"]
