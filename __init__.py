"""South African Credit Risk Volatility Engine.

Exposes a single end-to-end orchestrator entry point and package version metadata.
All components are fully compliant with:
    - SARB Directive 5/2017 (IFRS 9 ECL)
    - SARB Directive 6/2024 (CCyB 1% effective Jan 2026)
    - SARB Guidance Note 3/2016 (Credit risk management)
    - SARB Regulation 38 (Basel III capital requirements)
    - SARB Regulation 43 (Pillar 3 disclosure requirements)
"""

from __future__ import annotations

__version__ = "1.1.0"
__author__ = "SA Credit Risk Volatility Engine"
__all__ = ["__version__", "__author__"]
