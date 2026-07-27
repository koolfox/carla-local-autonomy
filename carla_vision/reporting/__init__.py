"""Manifest-driven research report generation and verification."""

from .builder import REPORT_RELEASE_SCHEMA_VERSION, build_report
from .contracts import REPORT_CONFIG_SCHEMA_VERSION, ReportConfig, load_report_config
from .verified import ReportIntegrityError, VerifiedReport, load_verified_report

__all__ = [
    "REPORT_CONFIG_SCHEMA_VERSION",
    "REPORT_RELEASE_SCHEMA_VERSION",
    "ReportConfig",
    "ReportIntegrityError",
    "VerifiedReport",
    "build_report",
    "load_report_config",
    "load_verified_report",
]
