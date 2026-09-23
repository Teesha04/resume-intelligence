"""Ingestion: resilient file discovery and text extraction."""

from .loader import discover_files, load_documents
from .parsers import SUPPORTED_EXTENSIONS, parse_file

__all__ = ["discover_files", "load_documents", "parse_file", "SUPPORTED_EXTENSIONS"]
