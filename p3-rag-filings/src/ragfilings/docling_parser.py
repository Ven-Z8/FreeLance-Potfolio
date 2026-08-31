"""IBM Docling Document Layout and TableFormer Parser Adapter."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DoclingParser:
    """Document layout and table extractor powered by IBM Docling (with fallback)."""

    def __init__(self, use_ocr: bool = True) -> None:
        self.use_ocr = use_ocr
        self._has_docling = False
        try:
            import docling  # type: ignore
            self._has_docling = True
        except ImportError:
            self._has_docling = False

    def parse_document(self, file_path: str | Path) -> dict[str, Any]:
        """Convert a document into structured Markdown and extracted table nodes."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document {file_path} not found")

        if self._has_docling:
            return self._parse_with_docling(path)
        return self._parse_fallback(path)

    def _parse_with_docling(self, path: Path) -> dict[str, Any]:
        from docling.document_converter import DocumentConverter  # type: ignore

        converter = DocumentConverter()
        result = converter.convert(str(path))
        doc = result.document

        markdown_text = doc.export_to_markdown()
        tables = []
        for i, table in enumerate(doc.tables):
            df = table.export_to_dataframe()
            tables.append({
                "index": i,
                "headers": list(df.columns),
                "rows": df.values.tolist(),
                "markdown": df.to_markdown(),
            })

        return {
            "source": str(path),
            "engine": "docling_tableformer",
            "markdown": markdown_text,
            "tables": tables,
            "figures": [{"index": i} for i, _ in enumerate(doc.pictures)],
        }

    def _parse_fallback(self, path: Path) -> dict[str, Any]:
        """Resilient fallback parser for standard HTML/text filings."""
        text = path.read_text(encoding="utf-8", errors="ignore")
        return {
            "source": str(path),
            "engine": "native_fallback",
            "markdown": text,
            "tables": [],
            "figures": [],
        }
