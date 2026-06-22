from __future__ import annotations

import re


DOI_PREFIX_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.IGNORECASE)
ARXIV_URL_RE = re.compile(r"^https?://arxiv\.org/(?:abs|pdf)/", re.IGNORECASE)


def normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = DOI_PREFIX_RE.sub("", value.strip())
    cleaned = cleaned.split("?")[0].strip().strip(".")
    return cleaned.lower() or None


def normalize_arxiv_id(value: str | None, *, strip_version: bool = False) -> str | None:
    if not value:
        return None
    cleaned = ARXIV_URL_RE.sub("", value.strip())
    cleaned = cleaned.removesuffix(".pdf")
    cleaned = cleaned.split("?")[0]
    cleaned = cleaned.replace("arXiv:", "").replace("arxiv:", "")
    if strip_version:
        cleaned = re.sub(r"v\d+$", "", cleaned)
    return cleaned or None


def arxiv_work_id(value: str | None) -> str | None:
    return normalize_arxiv_id(value, strip_version=True)


def normalize_title(value: str | None) -> str:
    return " ".join((value or "").lower().split())
