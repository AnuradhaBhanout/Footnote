"""Test that embed_specific() indexes the summary, not the pdf_url."""

from unittest.mock import MagicMock

import db.rag_index as rag_index


SUMMARY = "we propose a transformer architecture based on self attention"
PDF_URL = "https://arxiv.org/pdf/2301.00001"


def _papers_row():
    return ("2301.00001", "Attention Is All You Need", "Vaswani et al", SUMMARY, PDF_URL, "2026-01-01")




def test_embed_specific_uses_summary_not_pdf_url(monkeypatch):
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.side_effect = [
        [],
        [_papers_row()],
    ]

    monkeypatch.setattr(rag_index, "get_conn", lambda: conn)
    monkeypatch.setattr(rag_index, "put_conn", lambda c: None)

    idx = rag_index.HybridIndex()

    spy = MagicMock()
    monkeypatch.setattr(idx, "_upsert_to_db", spy)

    idx.embed_specific(["2301.00001"])

    texts = spy.call_args.kwargs["texts"]

    assert SUMMARY in texts[0],(f"embed_specific indexed the wrong column - summary missing. got: {texts[0]!r}")

    assert PDF_URL not in texts[0], (f"embed_specific indexed pdf_url instead of summary. got: {texts[0]!r}")