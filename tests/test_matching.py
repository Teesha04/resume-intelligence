"""Matching precision — the classic false positives must never appear."""

from src.matching import match_terms, scan_evidence
from src.schemas import EvidenceSource
from src.vocab import AI_TERMS, BACKEND_TERMS, PYTHON_TERMS


def test_java_does_not_match_javascript():
    hits = match_terms("JavaScript, React, Node.js", BACKEND_TERMS)
    assert "JavaScript" in hits and "Node.js" in hits
    assert "Java" not in hits


def test_java_matches_standalone():
    hits = match_terms("Java, Spring Boot", BACKEND_TERMS)
    assert "Java" in hits and "Spring Boot" in hits


def test_sql_does_not_match_postgresql():
    assert "SQL" not in match_terms("PostgreSQL 14", BACKEND_TERMS)
    assert "SQL" in match_terms("SQL queries", BACKEND_TERMS)


def test_multiword_separators():
    assert "Multi-Agent" in match_terms("multi agent workflow", AI_TERMS)
    assert "RAG" in match_terms("Retrieval-Augmented Generation (RAG)", AI_TERMS)


def test_generic_tools_token_does_not_match_tool_calling():
    # "Tools, Platforms & Technologies" must NOT imply tool calling.
    assert "Tool Calling" not in match_terms("Tools, Platforms & Technologies", AI_TERMS)


def test_empty_and_unrelated_text():
    assert match_terms("", PYTHON_TERMS) == []
    assert match_terms("Salesforce administration and CRM", PYTHON_TERMS) == []


def test_scan_evidence_assigns_source():
    assignments = [("skills", "Python, FastAPI"), ("projects", "Built a Python RAG agent")]
    evidence = scan_evidence(assignments, PYTHON_TERMS)
    by_term: dict[str, set] = {}
    for ev in evidence:
        by_term.setdefault(ev.term, set()).add(ev.source)
    assert EvidenceSource.SKILLS in by_term["FastAPI"]
    # Python appears in both a skills line and a project line.
    assert EvidenceSource.SKILLS in by_term["Python"]
    assert all(ev.context for ev in evidence)
