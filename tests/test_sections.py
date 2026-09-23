"""Section detection, including inline 'Heading: content' lines."""

from src.extract.sections import line_assignments, split_sections


def test_inline_heading_is_captured():
    text = "Skills: Python, FastAPI, Docker\nProjects\nBuilt a RAG agent"
    sections = split_sections(text)
    assert "Python" in sections["skills"]
    assert "RAG" in sections["projects"]


def test_headings_on_their_own_line():
    text = "Summary\nEngineer.\n\nExperience\nIntern at X\n\nProjects\nAgent\nDetails here"
    sections = split_sections(text)
    assert "Intern" in sections["experience"]
    assert "Agent" in sections["projects"]


def test_preamble_defaults_to_summary():
    text = "Jane Doe\njane@example.com\n\nSkills: Python"
    sections = split_sections(text)
    assert "Jane Doe" in sections["summary"]


def test_unknown_heading_goes_to_other_and_still_scanned():
    text = "Stuff\nPython RAG work\n"
    assignments = line_assignments(text)
    assert ("summary", "Python RAG work") in assignments


def test_long_skills_line_not_mistaken_for_heading():
    long_line = "Skills: " + ", ".join(["Python"] * 40)
    sections = split_sections(long_line)
    assert "Python" in sections["skills"]
