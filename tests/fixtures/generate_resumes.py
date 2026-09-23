"""Generate synthetic resumes for development, demos and tests.

Usage:
    python tests/fixtures/generate_resumes.py --out resumes_sample --count 50

Produces a mix that exercises every branch of the pipeline:
  * strong agentic / RAG / multi-agent candidates
  * Python backend with NO AI  (must be rejected)
  * JavaScript/React-only      (must be rejected)
  * thin LLM-wrapper projects  (must be penalised)
  * tutorial-only projects     (must be penalised)
  * general ML/CV (no LLM)     (eligible, lower AI depth)
  * no GitHub / private-style profiles
  * edge cases: corrupt PDF, empty TXT, exact duplicate, unsupported .png

PDFs require `reportlab` (dev-only). DOCX requires `python-docx`. If a writer
is unavailable the generator falls back to .txt so the set is always produced.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

try:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    _HAS_PDF = True
except Exception:  # noqa: BLE001
    _HAS_PDF = False

try:
    import docx

    _HAS_DOCX = True
except Exception:  # noqa: BLE001
    _HAS_DOCX = False


FIRST = ["Asha", "Ravi", "Meera", "Arjun", "Neha", "Karan", "Priya", "Dev", "Sara", "Vikram",
         "Ananya", "Rahul", "Isha", "Nikhil", "Tara", "Yash", "Diya", "Aditya", "Kavya", "Rohan",
         "Zara", "Manish", "Pooja", "Sanjay", "Lily", "Omar", "Hana", "Chen", "Diego", "Farah"]
LAST = ["Rao", "Mehta", "Iyer", "Kapoor", "Sharma", "Nair", "Gupta", "Singh", "Patel", "Reddy",
        "Bose", "Khan", "Verma", "Joshi", "Menon", "Das", "Chopra", "Pillai", "Malhotra", "Sinha"]


def _profile(name: str, email: str, github: str, body: str) -> str:
    gh_line = f"github.com/{github}" if github else ""
    return f"{name}\n{email}\n{gh_line}\n\n{body}".strip() + "\n"


# --- Archetypes ------------------------------------------------------------
# Each returns (body_text, tags) where tags drive expectations in tests.

def agentic_strong(i: int) -> tuple[str, list[str]]:
    return (
        """Summary
Backend-leaning engineer focused on LLM systems and Python services.

Skills: Python, FastAPI, PostgreSQL, Redis, Docker, GCP, LangGraph, LangChain, pgvector, pytest

Experience
Software Engineer Intern - Built async FastAPI services backed by PostgreSQL and Redis, added caching and unit tests, containerized with Docker and deployed to Cloud Run.

Projects
Agentic Support Triage
Built a multi-agent workflow in LangGraph that retrieves over 50k support docs using pgvector embeddings, calls internal ticketing tools, and maintains conversation state. Added an evaluation harness with regression tests and guardrails. Deployed on GCP Cloud Run with Redis-backed queueing and observability.
RAG Knowledge Assistant
Document Q&A service with a FastAPI backend, chunking pipeline, pgvector vector search, tool calling for live data, and a small evaluation set for answer quality.
""",
        ["agentic", "python", "cloud"],
    )


def rag_engineer(i: int) -> tuple[str, list[str]]:
    return (
        """Skills: Python, LlamaIndex, FastAPI, Qdrant, Docker, AWS, asyncio

Experience
Data Engineer Intern - Python ETL pipelines with async processing and a message queue processing 1M records/day.

Projects
Enterprise RAG Search
Built a retrieval pipeline using LlamaIndex with embeddings and vector search over internal wikis, plus an evaluation pipeline measuring answer relevance. FastAPI service deployed with Docker on AWS.
""",
        ["agentic", "python", "cloud"],
    )


def python_backend_no_ai(i: int) -> tuple[str, list[str]]:
    return (
        """Skills: Python, Django, Flask, PostgreSQL, MySQL, Redis, Docker, SQL

Experience
Backend Intern - Built Django REST APIs with PostgreSQL, Redis caching, Celery background jobs and pytest test suites.

Projects
Inventory API
REST API in Django with PostgreSQL, JWT auth, caching and pagination.
""",
        ["no_ai"],
    )


def frontend_only(i: int) -> tuple[str, list[str]]:
    return (
        """Skills: JavaScript, React, Next.js, TypeScript, Tailwind, HTML, CSS, Node.js, Express

Experience
Frontend Intern - Built responsive React dashboards consuming REST APIs, wrote Jest tests.

Projects
Analytics Dashboard
React + Next.js dashboard with charts, auth and API integration.
""",
        ["no_python", "no_ai"],
    )


def java_spring(i: int) -> tuple[str, list[str]]:
    return (
        """Skills: Java, Spring Boot, MySQL, Docker, REST API, Kafka

Experience
Software Intern - Built Spring Boot microservices with Kafka and MySQL.

Projects
Order Service
Spring Boot microservice with Kafka events and MySQL persistence.
""",
        ["no_python", "no_ai"],
    )


def thin_wrapper(i: int) -> tuple[str, list[str]]:
    return (
        """Skills: Python, OpenAI API, LangChain

Projects
Chatbot
A chatbot that calls the OpenAI API to answer user questions.
""",
        ["thin", "eligible"],
    )


def tutorial_only(i: int) -> tuple[str, list[str]]:
    return (
        """Skills: Python, LangChain, Streamlit

Projects
RAG Tutorial Follow-along
Followed a YouTube tutorial to build a basic RAG demo.
""",
        ["tutorial", "eligible"],
    )


def cv_ml(i: int) -> tuple[str, list[str]]:
    return (
        """Skills: Python, PyTorch, OpenCV, NumPy, Pandas, scikit-learn

Experience
ML Intern - Trained CNNs for image classification; built data preprocessing pipelines.

Projects
Defect Detection
Trained a CNN in PyTorch with OpenCV preprocessing to classify manufacturing defects.
""",
        ["general_ai", "python"],
    )


def fullstack_ai(i: int) -> tuple[str, list[str]]:
    return (
        """Skills: Python, FastAPI, React, Next.js, Docker, Supabase, LangChain, PostgreSQL

Experience
Full Stack Intern - Built React frontends over FastAPI backends with PostgreSQL.

Projects
AI Study Buddy
Next.js frontend over a Python FastAPI LangChain service with embeddings, vector search and a tool-calling agent for flashcards. Dockerized end to end.
""",
        ["agentic", "python", "cloud"],
    )


def data_python_ai(i: int) -> tuple[str, list[str]]:
    return (
        """Skills: Python, Pandas, NumPy, SQL, PostgreSQL, Airflow, scikit-learn, Docker, GCP

Experience
Data Intern - Built Python data pipelines on GCP with BigQuery and Airflow.

Projects
Churn Predictor with LLM Explanations
scikit-learn churn model plus an LLM layer (OpenAI API) generating explanations from feature importances, served via FastAPI with Postgres caching.
""",
        ["python", "cloud", "general_ai"],
    )


def devops_python(i: int) -> tuple[str, list[str]]:
    return (
        """Skills: Python, Docker, Kubernetes, Terraform, GCP, CI/CD, GitHub Actions, PostgreSQL

Experience
Platform Intern - Built Python automation, Terraform infra on GCP, CI/CD pipelines.

Projects
LLM Ops Platform
Served an open-source LLM behind a FastAPI gateway with embeddings-based semantic caching, Prometheus monitoring and autoscaling on GKE.
""",
        ["agentic", "python", "cloud"],
    )


def freshgrad_agentic(i: int) -> tuple[str, list[str]]:
    return (
        """Skills: Python, LangGraph, FastAPI, Redis, Docker

Projects
Multi-Agent Research Assistant
LangGraph multi-agent system with retrieval over arXiv using embeddings, planner and tool-calling agents, Redis session state, and an evaluation harness comparing summarisation quality.
""",
        ["agentic", "python"],
    )


def mobile_python_ai(i: int) -> tuple[str, list[str]]:
    return (
        """Skills: Python, Flask, OpenAI API, RAG, React Native, PostgreSQL

Projects
Recipe Assistant
Flask backend with a RAG pipeline over a recipe corpus using embeddings and vector search; React Native client.
""",
        ["agentic", "python"],
    )


def no_github_agentic(i: int) -> tuple[str, list[str]]:
    return (
        """Skills: Python, LlamaIndex, FastAPI, MongoDB, Docker

Projects
Multi-Agent Content Pipeline
LlamaIndex agents with tool calling and orchestration to draft, review and publish content; FastAPI backend with MongoDB state.
""",
        ["agentic", "python", "no_github"],
    )


ARCHETYPES = [
    agentic_strong, rag_engineer, python_backend_no_ai, frontend_only, java_spring,
    thin_wrapper, tutorial_only, cv_ml, fullstack_ai, data_python_ai, devops_python,
    freshgrad_agentic, mobile_python_ai, no_github_agentic,
]


def _write_pdf(path: Path, text: str) -> bool:
    if not _HAS_PDF:
        return False
    c = canvas.Canvas(str(path), pagesize=LETTER)
    width, height = LETTER
    y = height - 50
    for line in text.splitlines():
        for chunk in [line[i:i + 95] for i in range(0, max(1, len(line)), 95)] or [""]:
            if y < 50:
                c.showPage()
                y = height - 50
            c.drawString(40, y, chunk)
            y -= 14
    c.save()
    return True


def _write_docx(path: Path, text: str) -> bool:
    if not _HAS_DOCX:
        return False
    d = docx.Document()
    for line in text.splitlines():
        d.add_paragraph(line)
    d.save(str(path))
    return True


def generate(out_dir: Path, count: int = 50, seed: int = 7) -> dict:
    random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.iterdir():
        if f.is_file():
            f.unlink()

    used_names: set[str] = set()
    files: list[Path] = []
    tags_by_file: dict[str, list[str]] = {}

    for i in range(count):
        name = f"{FIRST[i % len(FIRST)]} {LAST[(i * 3) % len(LAST)]}"
        while name in used_names:
            name = f"{name.split()[0]} {LAST[random.randrange(len(LAST))]}"
        used_names.add(name)

        archetype = ARCHETYPES[i % len(ARCHETYPES)]
        body, tags = archetype(i)
        email = f"{name.split()[0].lower()}.{name.split()[1].lower()}@example.com"
        github = "" if "no_github" in tags else f"{name.split()[0].lower()}{i:02d}dev"
        text = _profile(name, email, github, body)

        fmt = i % 3
        if fmt == 0 and _HAS_PDF:
            path = out_dir / f"candidate_{i:02d}.pdf"
            _write_pdf(path, text)
        elif fmt == 1 and _HAS_DOCX:
            path = out_dir / f"candidate_{i:02d}.docx"
            _write_docx(path, text)
        else:
            path = out_dir / f"candidate_{i:02d}.txt"
            path.write_text(text, encoding="utf-8")

        files.append(path)
        tags_by_file[path.name] = tags

    # --- Edge cases --------------------------------------------------------
    corrupt = out_dir / "candidate_corrupt.pdf"
    corrupt.write_bytes(b"%PDF-1.4\nthis is not a real pdf body \x00\x01\x02")
    tags_by_file[corrupt.name] = ["corrupt"]
    files.append(corrupt)

    empty = out_dir / "candidate_empty.txt"
    empty.write_text("   \n\n", encoding="utf-8")
    tags_by_file[empty.name] = ["empty"]
    files.append(empty)

    # Exact duplicate of the first file (different name, same bytes).
    if files:
        dup = out_dir / f"candidate_00_copy{files[0].suffix}"
        shutil.copyfile(files[0], dup)
        tags_by_file[dup.name] = ["duplicate"]
        files.append(dup)

    # Unsupported type (should be skipped, not fail the batch).
    png = out_dir / "candidate_photo.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    tags_by_file[png.name] = ["unsupported"]
    files.append(png)

    (out_dir / "_tags.json").write_text(
        __import__("json").dumps(tags_by_file, indent=2), encoding="utf-8"
    )
    return tags_by_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic resumes.")
    parser.add_argument("--out", type=Path, default=Path("resumes_sample"))
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    tags = generate(args.out, args.count, args.seed)
    print(f"Generated {len(tags)} files in {args.out} "
          f"(pdf={_HAS_PDF}, docx={_HAS_DOCX})")


if __name__ == "__main__":
    main()
