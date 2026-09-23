"""Shared vocabulary for skill detection.

Single source of truth so extraction, eligibility and scoring all recognise
the same terms. `canonical` is what we show in output; `terms` are the
surface forms we match (regex, word-boundary, case-insensitive).

Matching strategy is deliberately conservative: we match whole tokens/phrases
(and common separators like `next.js`) to avoid false positives such as
"java" matching inside "javascript" (handled in matcher, see eligibility).
"""

from __future__ import annotations

# --- Python stack: genuine Python evidence -------------------------------
PYTHON_TERMS: dict[str, list[str]] = {
    "Python": ["python", "python3", "python 3"],
    "FastAPI": ["fastapi"],
    "Django": ["django"],
    "Flask": ["flask"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Pydantic": ["pydantic"],
    "SQLAlchemy": ["sqlalchemy"],
    "Celery": ["celery"],
    "asyncio": ["asyncio", "async/await", "async io"],
    "pytest": ["pytest"],
    "Streamlit": ["streamlit"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "SciPy": ["scipy"],
    "Poetry": ["poetry"],
    "uvicorn": ["uvicorn"],
    "BeautifulSoup": ["beautifulsoup", "bs4"],
    "Selenium": ["selenium"],
}

# --- AI / agentic evidence (eligibility gate) ----------------------------
AI_TERMS: dict[str, list[str]] = {
    "LangChain": ["langchain"],
    "LangGraph": ["langgraph"],
    "LlamaIndex": ["llamaindex", "llama-index"],
    "Google ADK": ["google adk", "agent development kit", "google agent development kit"],
    "CrewAI": ["crewai", "crew ai"],
    "AutoGen": ["autogen", "microsoft autogen"],
    "Semantic Kernel": ["semantic kernel"],
    "Haystack": ["haystack"],
    "DSPy": ["dspy"],
    "MCP": ["model context protocol", "mcp server", "mcp"],
    "OpenAI API": ["openai", "gpt-4", "gpt-3", "gpt4", "chatgpt api"],
    "Anthropic": ["anthropic", "claude api", "claude"],
    "Gemini": ["gemini", "google ai studio", "vertex ai"],
    "Hugging Face": ["hugging face", "huggingface", "transformers"],
    "RAG": ["rag", "retrieval-augmented", "retrieval augmented", "retrieval pipeline",
            "semantic chunking", "semantic search"],
    "Embeddings": ["embedding", "embeddings"],
    "Vector Search": ["vector search", "vector database", "vector store", "similarity search"],
    "Pinecone": ["pinecone"],
    "FAISS": ["faiss"],
    "Chroma": ["chroma", "chromadb"],
    "Weaviate": ["weaviate"],
    "Qdrant": ["qdrant"],
    "pgvector": ["pgvector"],
    "Tool Calling": ["tool calling", "function calling", "tool use", "tool invocation"],
    "Multi-Agent": ["multi-agent", "multi agent", "agentic workflow", "agent orchestration"],
    "AI Agents": ["agent", "agents", "llm agent", "ai agent"],
    "LLM": ["llm", "large language model", "language model"],
    "Prompt Engineering": ["prompt engineering", "prompting", "system prompt"],
    "Fine-tuning": ["fine-tun", "finetun", "lora", "peft"],
    "Evaluation": ["evaluation pipeline", "eval harness", "llm evaluation", "ragas", "guardrails"],
    "PyTorch": ["pytorch", "torch"],
    "TensorFlow": ["tensorflow"],
    "LangSmith": ["langsmith", "langfuse"],
    "Diffusers": ["diffusers", "stable diffusion"],
    "Computer Vision": ["computer vision", "opencv", "image classification", "yolo"],
    "NLP": ["nlp", "natural language processing", "named entity", "text classification"],
    # General AI / ML / DS (counts as an AI *project*, but scores low on the
    # agentic depth rubric unless there is real system depth).
    "Machine Learning": ["machine learning", "machinelearning", "supervised learning", "ml model", "ml models"],
    "Deep Learning": ["deep learning", "deeplearning", "neural network", "neural networks", "cnn", "rnn", "lstm"],
    "Artificial Intelligence": ["artificial intelligence", "artificialintelligence"],
    "XGBoost": ["xgboost", "lightgbm", "gradient boosting"],
    "Keras": ["keras"],
    "Data Science": ["data science"],
    "Generative AI": ["gen ai", "genai", "generative ai"],
}

# --- Supporting: backend / data -----------------------------------------
BACKEND_TERMS: dict[str, list[str]] = {
    "PostgreSQL": ["postgresql", "postgres", "psql"],
    "MySQL": ["mysql", "mariadb"],
    "MongoDB": ["mongodb", "mongo"],
    "Redis": ["redis"],
    "SQL": ["sql"],
    "REST API": ["rest api", "restful", "rest apis"],
    "GraphQL": ["graphql"],
    "Kafka": ["kafka"],
    "RabbitMQ": ["rabbitmq"],
    "WebSockets": ["websocket", "socket.io"],
    "gRPC": ["grpc"],
    "Node.js": ["node.js", "nodejs", "node"],
    "Express": ["express.js", "expressjs", "express"],
    "Java": ["java"],
    "Spring Boot": ["spring boot", "springboot", "spring"],
    "Go": ["golang", "go language"],
    "C++": ["c++", "cpp"],
    "TypeScript": ["typescript"],
    "JavaScript": ["javascript", "js"],
}

# --- Cloud / deployment / full stack ------------------------------------
CLOUD_TERMS: dict[str, list[str]] = {
    "GCP": ["gcp", "google cloud", "cloud run", "gke", "bigquery"],
    "AWS": ["aws", "amazon web services", "ec2", "s3 bucket", "lambda"],
    "Azure": ["azure", "azure functions"],
    "Docker": ["docker", "dockerfile", "container"],
    "Kubernetes": ["kubernetes", "k8s", "helm"],
    "Terraform": ["terraform"],
    "CI/CD": ["ci/cd", "github actions", "gitlab ci", "jenkins", "continuous integration"],
    "Vercel": ["vercel"],
    "Render": ["render.com", "render"],
    "Heroku": ["heroku"],
    "Nginx": ["nginx"],
    "Serverless": ["serverless", "cloud functions", "lambda functions"],
    "React": ["react", "react.js", "reactjs"],
    "Next.js": ["next.js", "nextjs"],
    "Tailwind": ["tailwind"],
    "Streamlit": ["streamlit"],
    "Firebase": ["firebase"],
    "Supabase": ["supabase"],
}

# --- Engineering depth signals ------------------------------------------
ENGINEERING_TERMS: dict[str, list[str]] = {
    "Testing": ["unit test", "unit tests", "testing", "test suite", "integration test", "pytest", "jest"],
    "Architecture": ["architecture", "design patterns", "clean architecture", "system design", "modular"],
    "Caching": ["caching", "cache layer", "memoization", "redis cache"],
    "Queues": ["message queue", "task queue", "celery", "rabbitmq", "kafka", "sqs", "background jobs"],
    "Observability": ["observability", "monitoring", "prometheus", "grafana", "logging", "tracing", "opentelemetry"],
    "Concurrency": ["concurrency", "concurrent", "async", "multithreading", "multiprocessing", "parallelism", "asyncio"],
    "Failure Handling": ["retry", "retries", "circuit breaker", "rate limiting", "error handling", "fault tolerance", "idempotent"],
    "Performance": ["optimization", "latency", "throughput", "profiling", "indexing", "load balancing"],
    "Security": ["authentication", "authorization", "oauth", "jwt", "encryption", "owasp"],
}

# Canonical AI terms that count as *core* LLM/agentic evidence for eligibility.
# These mirror the spec's examples (LangChain, LangGraph, ADK, RAG, vector
# search, tool-calling agents, multi-agent workflows, evaluation pipelines).
AGENTIC_CORE_TERMS: set[str] = {
    "LangChain", "LangGraph", "LlamaIndex", "Google ADK", "CrewAI", "AutoGen",
    "Semantic Kernel", "Haystack", "DSPy", "MCP", "OpenAI API", "Anthropic",
    "Gemini", "Hugging Face", "RAG", "Embeddings", "Vector Search", "Pinecone",
    "FAISS", "Chroma", "Weaviate", "Qdrant", "pgvector", "Tool Calling",
    "Multi-Agent", "AI Agents", "LLM", "Prompt Engineering", "Fine-tuning",
    "Evaluation", "LangSmith",
}

# Canonical AI terms that indicate general ML/CV/NLP but are not by themselves
# strong "agentic" evidence. A project using these still counts as an AI project.
GENERAL_AI_TERMS: set[str] = {
    "PyTorch", "TensorFlow", "Computer Vision", "NLP", "Diffusers",
    "Machine Learning", "Deep Learning", "Artificial Intelligence",
    "XGBoost", "Keras", "Data Science", "Generative AI",
}

# Category -> canonical terms, used to bucket matched skills for output.
ALL_SKILL_GROUPS: dict[str, dict[str, list[str]]] = {
    "python": PYTHON_TERMS,
    "ai": AI_TERMS,
    "backend": BACKEND_TERMS,
    "cloud": CLOUD_TERMS,
    "engineering": ENGINEERING_TERMS,
}
