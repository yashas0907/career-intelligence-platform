"""Skill taxonomy, normalization and matching knowledge base.

This module is the *deterministic backbone* of the matching engine:

1. `normalize_skill` maps any raw string to a canonical skill id
   (case-insensitive, alias-aware, e.g. "pytorch" and "PyTorch Framework" -> "pytorch").
2. `skill_category` classifies a canonical id (language / framework / tool / concept /
   soft / cloud / database).
3. `related_skills` provides a curated relatedness graph used to detect
   *transferable skills* (e.g. tensorflow <-> pytorch) without calling an LLM.
4. `SkillNormalizer.extract` pulls skills from free text via a fast Aho-Corasick-like
   scan using compiled word-boundary regexes.

The taxonomy is intentionally compact (~100 entries) — big enough to be genuinely
useful for tech/AI roles, small enough to audit by hand. It is data, not magic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Taxonomy entries: canonical_id -> metadata
# aliases: alternative spellings/variants that normalize to the canonical id
# related: transferable skill ids (bidirectional)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SkillMeta:
    canonical: str
    display: str
    category: str  # language | framework | library | tool | cloud | database | concept | soft | domain
    aliases: tuple[str, ...] = ()
    related: tuple[str, ...] = ()


_RAW: list[SkillMeta] = [
    # --- Languages ---
    SkillMeta("python", "Python", "language", ("python3", "python 3", "py"), ("cpython",)),
    SkillMeta("javascript", "JavaScript", "language", ("js", "ecmascript", "es6"), ("typescript",)),
    SkillMeta("typescript", "TypeScript", "language", ("ts",), ("javascript",)),
    SkillMeta("java", "Java", "language", ("java 8", "java8", "core java",), ()),
    SkillMeta("c", "C", "language", ("c language",), ("cpp",)),
    SkillMeta("cpp", "C++", "language", ("c++", "c plus plus"), ("c",)),
    SkillMeta("csharp", "C#", "language", ("c#", "c sharp"), ("dotnet",)),
    SkillMeta("go", "Go", "language", ("golang",), ()),
    SkillMeta("rust", "Rust", "language", (), ()),
    SkillMeta("sql", "SQL", "language", ("structured query language",), ("mysql", "postgresql")),
    SkillMeta("r", "R", "language", ("r language", "r programming",), ()),
    SkillMeta("scala", "Scala", "language", (), ()),
    SkillMeta("matlab", "MATLAB", "language", ("mat lab",), ()),
    SkillMeta("bash", "Bash", "language", ("shell", "shell scripting", "unix shell", "zsh"), ("linux",)),
    SkillMeta("php", "PHP", "language", (), ()),
    SkillMeta("kotlin", "Kotlin", "language", (), ("java",)),
    SkillMeta("swift", "Swift", "language", (), ()),
    SkillMeta("html", "HTML", "language", ("html5",), ("css",)),
    SkillMeta("css", "CSS", "language", ("css3",), ("html",)),

    # --- ML / AI frameworks ---
    SkillMeta("pytorch", "PyTorch", "framework", ("torch", "py torch"), ("tensorflow", "keras")),
    SkillMeta("tensorflow", "TensorFlow", "framework", ("tf",), ("pytorch", "keras")),
    SkillMeta("keras", "Keras", "framework", (), ("tensorflow", "pytorch")),
    SkillMeta("scikit-learn", "scikit-learn", "library", ("sklearn", "scikit learn"), ("numpy", "pandas")),
    SkillMeta("xgboost", "XGBoost", "library", ("x-gboost",), ("scikit-learn",)),
    SkillMeta("huggingface", "Hugging Face", "library", ("transformers", "hugging face", "hf transformers"), ("pytorch",)),
    SkillMeta("opencv", "OpenCV", "library", ("cv2",), ()),
    SkillMeta("nltk", "NLTK", "library", (), ("spacy",)),
    SkillMeta("spacy", "spaCy", "library", ("spacy nlp",), ("nltk",)),
    SkillMeta("gensim", "Gensim", "library", (), ()),
    SkillMeta("langchain", "LangChain", "framework", ("lang chain",), ("llm",)),
    SkillMeta("llamaindex", "LlamaIndex", "framework", ("llama index",), ("langchain",)),
    SkillMeta("yolo", "YOLO", "framework", ("yolov5", "yolov8", "yolo object detection"), ("opencv", "pytorch")),
    SkillMeta("spark", "Apache Spark", "framework", ("pyspark", "apache spark"), ("hadoop",)),
    SkillMeta("mlops", "MLOps", "concept", ("ml ops", "ml pipelines"), ("docker", "kubernetes")),

    # --- Data ---
    SkillMeta("numpy", "NumPy", "library", ("np",), ("pandas",)),
    SkillMeta("pandas", "pandas", "library", (), ("numpy",)),
    SkillMeta("matplotlib", "Matplotlib", "library", ("plt",), ("seaborn",)),
    SkillMeta("seaborn", "Seaborn", "library", (), ("matplotlib",)),
    SkillMeta("plotly", "Plotly", "library", (), ("matplotlib",)),
    SkillMeta("jupyter", "Jupyter", "tool", ("jupyter notebook", "jupyterlab", "ipython notebook"), ()),
    SkillMeta("excel", "Excel", "tool", ("ms excel", "microsoft excel"), ()),
    SkillMeta("powerbi", "Power BI", "tool", ("power-bi", "ms powerbi"), ("tableau",)),
    SkillMeta("tableau", "Tableau", "tool", (), ("powerbi",)),

    # --- Web / backend frameworks ---
    SkillMeta("django", "Django", "framework", (), ("flask", "fastapi")),
    SkillMeta("flask", "Flask", "framework", (), ("django", "fastapi")),
    SkillMeta("fastapi", "FastAPI", "framework", ("fast api",), ("flask", "django")),
    SkillMeta("react", "React", "framework", ("reactjs", "react.js", "react js"), ("nextjs",)),
    SkillMeta("nextjs", "Next.js", "framework", ("next.js", "next js"), ("react",)),
    SkillMeta("vue", "Vue.js", "framework", ("vuejs", "vue.js", "vue js"), ("react",)),
    SkillMeta("angular", "Angular", "framework", ("angularjs",), ("react",)),
    SkillMeta("nodejs", "Node.js", "framework", ("node.js", "node js", "node"), ("javascript",)),
    SkillMeta("express", "Express", "framework", ("express.js", "express js", "expressjs"), ("nodejs",)),
    SkillMeta("spring", "Spring Boot", "framework", ("spring", "springboot", "java spring"), ("java",)),
    SkillMeta("dotnet", ".NET", "framework", (".net", "asp.net", "dot net", "net core", ".net core"), ("csharp",)),
    SkillMeta("streamlit", "Streamlit", "framework", (), ("gradio",)),
    SkillMeta("gradio", "Gradio", "framework", (), ("streamlit",)),

    # --- Cloud / DevOps ---
    SkillMeta("aws", "AWS", "cloud", ("amazon web services", "ec2", "s3", "lambda", "sagemaker", "aws cloud"), ("azure", "gcp")),
    SkillMeta("azure", "Azure", "cloud", ("microsoft azure", "azure cloud"), ("aws", "gcp")),
    SkillMeta("gcp", "Google Cloud", "cloud", ("google cloud platform", "google cloud", "gcp"), ("aws", "azure")),
    SkillMeta("docker", "Docker", "tool", ("containers", "containerization", "docker compose"), ("kubernetes",)),
    SkillMeta("kubernetes", "Kubernetes", "tool", ("k8s", "kubectl"), ("docker",)),
    SkillMeta("git", "Git", "tool", ("github", "gitlab", "version control", "git version control"), ()),
    SkillMeta("jenkins", "Jenkins", "tool", ("ci/cd", "cicd", "ci cd", "continuous integration"), ("docker",)),
    SkillMeta("terraform", "Terraform", "tool", ("iac", "infrastructure as code"), ("docker",)),
    SkillMeta("linux", "Linux", "tool", ("unix", "ubuntu", "debian", "linux administration"), ("bash",)),
    SkillMeta("rest-api", "REST APIs", "concept", ("rest", "restful", "rest api", "restful api", "api development", "api design", "http api"), ("graphql",)),
    SkillMeta("graphql", "GraphQL", "concept", ("graph ql", "apollo graphql"), ("rest-api",)),

    # --- Databases ---
    SkillMeta("mysql", "MySQL", "database", ("my sql", "mariadb"), ("postgresql", "sql")),
    SkillMeta("postgresql", "PostgreSQL", "database", ("postgres", "postgre"), ("mysql", "sql")),
    SkillMeta("mongodb", "MongoDB", "database", ("mongo", "mongo db"), ("nosql",)),
    SkillMeta("nosql", "NoSQL", "concept", ("no sql", "no-sql"), ("mongodb",)),
    SkillMeta("redis", "Redis", "database", (), ("nosql",)),
    SkillMeta("sqlite", "SQLite", "database", ("sqllite",), ("sql",)),
    SkillMeta("elasticsearch", "Elasticsearch", "database", ("elastic search", "opensearch"), ("nosql",)),
    SkillMeta("vector-db", "Vector databases", "database", ("vector database", "vector db", "vector search", "faiss", "pinecone", "milvus", "weaviate", "chroma", "pgvector"), ("embeddings",)),

    # --- AI/ML concepts ---
    SkillMeta("machine-learning", "Machine Learning", "concept", ("ml", "machine learning algorithms", "supervised learning", "classic ml"), ("deep-learning", "scikit-learn")),
    SkillMeta("deep-learning", "Deep Learning", "concept", ("dl", "neural networks", "neural nets", "ann", "dnn", "cnn", "rnn"), ("machine-learning", "pytorch")),
    SkillMeta("nlp", "Natural Language Processing", "concept", ("natural language processing", "text mining", "computational linguistics"), ("machine-learning",)),
    SkillMeta("computer-vision", "Computer Vision", "concept", ("cv", "image processing", "image classification", "object detection", "image recognition"), ("deep-learning", "opencv")),
    SkillMeta("llm", "Large Language Models", "concept", ("large language models", "llms", "gpt", "gpt-4", "gpt-3", "chatgpt", "generative ai", "genai", "gen ai", "openai api"), ("nlp",)),
    SkillMeta("rag", "RAG", "concept", ("retrieval augmented generation", "retrieval-augmented generation"), ("llm", "embeddings")),
    SkillMeta("embeddings", "Embeddings", "concept", ("word embeddings", "sentence embeddings", "vector embeddings", "semantic similarity"), ("rag",)),
    SkillMeta("prompt-engineering", "Prompt Engineering", "concept", ("prompt design", "prompting", "prompts"), ("llm",)),
    SkillMeta("transformers", "Transformer models", "concept", ("transformer architecture", "attention mechanism", "bert", "self-attention"), ("nlp", "llm")),
    SkillMeta("reinforcement-learning", "Reinforcement Learning", "concept", ("rl", "q-learning"), ("machine-learning",)),
    SkillMeta("feature-engineering", "Feature Engineering", "concept", ("feature selection", "feature extraction"), ("machine-learning",)),
    SkillMeta("model-deployment", "Model Deployment", "concept", ("model serving", "deploying models", "production ml", "model inference", "inference optimization", "onnx"), ("mlops",)),
    SkillMeta("statistical-analysis", "Statistical Analysis", "concept", ("statistics", "statistical modeling", "hypothesis testing", "a/b testing", "ab testing", "regression analysis"), ("machine-learning",)),
    SkillMeta("data-visualization", "Data Visualization", "concept", ("dataviz", "data viz", "visualisation", "visualization"), ("matplotlib",)),
    SkillMeta("etl", "ETL / Data Pipelines", "concept", ("etl", "data pipelines", "data pipeline", "data engineering", "data wrangling", "data cleaning"), ("spark",)),
    SkillMeta("time-series", "Time Series Analysis", "concept", ("time series", "forecasting", "time-series forecasting"), ("statistical-analysis",)),

    # --- Domain ---
    SkillMeta("fintech", "FinTech", "domain", ("finance", "financial services", "fintech", "banking"), ()),
    SkillMeta("healthcare", "Healthcare / Medical", "domain", ("healthcare", "medical", "clinical", "health care", "biotech"), ()),
    SkillMeta("ecommerce", "E-commerce", "domain", ("e-commerce", "ecommerce", "retail", "online marketplace"), ()),
    SkillMeta("cybersecurity", "Cybersecurity", "domain", ("security", "infosec", "information security", "penetration testing"), ()),

    # --- Soft skills ---
    SkillMeta("communication", "Communication", "soft", ("communication skills", "verbal communication", "written communication", "presentation skills", "public speaking"), ()),
    SkillMeta("teamwork", "Teamwork", "soft", ("collaboration", "team player", "collaborative", "cross-functional", "cross functional"), ()),
    SkillMeta("problem-solving", "Problem Solving", "soft", ("problem solving", "analytical thinking", "analytical skills", "critical thinking"), ()),
    SkillMeta("leadership", "Leadership", "soft", ("leading teams", "team lead", "mentoring", "mentored", "people management"), ()),
    SkillMeta("ownership", "Ownership", "soft", ("self-starter", "initiative", "autonomy", "independent", "proactive"), ()),
    SkillMeta("adaptability", "Adaptability", "soft", ("fast-paced", "fast paced", "flexible", "agile mindset"), ()),
    SkillMeta("time-management", "Time Management", "soft", ("prioritization", "prioritisation", "deadline", "deadlines", "organizational skills"), ()),
]

TAXONOMY: dict[str, SkillMeta] = {m.canonical: m for m in _RAW}

# alias -> canonical  (longest aliases first to prefer specific matches)
_ALIAS_MAP: dict[str, str] = {}
for _m in _RAW:
    _ALIAS_MAP[_m.canonical] = _m.canonical
    for _a in _m.aliases:
        _ALIAS_MAP[_a.lower()] = _m.canonical

# related: bidirectional graph
_RELATED: dict[str, set[str]] = {}
for _m in _RAW:
    for _r in _m.related:
        _RELATED.setdefault(_m.canonical, set()).add(_r)
        _RELATED.setdefault(_r, set()).add(_m.canonical)


def normalize_skill(raw: str) -> str | None:
    """Return canonical skill id for a raw string, or None if unknown."""
    if not raw:
        return None
    key = raw.strip().lower()
    key = re.sub(r"\s+", " ", key)
    return _ALIAS_MAP.get(key)


def skill_display(skill_id: str) -> str:
    m = TAXONOMY.get(skill_id)
    return m.display if m else skill_id.replace("-", " ").title()


def skill_category(skill_id: str) -> str:
    m = TAXONOMY.get(skill_id)
    return m.category if m else "concept"


def related_skills(skill_id: str) -> set[str]:
    return set(_RELATED.get(skill_id, set()))


def canonical_related_pairs() -> list[tuple[str, str]]:
    return sorted({tuple(sorted((a, b))) for a, related in _RELATED.items() for b in related})


@dataclass
class SkillNormalizer:
    """Extract + normalize skills found in arbitrary text."""

    # regex per canonical id, longest alias first so "hugging face" wins over "face"
    _patterns: list[tuple[re.Pattern, str]] = field(default_factory=list, init=False, repr=False)
    _built: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._build()

    def _build(self) -> None:
        entries: list[tuple[str, str]] = []
        for meta in _RAW:
            entries.append((meta.canonical, meta.canonical))
            for alias in meta.aliases:
                entries.append((alias, meta.canonical))
        # Longest first; escape everything
        entries.sort(key=lambda t: len(t[0]), reverse=True)
        self._patterns = [
            (re.compile(r"(?<![\w./-])" + re.escape(term) + r"(?![\w/-])", re.IGNORECASE), canon)
            for term, canon in entries
        ]
        self._built = True

    def extract(self, text: str) -> dict[str, list[str]]:
        """Return {canonical_id: [evidence snippets]} for skills found in text."""
        found: dict[str, list[str]] = {}
        if not text:
            return found
        lowered = text.lower()
        for pattern, canon in self._patterns:
            m = pattern.search(lowered)
            if m:
                found.setdefault(canon, [])
                if not found[canon]:
                    start = max(0, m.start() - 40)
                    end = min(len(text), m.end() + 40)
                    snippet = text[start:end].replace("\n", " ").strip()
                    found[canon].append(snippet)
        return found


# module-level singleton used across the app
normalizer = SkillNormalizer()
