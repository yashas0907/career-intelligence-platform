"""Small labeled evaluation corpus for extraction quality measurement.

Ground truth is defined here explicitly so tests can compute precision/recall
per field — honest, measurable evaluation instead of accuracy claims.

To extend: add entries (resume_text, ground_truth) and run
`python tests/eval_extraction.py` for a report.
"""

EVAL_RESUMES: list[dict] = [
    {
        "name": "ml_student",
        "resume_text": """Priya Nair
priya.nair@email.com | +91 90000 11111 | github.com/priyanair

EDUCATION
B.Tech Computer Science
NIT Trichy, 2023 - 2027

SKILLS
Languages: Python, C++, SQL
Frameworks: PyTorch, FastAPI, React
Tools: Git, Docker, Jupyter

EXPERIENCE
Data Science Intern | FinEdge Analytics
May 2025 - July 2025
- Built churn models with scikit-learn; recall +9%
- Created ETL pipelines with pandas

PROJECTS
Music Genre Classifier
- CNN on spectrograms with TensorFlow; 92% accuracy
- Deployed with Docker on AWS

RAG Support Bot
- LangChain + FAISS retrieval over 5k FAQ documents
""",
        "ground_truth": {
            "name": "Priya Nair",
            "email": "priya.nair@email.com",
            "skills": {
                "python", "cpp", "sql", "pytorch", "fastapi", "react", "git",
                "docker", "jupyter", "scikit-learn", "pandas", "etl",
                "tensorflow", "aws", "langchain", "vector-db", "computer-vision",
                "machine-learning",
            },
            "projects": {"Music Genre Classifier", "RAG Support Bot"},
            "experience_titles": {"Data Science Intern"},
            "education_degrees": {"b.tech"},
        },
    },
    {
        "name": "backend_dev",
        "resume_text": """Rohan Mehta
rohan.mehta@gmail.com

EXPERIENCE
Software Engineer | PayU
Jan 2024 - Present
- Developed REST APIs in Java with Spring Boot
- Optimized PostgreSQL queries, cutting latency 40%

Backend Intern | Zerodha
Jun 2023 - Dec 2023
- Node.js microservices with Redis caching

EDUCATION
B.E. Information Technology
VTU, 2020 - 2024

SKILLS
Java, Spring, Node.js, PostgreSQL, Redis, Docker, Kubernetes, REST APIs, Git
""",
        "ground_truth": {
            "name": "Rohan Mehta",
            "email": "rohan.mehta@gmail.com",
            "skills": {
                "java", "spring", "nodejs", "postgresql", "redis", "docker",
                "kubernetes", "rest-api", "git", "sql",
            },
            "projects": set(),
            "experience_titles": {"Software Engineer", "Backend Intern"},
            "education_degrees": {"b.e"},
        },
    },
    {
        "name": "fresher_minimal",
        "resume_text": """Ananya Rao
ananya.rao@outlook.com

EDUCATION
B.Sc Data Science
Christ University, 2024 - 2027

SKILLS
Python, Excel, Power BI, communication

PROJECTS
Sales Dashboard
- Analyzed retail data with pandas and matplotlib
""",
        "ground_truth": {
            "name": "Ananya Rao",
            "email": "ananya.rao@outlook.com",
            "skills": {"python", "excel", "powerbi", "communication", "pandas", "matplotlib", "data-visualization"},
            "projects": {"Sales Dashboard"},
            "experience_titles": set(),
            "education_degrees": {"b.sc"},
        },
    },
]
