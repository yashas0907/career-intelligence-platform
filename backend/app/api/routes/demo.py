"""Demo data endpoint: lets visitors try the full product with one click,
without uploading a real resume. Sample text is clearly synthetic."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.entities import Resume
from app.schemas.schemas import ResumeProfileResponse
from app.services.document_parser import DocumentParseError, parse_document
from app.services.extractor import extract_resume

router = APIRouter(prefix="/demo", tags=["demo"])

DEMO_RESUME = """Aarav Sharma
Bengaluru, India | aarav.sharma@example.com | +91 98765 43210
github.com/aaravsharma | linkedin.com/in/aaravsharma

SUMMARY
Third-year Computer Science student passionate about machine learning and building data-driven products.

EDUCATION
B.Tech in Computer Science
Indian Institute of Technology, Bombay
2023 - 2027

TECHNICAL SKILLS
Languages: Python, C++, SQL, JavaScript
ML: PyTorch, scikit-learn, pandas, NumPy, TensorFlow
Tools: Git, Docker, Linux, Jupyter

EXPERIENCE
Machine Learning Intern | TraceAI Solutions
June 2025 - August 2025
- Built a churn prediction model using scikit-learn, improving recall by 12% over baseline
- Deployed a FastAPI service on AWS with Docker, serving 5k requests/day

PROJECTS
Semantic Document Search
- Built a RAG pipeline with LangChain, Hugging Face embeddings and FAISS vector search over 10k documents
- Achieved 89% answer faithfulness on a 200-question eval set

Real-time Object Detection
- Fine-tuned YOLOv8 on a custom dataset of 3k images; 42 FPS on edge GPU
- Used OpenCV for preprocessing and PyTorch for training

CERTIFICATIONS
Deep Learning Specialization - DeepLearning.AI (2024)

ACHIEVEMENTS
Winner, Smart India Hackathon 2024 (ML track)
"""

DEMO_JOBS = [
    {
        "title": "Machine Learning Intern",
        "description": """Machine Learning Intern

About the role
We are looking for a machine learning intern to join our applied ML team.

Requirements:
- Strong Python skills and experience with PyTorch or TensorFlow
- Familiarity with machine learning fundamentals and scikit-learn
- Experience building and deploying models with Docker
- Understanding of SQL and data pipelines
- Currently pursuing a Bachelor's degree in CS or related field

Nice to have:
- Experience with AWS or other cloud platforms
- NLP or computer vision project experience
- Kubernetes familiarity""",
    },
    {
        "title": "Backend Developer Intern",
        "description": """Backend Developer Intern

Requirements:
- Strong Python and Django or FastAPI experience
- Experience designing REST APIs
- PostgreSQL and Redis knowledge
- Docker for local development
- 1+ years of experience building web services

Preferred:
- GraphQL
- Kubernetes
- CI/CD pipelines with Jenkins""",
    },
    {
        "title": "Data Analyst Intern",
        "description": """Data Analyst Intern

Requirements:
- Strong SQL and Python skills
- Experience with pandas, statistics and data visualization
- Comfortable presenting insights to stakeholders
- Currently pursuing any degree

Nice to have:
- Tableau or Power BI
- Excel""",
    },
]


@router.get("/sample")
def get_demo_data() -> dict:
    """Return demo resume + job descriptions (no server-side state)."""
    return {"resume_text": DEMO_RESUME, "jobs": DEMO_JOBS}


@router.post("/seed", response_model=ResumeProfileResponse)
def seed_demo_resume(db: Session = Depends(get_db)) -> ResumeProfileResponse:
    """Parse + store the demo resume so the whole app is instantly explorable."""
    try:
        parsed = parse_document("demo-resume.txt", DEMO_RESUME.encode("utf-8"))
    except DocumentParseError as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Demo resume failed to parse: {exc}") from exc

    profile = extract_resume(parsed.text)
    resume = Resume(
        filename="demo-resume.txt",
        file_type="txt",
        file_size_bytes=len(DEMO_RESUME.encode("utf-8")),
        raw_text=parsed.text,
        parse_method="demo",
        extraction_status="ok",
        profile={k: v for k, v in profile.items() if not k.startswith("_")},
        warnings=["Demo resume loaded — upload your own any time to get real results."],
    )
    db.add(resume)
    db.commit()
    db.refresh(resume)

    return ResumeProfileResponse(
        resume_id=resume.id,
        filename=resume.filename,
        parse_method=resume.parse_method,
        extraction_status=resume.extraction_status,
        profile=resume.profile,
        warnings=resume.warnings,
        created_at=resume.created_at.isoformat(),
    )
