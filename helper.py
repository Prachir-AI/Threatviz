from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import JSONLoader
import tempfile

from langchain_community.vectorstores import FAISS
import os
import re
import json


url = [
    "https://mermaid.js.org/intro/syntax-reference.html",
    "https://mermaid.js.org/syntax/sequenceDiagram.html",
    "https://mermaid.js.org/syntax/classDiagram.html",
    "https://mermaid.js.org/syntax/stateDiagram.html",
    "https://mermaid.js.org/syntax/entityRelationshipDiagram.html",
    "https://mermaid.js.org/syntax/pie.html",
    "https://mermaid.js.org/syntax/c4.html",
    "https://mermaid.js.org/syntax/architecture.html"
    ]

pdf = "threat_models.pdf"
INDEX_PATH = "my_rag_db"

def vector_loader(file):
    docs = []

    # PDF
    if isinstance(file, str) and file.endswith('.pdf'):
        docs += PyPDFLoader(file).load()

    # URL
    elif isinstance(file, str) and file.startswith('https://'):
        docs += WebBaseLoader(file).load()

    # JSON dict
    elif isinstance(file, dict):
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
            json.dump(file, f)
            f.flush()
            loader = JSONLoader(
                file_path=f.name,
                jq_schema=".users[]",
                text_content=False
            )
            docs += loader.load()

    else:
        return "❌ File type not supported in RAG"

    # If we have documents, split and build embeddings
    if docs:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(INDEX_PATH)
        return vectorstore

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


if os.path.exists(INDEX_PATH):
    VECTORSTORE = load_vectorstore()
else:
    VECTORSTORE = vector_loader(pdf)


def retrieve_framework_context(query: str, k: int = 5) -> str:
    docs = VECTORSTORE.similarity_search(query, k=k)
    return "\n\n".join([d.page_content for d in docs])

def sanitize_cve_id(cve):
    clean_input = cve.strip().upper()
    pattern = r"^CVE-\d{4}-\d{4,6}$"
    
    if not re.match(pattern, cve):
        raise ValueError("Invalid CVE format. Expected CVE-YYYY-NNNN")
    return cve


def sanitize_llm(content: str) -> str:
    if not content:
        return ""

    # Remove markdown headers (#, ##, ### ...)
    content = re.sub(r'^\s*#{1,6}\s*', '', content, flags=re.MULTILINE)

    # Remove <think>...</think> or any similar tags
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)

    # Remove bold/italic markers (**text**, *text*, __text__, _text_)
    content = re.sub(r'(\*\*|__)(.*?)\1', r'\2', content)
    content = re.sub(r'(\*|_)(.*?)\1', r'\2', content)

    # Remove inline code markers (`code`)
    content = re.sub(r'`([^`]*)`', r'\1', content)

    # Remove code fences like ```mermaid or ```html
    content = re.sub(r'```(?:mermaid|html)?\s*', '', content)
    content = re.sub(r'```', '', content)

    # Remove any remaining excessive whitespace
    content = re.sub(r'\n{3,}', '\n\n', content)

    # Strip leading/trailing whitespace
    content = content.strip()

    return content




def normalize_text_field(field):
    """
    Converts strings or lists into a clean list of non-empty lines.
    Keeps multi-line paragraphs intact when they are already strings.
    """
    if isinstance(field, str):
        # Split only on newlines that are real paragraph breaks
        lines = [line.strip() for line in field.splitlines() if line.strip()]
    elif isinstance(field, list):
        # Flatten list of strings into single lines
        lines = []
        for item in field:
            if isinstance(item, str):
                lines.extend([line.strip() for line in item.splitlines() if line.strip()])
            else:
                lines.append(str(item).strip())
    else:
        lines = []
    return lines



def render_security_report_html(report: dict) -> str:
    """
    Renders a security report JSON into HTML.
    Each section (Executive Summary, Detailed Analysis, Risk Assessment, Threat Model, Mitigation)
    is rendered in its own block. Lists and numbered items are converted to HTML <li>.
    Mermaid diagrams are rendered properly.
    """
    def normalize_text_field(field):
        """Convert string or list into clean list of lines."""
        lines = []
        if isinstance(field, str):
            lines = [line.strip() for line in field.splitlines() if line.strip()]
        elif isinstance(field, list):
            for item in field:
                if isinstance(item, str):
                    lines.extend([line.strip() for line in item.splitlines() if line.strip()])
                else:
                    lines.append(str(item).strip())
        return lines

    def generate_list(items):
        """Convert list of strings to HTML list."""
        if not items:
            return ""
        html = "<ul>\n"
        for item in items:
            html += f"  <li>{item}</li>\n"
        html += "</ul>"
        return html

    def split_mermaid_blocks(code: str):
        """Split mermaid code into separate blocks for rendering."""
        diagram_keywords = (
            "graph", "flowchart", "sequenceDiagram", "erDiagram",
            "classDiagram", "stateDiagram", "stateDiagram-v2", "pie"
        )
        blocks = []
        current = []
        for line in code.splitlines():
            stripped = line.strip()
            if any(stripped.startswith(k) for k in diagram_keywords):
                if current:
                    blocks.append("\n".join(current))
                    current = []
            if stripped:
                current.append(line)
        if current:
            blocks.append("\n".join(current))
        return blocks

    # Extract and normalize fields
    title = report.get("TITLE", "Security Report")
    executive_summary = normalize_text_field(report.get("EXECUTIVE_SUMMARY", []))
    detailed_analysis = normalize_text_field(report.get("DETAILED_ANALYSIS", []))
    risk_assessment_list = normalize_text_field(report.get("RISK_ASSESSMENT", []))
    mitigation_list = normalize_text_field(report.get("MITIGATION", []))
    threat_model_code = report.get("THREAT_MODEL", "")

    # Process mermaid diagrams
    threat_blocks = split_mermaid_blocks(threat_model_code)
    threat_model_html = ""
    for block in threat_blocks:
        threat_model_html += f'<div class="mermaid">\n{block.strip()}\n</div>\n'

    # Render HTML
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>mermaid.initialize({{ startOnLoad: true }});</script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f9f9f9; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; }}
        .section {{ background: #fff; border-radius: 8px; padding: 15px 20px; margin-bottom: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }}
        ul {{ padding-left: 20px; }}
        li {{ margin-bottom: 8px; }}
        .mermaid {{ background: #fff; border-radius: 8px; padding: 10px; margin-top: 10px; }}
    </style>
</head>
<body>
    <h1>{title}</h1>

    <div class="section">
        <h2>Executive Summary</h2>
        {generate_list(executive_summary)}
    </div>

    <div class="section">
        <h2>Detailed Analysis</h2>
        {generate_list(detailed_analysis)}
    </div>

    <div class="section">
        <h2>Risk Assessment</h2>
        {generate_list(risk_assessment_list)}
    </div>

    <div class="section">
        <h2>Threat Model</h2>
        {threat_model_html if threat_model_html else "<p><b>No threat model diagrams available</b></p>"}
    </div>

    <div class="section">
        <h2>Mitigation</h2>
        {generate_list(mitigation_list)}
    </div>
</body>
</html>
"""
    return html
