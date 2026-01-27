"""
Build a search index from RST docs and Python docstrings for AI search.
Usage: python build_search_index.py
Output: _static/search_index.json
"""

import ast
import json
import re
from pathlib import Path


def clean_rst_text(text: str) -> str:
    """Remove RST markup and clean text for indexing."""
    text = re.sub(r"\.\.\s+\w+::[^\n]*\n", "", text)
    text = re.sub(r":\w+:`[^`]+`", "", text)
    text = re.sub(r"\.\.\s+_[^:]+:", "", text)
    text = re.sub(r"\.\.\s+\|[^|]+\|[^\n]+", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_title(text: str) -> str:
    """Extract document title from RST content."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            if (
                next_line
                and len(next_line) >= len(line.strip())
                and set(next_line.strip()) <= {"=", "-", "~", "^"}
            ):
                return line.strip()
    return "Untitled"


def title_to_anchor(title: str) -> str:
    """Convert section title to Sphinx anchor ID."""
    anchor = title.lower().strip()
    anchor = re.sub(r"[^a-z0-9\s-]", "", anchor)
    anchor = re.sub(r"\s+", "-", anchor)
    anchor = re.sub(r"-+", "-", anchor)
    return anchor.strip("-")


def extract_sections_from_rst(text: str) -> list[dict]:
    """Extract sections from RST content with anchor links."""
    lines = text.split("\n")
    sections = []
    current_section = None
    current_content = []
    heading_chars = {"=", "-", "~", "^", '"', "'"}

    i = 0
    while i < len(lines):
        line = lines[i]
        if i + 1 < len(lines) and line.strip():
            next_line = lines[i + 1]
            if (
                next_line
                and len(next_line.strip()) >= len(line.strip())
                and len(set(next_line.strip())) == 1
                and set(next_line.strip()) <= heading_chars
            ):
                if current_section:
                    sections.append(
                        {
                            "title": current_section,
                            "anchor": title_to_anchor(current_section),
                            "content": clean_rst_text("\n".join(current_content)),
                        }
                    )
                current_section = line.strip()
                current_content = []
                i += 2
                continue
        if current_section:
            current_content.append(line)
        i += 1

    if current_section and current_content:
        sections.append(
            {
                "title": current_section,
                "anchor": title_to_anchor(current_section),
                "content": clean_rst_text("\n".join(current_content)),
            }
        )
    return sections


def extract_docstrings_from_python(file_path: Path) -> list[dict]:
    """Extract docstrings from a Python file using AST."""
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    docstrings = []
    module_doc = ast.get_docstring(tree)
    if module_doc:
        docstrings.append(
            {"name": file_path.stem, "type": "module", "docstring": module_doc}
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node)
            if doc:
                docstrings.append(
                    {"name": node.name, "type": "class", "docstring": doc}
                )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_") and node.name != "__init__":
                continue
            doc = ast.get_docstring(node)
            if doc:
                docstrings.append(
                    {"name": node.name, "type": "function", "docstring": doc}
                )
    return docstrings


def build_index_from_rst(doc_dir: Path) -> list[dict]:
    """Build search index from RST files, split into sections."""
    index = []
    for rst_file in doc_dir.rglob("*.rst"):
        if "_templates" in str(rst_file) or "apidoc_templates" in str(rst_file):
            continue
        try:
            content = rst_file.read_text(encoding="utf-8")
        except Exception:
            continue

        html_path = str(rst_file.relative_to(doc_dir)).replace(".rst", ".html")
        sections = extract_sections_from_rst(content)

        if not sections:
            index.append(
                {
                    "title": extract_title(content),
                    "path": html_path,
                    "content": clean_rst_text(content),
                }
            )
            print(f"Indexed RST: {extract_title(content)} ({html_path})")
        else:
            doc_title = sections[0]["title"]
            for i, section in enumerate(sections):
                path = html_path if i == 0 else f"{html_path}#{section['anchor']}"
                title = (
                    section["title"] if i == 0 else f"{section['title']} ({doc_title})"
                )
                if len(section["content"].strip()) > 50:
                    index.append(
                        {"title": title, "path": path, "content": section["content"]}
                    )
                    print(f"Indexed RST: {title} ({path})")
    return index


def build_index_from_python(source_dirs: list[Path], base_dir: Path) -> list[dict]:
    """Build search index from Python docstrings."""
    index = []
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        for py_file in source_dir.rglob("*.py"):
            if "test" in py_file.name.lower() or "__pycache__" in str(py_file):
                continue
            if py_file.name.startswith("_") and py_file.name != "__init__.py":
                continue

            docstrings = extract_docstrings_from_python(py_file)
            if not docstrings:
                continue

            try:
                module_path = (
                    str(py_file.relative_to(base_dir))
                    .replace("/", ".")
                    .replace(".py", "")
                )
            except ValueError:
                module_path = py_file.stem

            combined = [
                f"## {d['type'].title()}: {d['name']}\n{d['docstring']}"
                for d in docstrings
            ]
            title = module_path.split(".")[-1]
            for d in docstrings:
                if d["type"] == "class":
                    title = f"{d['name']} ({module_path})"
                    break

            index.append(
                {
                    "title": title,
                    "path": f"reference/api/{module_path}.html",
                    "content": "\n\n".join(combined),
                    "module": module_path,
                }
            )
            print(f"Indexed Python: {title}")
    return index


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    print("Building search index...\n")

    rst_index = build_index_from_rst(script_dir)
    python_index = build_index_from_python(
        [
            project_root / "ecoli",
            project_root / "wholecell",
            project_root / "reconstruction",
            project_root / "runscripts",
            project_root / "validation",
        ],
        project_root,
    )

    index = rst_index + python_index
    output_path = script_dir / "_static" / "search_index.json"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(
        f"\nCreated index: {len(rst_index)} RST + {len(python_index)} Python = {len(index)} total"
    )


if __name__ == "__main__":
    main()
