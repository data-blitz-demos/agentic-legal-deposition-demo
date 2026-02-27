from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_generate_internal_docs_contains_key_artifacts_and_functions(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    output = tmp_path / "code_function_reference.md"
    script = repo_root / "scripts" / "generate_internal_docs.py"

    subprocess.run(
        [sys.executable, str(script), "--output", str(output)],
        check=True,
        cwd=repo_root,
    )

    text = output.read_text(encoding="utf-8")
    assert "## `backend/app/main.py`" in text
    assert "## `frontend/app.js`" in text
    assert "`load_graph_rag_ontology`" in text
    assert "`loadOntologyGraph`" in text
