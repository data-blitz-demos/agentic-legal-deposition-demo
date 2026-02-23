from __future__ import annotations

"""Generate UML-style architecture and processing diagrams as PNG files."""

import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "uml"
OUT_DIR.mkdir(parents=True, exist_ok=True)


try:
    FONT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 24)
    FONT_BOLD = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 26)
    FONT_TITLE = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 40)
except OSError:
    FONT = ImageFont.load_default()
    FONT_BOLD = ImageFont.load_default()
    FONT_TITLE = ImageFont.load_default()


def wrapped(text: str, width: int) -> str:
    """Word-wrap text to a target character width."""

    return "\n".join(textwrap.wrap(text, width=width))


def center_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font, fill: str = "#0F172A"):
    """Draw multiline text centered inside a rectangular box."""

    left, top, right, bottom = box
    w, h = draw.multiline_textbbox((0, 0), text, font=font, spacing=6)[2:]
    x = left + (right - left - w) / 2
    y = top + (bottom - top - h) / 2
    draw.multiline_text((x, y), text, font=font, fill=fill, align="center", spacing=6)


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    body: str,
    fill: str,
    outline: str = "#334155",
):
    """Draw a rounded component box with a header title and body text."""

    draw.rounded_rectangle(box, radius=18, fill=fill, outline=outline, width=3)
    l, t, r, b = box
    header = (l + 12, t + 10, r - 12, t + 64)
    draw.rounded_rectangle(header, radius=12, fill="#E2E8F0", outline=outline, width=2)
    center_text(draw, header, title, FONT_BOLD)
    body_box = (l + 18, t + 72, r - 18, b - 12)
    center_text(draw, body_box, wrapped(body, 30), FONT)


def draw_activity_node(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    fill: str = "#FFFFFF",
    outline: str = "#334155",
):
    """Draw a simple activity/step node used in process diagrams."""

    draw.rounded_rectangle(box, radius=14, fill=fill, outline=outline, width=3)
    center_text(draw, box, wrapped(text, 32), FONT)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], label: str = "", color: str = "#0F172A"):
    """Draw a directional connector with optional label."""

    draw.line([start, end], fill=color, width=4)
    ex, ey = end
    sx, sy = start
    dx = ex - sx
    dy = ey - sy
    length = max((dx * dx + dy * dy) ** 0.5, 1)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    p1 = (ex - 22 * ux + 10 * px, ey - 22 * uy + 10 * py)
    p2 = (ex - 22 * ux - 10 * px, ey - 22 * uy - 10 * py)
    draw.polygon([end, p1, p2], fill=color)

    if label:
        mx = (sx + ex) / 2
        my = (sy + ey) / 2
        box = draw.textbbox((0, 0), label, font=FONT)
        tw = box[2] - box[0]
        th = box[3] - box[1]
        pad = 8
        rect = (mx - tw / 2 - pad, my - th / 2 - pad, mx + tw / 2 + pad, my + th / 2 + pad)
        draw.rounded_rectangle(rect, radius=8, fill="#FEF9C3", outline="#CA8A04", width=2)
        draw.text((mx - tw / 2, my - th / 2), label, font=FONT, fill="#713F12")


def architecture_diagram() -> None:
    """Render and save the component architecture UML PNG."""

    img = Image.new("RGB", (2800, 1800), "#F8FAFC")
    d = ImageDraw.Draw(img)

    d.text((70, 35), "UML Component Architecture", font=FONT_TITLE, fill="#0B3A53")
    d.text((70, 90), "Legal Deposition Analysis Demo (FastAPI + UI + LLM Providers + CouchDB)", font=FONT, fill="#334155")

    # Swimlane backgrounds
    lanes = [
        ((40, 140, 760, 1740), "Client/Operator", "#E0F2FE"),
        ((790, 140, 2060, 1740), "API Service (FastAPI)", "#E2E8F0"),
        ((2090, 140, 2760, 1740), "External Systems", "#F1F5F9"),
    ]
    for box, label, color in lanes:
        d.rounded_rectangle(box, radius=22, fill=color, outline="#94A3B8", width=3)
        d.text((box[0] + 18, box[1] + 14), label, font=FONT_BOLD, fill="#0F172A")

    ui = (110, 260, 690, 560)
    static_assets = (110, 610, 690, 860)
    user = (110, 910, 690, 1200)

    draw_box(d, ui, "Web UI (frontend)", "Case controls\nLLM dropdown + readiness\nTimeline, detail, chat", "#DBEAFE")
    draw_box(d, static_assets, "Static Assets", "index.html\napp.js\nstyles.css\nlogo", "#DBEAFE")
    draw_box(d, user, "Legal User", "Chooses model\nRuns ingest\nRuns chat/reason", "#DBEAFE")

    main_api = (860, 240, 1450, 540)
    workflow = (860, 590, 1450, 940)
    chat = (860, 990, 1450, 1290)
    llm_gateway = (1500, 240, 1980, 540)
    prompts = (1500, 590, 1980, 900)
    couch_client = (1500, 950, 1980, 1290)
    startup_gate = (860, 1340, 1980, 1640)

    draw_box(d, main_api, "main.py Routes", "/api/ingest-case\n/api/depositions\n/api/chat\n/api/reason-contradiction\n/api/llm-options", "#E2E8F0")
    draw_box(d, workflow, "DepositionWorkflow", "Map deposition\nAssess contradictions\nPersist assessments", "#E2E8F0")
    draw_box(d, chat, "AttorneyChatService", "Context assembly\nChat response\nFocused contradiction reasoning", "#E2E8F0")
    draw_box(d, llm_gateway, "LLM Gateway (llm.py)", "Model selection\nOperational readiness checks\nActionable fix messages", "#E2E8F0")
    draw_box(d, prompts, "Prompt Registry", "backend/prompts/*.txt\nSystem + User templates\nVersion-controlled independently", "#E2E8F0")
    draw_box(d, couch_client, "CouchDBClient", "ensure_db / save / update\nlist_depositions / get_doc", "#E2E8F0")
    draw_box(d, startup_gate, "Startup Guard", "At boot: resolve default model\nProbe model operational status\nAbort app startup if LLM unavailable", "#E2E8F0")

    openai = (2170, 280, 2680, 600)
    ollama = (2170, 650, 2680, 1020)
    couchdb = (2170, 1070, 2680, 1380)
    files = (2170, 1430, 2680, 1700)

    draw_box(d, openai, "OpenAI API", "ChatGPT models\n(e.g., gpt-5.2)", "#F8FAFC")
    draw_box(d, ollama, "Ollama Server", "Local models\n(gpt-oss:20b, llama3.3:latest)", "#F8FAFC")
    draw_box(d, couchdb, "CouchDB", "Deposition docs\nContradictions\nScores/flags", "#F8FAFC")
    draw_box(d, files, "Deposition Files", ".txt corpus\nMounted as /data/depositions", "#F8FAFC")

    arrow(d, (690, 410), (860, 390), "HTTPS API")
    arrow(d, (690, 730), (860, 390), "served by /")
    arrow(d, (400, 1200), (400, 560), "interacts")
    arrow(d, (1450, 390), (1500, 390), "LLM options + validation")
    arrow(d, (1450, 770), (1500, 760), "uses prompts + llm")
    arrow(d, (1450, 1140), (1500, 1140), "uses prompts + llm")
    arrow(d, (1500, 480), (2170, 420), "provider=openai")
    arrow(d, (1500, 520), (2170, 770), "provider=ollama")
    arrow(d, (1980, 1130), (2170, 1230), "CRUD")
    arrow(d, (2170, 1550), (860, 760), "ingest input")
    arrow(d, (1200, 1340), (1200, 540), "pre-flight\nreadiness")

    out = OUT_DIR / "architecture-uml.png"
    img.save(out)


def processing_diagram() -> None:
    """Render and save the process/activity UML PNG."""

    img = Image.new("RGB", (3000, 1900), "#FFFFFF")
    d = ImageDraw.Draw(img)

    d.text((70, 35), "UML Processing Model (Activity + Sequence View)", font=FONT_TITLE, fill="#0B3A53")
    d.text((70, 90), "Startup gate, model readiness, ingest pipeline, chat pipeline, contradiction reasoning", font=FONT, fill="#334155")

    # Lane columns
    columns = [
        (80, 520, "User/UI", "#E0F2FE"),
        (540, 1180, "FastAPI + Services", "#F1F5F9"),
        (1200, 1780, "LLM Layer", "#ECFEFF"),
        (1800, 2920, "Data Layer", "#F8FAFC"),
    ]
    for x1, x2, label, color in columns:
        d.rounded_rectangle((x1, 150, x2, 1820), radius=18, fill=color, outline="#94A3B8", width=3)
        d.text((x1 + 12, 162), label, font=FONT_BOLD, fill="#0F172A")

    # Activity nodes
    nodes: dict[str, tuple[int, int, int, int]] = {
        "start": (90, 250, 500, 360),
        "startup_check": (560, 250, 1160, 390),
        "llm_probe": (1220, 250, 1760, 390),
        "startup_decision": (560, 430, 1160, 560),
        "fail_fast": (560, 600, 1160, 760),
        "ui_refresh": (90, 820, 500, 960),
        "llm_options": (560, 820, 1160, 980),
        "ingest_req": (90, 1030, 500, 1160),
        "validate_req": (560, 1030, 1160, 1160),
        "map_assess": (1220, 980, 1760, 1260),
        "db_write": (1820, 980, 2900, 1210),
        "ingest_resp": (560, 1220, 1160, 1340),
        "chat_req": (90, 1400, 500, 1520),
        "chat_flow": (560, 1400, 1160, 1520),
        "llm_chat": (1220, 1380, 1760, 1540),
        "reason_req": (90, 1580, 500, 1720),
        "reason_flow": (560, 1580, 1160, 1720),
        "llm_reason": (1220, 1580, 1760, 1740),
        "end": (1820, 1580, 2900, 1780),
    }

    labels = {
        "start": "App Boot",
        "startup_check": "Resolve default provider/model",
        "llm_probe": "Readiness probe\n(structured output test)",
        "startup_decision": "LLM operational?",
        "fail_fast": "Abort startup\nReturn explicit error\n+ Possible fix",
        "ui_refresh": "User clicks\nRefresh Models",
        "llm_options": "GET /api/llm-options\nAnnotate each option\noperational|error|fix",
        "ingest_req": "POST /api/ingest-case\n(case + directory + llm)",
        "validate_req": "Validate selected LLM\n(503 if unavailable)",
        "map_assess": "Workflow:\n1) map deposition\n2) assess contradictions\n3) reassess case",
        "db_write": "Persist docs and\nupdated scores in CouchDB",
        "ingest_resp": "Return ingest results",
        "chat_req": "POST /api/chat",
        "chat_flow": "Validate LLM + load context\n(target + peers)",
        "llm_chat": "Generate attorney response\n(no silent fallback)",
        "reason_req": "POST /api/reason-contradiction",
        "reason_flow": "Validate LLM + focus single\ncontradiction item",
        "llm_reason": "Generate focused reasoning\n(no silent fallback)",
        "end": "Responses rendered\nin UI",
    }

    for key, box in nodes.items():
        draw_activity_node(d, box, labels[key], fill="#FFFFFF")

    # Flow arrows
    arrow(d, (500, 305), (560, 320), "startup")
    arrow(d, (1160, 320), (1220, 320), "probe")
    arrow(d, (1490, 390), (860, 430), "result")
    arrow(d, (860, 560), (860, 600), "no")
    arrow(d, (1160, 500), (1220, 900), "yes")

    arrow(d, (500, 890), (560, 900), "refresh")
    arrow(d, (860, 980), (860, 1030), "select model")

    arrow(d, (500, 1095), (560, 1095), "ingest")
    arrow(d, (1160, 1095), (1220, 1120), "execute")
    arrow(d, (1760, 1120), (1820, 1080), "persist")
    arrow(d, (860, 1160), (860, 1220), "ok")

    arrow(d, (500, 1460), (560, 1460), "chat")
    arrow(d, (1160, 1460), (1220, 1460), "invoke")
    arrow(d, (500, 1650), (560, 1650), "reason")
    arrow(d, (1160, 1650), (1220, 1660), "invoke")

    arrow(d, (1760, 1460), (1820, 1660), "response")
    arrow(d, (1760, 1660), (1820, 1680), "response")

    out = OUT_DIR / "processing-model-uml.png"
    img.save(out)


def main() -> None:
    """Generate all UML diagrams into ``docs/uml``."""

    architecture_diagram()
    processing_diagram()
    print(f"Generated diagrams in {OUT_DIR}")


if __name__ == "__main__":
    main()
