from __future__ import annotations

"""Generate UML sequence diagrams as PNG files."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "uml"
OUT_DIR.mkdir(parents=True, exist_ok=True)


try:
    FONT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
    FONT_BOLD = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 24)
    FONT_TITLE = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 36)
except OSError:
    FONT = ImageFont.load_default()
    FONT_BOLD = ImageFont.load_default()
    FONT_TITLE = ImageFont.load_default()


def draw_title(draw: ImageDraw.ImageDraw, title: str, subtitle: str = "") -> None:
    draw.text((56, 28), title, font=FONT_TITLE, fill="#0B3A53")
    if subtitle:
        draw.text((56, 76), subtitle, font=FONT, fill="#334155")


def participant_centers(width: int, participants: list[str], margin: int = 90) -> list[int]:
    if len(participants) == 1:
        return [width // 2]
    span = width - (margin * 2)
    step = span / (len(participants) - 1)
    return [int(margin + i * step) for i in range(len(participants))]


def draw_participants(
    draw: ImageDraw.ImageDraw,
    participants: list[str],
    xs: list[int],
    head_y: int,
    lane_top: int,
    lane_bottom: int,
) -> None:
    for name, x in zip(participants, xs):
        head = (x - 120, head_y, x + 120, head_y + 60)
        draw.rounded_rectangle(head, radius=12, fill="#E2E8F0", outline="#334155", width=2)
        bbox = draw.textbbox((0, 0), name, font=FONT_BOLD)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((x - tw / 2, head_y + (60 - th) / 2), name, font=FONT_BOLD, fill="#0F172A")
        for y in range(lane_top, lane_bottom, 18):
            draw.line([(x, y), (x, min(y + 10, lane_bottom))], fill="#64748B", width=2)


def arrow_head(
    draw: ImageDraw.ImageDraw,
    end: tuple[int, int],
    direction: int,
    color: str,
) -> None:
    ex, ey = end
    if direction > 0:
        points = [(ex, ey), (ex - 14, ey - 7), (ex - 14, ey + 7)]
    else:
        points = [(ex, ey), (ex + 14, ey - 7), (ex + 14, ey + 7)]
    draw.polygon(points, fill=color)


def draw_message(
    draw: ImageDraw.ImageDraw,
    x1: int,
    x2: int,
    y: int,
    text: str,
    dashed: bool = False,
    color: str = "#0F172A",
) -> None:
    if dashed:
        start = min(x1, x2)
        end = max(x1, x2)
        for x in range(start, end, 18):
            seg_end = min(x + 10, end)
            draw.line([(x, y), (seg_end, y)], fill=color, width=3)
    else:
        draw.line([(x1, y), (x2, y)], fill=color, width=3)

    direction = 1 if x2 >= x1 else -1
    arrow_head(draw, (x2, y), direction, color)

    tx = (x1 + x2) / 2
    bbox = draw.textbbox((0, 0), text, font=FONT)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    rect = (tx - tw / 2 - 8, y - th - 16, tx + tw / 2 + 8, y - 8)
    draw.rounded_rectangle(rect, radius=8, fill="#FEF9C3", outline="#CA8A04", width=2)
    draw.text((tx - tw / 2, y - th - 12), text, font=FONT, fill="#713F12")


def sequence_diagram(
    filename: str,
    title: str,
    subtitle: str,
    participants: list[str],
    messages: list[tuple[str, str, str, bool]],
    width: int = 2200,
    height: int = 1400,
) -> None:
    img = Image.new("RGB", (width, height), "#F8FAFC")
    d = ImageDraw.Draw(img)

    draw_title(d, title, subtitle)

    head_y = 140
    lane_top = 220
    lane_bottom = height - 90
    xs = participant_centers(width, participants)
    index = {name: i for i, name in enumerate(participants)}

    draw_participants(d, participants, xs, head_y, lane_top, lane_bottom)

    y = 280
    step = 96
    for sender, receiver, text, dashed in messages:
        x1 = xs[index[sender]]
        x2 = xs[index[receiver]]
        draw_message(d, x1, x2, y, text, dashed=dashed)
        y += step

    img.save(OUT_DIR / filename)


def startup_sequence() -> None:
    sequence_diagram(
        filename="sequence-startup-readiness-uml.png",
        title="UML Sequence - Startup Readiness",
        subtitle="Boot-time default model validation with explicit fail-fast path",
        participants=["App Boot", "FastAPI", "LLM Gateway", "Provider"],
        messages=[
            ("App Boot", "FastAPI", "start service", False),
            ("FastAPI", "LLM Gateway", "resolve default provider/model", False),
            ("LLM Gateway", "Provider", "readiness probe", False),
            ("Provider", "LLM Gateway", "probe result", True),
            ("LLM Gateway", "FastAPI", "operational OR error+fix", True),
            ("FastAPI", "App Boot", "serve API OR abort startup", True),
        ],
        width=1800,
        height=980,
    )


def ingest_sequence() -> None:
    sequence_diagram(
        filename="sequence-ingest-case-uml.png",
        title="UML Sequence - Ingest Case",
        subtitle="Folder selection, LLM validation, mapping, contradiction assessment, persistence",
        participants=["UI", "FastAPI", "LLM Gateway", "Workflow", "Provider", "CouchDB"],
        messages=[
            ("UI", "FastAPI", "POST /api/ingest-case", False),
            ("FastAPI", "LLM Gateway", "validate selected model", False),
            ("LLM Gateway", "Provider", "operational probe", False),
            ("Provider", "LLM Gateway", "ready/unavailable", True),
            ("FastAPI", "Workflow", "run map+assess", False),
            ("Workflow", "Provider", "map deposition + assess contradictions", False),
            ("Provider", "Workflow", "structured outputs", True),
            ("Workflow", "CouchDB", "upsert docs + scores", False),
            ("CouchDB", "FastAPI", "persisted", True),
            ("FastAPI", "UI", "ingest results", True),
        ],
        width=2400,
        height=1400,
    )


def chat_sequence() -> None:
    sequence_diagram(
        filename="sequence-chat-uml.png",
        title="UML Sequence - Attorney Chat",
        subtitle="Validated chat request with context loading and LLM response",
        participants=["UI", "FastAPI", "LLM Gateway", "CouchDB", "Chat Service", "Provider"],
        messages=[
            ("UI", "FastAPI", "POST /api/chat", False),
            ("FastAPI", "LLM Gateway", "validate selected model", False),
            ("FastAPI", "CouchDB", "load target deposition + peers", False),
            ("CouchDB", "FastAPI", "deposition context", True),
            ("FastAPI", "Chat Service", "build prompts + context", False),
            ("Chat Service", "Provider", "chat completion", False),
            ("Provider", "Chat Service", "detailed text response", True),
            ("Chat Service", "FastAPI", "response payload", True),
            ("FastAPI", "UI", "render response", True),
        ],
        width=2400,
        height=1320,
    )


def reason_sequence() -> None:
    sequence_diagram(
        filename="sequence-reason-contradiction-uml.png",
        title="UML Sequence - Focused Contradiction Re-Analysis",
        subtitle="Single-contradiction deep reasoning from UI click to rendered output",
        participants=["UI", "FastAPI", "LLM Gateway", "CouchDB", "Chat Service", "Provider"],
        messages=[
            ("UI", "FastAPI", "POST /api/reason-contradiction", False),
            ("FastAPI", "LLM Gateway", "validate selected model", False),
            ("FastAPI", "CouchDB", "load deposition + peers", False),
            ("CouchDB", "FastAPI", "context docs", True),
            ("FastAPI", "Chat Service", "reason_about_contradiction", False),
            ("Chat Service", "Provider", "focused reasoning completion", False),
            ("Provider", "Chat Service", "detailed analysis", True),
            ("Chat Service", "FastAPI", "reason response", True),
            ("FastAPI", "UI", "render focused analysis", True),
        ],
        width=2400,
        height=1320,
    )


def main() -> None:
    for png in OUT_DIR.glob("*.png"):
        png.unlink(missing_ok=True)

    startup_sequence()
    ingest_sequence()
    chat_sequence()
    reason_sequence()
    print(f"Generated sequence diagrams in {OUT_DIR}")


if __name__ == "__main__":
    main()
