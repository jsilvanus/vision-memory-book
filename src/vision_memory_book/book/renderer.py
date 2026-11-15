"""HTML and PDF memory book renderer preferring Final-Touch (=d) originals."""

from __future__ import annotations

from html import escape
import io
import json
from pathlib import Path
from typing import Any

from fpdf import FPDF
from PIL import Image, ImageOps

from vision_memory_book.db.repository import Repository


def _resolve_photo_path(photo: dict[str, Any]) -> tuple[Path | None, str]:
    """Resolve best available image path, preferring original_path over cache_path."""
    original_raw = photo.get("original_path")
    if original_raw:
        orig_path = Path(str(original_raw))
        if orig_path.is_file():
            return orig_path, "original"

    cache_raw = photo.get("cache_path")
    if cache_raw:
        cache_path = Path(str(cache_raw))
        if cache_path.is_file():
            return cache_path, "cache"

    return None, "missing"


def _pdf_safe_text(text: str) -> str:
    """Normalize text for fpdf2 core fonts (latin-1) without raising encoding errors."""
    replacements = {
        "\u2014": "-",
        "\u2013": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
    }
    normalized = text
    for src, dst in replacements.items():
        normalized = normalized.replace(src, dst)
    return normalized.encode("latin-1", "replace").decode("latin-1")


def _image_to_jpeg_stream(image_path: Path) -> tuple[io.BytesIO, int, int]:
    """Load an image (WebP, JPEG, PNG, etc.), auto-orient, and return a JPEG BytesIO + dimensions."""
    with Image.open(image_path) as img:
        oriented = ImageOps.exif_transpose(img)
        if oriented.mode != "RGB":
            oriented = oriented.convert("RGB")
        width, height = oriented.size
        buf = io.BytesIO()
        oriented.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        return buf, width, height


class BookRenderer:
    """Renders a persisted Memory Book into print-styled HTML and PDF."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)

    def render(self, book: dict[str, Any], repo: Repository) -> dict[str, Any]:
        """Render HTML and PDF files for `book`, update DB paths, and return updated book."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        book_id = int(book["id"])
        year = int(book["year"])
        title = str(book.get("title") or f"{year} Memory Book")

        raw_outline = book.get("outline_json") or "{}"
        outline: dict[str, Any] = (
            json.loads(raw_outline) if isinstance(raw_outline, str) else dict(raw_outline)
        )

        resolved_chapters = self._resolve_chapters(outline, repo)

        html_path = self.output_dir / f"{year}_book_{book_id}.html"
        pdf_path = self.output_dir / f"{year}_book_{book_id}.pdf"

        html_content = self._render_html(
            year=year,
            title=title,
            subtitle=str(outline.get("subtitle") or f"Highlights from {year}"),
            intro=str(outline.get("intro") or ""),
            chapters=resolved_chapters,
        )
        html_path.write_text(html_content, encoding="utf-8")

        self._render_pdf(
            pdf_path=pdf_path,
            year=year,
            title=title,
            subtitle=str(outline.get("subtitle") or f"Highlights from {year}"),
            intro=str(outline.get("intro") or ""),
            chapters=resolved_chapters,
        )

        repo.update_book_paths(
            book_id=book_id,
            html_path=str(html_path),
            pdf_path=str(pdf_path),
        )
        updated = repo.get_book(book_id)
        if updated is not None:
            return updated
        result = dict(book)
        result["html_path"] = str(html_path)
        result["pdf_path"] = str(pdf_path)
        return result

    def _resolve_chapters(
        self,
        outline: dict[str, Any],
        repo: Repository,
    ) -> list[dict[str, Any]]:
        resolved: list[dict[str, Any]] = []
        for idx, ch in enumerate(outline.get("chapters") or [], start=1):
            ch_title = str(ch.get("title") or f"Chapter {idx}")
            ch_intro = str(ch.get("intro") or "")
            raw_captions = ch.get("captions") or {}
            photo_ids = [int(pid) for pid in (ch.get("photo_ids") or [])]

            photos_out: list[dict[str, Any]] = []
            for pid in photo_ids:
                photo = repo.get_photo(pid)
                if photo is None:
                    continue
                img_path, source_kind = _resolve_photo_path(photo)
                caption = (
                    raw_captions.get(str(pid))
                    or raw_captions.get(pid)
                    or photo.get("description")
                    or photo.get("filename")
                    or f"Photo #{pid}"
                )
                photos_out.append(
                    {
                        "id": pid,
                        "filename": photo.get("filename") or f"photo_{pid}",
                        "create_time": photo.get("create_time") or "",
                        "caption": str(caption),
                        "image_path": img_path,
                        "source_kind": source_kind,
                    }
                )

            resolved.append(
                {
                    "index": idx,
                    "title": ch_title,
                    "intro": ch_intro,
                    "photos": photos_out,
                }
            )
        return resolved

    def _render_html(
        self,
        year: int,
        title: str,
        subtitle: str,
        intro: str,
        chapters: list[dict[str, Any]],
    ) -> str:
        toc_items = "\n".join(
            f'<li><a href="#chapter-{ch["index"]}">{escape(ch["title"])}</a> '
            f'<span class="toc-count">({len(ch["photos"])} photos)</span></li>'
            for ch in chapters
        )

        chapter_sections: list[str] = []
        for ch in chapters:
            figures: list[str] = []
            for item in ch["photos"]:
                img_path: Path | None = item["image_path"]
                if img_path is not None:
                    img_html = (
                        f'<img src="{escape(img_path.resolve().as_uri())}" '
                        f'alt="{escape(item["caption"])}" loading="lazy" />'
                    )
                else:
                    img_html = '<div class="missing-photo">Image unavailable on disk</div>'

                meta_parts = [p for p in (item["create_time"][:10], item["filename"]) if p]
                meta_str = " &middot; ".join(escape(m) for m in meta_parts)
                figures.append(
                    f"""
                    <figure class="photo-card" data-source="{escape(item['source_kind'])}">
                        {img_html}
                        <figcaption>
                            <p class="caption">{escape(item["caption"])}</p>
                            <p class="meta">{meta_str}</p>
                        </figcaption>
                    </figure>
                    """
                )

            chapter_sections.append(
                f"""
                <section class="chapter" id="chapter-{ch['index']}">
                    <header class="chapter-header">
                        <span class="chapter-num">Chapter {ch['index']}</span>
                        <h2>{escape(ch['title'])}</h2>
                        {f'<p class="chapter-intro">{escape(ch["intro"])}</p>' if ch['intro'] else ''}
                    </header>
                    <div class="photo-grid">
                        {''.join(figures)}
                    </div>
                </section>
                """
            )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>{escape(title)}</title>
<style>
  @page {{
    size: A4;
    margin: 18mm 16mm;
  }}
  :root {{
    --ink: #1f2937;
    --muted: #6b7280;
    --paper: #faf8f5;
    --card: #ffffff;
    --accent: #8c5a3c;
    --border: #e5e0d8;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: Georgia, "Times New Roman", serif;
    color: var(--ink);
    background: var(--paper);
    margin: 0;
    padding: 0;
    line-height: 1.55;
  }}
  .cover {{
    max-width: 900px;
    margin: 48px auto 32px;
    padding: 64px 48px;
    background: var(--card);
    border: 1px solid var(--border);
    text-align: center;
    page-break-after: always;
  }}
  .cover .year-badge {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    letter-spacing: 0.2em;
    font-size: 0.85rem;
    text-transform: uppercase;
    color: var(--accent);
  }}
  .cover h1 {{
    font-size: 2.6rem;
    margin: 16px 0 8px;
    font-weight: 600;
  }}
  .cover .subtitle {{
    font-size: 1.15rem;
    color: var(--muted);
    margin: 0 0 24px;
    font-style: italic;
  }}
  .cover .intro {{
    max-width: 620px;
    margin: 0 auto;
    font-size: 1.05rem;
  }}
  .toc {{
    max-width: 900px;
    margin: 0 auto 40px;
    padding: 32px 40px;
    background: var(--card);
    border: 1px solid var(--border);
    page-break-after: always;
  }}
  .toc h2 {{
    margin-top: 0;
    color: var(--accent);
  }}
  .toc ol {{
    padding-left: 20px;
  }}
  .toc li {{
    margin: 10px 0;
  }}
  .toc a {{
    color: var(--ink);
    text-decoration: none;
    font-weight: 600;
  }}
  .toc-count {{
    color: var(--muted);
    font-size: 0.9rem;
  }}
  .chapter {{
    max-width: 900px;
    margin: 0 auto 48px;
    padding: 32px 40px;
    background: var(--card);
    border: 1px solid var(--border);
    page-break-before: always;
  }}
  .chapter-num {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 0.8rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent);
  }}
  .chapter-header h2 {{
    margin: 6px 0 10px;
    font-size: 1.8rem;
  }}
  .chapter-intro {{
    color: var(--muted);
    margin: 0 0 24px;
    font-style: italic;
  }}
  .photo-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
    gap: 24px;
  }}
  .photo-card {{
    margin: 0;
    border: 1px solid var(--border);
    background: #fff;
    padding: 12px;
    break-inside: avoid;
  }}
  .photo-card img {{
    width: 100%;
    height: auto;
    max-height: 360px;
    object-fit: cover;
    display: block;
  }}
  .missing-photo {{
    height: 180px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #f3f4f6;
    color: var(--muted);
    font-size: 0.9rem;
  }}
  figcaption {{
    margin-top: 10px;
  }}
  .caption {{
    margin: 0 0 4px;
    font-size: 0.96rem;
  }}
  .meta {{
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 0.78rem;
    color: var(--muted);
  }}
  @media print {{
    body {{ background: #fff; }}
    .cover, .toc, .chapter {{ border: none; margin: 0; padding: 0; }}
  }}
</style>
</head>
<body>
  <section class="cover">
    <div class="year-badge">{year} Memory Book</div>
    <h1>{escape(title)}</h1>
    <p class="subtitle">{escape(subtitle)}</p>
    {f'<p class="intro">{escape(intro)}</p>' if intro else ''}
  </section>
  <section class="toc">
    <h2>Contents</h2>
    <ol>
      {toc_items}
    </ol>
  </section>
  {''.join(chapter_sections)}
</body>
</html>
"""

    def _render_pdf(
        self,
        pdf_path: Path,
        year: int,
        title: str,
        subtitle: str,
        intro: str,
        chapters: list[dict[str, Any]],
    ) -> None:
        pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.set_auto_page_break(auto=True, margin=18)

        # Cover Page
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)
        pdf.set_text_color(140, 90, 60)
        pdf.ln(55)
        pdf.cell(0, 8, _pdf_safe_text(f"{year} MEMORY BOOK"), new_x="LMARGIN", new_y="NEXT", align="C")

        pdf.set_font("Helvetica", style="B", size=24)
        pdf.set_text_color(31, 41, 55)
        pdf.ln(4)
        pdf.multi_cell(0, 12, _pdf_safe_text(title), align="C")

        if subtitle:
            pdf.set_font("Helvetica", style="I", size=13)
            pdf.set_text_color(107, 114, 128)
            pdf.ln(3)
            pdf.multi_cell(0, 7, _pdf_safe_text(subtitle), align="C")

        if intro:
            pdf.set_font("Helvetica", size=11)
            pdf.set_text_color(55, 65, 81)
            pdf.ln(10)
            pdf.multi_cell(0, 6, _pdf_safe_text(intro), align="C")

        usable_width = pdf.w - pdf.l_margin - pdf.r_margin

        for ch in chapters:
            pdf.add_page()
            pdf.set_font("Helvetica", style="B", size=10)
            pdf.set_text_color(140, 90, 60)
            pdf.cell(
                0,
                6,
                _pdf_safe_text(f"CHAPTER {ch['index']}"),
                new_x="LMARGIN",
                new_y="NEXT",
            )

            pdf.set_font("Helvetica", style="B", size=18)
            pdf.set_text_color(31, 41, 55)
            pdf.multi_cell(0, 9, _pdf_safe_text(ch["title"]))

            if ch["intro"]:
                pdf.set_font("Helvetica", style="I", size=11)
                pdf.set_text_color(107, 114, 128)
                pdf.ln(2)
                pdf.multi_cell(0, 6, _pdf_safe_text(ch["intro"]))

            pdf.ln(6)

            for item in ch["photos"]:
                img_path: Path | None = item["image_path"]
                if img_path is not None:
                    try:
                        stream, img_w, img_h = _image_to_jpeg_stream(img_path)
                        max_img_w = usable_width
                        max_img_h = 115.0
                        aspect = img_h / max(img_w, 1)
                        render_w = max_img_w
                        render_h = render_w * aspect
                        if render_h > max_img_h:
                            render_h = max_img_h
                            render_w = render_h / max(aspect, 1e-6)

                        if pdf.get_y() + render_h + 22 > pdf.h - pdf.b_margin:
                            pdf.add_page()

                        x_pos = pdf.l_margin + (usable_width - render_w) / 2.0
                        pdf.image(stream, x=x_pos, w=render_w, h=render_h)
                        pdf.ln(3)
                    except Exception:
                        pdf.set_font("Helvetica", style="I", size=9)
                        pdf.set_text_color(156, 163, 175)
                        pdf.cell(
                            0,
                            6,
                            _pdf_safe_text(f"[Could not render image: {item['filename']}]"),
                            new_x="LMARGIN",
                            new_y="NEXT",
                        )

                pdf.set_font("Helvetica", size=10)
                pdf.set_text_color(31, 41, 55)
                pdf.multi_cell(0, 5.5, _pdf_safe_text(item["caption"]))

                meta_parts = [p for p in (item["create_time"][:10], item["filename"]) if p]
                if meta_parts:
                    pdf.set_font("Helvetica", size=8)
                    pdf.set_text_color(107, 114, 128)
                    pdf.cell(
                        0,
                        4.5,
                        _pdf_safe_text(" | ".join(meta_parts)),
                        new_x="LMARGIN",
                        new_y="NEXT",
                    )

                pdf.ln(6)

        pdf.output(str(pdf_path))
