"""LLM-driven yearly memory book curator and chapter/group builder."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import json
import logging
import re
from typing import Any

from vision_memory_book.ai.base import AIProvider
from vision_memory_book.db.repository import Repository

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)


def _strip_json_fences(raw_text: str) -> str:
    stripped = raw_text.strip()
    match = _FENCE_RE.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped


def _month_label(month_key: str) -> str:
    if month_key == "undated":
        return "Undated Memories"
    try:
        dt = datetime.strptime(month_key, "%Y-%m")
        return dt.strftime("%B %Y")
    except ValueError:
        return month_key


def _fallback_caption(photo: dict[str, Any]) -> str:
    desc = str(photo.get("description") or "").strip()
    if desc:
        return desc
    filename = str(photo.get("filename") or "").strip()
    if filename:
        return filename
    return f"Photo #{photo['id']}"


class MemoryBookGenerator:
    """Organizes a year's photos into thematic/chronological chapters via a local LLM."""

    def __init__(self, repo: Repository, text_provider: AIProvider) -> None:
        self.repo = repo
        self.text_provider = text_provider

    async def generate_year_book(
        self,
        year: int,
        title: str | None = None,
        max_chapters: int = 6,
        max_photos_per_chapter: int = 12,
    ) -> dict[str, Any]:
        """Curate a yearly memory book, persist chapter groups, and save the book record."""
        photos = self.repo.search_photos(year=year, limit=500)
        if not photos:
            raise ValueError(f"No photos found for year {year}")

        photo_by_id: dict[int, dict[str, Any]] = {int(p["id"]): p for p in photos}
        outline = await self._curate_outline(
            year=year,
            photos=photos,
            photo_by_id=photo_by_id,
            title=title,
            max_chapters=max_chapters,
            max_photos_per_chapter=max_photos_per_chapter,
        )

        chapters_with_groups: list[dict[str, Any]] = []
        for chapter in outline["chapters"]:
            chapter_title = str(chapter["title"]).strip()
            chapter_intro = str(chapter.get("intro") or "").strip()
            photo_ids: list[int] = [int(pid) for pid in chapter["photo_ids"]]
            raw_captions: dict[str, str] = chapter.get("captions") or {}
            captions_by_pid: dict[int, str] = {}
            for pid in photo_ids:
                cap = raw_captions.get(str(pid)) or raw_captions.get(pid)  # type: ignore[arg-type]
                if cap and str(cap).strip():
                    captions_by_pid[pid] = str(cap).strip()
                else:
                    captions_by_pid[pid] = _fallback_caption(photo_by_id[pid])

            group_id = self.repo.create_group(
                name=chapter_title,
                year=year,
                description=chapter_intro,
                kind="book_chapter",
            )
            self.repo.add_photos_to_group(
                group_id=group_id,
                photo_ids=photo_ids,
                captions=captions_by_pid,
            )

            chapters_with_groups.append(
                {
                    "title": chapter_title,
                    "intro": chapter_intro,
                    "group_id": group_id,
                    "photo_ids": photo_ids,
                    "captions": {str(pid): captions_by_pid[pid] for pid in photo_ids},
                }
            )

        final_title = (title or "").strip() or str(outline.get("title") or "").strip() or f"{year} Memory Book"
        final_outline: dict[str, Any] = {
            "year": year,
            "title": final_title,
            "subtitle": str(outline.get("subtitle") or f"Highlights from {year}").strip(),
            "intro": str(outline.get("intro") or f"A curated collection of memories from {year}.").strip(),
            "chapters": chapters_with_groups,
        }

        book_id = self.repo.save_book(
            year=year,
            title=final_title,
            outline_json=final_outline,
        )
        saved = self.repo.get_book(book_id)
        if saved is None:
            raise RuntimeError(f"Failed to retrieve saved book {book_id}")
        return saved

    async def _curate_outline(
        self,
        year: int,
        photos: list[dict[str, Any]],
        photo_by_id: dict[int, dict[str, Any]],
        title: str | None,
        max_chapters: int,
        max_photos_per_chapter: int,
    ) -> dict[str, Any]:
        catalog = [
            {
                "id": int(p["id"]),
                "create_time": p.get("create_time"),
                "filename": p.get("filename"),
                "description": p.get("description") or "",
                "tags": p.get("tags") or [],
            }
            for p in photos
        ]

        prompt = (
            f"You are curating a yearly photo memory book for {year}.\n"
            f"Organize the provided photos into at most {max_chapters} thematic or chronological chapters, "
            f"with at most {max_photos_per_chapter} photos per chapter.\n"
            "Use ONLY photo IDs that exist in the catalog below.\n"
            "Respond with a single valid JSON object (no markdown commentary) with this schema:\n"
            "{\n"
            '  "title": "Book title",\n'
            '  "subtitle": "Short subtitle",\n'
            '  "intro": "Warm 1-2 sentence book introduction",\n'
            '  "chapters": [\n'
            "    {\n"
            '      "title": "Chapter title",\n'
            '      "intro": "Short 1-2 sentence chapter introduction",\n'
            '      "photo_ids": [1, 2],\n'
            '      "captions": {"1": "Caption for photo 1", "2": "Caption for photo 2"}\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            f"Preferred title: {title or f'{year} Memory Book'}\n"
            f"Photo catalog ({len(catalog)} photos):\n"
            f"{json.dumps(catalog, ensure_ascii=False)}"
        )

        messages = [
            {
                "role": "system",
                "content": "You are a thoughtful photo book editor. Return strictly valid JSON.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            raw_response = await self.text_provider.chat(messages)
            parsed = self._parse_and_validate_outline(
                raw_response=raw_response,
                photo_by_id=photo_by_id,
                max_chapters=max_chapters,
                max_photos_per_chapter=max_photos_per_chapter,
            )
            if parsed is not None:
                return parsed
        except Exception as exc:
            logger.warning("LLM book curation failed (%s); using chronological fallback", exc)

        return self._deterministic_chronological_fallback(
            year=year,
            photos=photos,
            title=title,
            max_chapters=max_chapters,
            max_photos_per_chapter=max_photos_per_chapter,
        )

    def _parse_and_validate_outline(
        self,
        raw_response: str,
        photo_by_id: dict[int, dict[str, Any]],
        max_chapters: int,
        max_photos_per_chapter: int,
    ) -> dict[str, Any] | None:
        cleaned = _strip_json_fences(raw_response)
        if not cleaned:
            return None

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        raw_chapters = data.get("chapters")
        if not isinstance(raw_chapters, list) or not raw_chapters:
            return None

        valid_chapters: list[dict[str, Any]] = []
        for idx, ch in enumerate(raw_chapters):
            if len(valid_chapters) >= max_chapters:
                break
            if not isinstance(ch, dict):
                continue

            raw_pids = ch.get("photo_ids")
            if not isinstance(raw_pids, list):
                continue

            seen_pids: set[int] = set()
            clean_pids: list[int] = []
            for raw_pid in raw_pids:
                try:
                    pid = int(raw_pid)
                except (TypeError, ValueError):
                    continue
                if pid in photo_by_id and pid not in seen_pids:
                    seen_pids.add(pid)
                    clean_pids.append(pid)
                    if len(clean_pids) >= max_photos_per_chapter:
                        break

            if not clean_pids:
                continue

            ch_title = str(ch.get("title") or f"Chapter {idx + 1}").strip()
            ch_intro = str(ch.get("intro") or "").strip()
            raw_captions = ch.get("captions") if isinstance(ch.get("captions"), dict) else {}
            clean_captions: dict[str, str] = {}
            for pid in clean_pids:
                cap = raw_captions.get(str(pid)) or raw_captions.get(pid)
                if cap and str(cap).strip():
                    clean_captions[str(pid)] = str(cap).strip()
                else:
                    clean_captions[str(pid)] = _fallback_caption(photo_by_id[pid])

            valid_chapters.append(
                {
                    "title": ch_title,
                    "intro": ch_intro,
                    "photo_ids": clean_pids,
                    "captions": clean_captions,
                }
            )

        if not valid_chapters:
            return None

        return {
            "title": str(data.get("title") or "").strip(),
            "subtitle": str(data.get("subtitle") or "").strip(),
            "intro": str(data.get("intro") or "").strip(),
            "chapters": valid_chapters,
        }

    def _deterministic_chronological_fallback(
        self,
        year: int,
        photos: list[dict[str, Any]],
        title: str | None,
        max_chapters: int,
        max_photos_per_chapter: int,
    ) -> dict[str, Any]:
        sorted_photos = sorted(
            photos,
            key=lambda p: (str(p.get("create_time") or "9999-99-99"), int(p["id"])),
        )

        by_month: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for photo in sorted_photos:
            ctime = str(photo.get("create_time") or "")
            month_key = ctime[:7] if len(ctime) >= 7 and ctime[:4].isdigit() else "undated"
            by_month[month_key].append(photo)

        month_keys = sorted(by_month.keys())
        # If more months than max_chapters, chunk months evenly into <= max_chapters buckets
        buckets: list[tuple[str, list[dict[str, Any]]]] = []
        if len(month_keys) <= max_chapters:
            for mk in month_keys:
                buckets.append((_month_label(mk), by_month[mk]))
        else:
            chunk_size = (len(month_keys) + max_chapters - 1) // max_chapters
            for i in range(0, len(month_keys), chunk_size):
                group_keys = month_keys[i : i + chunk_size]
                combined: list[dict[str, Any]] = []
                for mk in group_keys:
                    combined.extend(by_month[mk])
                if len(group_keys) == 1:
                    label = _month_label(group_keys[0])
                else:
                    label = f"{_month_label(group_keys[0])} - {_month_label(group_keys[-1])}"
                buckets.append((label, combined))

        chapters: list[dict[str, Any]] = []
        for ch_title, ch_photos in buckets[:max_chapters]:
            selected = ch_photos[:max_photos_per_chapter]
            pids = [int(p["id"]) for p in selected]
            captions = {str(p["id"]): _fallback_caption(p) for p in selected}
            chapters.append(
                {
                    "title": ch_title,
                    "intro": f"Memories captured during {ch_title}.",
                    "photo_ids": pids,
                    "captions": captions,
                }
            )

        return {
            "title": (title or "").strip() or f"{year} Memory Book",
            "subtitle": f"A chronological look back at {year}",
            "intro": f"Highlights and moments from {year}.",
            "chapters": chapters,
        }
