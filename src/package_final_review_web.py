"""Prepare a bounded, read-only web copy of the final regression review.

Run from the project root with ``python src/package_final_review_web.py``.
This command does not train, commit, push, request a network resource or deploy.
It follows local HTML links, preserves project-relative paths and fingerprints
the original bytes separately from the public copies carrying historical notes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import stat
from urllib.parse import parse_qsl, unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = Path("dashboard_data/final_review_20261018")
ENTRYPOINT = "delivery/final_regression_local_20261018/index.html"
STUDY = "results/final_regression_20261018/"
FEEDBACK = "results/professor_feedback_20261018/"
PUBLICATION_DATE = "2026-10-18"
ENTRYPOINTS = (
    ENTRYPOINT,
    FEEDBACK + "informe.html",
    STUDY + "report.html",
    STUDY + "dm_original_vs_optimized_w21_2025/report.html",
    STUDY + "window_review_original_vs_optimized_2025/report.html",
    STUDY + "scenario_review_w21/report.html",
)
ADDITIONAL_INPUTS = (
    STUDY + "evaluation/macro_by_window.csv",
    STUDY + "evaluation/macro_metrics.csv",
    STUDY + "evaluation/directed_dm_all_years.csv",
    STUDY + "outer_fold_mean_sd.csv",
    STUDY + "selected_inputs.csv",
    STUDY + "review_summary.json",
    STUDY + "protected_before.json",
    STUDY + "selection_freeze.json",
    STUDY + "validation.json",
    STUDY + "final_delivery_review.json",
    STUDY + "window_review_original_vs_optimized_2025/dm_all_112.csv",
    STUDY + "window_review_original_vs_optimized_2025/window_summary.csv",
    STUDY + "window_review_original_vs_optimized_2025/window_optimizer_summary.csv",
    STUDY + "window_review_original_vs_optimized_2025/summary.json",
    STUDY + "dm_original_vs_optimized_w21_2025/dm_all_28.csv",
    STUDY + "explain_best/plots/global_bar.png",
    STUDY + "explain_best/plots/global_beeswarm.png",
    FEEDBACK + "diagnostics/intraday_development.png",
    FEEDBACK + "diagnostics/intraday_summary.csv",
    FEEDBACK + "diagnostics/metadata.json",
    FEEDBACK + "review_manifest.json",
    "dashboard_data/comparative_audit.json",
    "results/improved_classical_2023_2025/verification.json",
    "delivery/final_regression_local_20261018/pdf/presentacion_final_10min.pdf",
)
SOURCE_DOWNLOADS = frozenset((
    "src/run_final_regression.py", "src/finish_final_regression.py",
    "src/final_regression_core.py", "src/final_regression_search.py",
))
EXACT_INPUTS = SOURCE_DOWNLOADS | frozenset((
    "requirements-final-regression.txt", "dashboard_data/comparative_audit.json",
    "results/improved_classical_2023_2025/verification.json",
))
PREFIXES = (Path("results/final_regression_20261018"),
            Path("results/professor_feedback_20261018"),
            Path("delivery/final_regression_local_20261018"))
EXTENSIONS = frozenset((".html", ".csv", ".json", ".png", ".svg", ".pdf",
                        ".tex", ".ipynb", ".py", ".txt", ".md", ".css",
                        ".js", ".ico", ".woff", ".woff2"))
BINARY_EXTENSIONS = frozenset((".png", ".pdf", ".ico", ".woff", ".woff2"))
DENIED_PARTS = frozenset((
    ".git", ".github", ".aws", ".codex", ".agents", ".env", "secrets",
    "credentials", "node_modules", "__pycache__", ".npm-cache", ".venv-repro",
    "vol_venv", "venv", ".venv", "env", "cache", "caches", "models",
    "checkpoints", "tasks", "raw", "stage_logs", "_build",
))
MAX_FILE_BYTES = 24 * 1024 * 1024
MAX_TOTAL_BYTES = 160 * 1024 * 1024
MAX_FILES = 1000

HISTORICAL_REPLACEMENTS = (
    ("GitHub, Render y el modelo vigente no se actualizan.",
     "La revisión se publica el 18/10/2026; el modelo HAR-Ridge + XGBoost original continúa vigente."),
    ("Trabajo local. Modelos originales, plan primario, GitHub y Render sin cambios.",
     "Estudio local previo a la publicación de esta revisión el 18/10/2026; se conservan los modelos originales y el plan primario."),
    ("GitHub, Render, los modelos publicados y la presentación original permanecen sin cambios.",
     "Al generarse este informe local, GitHub y Render todavía no se habían actualizado. Esta copia publica la revisión el 18/10/2026 y conserva los modelos vigentes y la presentación original."),
    ("GitHub, Render y la presentación TeX permanecen sin cambios.",
     "La fuente se generó antes de publicar esta revisión el 18/10/2026; se conservan el modelo vigente y la presentación TeX original."),
    ("GitHub y Render permanecen sin cambios.",
     "Esta revisión se publica el 18/10/2026; las fuentes corresponden al estudio local previo."),
    ("GitHub y Render sin cambios.",
     "Esta revisión se publica el 18/10/2026; las fuentes corresponden al estudio local previo."),
    ("GitHub y Render no se modificaron.",
     "Los informes se generaron antes de publicar esta revisión el 18/10/2026; el modelo original sigue vigente."),
    ("Los resultados siguen siendo locales.",
     "Los resultados del estudio local se publican como revisión el 18/10/2026; el modelo original continúa vigente."),
    ("Los resultados siguen locales.",
     "Los resultados del estudio local se publican como revisión el 18/10/2026; el modelo original continúa vigente."),
    ("Publicación no autorizada",
     "Estado de la fuente al generarse: publicación no autorizada entonces; esta revisión se publica el 18/10/2026 sin reemplazar el modelo vigente"),
)

SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:[A-Z ]*PRIVATE KEY|OPENSSH PRIVATE KEY)-----"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"\bsk_live_[A-Za-z0-9]{16,}\b"),
    re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|password|client[_-]?secret)"
               r"[\"']?\s*[:=]\s*[\"']([^\"'\s]{8,})[\"']"),
    re.compile(r"https?://[^\s\"'<>/]+:[^\s\"'<>@]+@", re.I),
    re.compile(r"https://api\.render\.com/deploy/[^\s\"'<>]+[?&]key=", re.I),
)


def digest(content: bytes) -> str:
    """Return an exact-byte SHA-256 fingerprint."""
    return hashlib.sha256(content).hexdigest()


def _is_reparse(path: Path) -> bool:
    info = path.lstat()
    return path.is_symlink() or bool(getattr(info, "st_file_attributes", 0)
                                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _no_reparse(root: Path, path: Path) -> None:
    """Reject symlinks/junctions before following a source or output path."""
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        if current.exists() and _is_reparse(current):
            raise ValueError("Symlinks and junctions are not permitted in the web package")


def _source(root: Path, candidate: Path) -> tuple[Path, str]:
    lexical = candidate.absolute()
    try:
        _no_reparse(root, lexical)
    except ValueError:
        raise ValueError("A linked path escapes the permitted project tree") from None
    path = lexical.resolve()
    if not path.is_relative_to(root):
        raise ValueError("A linked path escapes the permitted project tree")
    relative = path.relative_to(root)
    name = relative.as_posix()
    folded = tuple(part.casefold() for part in relative.parts)
    if (any(part in DENIED_PARTS or part.startswith((".env.", "explain_global_failed", "explain_global_failure"))
            for part in folded)
            or any("secret" in part or "credential" in part for part in folded)):
        raise ValueError("A private, cached or training-artifact path was requested")
    if name not in EXACT_INPUTS and not any(relative.is_relative_to(prefix) for prefix in PREFIXES):
        raise ValueError("A linked path is outside the explicit publication allowlist")
    if path.suffix.lower() not in EXTENSIONS:
        raise ValueError("A linked file has an extension excluded from publication")
    if path.suffix.lower() == ".py" and name not in SOURCE_DOWNLOADS:
        raise ValueError("Only the four documented Python source downloads are permitted")
    if not path.is_file():
        raise ValueError("Missing required or linked file: " + name)
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("A linked file exceeds the bounded publication size: " + name)
    return path, name


def _check_secrets(text: str, relative: str) -> None:
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        # Never include the matching value in an exception or console output.
        raise ValueError("Sensitive content detected in publication candidate: " + relative)


class _Page(HTMLParser):
    """Collect dependency links and visible text offsets without serializing HTML."""
    def __init__(self, content: str):
        super().__init__(convert_charrefs=False)
        self.links: list[str] = []
        self.ids: set[str] = set()
        self.data_spans: list[tuple[int, str]] = []
        self.line_starts = [0]
        self.line_starts.extend(match.end() for match in re.finditer("\n", content))
        self.hidden: list[str] = []
        self.insert_at: int | None = None
        self.body_at: int | None = None
        self.has_banner = False
        self.feed(content)

    def _offset(self) -> int:
        line, column = self.getpos()
        return self.line_starts[line - 1] + column

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag in ("script", "style", "pre", "code"):
            self.hidden.append(tag)
        if tag == "base":
            raise ValueError("HTML base URLs are not permitted in the bounded web package")
        if values.get("id"):
            self.ids.add(values["id"])
        self.has_banner |= values.get("id") == "public-review-context"
        if tag == "body":
            self.body_at = self._offset() + len(self.get_starttag_text())
        if self.insert_at is None and tag in ("h1", "main", "header"):
            self.insert_at = self._offset()
        for attribute in ("href", "src", "poster"):
            if values.get(attribute):
                self.links.append(values[attribute])
        if values.get("srcset"):
            raise ValueError("Explicit srcset expansion is required before publishing this HTML")

    def handle_endtag(self, tag):
        if self.hidden and tag in self.hidden:
            index = len(self.hidden) - 1 - self.hidden[::-1].index(tag)
            del self.hidden[index:]

    def handle_data(self, data):
        if not self.hidden:
            self.data_spans.append((self._offset(), data))


def _linked_target(root: Path, owner: Path, link: str):
    """Resolve local links inside the package; inventory external URLs, never fetch."""
    url = urlsplit(link)
    scheme = url.scheme.casefold()
    if scheme == "data":
        if not re.match(r"data:image/(?:png|jpeg|gif|svg\+xml|webp);", link, re.I):
            raise ValueError("An unsupported embedded URL was encountered")
        return None, None
    if scheme or url.netloc:
        if scheme not in ("http", "https", "mailto") and not (not scheme and url.netloc):
            raise ValueError("An unsupported URL scheme was encountered")
        if url.username or url.password or any(key.casefold() in
                {"key", "token", "api_key", "secret", "password"} for key, _ in parse_qsl(url.query)):
            raise ValueError("A credential-bearing URL cannot be published")
        return None, link
    path = unquote(url.path)
    if path.startswith(("/", "\\")) or "\\" in path or "\x00" in path:
        raise ValueError("A local link escapes the publication URL prefix")
    target, _ = _source(root, owner.parent / path if path else owner)
    return (target, unquote(url.fragment)), None


def _public_html(text: str, source_hash: str):
    page = _Page(text)
    if page.has_banner:
        raise ValueError("An original report unexpectedly already contains the public-copy banner")
    edits, changes = [], []
    for offset, original_text in page.data_spans:
        changed = original_text
        for original, replacement in HISTORICAL_REPLACEMENTS:
            count = changed.count(original)
            if count:
                changed = changed.replace(original, replacement)
                changes.append({"original": original, "replacement": replacement, "count": count})
        if changed != original_text:
            edits.append((offset, len(original_text), changed))
    banner = (
        '<aside id="public-review-context" role="note" '
        'style="box-sizing:border-box;clear:both;max-width:1250px;margin:16px auto;'
        'padding:16px 18px;background:#edf4f7;border:2px solid #087f8c;'
        'color:#17324d;font:15px/1.5 system-ui,sans-serif;text-align:left" '
        f'data-source-sha256="{escape(source_hash)}">'
        '<strong>Revisión publicada · 18 de octubre de 2026.</strong> '
        'Estos informes documentan el estudio local realizado antes de esta publicación. '
        '<strong>HAR-Ridge + XGBoost original continúa como modelo vigente:</strong> '
        'publicar esta revisión no sustituye el modelo ni demuestra una mejora confirmada. '
        'Los resultados de 2025 y 2026 son retrospectivos. Las referencias históricas '
        'a «solo local» o «sin cambios en GitHub/Render» describen el estado al generar '
        'las fuentes. Los manifiestos y archivos descargables conservan ese contexto previo.'
        '</aside>'
    )
    insertion = page.body_at if page.body_at is not None else page.insert_at
    if insertion is None:
        match = re.search(r"</head\s*>", text, re.I)
        insertion = match.end() if match else 0
    edits.append((insertion, 0, banner))
    for offset, size, replacement in sorted(edits, key=lambda row: row[0], reverse=True):
        text = text[:offset] + replacement + text[offset + size:]
    return text.encode("utf-8"), [{"kind": "historical_publication_note", "source_sha256": source_hash}] + changes


def _existing_files(root: Path, directory: Path) -> set[str]:
    if not directory.exists():
        return set()
    result, queue = set(), [directory]
    while queue:
        current = queue.pop()
        for path in current.iterdir():
            _no_reparse(root, path)
            if path.is_dir():
                queue.append(path)
            elif path.is_file():
                result.add(path.relative_to(directory).as_posix())
    return result


def package_review(project_root: Path = ROOT, *, entrypoints=None, additional_inputs=None) -> dict:
    """Validate the full dependency graph, then write only the dedicated web copy.

    ``entrypoints`` and ``additional_inputs`` permit small fixture graphs in tests;
    every path still passes the same production allowlist and output safeguards.
    Sources are never rewritten. Existing unexpected public files cause an error
    instead of being deleted or silently left accessible through the web route.
    """
    root = Path(project_root).resolve(strict=True)
    output = root / DESTINATION
    _no_reparse(root, output)
    files_root = output / "files"
    seeds = tuple(ENTRYPOINTS if entrypoints is None else entrypoints)
    extras = tuple(ADDITIONAL_INPUTS if additional_inputs is None else additional_inputs)
    if not seeds or ENTRYPOINT not in seeds:
        raise ValueError("The public review entrypoint must be included in the graph")
    queue = [root / name for name in seeds + extras]
    originals, names, pages, edges, external = {}, {}, {}, [], set()
    total_source = 0
    while queue:
        path, relative = _source(root, queue.pop())
        if path in originals:
            continue
        raw = path.read_bytes()
        total_source += len(raw)
        if len(originals) >= MAX_FILES or total_source > MAX_TOTAL_BYTES:
            raise ValueError("The publication graph exceeds its bounded file/byte budget")
        originals[path], names[path] = raw, relative
        if path.suffix.lower() not in BINARY_EXTENSIONS:
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                raise ValueError("A text publication candidate is not UTF-8: " + relative) from None
            _check_secrets(text, relative)
            if path.suffix.lower() == ".html":
                page = _Page(text)
                pages[path] = page
                for link in page.links:
                    target, remote = _linked_target(root, path, link)
                    if remote:
                        external.add(remote)
                    elif target:
                        target_path, fragment = target
                        queue.append(target_path)
                        edges.append((path, target_path, fragment))
    for owner, target, fragment in edges:
        if fragment and target.suffix.lower() == ".html" and fragment not in pages[target].ids:
            raise ValueError("A local HTML fragment does not exist: " + names[owner])
    public, records = {}, {}
    for path, raw in originals.items():
        relative = names[path]
        source_hash = digest(raw)
        rendered, changes = (_public_html(raw.decode("utf-8-sig"), source_hash)
                             if path.suffix.lower() == ".html" else (raw, []))
        public[relative] = rendered
        records[relative] = {"source_sha256": source_hash, "public_sha256": digest(rendered),
                             "source_bytes": len(raw), "public_bytes": len(rendered),
                             "transformations": changes}
    unexpected = _existing_files(root, files_root) - set(public)
    if unexpected:
        raise ValueError("Unexpected stale files require review before rebuilding the public package")
    # All graph, type, content, fragment and size checks precede output mutations.
    files_root.mkdir(parents=True, exist_ok=True)
    for relative, raw in sorted(public.items()):
        target = files_root / relative
        _no_reparse(root, target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    changed_sources = [names[path] for path, raw in originals.items() if digest(path.read_bytes()) != digest(raw)]
    if changed_sources:
        raise ValueError("A source changed during packaging; no valid bundle manifest was written")
    for relative, record in records.items():
        if digest((files_root / relative).read_bytes()) != record["public_sha256"]:
            raise ValueError("A public copy does not match its fingerprint")
    source_map = {name: row["source_sha256"] for name, row in sorted(records.items())}
    public_map = {name: row["public_sha256"] for name, row in sorted(records.items())}
    manifest = {
        "format_version": 1, "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "publication_date": PUBLICATION_DATE, "entrypoint": ENTRYPOINT,
        "url_prefix": "/final-review/", "files_directory": "files",
        "publication_context": {"review_only": True, "incumbent_model": "HAR-Ridge + XGBoost original",
                                "model_replaced": False, "packager_performed_deployment": False,
                                "historical_source_metadata_preserved": True},
        "source_hash": digest(json.dumps(source_map, sort_keys=True).encode("utf-8")),
        "public_hash": digest(json.dumps(public_map, sort_keys=True).encode("utf-8")),
        "source_files_unchanged": True, "file_count": len(records),
        "total_source_bytes": sum(row["source_bytes"] for row in records.values()),
        "total_public_bytes": sum(row["public_bytes"] for row in records.values()),
        "html_pages": len(pages), "local_link_count": len(edges),
        "external_links": sorted(external), "broken_local_links": [],
        "entrypoint_reports": list(seeds), "explicit_inputs": list(extras),
        "excluded_artifact_types": ["models", "checkpoints", "raw minute data", "NPZ", "NPY",
                                    "joblib", "compressed complete forecast tables", "build caches"],
        "python_download_allowlist": sorted(SOURCE_DOWNLOADS),
        "files": dict(sorted(records.items())),
    }
    (output / "bundle_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    manifest = package_review()
    print(json.dumps({"status": manifest["status"], "files": manifest["file_count"],
                      "html_pages": manifest["html_pages"], "local_links": manifest["local_link_count"],
                      "source_mebibytes": round(manifest["total_source_bytes"] / 1024**2, 2),
                      "public_mebibytes": round(manifest["total_public_bytes"] / 1024**2, 2),
                      "sources_unchanged": manifest["source_files_unchanged"],
                      "manifest": (DESTINATION / "bundle_manifest.json").as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
