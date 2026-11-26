#!/usr/bin/env python3
"""Check that the links and DOIs in the tracked sources resolve.

A DOI is looked up in the doi.org handle service, which answers
``responseCode`` 1 for a registered DOI and 100 for an unknown one. A
publisher page is not fetched, because publishers answer scripts with 403
whether the DOI is right or not.

Any other URL fails on 404, 410 or a name that does not resolve. A 403, a
429 or a timeout is a warning, since those come from rate limits and bot
walls. Pass ``--offline`` to list the links without a request.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUFFIXES = (
    ".org", ".md", ".rs", ".h", ".hpp", ".cpp", ".py", ".pyi", ".toml",
    ".cff", ".bib", ".yml", ".el",
)
SKIP_PREFIXES = ("bench/", "benches/", "target/", "docs/source/crates/")
SKIP_FILES = {"Cargo.lock"}
SKIP_HOSTS = {"crates.io", "localhost", "127.0.0.1", "example.com"}
URL = re.compile(r"https?://[^\s<>\"')\]}`~|]+")
DOI_URL = re.compile(r"https?://(?:dx\.)?doi\.org/(10\.\d{4,9}/[^\s\[\]<>\"'`|}]+)", re.I)
DOI_ROLE = re.compile(r"\bdoi:(10\.\d{4,9}/[^\s,;\]\)]+)")
BIB_DOI = re.compile(r"^\s*doi\s*=\s*\{([^}]+)\}", re.M | re.I)
AGENT = "Mozilla/5.0 (compatible; rgmin-link-check)"


def tracked_files() -> list[Path]:
    out = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
    files = []
    for name in out.splitlines():
        if name in SKIP_FILES or name.startswith(SKIP_PREFIXES):
            continue
        if name.startswith("docs/source/") and name.endswith(".rst"):
            continue
        if name.endswith(SUFFIXES):
            files.append(ROOT / name)
    return files


def clean_doi(raw: str) -> str:
    doi = urllib.parse.unquote(raw).rstrip(".,;:*)")
    if doi.count("(") > doi.count(")"):
        doi += ")"
    return doi.lower()


def collect(files: list[Path]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    urls: dict[str, set[str]] = {}
    dois: dict[str, set[str]] = {}
    for path in files:
        rel = str(path.relative_to(ROOT))
        text = path.read_text(errors="ignore")
        for match in DOI_URL.finditer(text):
            dois.setdefault(clean_doi(match.group(1)), set()).add(rel)
        for match in DOI_ROLE.finditer(text):
            dois.setdefault(clean_doi(match.group(1)), set()).add(rel)
        for match in BIB_DOI.finditer(text):
            dois.setdefault(clean_doi(match.group(1)), set()).add(rel)
        for match in URL.finditer(text):
            url = match.group(0).rstrip(".,;:*")
            host = urllib.parse.urlparse(url).hostname or ""
            if DOI_URL.match(url) or host in SKIP_HOSTS or host.endswith("doi.org"):
                continue
            if "${" in url or "{{" in url or "@" in url.split("/", 3)[-1]:
                continue
            urls.setdefault(url, set()).add(rel)
    return urls, dois


def fetch(request: urllib.request.Request) -> tuple[int | None, str]:
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, response.read(65536).decode("utf-8", "ignore")
        except urllib.error.HTTPError as err:
            if err.code == 429 and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            return err.code, ""
        except (urllib.error.URLError, TimeoutError, OSError) as err:
            reason = getattr(err, "reason", err)
            if "Name or service not known" in str(reason) or "nodename" in str(reason):
                return -1, str(reason)
            if attempt == 2:
                return None, str(reason)
            time.sleep(1 + attempt)
    return None, "unreachable"


def check_doi(doi: str) -> tuple[str, str]:
    quoted = urllib.parse.quote(doi, safe="/")
    request = urllib.request.Request(
        f"https://doi.org/api/handles/{quoted}", headers={"User-Agent": AGENT}
    )
    status, body = fetch(request)
    if status is None:
        return "warn", f"no answer from doi.org: {body}"
    if status == 404:
        return "fail", "doi.org does not know this DOI"
    if status != 200:
        return "warn", f"doi.org answered HTTP {status}"
    try:
        code = json.loads(body).get("responseCode")
    except json.JSONDecodeError:
        return "warn", "doi.org answered with something that is not JSON"
    if code == 1:
        return "ok", ""
    return "fail", f"doi.org responseCode {code}"


def check_url(url: str) -> tuple[str, str]:
    last: tuple[int | None, str] = (None, "")
    for method in ("HEAD", "GET"):
        request = urllib.request.Request(
            url, method=method, headers={"User-Agent": AGENT}
        )
        status, body = fetch(request)
        last = (status, body)
        if status is not None and 200 <= status < 400:
            return "ok", ""
        if status in (404, 410) and method == "GET":
            return "fail", f"HTTP {status}"
        if status == -1:
            return "fail", f"name does not resolve: {body}"
    status, body = last
    if status is None:
        return "warn", f"no answer: {body}"
    return "warn", f"HTTP {status}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--offline", action="store_true", help="list, do not fetch")
    args = parser.parse_args()
    urls, dois = collect(tracked_files())
    print(f"{len(dois)} DOIs and {len(urls)} other URLs")
    if args.offline:
        return 0

    failed = 0
    jobs: list[tuple[str, str, set[str]]] = [("doi", d, f) for d, f in sorted(dois.items())]
    jobs += [("url", u, f) for u, f in sorted(urls.items())]

    def run(job: tuple[str, str, set[str]]) -> tuple[str, str, str, str, set[str]]:
        kind, target, where = job
        verdict, note = check_doi(target) if kind == "doi" else check_url(target)
        return verdict, note, kind, target, where

    with futures.ThreadPoolExecutor(max_workers=8) as pool:
        for verdict, note, kind, target, where in pool.map(run, jobs):
            if verdict == "ok":
                continue
            label = "ERROR" if verdict == "fail" else "warning"
            name = f"doi:{target}" if kind == "doi" else target
            print(f"{label}: {name}: {note} ({', '.join(sorted(where)[:3])})")
            failed += verdict == "fail"
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
