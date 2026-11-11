#!/usr/bin/env bash
# Download all open-access arXiv PDFs cited by the manuscript into references/pdfs/.
# Respect arXiv's terms: this is for personal research; do not redistribute the PDFs.
set -u
cd "$(dirname "$0")"
mkdir -p pdfs

PY=python
if ! command -v "$PY" >/dev/null 2>&1; then PY=python3; fi

"$PY" - <<'PYCEPT'
import json, os, subprocess, sys, time
refs = json.load(open("_refs_parsed.json", encoding="utf-8"))
ids = [r["arxiv"] for r in refs if r.get("arxiv")]
print(f"{len(ids)} arXiv items to fetch")
os.makedirs("pdfs", exist_ok=True)
ok = 0
for i, aid in enumerate(ids, 1):
    dst = f"pdfs/arxiv_{aid}.pdf"
    if os.path.exists(dst) and os.path.getsize(dst) > 20000:
        print(f"[{i}/{len(ids)}] {aid} exists, skip"); continue
    url = f"https://arxiv.org/pdf/{aid}"
    rc = subprocess.run(["curl", "-sSL", "--retry", "3", "--retry-delay", "2",
                         "-m", "600", "-o", dst, url]).returncode
    sz = os.path.getsize(dst) if os.path.exists(dst) else 0
    # verify it's a PDF
    head = b""
    if sz:
        head = open(dst, "rb").read(4)
    if rc == 0 and head == b"%PDF":
        print(f"[{i}/{len(ids)}] {aid} OK {sz//1024}KB"); ok += 1
    else:
        print(f"[{i}/{len(ids)}] {aid} FAIL rc={rc} size={sz}")
        if os.path.exists(dst): os.remove(dst)
    time.sleep(2)
print(f"done: {ok}/{len(ids)} downloaded")
PYCEPT
