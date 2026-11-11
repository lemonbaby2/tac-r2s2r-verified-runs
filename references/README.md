# References — cited papers of the TAC-R2S2R manuscript

This folder packages the **36 references** cited by the manuscript.

| File | What it is |
|---|---|
| `BIBLIOGRAPHY.md` | Human-readable table of all 36 references (title, authors, arXiv/DOI link) |
| `references.bib` | BibTeX file (auto-derived; every entry keeps the original citation in `note`) |
| `arxiv_metadata.json` | Title / authors / abstract / date for the arXiv items, fetched from the arXiv API |
| `MANIFEST.md` | Coverage table: which refs are arXiv (fetchable) vs DOI-only |
| `fetch_pdfs.sh` | One-command downloader for the open-access arXiv PDFs |

## Why full PDFs are not committed here

The 36 papers are **hundreds of MB** in total. Committing them to git would bloat
the repository and exceed sensible hosting limits (GitHub/Gitee). Instead:

- The complete **bibliographic record** is committed (small, diffable, citable).
- Open-access full texts are fetched on demand with `./fetch_pdfs.sh`, which
  writes into `pdfs/` (git-ignored).

## Fetch the PDFs

```bash
cd references
./fetch_pdfs.sh            # downloads every arXiv item into pdfs/
```

On a slow link (~40 KB/s), run it overnight or on a machine with a fast
upstream. Set `HTTPS_PROXY` if you need a proxy:

```bash
HTTPS_PROXY=http://127.0.0.1:7891 ./fetch_pdfs.sh
```

## Licensing of the cited papers

The papers remain under **their own** licenses and copyright. This folder only
records citations and links; downloading a paper is your own responsibility to
respect its license. The DOIs/arXiv links are provided for that purpose.
