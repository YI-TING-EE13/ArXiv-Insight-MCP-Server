# ArXiv Insight MCP Server

An MCP server for academic-paper demos. It exposes arXiv search, citation,
PDF/text retrieval, a recent-search resource, and reusable review prompts.

## What It Provides

Tools:

- `health_check`: local health and cache status, no network required.
- `search_arxiv`: search arXiv by topic, category, sort order, year range, and offset.
- `get_paper_fulltext`: download and extract paper text, using local cache.
- `download_pdf`: download a PDF into the local `downloads/` directory only.
- `get_bibtex`: generate BibTeX for an arXiv paper ID.
- `extract_section`: extract a named section from a cached or downloaded paper.

Resources:

- `papers://recent`: IDs and titles from the most recent `search_arxiv` result.

Prompts:

- `review_paper`: standard review workflow for one paper.
- `compare_papers`: comparison workflow for multiple papers.

## Requirements

- Python 3.12+
- `uv`
- Network access to arXiv for search, PDF, and BibTeX tools

The project currently uses MCP Python SDK v1.x through `mcp[cli]>=1.25.0`.
Before refreshing dependencies, consider adding a `<2` upper bound while SDK v2
remains a separate pre-release line.

## Install

```powershell
uv sync --frozen
```

## Run

```powershell
uv run main.py
```

The server uses stdio transport by default, which is the expected mode for local
MCP clients.

## MCP Client Configuration

From the sibling `mcp-client` or `chainlit-mcp-client` repo, use:

```json
{
  "mcpServers": {
    "arxiv-insight": {
      "command": "uv",
      "args": ["--directory", "../mcp-server", "run", "main.py"]
    }
  }
}
```

For absolute Windows paths, escape backslashes in JSON.

## Test

Local smoke tests do not contact arXiv or require extra dev dependencies:

```powershell
uv run python scripts/smoke_test.py
```

Manual MCP Inspector flow:

```powershell
npx -y @modelcontextprotocol/inspector
```

In the Inspector, configure a stdio server with command `uv` and args:

```text
--directory C:\absolute\path\to\mcp-server run main.py
```

First call `health_check`, then `search_arxiv` with a small `max_results` such
as `3`.

## Demo Baseline

This server has been verified through the sibling `../mcp-client` Chainlit demo
on Windows with LM Studio Local Server:

- Endpoint: `http://localhost:1234/v1`
- Model identifier: `google/gemma-4-e4b`
- Resource shown in Chainlit: `papers://recent`
- Successful MCP tool calls: `health_check`, then `search_arxiv`

## Data And Security Notes

- `paper_cache/`, `downloads/`, `metadata_db.json`, and `Log.txt` are local runtime artifacts.
- `download_pdf` validates the arXiv ID and restricts writes to `downloads/`.
- arXiv calls are rate-limited to a 3-second interval through the shared client state.
- Tool errors are returned as text for demo readability; production servers should consider structured error output and stricter auditing.

## Troubleshooting

- If `uv run python scripts/smoke_test.py` cannot import dependencies, run `uv sync --frozen` first.
- If arXiv tools fail but `health_check` works, check network connectivity and arXiv availability.
- If PDF extraction is slow, repeat the same paper ID; cached text should be reused from `paper_cache/`.
