# ArXiv Insight MCP Server

An intelligent Model Context Protocol (MCP) server that empowers LLMs to search, read, analyze, and manage academic papers from arXiv.

## Features

- **🔍 Smart Search**: Search arXiv papers by topic with optional category filtering (e.g., `cs.AI`, `cs.CV`).
- **📖 Full Text Access**: Retrieve optimized full text of papers. Includes local caching for 10x speedup on repeat access.
- **📥 PDF Download**: Download original PDF files to your local machine. **Securely restricted** to the project's download directory.
- **📝 BibTeX Generation**: Generate standard BibTeX citations for your papers.
- **✂️ Section Extraction**: Smartly extract specific sections like "Introduction", "Methodology", or "Conclusion".
- **🔗 Reference Discovery**: Automatically detects and links other arXiv papers referenced in the text.
- **🛡️ Robust & Secure**: Implements rate limiting, path traversal protection, and persistent search history.

## Prerequisites

- **Python 3.12+**
- **[uv](https://github.com/astral-sh/uv)** (Recommended for dependency management)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YI-TING-EE13/ArXiv-Insight-MCP-Server.git
   cd ArXiv-Insight-MCP-Server
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

## Usage

### Running the Server

You can run the server directly using `uv`:

```bash
uv run main.py
```

### Configuring with MCP Clients (e.g., Claude Desktop, LM Studio)

To use this server with an MCP client, you need to add it to your client's configuration file (e.g., `claude_desktop_config.json` or LM Studio's MCP settings).

**Configuration Example (JSON):**

Replace `C:\\Path\\To\\arxiv-insight-mcp` with the absolute path to your project directory.

```json
{
  "mcpServers": {
    "arxiv-insight": {
      "command": "uv",
      "args": [
        "--directory",
        "C:\\Path\\To\\arxiv-insight-mcp",
        "run",
        "main.py"
      ]
    }
  }
}
```

**Note for Windows Users:** Ensure you use double backslashes `\\` in the path.

## Tools Available

| Tool | Description |
|------|-------------|
| `search_arxiv` | Search for papers. Arguments: `topic`, `max_results`, `category`. |
| `get_paper_fulltext` | Get the full text content of a paper. |
| `download_pdf` | Download the PDF file. Arguments: `paper_id`, `save_dir`. |
| `get_bibtex` | Get the BibTeX citation string. |
| `extract_section` | Extract a specific section from the paper. |

## Prompts

- **`review_paper`**: Generates a deep review structure (Contribution, Methodology, Limitations).
- **`compare_papers`**: Compares multiple papers' contributions and methodologies.

## Project Structure

- `main.py`: Entry point for the server.
- `arxiv_insight.py`: Main MCP server implementation.
- `paper_cache/`: Directory where parsed paper texts are cached (ignored by git).
- `downloads/`: Default directory for downloaded PDFs (ignored by git).
- `metadata_db.json`: Persistent storage for search history and rate limiting state (ignored by git).
