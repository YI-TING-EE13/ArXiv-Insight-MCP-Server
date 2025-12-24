import asyncio
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import arxiv
import fitz  # PyMuPDF
import httpx
from mcp.server.fastmcp import FastMCP, Context
from mcp.types import SamplingMessage, TextContent

# --- Configuration & Setup ---

# Directory to store parsed paper text to improve performance
CACHE_DIR = Path("paper_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Initialize the FastMCP server
mcp = FastMCP("arxiv-insight")

# --- State Management ---

class ServerState:
    def __init__(self):
        self.last_request_time = 0
        self.rate_limit_delay = 3.0
        self.store_file = Path("metadata_db.json")
        self.resources = self._load_store()
        self.arxiv_client = arxiv.Client(delay_seconds=3.0, num_retries=3)

    def _load_store(self) -> Dict[str, Any]:
        if self.store_file.exists():
            try:
                return json.loads(self.store_file.read_text(encoding='utf-8'))
            except Exception:
                return {"recent_searches": []}
        return {"recent_searches": []}

    def save_store(self):
        try:
            self.store_file.write_text(json.dumps(self.resources, indent=2), encoding='utf-8')
        except Exception as e:
            sys.stderr.write(f"Error saving store: {e}\n")

    async def wait_for_rate_limit(self):
        """Ensure we don't hit arXiv API too fast (3 seconds interval)."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

state = ServerState()

# --- Helper Functions ---

def extract_arxiv_refs(text: str) -> List[str]:
    """
    Find other arXiv IDs mentioned in the text using Regex.
    """
    pattern = r"\b\d{4}\.\d{4,5}(?:v\d+)?\b"
    found = re.findall(pattern, text)
    return list(set(found))

def optimize_markdown(text: str) -> str:
    """
    Basic cleanup for PDF text to improve LLM readability.
    """
    # Remove potential header/footer artifacts containing arXiv ID and page numbers
    text = re.sub(r'arXiv:\d+\.\d+v\d+\s+\[.*?\]\s+\d+\s+\w+\s+\d+', '', text)
    
    # Fix hyphenated words at line breaks (e.g., "algo-\nrithm" -> "algorithm")
    text = re.sub(r'-\n\s*', '', text)
    
    # Collapse multiple newlines into double newlines (paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove single newlines within paragraphs (common in PDF text extraction)
    # This is a heuristic: if a line ends with a character that isn't punctuation, join it.
    # But simple approach: replace single newline with space if not preceded by punctuation?
    # For safety, let's just collapse excessive whitespace for now.
    
    return text

def _sync_search_arxiv(query: str, max_results: int, offset: int, sort_by: str) -> List[Dict[str, Any]]:
    """Synchronous arXiv search to be run in a thread."""
    
    # Map sort_by string to arxiv.SortCriterion
    criterion = arxiv.SortCriterion.Relevance
    if sort_by == "submitted":
        criterion = arxiv.SortCriterion.SubmittedDate
    elif sort_by == "updated":
        criterion = arxiv.SortCriterion.LastUpdatedDate
        
    # Use shared client
    search = arxiv.Search(
        query=query, 
        max_results=max_results, 
        sort_by=criterion
    )
    
    results = []
    for paper in state.arxiv_client.results(search, offset=offset):
        results.append({
            "id": paper.get_short_id(),
            "title": paper.title,
            "authors": [a.name for a in paper.authors],
            "published": paper.published.strftime("%Y-%m-%d"),
            "summary": paper.summary[:200] + "..." if len(paper.summary) > 200 else paper.summary,
            "pdf_url": paper.pdf_url,
            "category": paper.primary_category or ""
        })
    return results

def _sync_get_bibtex(paper_id: str) -> str:
    """Synchronous BibTeX retrieval."""
    # Use shared client
    search = arxiv.Search(id_list=[paper_id])
    try:
        paper = next(state.arxiv_client.results(search))
    except StopIteration:
        return f"Error: Paper {paper_id} not found."
        
    authors = " and ".join([a.name for a in paper.authors])
    year = paper.published.year
    
    return (
        f"@misc{{{paper.get_short_id()},\n"
        f"      title={{{paper.title}}},\n"
        f"      author={{{authors}}},\n"
        f"      year={{{year}}},\n"
        f"      eprint={{{paper.get_short_id()}}},\n"
        f"      archivePrefix={{arXiv}},\n"
        f"      primaryClass={{{paper.primary_category}}}\n"
        f"}}"
    )

# --- Tools ---

@mcp.tool()
async def search_arxiv(
    topic: str, 
    max_results: int = 100, 
    offset: int = 0, 
    category: str = "",
    sort_by: str = "relevance",
    start_year: int = 0,
    end_year: int = 0
) -> str:
    """
    Search for papers on arXiv and cache metadata for Resources.
    
    Args:
        topic: The search query. Supports advanced prefixes like 'ti:' (title), 'au:' (author), 'abs:' (abstract).
        max_results: Maximum number of results to return (default: 100, max: 300).
        offset: The index of the first result to return (for pagination).
        category: Optional category filter (e.g., 'cs.AI').
        sort_by: Sort order. Options: 'relevance' (default), 'submitted', 'updated'.
        start_year: Filter by submission year (start). Set to 0 to ignore.
        end_year: Filter by submission year (end). Set to 0 to ignore.
    """
    sys.stderr.write(f"DEBUG: search_arxiv called with topic='{topic}', category='{category}', offset={offset}, sort_by='{sort_by}', years={start_year}-{end_year}\n")
    
    # Heuristic: If topic is a simple list of words, join with AND to enforce all keywords.
    # Also wrap in parentheses to ensure it groups correctly against other filters.
    
    # Check if topic contains special operators
    special_chars = ["AND", "OR", "NOT", "(", ")", ":", '"']
    is_simple_query = not any(char in topic for char in special_chars)
    
    if is_simple_query:
        # Replace spaces with AND to enforce all keywords
        terms = topic.split()
        if len(terms) > 1:
            topic_query = " AND ".join(terms)
            topic_part = f"({topic_query})"
        else:
            topic_part = topic
    else:
        # Just wrap in parentheses for safety
        topic_part = f"({topic})"

    query_parts = [topic_part]
    if category:
        query_parts.append(f"cat:{category}")
    
    if start_year and start_year > 0:
        # Format: submittedDate:[YYYYMMDDHHMM TO YYYYMMDDHHMM]
        # We use broad range for the year
        start_date = f"{start_year}01010000"
        end_date = f"{end_year}12312359" if end_year and end_year > 0 else f"{start_year}12312359"
        # If end_year is not provided but start_year is, assume just that year? 
        # Or if user wants "since 2020", they should provide end_year=2025 (current).
        # Let's assume if end_year is missing, it means "from start_year until now".
        if not end_year or end_year == 0:
             # Use a far future date or current year. Let's use 2099 to be safe for "until now"
             end_date = "209912312359"
        
        query_parts.append(f"submittedDate:[{start_date} TO {end_date}]")
    
    query = " AND ".join(query_parts)
    
    # Enforce safety limit
    if max_results > 300:
        sys.stderr.write(f"Warning: max_results {max_results} exceeds limit. Capping at 300.\n")
        max_results = 300
    
    # Rate limiting
    await state.wait_for_rate_limit()
    
    # Run blocking arXiv call in a separate thread
    try:
        results = await asyncio.to_thread(_sync_search_arxiv, query, max_results, offset, sort_by)
        
        # Update state and persist
        state.resources["recent_searches"] = results
        state.save_store()
        
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error searching arXiv: {str(e)}"

@mcp.tool()
async def get_bibtex(paper_id: str) -> str:
    """
    Get the BibTeX citation for a specific arXiv paper.
    """
    await state.wait_for_rate_limit()
    try:
        return await asyncio.to_thread(_sync_get_bibtex, paper_id)
    except Exception as e:
        return f"Error retrieving BibTeX: {str(e)}"

async def _get_raw_fulltext_only(paper_id: str) -> str:
    """
    Internal helper to retrieve full text without sampling/summarization.
    """
    cache_path = CACHE_DIR / f"{paper_id}.txt"

    if cache_path.exists():
        sys.stderr.write(f"DEBUG: Loading {paper_id} from local cache...\n")
        return cache_path.read_text(encoding="utf-8")

    pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
    
    try:
        await state.wait_for_rate_limit()
        
        # Use httpx for async download
        async with httpx.AsyncClient() as client:
            response = await client.get(pdf_url, follow_redirects=True, timeout=30.0)
            response.raise_for_status()
            pdf_content = response.content

        # Process PDF
        def process_pdf(content):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(content)
                temp_path = temp_pdf.name
            
            raw_text = ""
            try:
                with fitz.open(temp_path) as doc:
                    for page in doc:
                        raw_text += page.get_text()
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            return raw_text

        raw_text = await asyncio.to_thread(process_pdf, pdf_content)
        clean_text = optimize_markdown(raw_text)
        
        refs = extract_arxiv_refs(clean_text)
        ref_header = f"\n\n--- [Automatically Detected References] ---\n{', '.join(refs)}\n\n" if refs else ""
        final_content = clean_text + ref_header
        
        # Save to cache
        cache_path.write_text(final_content, encoding="utf-8")
        
        return final_content
        
    except Exception as e:
        raise e

@mcp.tool()
async def get_paper_fulltext(ctx: Context, paper_id: str) -> str:
    """
    Retrieve the full text of a paper. Checks local cache first.
    """
    try:
        final_content = await _get_raw_fulltext_only(paper_id)

        if len(final_content) > 15000:
            summary_request = (
                f"The following is the raw text content of an academic paper ({paper_id}). "
                f"It is too long to display fully. Please provide a structured summary including:\n"
                f"1. **Core Contribution**: What is the main problem and solution?\n"
                f"2. **Methodology**: Key technical details (architecture, algorithms).\n"
                f"3. **Key Results**: Main metrics and findings.\n\n"
                f"Paper Content (Truncated):\n{final_content[:50000]}"
            )
            try:
                # Use ctx.session.create_message with correct types for Sampling
                result = await ctx.session.create_message(
                    messages=[
                        SamplingMessage(
                            role="user",
                            content=TextContent(type="text", text=summary_request)
                        )
                    ],
                    max_tokens=2000
                )
                final_content = result.content.text
            except Exception as e:
                sys.stderr.write(f"Sampling failed, falling back to truncation: {e}\n")
                final_content = final_content[:15000] + "\n\n[Content truncated due to length and sampling failure]"

        return final_content
        
    except Exception as e:
        return f"Error retrieving paper {paper_id}: {str(e)}"

@mcp.tool()
async def download_pdf(paper_id: str, save_dir: str = "downloads") -> str:
    """
    Download a paper's PDF to a local directory.
    Securely restricts downloads to the project's 'downloads' folder and its subdirectories.
    """
    # Security: Enforce downloads directory
    base_dir = Path("downloads").resolve()
    base_dir.mkdir(exist_ok=True)
    
    # We treat save_dir as a subdirectory name if provided, but ensure it stays inside base_dir
    # Or simply ignore user path and use base_dir to be safest.
    # Let's allow subdirectories but validate path.
    
    try:
        # Resolve the target directory
        target_dir = (base_dir / save_dir).resolve()
        
        # Check if target_dir is relative to base_dir (prevents ../../ attacks)
        if not str(target_dir).startswith(str(base_dir)):
            return f"Error: Security violation. Cannot save outside of {base_dir}"
            
        target_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{paper_id}.pdf"
        file_path = target_dir / filename
        pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
        
        await state.wait_for_rate_limit()
        
        async with httpx.AsyncClient() as client:
            response = await client.get(pdf_url, follow_redirects=True, timeout=30.0)
            response.raise_for_status()
            file_path.write_bytes(response.content)
            
        return f"Successfully downloaded {paper_id} to {file_path}"
        
    except Exception as e:
        return f"Error downloading PDF: {str(e)}"

@mcp.tool()
async def extract_section(ctx: Context, paper_id: str, section_name: str) -> str:
    """
    Extract a specific section (e.g., Abstract, Introduction, Conclusion) from a paper.
    """
    try:
        full_text = await _get_raw_fulltext_only(paper_id)
    except Exception as e:
        return f"Error retrieving paper for section extraction: {str(e)}"
    
    lines = full_text.split('\n')
    start_idx = -1
    end_idx = -1
    target = section_name.lower()
    
    for i, line in enumerate(lines):
        clean_line = line.strip().lower()
        if target in clean_line and len(clean_line) < 50:
            start_idx = i
            break
            
    if start_idx == -1:
        return f"Section '{section_name}' not found in {paper_id}."
        
    for i in range(start_idx + 1, len(lines)):
        line = lines[i].strip()
        if not line: continue
        if re.match(r'^\d+\.\s+[A-Z]', line) or line.lower() in ['references', 'conclusion', 'conclusions', 'acknowledgments']:
             end_idx = i
             break
             
    if end_idx == -1:
        end_idx = min(start_idx + 100, len(lines))
        
    return "\n".join(lines[start_idx:end_idx])

# --- Resources & Prompts ---

@mcp.resource("papers://recent")
def get_recent_papers() -> str:
    """
    Resource to list IDs and titles from the most recent search.
    """
    recent = state.resources.get("recent_searches", [])
    if not recent: return "No recent search results."
    return "Recent Results:\n" + "\n".join([f"- [{p['id']}] {p['title']}" for p in recent])

@mcp.prompt()
def review_paper(arxiv_id: str) -> str:
    """
    Prompt template for a standard academic review workflow.
    """
    return f"Deep review of {arxiv_id}. 1. Contribution 2. Methodology 3. Limitations. Use get_paper_fulltext tool."

@mcp.prompt()
def compare_papers(paper_ids: str) -> str:
    """
    Create a prompt to compare multiple papers.
    """
    return (
        f"Please compare the following papers: {paper_ids}.\n"
        f"1. Summarize the main contribution of each.\n"
        f"2. Compare their methodologies.\n"
        f"3. Discuss their results and which one performs better (if applicable).\n"
        f"Use the 'get_paper_fulltext' tool to read the content of each paper."
    )

if __name__ == "__main__":
    mcp.run(transport='stdio')