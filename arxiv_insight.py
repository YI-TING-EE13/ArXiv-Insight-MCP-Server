import arxiv
import fitz  # PyMuPDF
import requests
import tempfile
import os
import re
import json
from typing import List, Dict, Any
from mcp.server.fastmcp import FastMCP, Context

# --- Configuration & Setup ---

# Directory to store parsed paper text to improve performance
# This acts as a persistent cache to avoid re-downloading PDFs
CACHE_DIR = "paper_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Global variable to store recent search results for Resources
# Note: In a multi-user environment, global state should be avoided or managed per-session.
recent_search_results = []

# Initialize the FastMCP server with the name "arxiv-insight"
mcp = FastMCP("arxiv-insight")

# --- Helper Functions ---

def extract_arxiv_refs(text: str) -> List[str]:
    """
    Find other arXiv IDs mentioned in the text using Regex.
    
    Args:
        text (str): The text content of the paper.
        
    Returns:
        List[str]: A list of unique arXiv IDs found in the text.
    """
    # Pattern for modern arXiv IDs (post-2007), e.g., 2310.12345
    # Matches YYMM.NNNNN optionally followed by version vN
    pattern = r"\b\d{4}\.\d{4,5}(?:v\d+)?\b"
    found = re.findall(pattern, text)
    # Return unique IDs to avoid duplicates
    return list(set(found))

def optimize_markdown(text: str) -> str:
    """
    Basic cleanup for PDF text to improve LLM readability.
    Removes common artifacts like headers/footers or excessive whitespace.
    
    Args:
        text (str): Raw text extracted from PDF.
        
    Returns:
        str: Cleaned text.
    """
    # Remove potential header/footer artifacts containing arXiv ID and page numbers
    text = re.sub(r'arXiv:\d+\.\d+v\d+\s+\[.*?\]\s+\d+\s+\w+\s+\d+', '', text)
    # Collapse multiple newlines into double newlines (paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text

# --- Tools ---

@mcp.tool()
def search_arxiv(topic: str, max_results: int = 3, category: str = "") -> str:
    """
    Search for papers on arXiv and cache metadata for Resources.
    
    Args:
        topic (str): The search query. Keep it simple and focused (e.g., "gesture recognition edge devices"). Avoid overly long sentences.
        max_results (int): Maximum number of results to return. Default is 3.
        category (str): Optional arXiv category filter (e.g., 'cs.AI', 'cs.CL'). Defaults to empty string.
        
    Returns:
        str: A JSON string containing a list of paper metadata (id, title, summary, pdf_url).
    """
    global recent_search_results
    client = arxiv.Client()
    
    query = topic
    if category:
        query = f"{topic} AND cat:{category}"
        
    # Search arXiv sorted by relevance
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    
    results = []
    for paper in client.results(search):
        results.append({
            "id": paper.get_short_id(),
            "title": paper.title,
            "summary": paper.summary,
            "pdf_url": paper.pdf_url
        })
    
    # Update the global cache of recent results
    recent_search_results = results
    return json.dumps(results, indent=2)

@mcp.tool()
def get_bibtex(paper_id: str) -> str:
    """
    Get the BibTeX citation for a specific arXiv paper.
    
    Args:
        paper_id (str): The arXiv ID of the paper.
        
    Returns:
        str: The BibTeX citation string.
    """
    client = arxiv.Client()
    search = arxiv.Search(id_list=[paper_id])
    
    try:
        paper = next(client.results(search))
    except StopIteration:
        return f"Error: Paper {paper_id} not found."
        
    authors = " and ".join([a.name for a in paper.authors])
    year = paper.published.year
    
    bibtex = (
        f"@misc{{{paper.get_short_id()},\n"
        f"      title={{{paper.title}}},\n"
        f"      author={{{authors}}},\n"
        f"      year={{{year}}},\n"
        f"      eprint={{{paper.get_short_id()}}},\n"
        f"      archivePrefix={{arXiv}},\n"
        f"      primaryClass={{{paper.primary_category}}}\n"
        f"}}"
    )
    return bibtex

@mcp.tool()
async def get_paper_fulltext(ctx: Context, paper_id: str) -> str:
    """
    Retrieve the full text of a paper. 
    Checks local cache first to boost speed significantly.
    Also extracts reference links for cross-paper discovery.
    
    Args:
        ctx (Context): The MCP context object (used for sampling/logging).
        paper_id (str): The arXiv ID of the paper (e.g., "2310.12345").
        
    Returns:
        str: The full text of the paper, or a summary if the paper is too long.
    """
    cache_path = os.path.join(CACHE_DIR, f"{paper_id}.txt")

    # --- Optimization: Local Caching ---
    # Principle: Avoid re-downloading and re-parsing if we already have the data
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            print(f"Loading {paper_id} from local cache...")
            return f.read()

    # --- Normal Fetching Logic ---
    pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
    try:
        # Download the PDF
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status() # Ensure we got a valid response
        
        # Save to a temporary file because PyMuPDF (fitz) works best with file paths
        # delete=False is required on Windows to allow opening the file again
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(response.content)
            temp_path = temp_pdf.name

        # Extract text from the PDF
        raw_text = ""
        with fitz.open(temp_path) as doc:
            for page in doc:
                raw_text += page.get_text()
        
        # Clean up the temporary file
        os.remove(temp_path)

        # Clean and optimize the text
        clean_text = optimize_markdown(raw_text)
        
        # --- Feature: Auto-linking References ---
        # Detect other arXiv IDs cited in the paper
        refs = extract_arxiv_refs(clean_text)
        ref_header = f"\n\n--- [Automatically Detected References] ---\n{', '.join(refs)}\n\n" if refs else ""
        
        final_content = clean_text + ref_header

        # --- Handling Long Text ---
        # If the text is too long for typical context windows, use the LLM to summarize it.
        # Note: This uses the MCP sampling capability via ctx.create_message
        if len(final_content) > 15000:
            summary_request = (
                f"The following paper ({paper_id}) is long. Please summarize the core "
                f"methodology and results in 2000 chars:\n\n{final_content[:50000]}"
            )
            # Request the host (LLM) to summarize the content
            final_content = await ctx.create_message(summary_request)

        # Save the result (full text or summary) to cache for future use
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(final_content)

        return final_content
        
    except Exception as e:
        return f"Error retrieving paper {paper_id}: {str(e)}"

@mcp.tool()
def download_pdf(paper_id: str, save_dir: str = "downloads") -> str:
    """
    Download a paper's PDF to a local directory.
    
    Args:
        paper_id (str): The arXiv ID of the paper.
        save_dir (str): Directory to save the PDF. Defaults to "downloads".
        
    Returns:
        str: The path to the downloaded file.
    """
    os.makedirs(save_dir, exist_ok=True)
    pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
    filename = f"{paper_id}.pdf"
    file_path = os.path.join(save_dir, filename)
    
    try:
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(response.content)
        return f"Successfully downloaded {paper_id} to {file_path}"
    except Exception as e:
        return f"Error downloading PDF: {str(e)}"

@mcp.tool()
async def extract_section(ctx: Context, paper_id: str, section_name: str) -> str:
    """
    Extract a specific section (e.g., Abstract, Introduction, Conclusion) from a paper.
    
    Args:
        ctx (Context): MCP Context.
        paper_id (str): The arXiv ID.
        section_name (str): The name of the section to extract (case-insensitive).
        
    Returns:
        str: The content of the section.
    """
    # Reuse the fulltext retrieval logic
    full_text = await get_paper_fulltext(ctx, paper_id)
    
    # Simple heuristic extraction
    lines = full_text.split('\n')
    start_idx = -1
    end_idx = -1
    
    # Normalize section name for search
    target = section_name.lower()
    
    # Look for a line that contains the section name and is short (likely a header)
    for i, line in enumerate(lines):
        clean_line = line.strip().lower()
        # Check if line matches target, allowing for numbering (e.g., "1. introduction")
        if target in clean_line and len(clean_line) < 50:
            start_idx = i
            break
            
    if start_idx == -1:
        return f"Section '{section_name}' not found in {paper_id}."
        
    # Find end of section (next header)
    for i in range(start_idx + 1, len(lines)):
        line = lines[i].strip()
        if not line: continue
        
        # Check for next header candidates (starts with number or standard section name)
        if re.match(r'^\d+\.\s+[A-Z]', line) or line.lower() in ['references', 'conclusion', 'conclusions', 'acknowledgments']:
             end_idx = i
             break
             
    if end_idx == -1:
        # If no next header found, take a reasonable chunk (e.g., 100 lines) or the rest
        end_idx = min(start_idx + 100, len(lines))
        
    return "\n".join(lines[start_idx:end_idx])

# --- Resources & Prompts ---

@mcp.resource("papers://recent")
def get_recent_papers() -> str:
    """
    Resource to list IDs and titles from the most recent search.
    This allows the LLM to "see" what was just searched without re-running the tool.
    """
    if not recent_search_results: return "No recent search results."
    return "Recent Results:\n" + "\n".join([f"- [{p['id']}] {p['title']}" for p in recent_search_results])

@mcp.prompt()
def review_paper(arxiv_id: str) -> str:
    """
    Prompt template for a standard academic review workflow.
    
    Args:
        arxiv_id (str): The ID of the paper to review.
    """
    return f"Deep review of {arxiv_id}. 1. Contribution 2. Methodology 3. Limitations. Use get_paper_fulltext tool."

@mcp.prompt()
def compare_papers(paper_ids: str) -> str:
    """
    Create a prompt to compare multiple papers.
    
    Args:
        paper_ids (str): Comma-separated list of arXiv IDs (e.g., "2310.0001, 2310.0002").
    """
    return (
        f"Please compare the following papers: {paper_ids}.\n"
        f"1. Summarize the main contribution of each.\n"
        f"2. Compare their methodologies.\n"
        f"3. Discuss their results and which one performs better (if applicable).\n"
        f"Use the 'get_paper_fulltext' tool to read the content of each paper."
    )

if __name__ == "__main__":
    # Run the MCP server using stdio transport
    mcp.run(transport='stdio')