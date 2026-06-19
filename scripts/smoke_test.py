import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arxiv_insight import mcp


def run(coro):
    return asyncio.run(coro)


def main() -> None:
    tools = run(mcp.list_tools())
    tool_names = {tool.name for tool in tools}
    expected_tools = {
        "health_check",
        "search_arxiv",
        "get_paper_fulltext",
        "download_pdf",
        "get_bibtex",
        "extract_section",
    }
    missing_tools = expected_tools - tool_names
    if missing_tools:
        raise SystemExit(f"Missing tools: {sorted(missing_tools)}")

    resources = run(mcp.list_resources())
    resource_uris = {str(resource.uri) for resource in resources}
    if "papers://recent" not in resource_uris:
        raise SystemExit("Missing resource: papers://recent")

    prompts = run(mcp.list_prompts())
    prompt_names = {prompt.name for prompt in prompts}
    for prompt_name in ("review_paper", "compare_papers"):
        if prompt_name not in prompt_names:
            raise SystemExit(f"Missing prompt: {prompt_name}")

    result = run(mcp.call_tool("health_check", {}))
    content, structured = result
    payload = structured.get("result") or json.loads(content[0].text)
    if payload["status"] != "ok":
        raise SystemExit(f"Unexpected health status: {payload}")

    print("server smoke test passed")


if __name__ == "__main__":
    main()
