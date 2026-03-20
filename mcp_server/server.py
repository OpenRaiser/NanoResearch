"""MCP stdio server — exposes NanoResearch tools via Model Context Protocol."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any

from mcp_server.tools.arxiv_search import search_arxiv
from mcp_server.tools.semantic_scholar import search_semantic_scholar, get_paper_details
from mcp_server.tools.github_search import search_repos as search_github_repos
from mcp_server.tools.latex_gen import generate_latex, generate_full_paper
from mcp_server.tools.pdf_compile import compile_pdf
from mcp_server.tools.figure_gen import generate_figure
from mcp_server.tools.web_search import search_web
from mcp_server.tools.paperswithcode import search_tasks as pwc_search_tasks, get_sota as pwc_get_sota
from mcp_server.tools.pdf_reader import download_and_extract as pdf_download_and_extract

logger = logging.getLogger(__name__)


# Tool registry: name → (handler, description, input_schema)
TOOLS: dict[str, dict[str, Any]] = {
    "search_arxiv": {
        "description": "Search arXiv for academic papers. Returns paper metadata (title, authors, abstract, etc.).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "default": 20, "description": "Max papers to return"},
            },
            "required": ["query"],
        },
    },
    "search_semantic_scholar": {
        "description": "Search Semantic Scholar for papers and citation data.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "default": 20},
            },
            "required": ["query"],
        },
    },
    "get_paper_details": {
        "description": "Get detailed information about a specific paper by its Semantic Scholar or arXiv ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "Paper ID (Semantic Scholar ID or arXiv:XXXX.XXXXX format)",
                },
            },
            "required": ["paper_id"],
        },
    },
    "search_github": {
        "description": "Search GitHub for relevant code repositories.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "default": 5, "description": "Max repos to return"},
                "language": {"type": "string", "default": "Python", "description": "Programming language filter"},
            },
            "required": ["query"],
        },
    },
    "generate_latex": {
        "description": "Generate LaTeX document from a Jinja2 template with provided data.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "template_name": {"type": "string", "description": "Template file, e.g. 'paper.tex.j2'"},
                "data": {"type": "object", "description": "Template variables"},
                "template_format": {
                    "type": "string",
                    "default": "arxiv",
                    "description": "Template format (auto-discovered)",
                },
            },
            "required": ["template_name", "data"],
        },
    },
    "compile_pdf": {
        "description": "Compile a .tex file to PDF using pdflatex or tectonic.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tex_path": {"type": "string", "description": "Path to .tex file"},
                "bibtex": {"type": "boolean", "default": True},
            },
            "required": ["tex_path"],
        },
    },
    "generate_figure": {
        "description": "Generate a figure (bar chart, line chart, grouped bar, heatmap, table, or placeholder) as PNG.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "figure_type": {
                    "type": "string",
                    "enum": [
                        "bar_chart", "line_chart", "grouped_bar", "heatmap",
                        "table", "placeholder",
                    ],
                },
                "data": {"type": "object", "description": "Figure data"},
                "output_path": {"type": "string", "description": "Output PNG path"},
                "title": {"type": "string", "default": ""},
            },
            "required": ["figure_type", "data", "output_path"],
        },
    },
    "search_web": {
        "description": "Search the web using DuckDuckGo. Returns titles, URLs, and snippets.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "default": 10, "description": "Max results to return"},
            },
            "required": ["query"],
        },
    },
    "search_pwc_tasks": {
        "description": "Search Papers With Code for ML tasks and benchmarks.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Task search query"},
                "max_results": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    },
    "get_pwc_sota": {
        "description": "Get SOTA leaderboard results from Papers With Code for a given task.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Papers With Code task ID"},
                "dataset": {"type": "string", "description": "Optional dataset name filter"},
                "max_results": {"type": "integer", "default": 20},
            },
            "required": ["task_id"],
        },
    },
    "read_pdf": {
        "description": "Download and extract full text from a PDF URL. Returns structured text with sections.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pdf_url": {"type": "string", "description": "URL to the PDF file"},
                "max_pages": {"type": "integer", "default": 30, "description": "Max pages to process"},
            },
            "required": ["pdf_url"],
        },
    },
}

# Dynamically populate the template_format enum from discovered templates
try:
    from nanoresearch.templates import get_available_formats as _get_fmts
    TOOLS["generate_latex"]["inputSchema"]["properties"]["template_format"]["enum"] = _get_fmts()
except Exception:
    pass  # graceful fallback if templates package is unavailable


async def handle_tool_call(name: str, arguments: dict) -> Any:
    """Dispatch a tool call to the appropriate handler."""
    if name not in TOOLS:
        raise ValueError(f"Unknown tool: {name}. Available: {list(TOOLS)}")

    # Validate required arguments
    schema = TOOLS[name].get("inputSchema", {})
    required = schema.get("required", [])
    for req in required:
        if req not in arguments:
            raise ValueError(f"Missing required argument '{req}' for tool '{name}'")

    if name == "search_arxiv":
        query = arguments["query"]
        if not query or not query.strip():
            raise ValueError("search_arxiv: 'query' must be a non-empty string")
        return await search_arxiv(query, arguments.get("max_results", 20))

    elif name == "search_semantic_scholar":
        query = arguments["query"]
        if not query or not query.strip():
            raise ValueError("search_semantic_scholar: 'query' must be a non-empty string")
        return await search_semantic_scholar(query, arguments.get("max_results", 20))

    elif name == "get_paper_details":
        paper_id = arguments["paper_id"]
        if not paper_id or not paper_id.strip():
            raise ValueError("get_paper_details: 'paper_id' must be a non-empty string")
        return await get_paper_details(paper_id)

    elif name == "search_github":
        query = arguments["query"]
        if not query or not query.strip():
            raise ValueError("search_github: 'query' must be a non-empty string")
        return await search_github_repos(
            query,
            max_results=arguments.get("max_results", 5),
            language=arguments.get("language", "Python"),
        )

    elif name == "generate_latex":
        return generate_latex(
            arguments["template_name"],
            arguments["data"],
            arguments.get("template_format", "arxiv"),
        )

    elif name == "compile_pdf":
        return await compile_pdf(arguments["tex_path"], bibtex=arguments.get("bibtex", True))

    elif name == "generate_figure":
        return generate_figure(
            arguments["figure_type"],
            arguments["data"],
            arguments["output_path"],
            arguments.get("title", ""),
        )

    elif name == "search_web":
        query = arguments["query"]
        if not query or not query.strip():
            raise ValueError("search_web: 'query' must be a non-empty string")
        return await search_web(query, arguments.get("max_results", 10))

    elif name == "search_pwc_tasks":
        query = arguments["query"]
        if not query or not query.strip():
            raise ValueError("search_pwc_tasks: 'query' must be a non-empty string")
        return await pwc_search_tasks(query, arguments.get("max_results", 10))

    elif name == "get_pwc_sota":
        task_id = arguments["task_id"]
        if not task_id or not task_id.strip():
            raise ValueError("get_pwc_sota: 'task_id' must be a non-empty string")
        return await pwc_get_sota(
            task_id,
            dataset=arguments.get("dataset"),
            max_results=arguments.get("max_results", 20),
        )

    elif name == "read_pdf":
        pdf_url = arguments["pdf_url"]
        if not pdf_url or not pdf_url.strip():
            raise ValueError("read_pdf: 'pdf_url' must be a non-empty string")
        return await pdf_download_and_extract(pdf_url, arguments.get("max_pages", 30))

    raise ValueError(f"Unhandled tool: {name}")  # pragma: no cover


async def serve_stdio() -> None:
    """Run the MCP server over stdio (JSON-RPC 2.0)."""
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    loop = asyncio.get_running_loop()
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    writer_transport, writer_protocol = await loop.connect_write_pipe(
        asyncio.streams.FlowControlMixin, sys.stdout
    )
    writer = asyncio.StreamWriter(writer_transport, writer_protocol, None, loop)

    async def send_response(response: dict) -> None:
        data = json.dumps(response) + "\n"
        writer.write(data.encode())
        await writer.drain()

    while True:
        line = await reader.readline()
        if not line:
            break
        try:
            request = json.loads(line.decode())
        except json.JSONDecodeError:
            continue

        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params", {})

        if method == "initialize":
            await send_response({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "nanoresearch", "version": "0.1.0"},
                    "capabilities": {"tools": {"listChanged": False}},
                },
            })
        elif method == "tools/list":
            tools_list = [
                {"name": name, "description": spec["description"], "inputSchema": spec["inputSchema"]}
                for name, spec in TOOLS.items()
            ]
            await send_response({"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools_list}})
        elif method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            try:
                result = await handle_tool_call(tool_name, arguments)
                content_text = json.dumps(result, ensure_ascii=False, default=str)
                await send_response({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": content_text}]},
                })
            except ValueError as e:
                # Input validation errors
                logger.warning("Tool %s validation error: %s", tool_name, e)
                await send_response({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Validation error: {e}"}],
                        "isError": True,
                    },
                })
            except Exception as e:
                # Unexpected errors
                logger.error("Tool %s failed: %s", tool_name, e, exc_info=True)
                await send_response({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error: {type(e).__name__}: {e}"}],
                        "isError": True,
                    },
                })
        elif method == "notifications/initialized":
            pass  # no response needed for notifications
        else:
            if req_id is not None:
                await send_response({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                })


def main() -> None:
    asyncio.run(serve_stdio())


if __name__ == "__main__":
    main()
