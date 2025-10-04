"""
CORE Memory Commands

Commands for managing HeySol CORE Memory operations including
ingestion, search, space management, and analytics.
"""

import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.logging_config import get_logger

console = Console()
logger = get_logger(__name__)

app = typer.Typer(
    name="memory",
    help="CORE Memory operations and management",
    add_completion=True,
)


@app.command("ingest")
def ingest_data(
    message: str = typer.Argument(..., help="Data to ingest into memory"),
    space_id: Optional[str] = typer.Option(None, help="Target memory space ID"),
    source: Optional[str] = typer.Option(None, help="Data source identifier"),
    metadata: Optional[str] = typer.Option(None, help="JSON metadata string"),
    use_mcp: bool = typer.Option(True, help="Prefer MCP over direct API"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Ingest data into CORE Memory.

    Stores clinical data, summaries, or any information in HeySol CORE Memory
    with optional metadata and space targeting.
    """
    console.print(f"[bold blue]💾 Ingesting data into CORE Memory[/bold blue]")

    try:
        # Import HeySol client
        from heysol import HeySolClient
        from config.heysol_config import get_config

        # Get configuration
        config = get_config()
        api_key = config.get_api_key("gpt-4")  # Use any model for API key

        if not api_key:
            console.print("[red]❌ No HeySol API key configured[/red]")
            console.print("[yellow]💡 Set HEYSOL_API_KEY environment variable[/yellow]")
            raise typer.Exit(1)

        # Initialize client
        client = HeySolClient(
            api_key=api_key,
            prefer_mcp=use_mcp
        )

        if verbose:
            console.print(f"[blue]🎯 Target space: {space_id or 'default'}[/blue]")
            console.print(f"[blue]📧 Source: {source or 'cli'}[/blue]")
            console.print(f"[blue]🧠 Using MCP: {use_mcp}[/blue]")

        # Parse metadata if provided
        parsed_metadata = None
        if metadata:
            try:
                import json
                parsed_metadata = json.loads(metadata)
                console.print("[blue]📋 Metadata parsed successfully[/blue]")
            except json.JSONDecodeError as e:
                console.print(f"[red]❌ Invalid metadata JSON: {e}[/red]")
                raise typer.Exit(1)

        console.print("[blue]📤 Sending data to CORE Memory...[/blue]")

        # Ingest data
        result = client.ingest(
            message=message,
            space_id=space_id,
            source=source or "mcode_cli",
            metadata=parsed_metadata
        )

        console.print("[green]✅ Data ingested successfully[/green]")

        if verbose and result:
            console.print(f"[blue]📊 Result: {result}[/blue]")

        # Cleanup
        client.close()

    except ImportError as e:
        console.print(f"[red]❌ Import error: {e}[/red]")
        console.print("[yellow]💡 Ensure heysol-api-client is installed[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Ingestion failed: {e}[/red]")
        logger.exception("Memory ingestion error")
        raise typer.Exit(1)


@app.command("search")
def search_memory(
    query: str = typer.Argument(..., help="Search query"),
    space_ids: Optional[str] = typer.Option(None, help="Comma-separated space IDs to search"),
    limit: int = typer.Option(10, help="Maximum results to return"),
    use_mcp: bool = typer.Option(True, help="Prefer MCP over direct API"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Search CORE Memory for relevant information.

    Performs semantic search across memory spaces to find relevant clinical
    data, summaries, or stored information.
    """
    console.print(f"[bold blue]🔍 Searching CORE Memory: '{query}'[/bold blue]")

    try:
        # Import HeySol client
        from heysol import HeySolClient
        from config.heysol_config import get_config

        # Get configuration
        config = get_config()
        api_key = config.get_api_key("gpt-4")

        if not api_key:
            console.print("[red]❌ No HeySol API key configured[/red]")
            raise typer.Exit(1)

        # Parse space IDs
        space_list = None
        if space_ids:
            space_list = [s.strip() for s in space_ids.split(",") if s.strip()]

        # Initialize client
        client = HeySolClient(api_key=api_key, prefer_mcp=use_mcp)

        if verbose:
            console.print(f"[blue]🎯 Search spaces: {space_list or 'all'}[/blue]")
            console.print(f"[blue]📊 Result limit: {limit}[/blue]")
            console.print(f"[blue]🧠 Using MCP: {use_mcp}[/blue]")

        console.print("[blue]🔎 Performing search...[/blue]")

        # Perform search
        results = client.search(
            query=query,
            space_ids=space_list,
            limit=limit
        )

        episodes = results.get("episodes", [])

        console.print(f"[green]✅ Search completed - found {len(episodes)} results[/green]")

        if episodes:
            # Display results in a table
            table = Table(title="Search Results")
            table.add_column("Content", style="cyan", no_wrap=False)
            table.add_column("Score", style="magenta", justify="right")
            table.add_column("Source", style="green")

            for episode in episodes:
                content = episode.get("content", "")[:100]
                score = episode.get("score", "N/A")
                metadata = episode.get("metadata", {})
                source = metadata.get("source", "unknown")

                table.add_row(
                    content + ("..." if len(content) == 100 else ""),
                    str(score),
                    source
                )

            console.print(table)

            if verbose:
                console.print(f"\n[blue]📊 Full results: {results}[/blue]")
        else:
            console.print("[yellow]⚠️ No results found for the query[/yellow]")

        # Cleanup
        client.close()

    except Exception as e:
        console.print(f"[red]❌ Search failed: {e}[/red]")
        logger.exception("Memory search error")
        raise typer.Exit(1)


@app.command("spaces")
def list_spaces(
    use_mcp: bool = typer.Option(True, help="Prefer MCP over direct API"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    List available memory spaces.

    Displays all CORE Memory spaces with their details and statistics.
    """
    console.print("[bold blue]📂 Listing CORE Memory spaces[/bold blue]")

    try:
        # Import HeySol client
        from heysol import HeySolClient
        from config.heysol_config import get_config

        # Get configuration
        config = get_config()
        api_key = config.get_api_key("gpt-4")

        if not api_key:
            console.print("[red]❌ No HeySol API key configured[/red]")
            raise typer.Exit(1)

        # Initialize client
        client = HeySolClient(api_key=api_key, prefer_mcp=use_mcp)

        console.print("[blue]📊 Fetching space information...[/blue]")

        # Get spaces
        spaces = client.get_spaces()

        console.print(f"[green]✅ Found {len(spaces)} memory spaces[/green]")

        if spaces:
            # Display spaces in a table
            table = Table(title="Memory Spaces")
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Name", style="green")
            table.add_column("Description", style="yellow", no_wrap=False)

            for space in spaces:
                if isinstance(space, dict):
                    space_id = space.get("id", "unknown")
                    name = space.get("name", "unnamed")
                    description = space.get("description", "")
                    table.add_row(space_id[:16] + "...", name, description)

            console.print(table)

            if verbose:
                console.print(f"\n[blue]📊 Full space data: {spaces}[/blue]")
        else:
            console.print("[yellow]⚠️ No memory spaces found[/yellow]")

        # Cleanup
        client.close()

    except Exception as e:
        console.print(f"[red]❌ Failed to list spaces: {e}[/red]")
        logger.exception("List spaces error")
        raise typer.Exit(1)


@app.command("create-space")
def create_space(
    name: str = typer.Argument(..., help="Name for the new memory space"),
    description: str = typer.Option("", help="Description of the memory space"),
    use_mcp: bool = typer.Option(True, help="Prefer MCP over direct API"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Create a new memory space.

    Creates a dedicated space in CORE Memory for organizing related data.
    """
    console.print(f"[bold blue]🏗️ Creating memory space: {name}[/bold blue]")

    try:
        # Import HeySol client
        from heysol import HeySolClient
        from config.heysol_config import get_config

        # Get configuration
        config = get_config()
        api_key = config.get_api_key("gpt-4")

        if not api_key:
            console.print("[red]❌ No HeySol API key configured[/red]")
            raise typer.Exit(1)

        # Initialize client
        client = HeySolClient(api_key=api_key, prefer_mcp=use_mcp)

        if verbose:
            console.print(f"[blue]📝 Description: {description or 'none'}[/blue]")
            console.print(f"[blue]🧠 Using MCP: {use_mcp}[/blue]")

        console.print("[blue]🏗️ Creating space...[/blue]")

        # Create space
        space_id = client.create_space(name=name, description=description)

        console.print("[green]✅ Memory space created successfully[/green]")
        console.print(f"[green]🆔 Space ID: {space_id}[/green]")

        # Cleanup
        client.close()

    except Exception as e:
        console.print(f"[red]❌ Failed to create space: {e}[/red]")
        logger.exception("Create space error")
        raise typer.Exit(1)


@app.command("stats")
def memory_stats(
    use_mcp: bool = typer.Option(True, help="Prefer MCP over direct API"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Display CORE Memory statistics.

    Shows comprehensive statistics about memory usage, spaces, and performance.
    """
    console.print("[bold blue]📊 CORE Memory Statistics[/bold blue]")

    try:
        # Import HeySol client
        from heysol import HeySolClient
        from config.heysol_config import get_config

        # Get configuration
        config = get_config()
        api_key = config.get_api_key("gpt-4")

        if not api_key:
            console.print("[red]❌ No HeySol API key configured[/red]")
            raise typer.Exit(1)

        # Initialize client
        client = HeySolClient(api_key=api_key, prefer_mcp=use_mcp)

        console.print("[blue]📊 Gathering statistics...[/blue]")

        # Get spaces for basic stats
        spaces = client.get_spaces()

        console.print("[green]✅ Statistics retrieved[/green]")

        # Display statistics
        console.print(f"[cyan]📂 Total Spaces: {len(spaces)}[/cyan]")
        console.print(f"[cyan]🧠 MCP Available: {client.is_mcp_available()}[/cyan]")

        if spaces and verbose:
            console.print("\n[blue]📋 Space Details:[/blue]")
            for space in spaces:
                if isinstance(space, dict):
                    name = space.get("name", "unnamed")
                    space_id = space.get("id", "unknown")
                    console.print(f"  • {name} (ID: {space_id[:16]}...)")

        # Cleanup
        client.close()

    except Exception as e:
        console.print(f"[red]❌ Failed to get statistics: {e}[/red]")
        logger.exception("Memory stats error")
        raise typer.Exit(1)