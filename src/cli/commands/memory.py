"""
CORE Memory Commands

Commands for managing HeySol CORE Memory operations including
ingestion, search, space management, and analytics.
"""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.heysol_client import OncoCoreClient
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
    console.print("[bold blue]💾 Ingesting data into CORE Memory[/bold blue]")

    # Initialize OncoCoreClient
    client = OncoCoreClient.from_env(prefer_mcp=use_mcp)

    if verbose:
        console.print(f"[blue]🎯 Target space: {space_id or 'default'}[/blue]")
        console.print(f"[blue]📧 Source: {source or 'cli'}[/blue]")
        console.print(f"[blue]🧠 Using MCP: {use_mcp}[/blue]")

    # Parse metadata if provided
    parsed_metadata = None
    if metadata:
        import json

        parsed_metadata = json.loads(metadata)
        console.print("[blue]📋 Metadata parsed successfully[/blue]")

    console.print("[blue]📤 Sending data to CORE Memory...[/blue]")

    # Ingest data
    result = client.ingest(
        message=message,
        space_id=space_id,
        source=source or "mcode_cli",
        metadata=parsed_metadata,
    )

    console.print("[green]✅ Data ingested successfully[/green]")

    if verbose and result:
        console.print(f"[blue]📊 Result: {result}[/blue]")

    # Cleanup
    client.close()


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
        # Parse space IDs
        space_list = None
        if space_ids:
            space_list = [s.strip() for s in space_ids.split(",") if s.strip()]

        # Initialize client
        client = OncoCoreClient.from_env(prefer_mcp=use_mcp)

        if verbose:
            console.print(f"[blue]🎯 Search spaces: {space_list or 'all'}[/blue]")
            console.print(f"[blue]📊 Result limit: {limit}[/blue]")
            console.print(f"[blue]🧠 Using MCP: {use_mcp}[/blue]")

        console.print("[blue]🔎 Performing search...[/blue]")

        # Perform search
        results = client.search(query=query, space_ids=space_list, limit=limit)

        episodes = results.episodes

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

                table.add_row(content + ("..." if len(content) == 100 else ""), str(score), source)

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
        # Initialize client
        client = OncoCoreClient.from_env(prefer_mcp=use_mcp)

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
        # Initialize client
        client = OncoCoreClient.from_env(prefer_mcp=use_mcp)

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


@app.command("logs-by-source")
def get_logs_by_source(
    source: str = typer.Argument(..., help="Source identifier to filter logs"),
    space_id: Optional[str] = typer.Option(None, help="Space ID to filter logs"),
    limit: int = typer.Option(100, help="Maximum number of logs to retrieve"),
    use_mcp: bool = typer.Option(True, help="Prefer MCP over direct API"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    List logs for a specific source/user.

    Retrieves and displays all memory logs associated with the specified source identifier.
    """
    console.print(f"[bold blue]📋 Getting logs for source: '{source}'[/bold blue]")

    try:
        # Initialize client
        client = OncoCoreClient.from_env(prefer_mcp=use_mcp)

        if verbose:
            console.print(f"[blue]🎯 Space ID: {space_id or 'all'}[/blue]")
            console.print(f"[blue]📊 Limit: {limit}[/blue]")
            console.print(f"[blue]🧠 Using MCP: {use_mcp}[/blue]")

        console.print("[blue]🔍 Retrieving logs...[/blue]")

        # Get logs by source
        result = client.get_logs_by_source(source=source, space_id=space_id, limit=limit)

        logs = result.get("logs", [])
        total_count = result.get("total_count", 0)

        console.print(f"[green]✅ Found {len(logs)} logs (total: {total_count})[/green]")

        if logs:
            # Display logs in a table
            table = Table(title=f"Logs for source '{source}'")
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Content", style="green", no_wrap=False)
            table.add_column("Created", style="yellow", justify="center")
            table.add_column("Space ID", style="magenta", no_wrap=True)

            for log in logs:
                if isinstance(log, dict):
                    log_id = log.get("id", "unknown")[:16] + "..."
                    content = log.get("content", "")[:50]
                    created_at = log.get("created_at", "unknown")
                    space_id_val = log.get("space_id", "none")[:16] + "..."
                    table.add_row(
                        log_id,
                        content + ("..." if len(content) == 50 else ""),
                        created_at,
                        space_id_val,
                    )

            console.print(table)

            if verbose:
                console.print(f"\n[blue]📊 Full result: {result}[/blue]")
        else:
            console.print("[yellow]⚠️ No logs found for this source[/yellow]")

        # Cleanup
        client.close()

    except Exception as e:
        console.print(f"[red]❌ Failed to get logs: {e}[/red]")
        logger.exception("Get logs by source error")
        raise typer.Exit(1)


@app.command("delete-logs-by-source")
def delete_logs_by_source(
    source: str = typer.Argument(..., help="Source identifier to delete logs for"),
    space_id: Optional[str] = typer.Option(None, help="Space ID to filter logs"),
    no_confirm: bool = typer.Option(False, "--no-confirm", help="Skip confirmation prompt"),
    use_mcp: bool = typer.Option(True, help="Prefer MCP over direct API"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Delete all logs for a specific source/user.

    Permanently removes all memory logs associated with the specified source identifier.
    Requires confirmation unless --no-confirm is used.
    """
    console.print(f"[bold red]🗑️ Deleting logs for source: '{source}'[/bold red]")

    try:
        # Initialize client
        client = OncoCoreClient.from_env(prefer_mcp=use_mcp)

        if verbose:
            console.print(f"[blue]🎯 Space ID: {space_id or 'all'}[/blue]")
            console.print(f"[blue]🧠 Using MCP: {use_mcp}[/blue]")

        # Get count of logs to be deleted for confirmation
        if not no_confirm:
            console.print("[blue]🔍 Checking logs to be deleted...[/blue]")
            result = client.get_logs_by_source(source=source, space_id=space_id, limit=1000)
            log_count = result.get("total_count", 0)

            if log_count == 0:
                console.print("[yellow]⚠️ No logs found for this source[/yellow]")
                client.close()
                return

            console.print(f"[yellow]⚠️ This will delete {log_count} logs permanently[/yellow]")

            # Confirmation prompt
            if not typer.confirm("Are you sure you want to continue?"):
                console.print("[blue]❌ Operation cancelled[/blue]")
                client.close()
                return

        console.print("[red]🗑️ Deleting logs...[/red]")

        # Delete logs by source
        result = client.delete_logs_by_source(source=source, space_id=space_id, confirm=no_confirm)

        deleted_count = result.get("deleted_count", 0)
        console.print(f"[green]✅ Deleted {deleted_count} logs successfully[/green]")

        if verbose:
            console.print(f"[blue]📊 Result: {result}[/blue]")

        # Cleanup
        client.close()

    except Exception as e:
        console.print(f"[red]❌ Failed to delete logs: {e}[/red]")
        logger.exception("Delete logs by source error")
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
        # Initialize client
        client = OncoCoreClient.from_env(prefer_mcp=use_mcp)

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


@app.command("get-ingestion-logs")
def get_ingestion_logs(
    space_id: Optional[str] = typer.Option(None, help="Space ID to filter logs"),
    limit: int = typer.Option(100, help="Maximum number of logs to retrieve"),
    offset: int = typer.Option(0, help="Number of logs to skip"),
    status: Optional[str] = typer.Option(None, help="Filter by status (success, error, pending)"),
    start_date: Optional[str] = typer.Option(None, help="Start date (ISO format)"),
    end_date: Optional[str] = typer.Option(None, help="End date (ISO format)"),
    use_mcp: bool = typer.Option(True, help="Prefer MCP over direct API"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    List ingestion logs with optional filters.

    Retrieves and displays ingestion logs from CORE Memory with support
    for filtering by space, status, date range, and pagination.
    """
    console.print("[bold blue]📋 Getting ingestion logs[/bold blue]")

    try:
        # Initialize client
        client = OncoCoreClient.from_env(prefer_mcp=use_mcp)

        if verbose:
            console.print(f"[blue]🎯 Space ID: {space_id or 'all'}[/blue]")
            console.print(f"[blue]📊 Limit: {limit}, Offset: {offset}[/blue]")
            console.print(f"[blue]🔍 Status: {status or 'all'}[/blue]")
            console.print(
                f"[blue]📅 Date range: {start_date or 'none'} to {end_date or 'none'}[/blue]"
            )
            console.print(f"[blue]🧠 Using MCP: {use_mcp}[/blue]")

        console.print("[blue]🔍 Retrieving ingestion logs...[/blue]")

        # Get ingestion logs
        logs = client.get_ingestion_logs(
            space_id=space_id,
            limit=limit,
            offset=offset,
            status=status,
            start_date=start_date,
            end_date=end_date,
        )

        console.print(f"[green]✅ Found {len(logs)} ingestion logs[/green]")

        if logs:
            # Display logs in a table
            table = Table(title="Ingestion Logs")
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Status", style="green")
            table.add_column("Space ID", style="yellow", no_wrap=True)
            table.add_column("Created", style="magenta")
            table.add_column("Message", style="blue", no_wrap=False)

            for log in logs:
                if isinstance(log, dict):
                    log_id = log.get("id", "unknown")[:16] + "..."
                    log_status = log.get("status", "unknown")
                    space_id_val = log.get("space_id", "none")[:16] + "..."
                    created_at = log.get("created_at", "unknown")
                    message = log.get("message", "")[:50]
                    table.add_row(
                        log_id,
                        log_status,
                        space_id_val,
                        created_at,
                        message + ("..." if len(message) == 50 else ""),
                    )

            console.print(table)

            if verbose:
                console.print(f"\n[blue]📊 Full logs data: {logs}[/blue]")
        else:
            console.print("[yellow]⚠️ No ingestion logs found[/yellow]")

        # Cleanup
        client.close()

    except Exception as e:
        console.print(f"[red]❌ Failed to get ingestion logs: {e}[/red]")
        logger.exception("Get ingestion logs error")
        raise typer.Exit(1)


@app.command("get-log")
def get_log(
    log_id: str = typer.Argument(..., help="Log entry ID to retrieve"),
    use_mcp: bool = typer.Option(True, help="Prefer MCP over direct API"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Get specific log entry by ID.

    Retrieves detailed information about a specific ingestion log entry.
    """
    console.print(f"[bold blue]📋 Getting log entry: {log_id}[/bold blue]")

    try:
        # Initialize client
        client = OncoCoreClient.from_env(prefer_mcp=use_mcp)

        if verbose:
            console.print(f"[blue]🧠 Using MCP: {use_mcp}[/blue]")

        console.print("[blue]🔍 Retrieving log entry...[/blue]")

        # Get specific log
        log = client.get_specific_log(log_id)

        if log:
            console.print("[green]✅ Log entry found[/green]")

            # Display log details
            console.print(f"[cyan]ID:[/cyan] {log.get('id', 'unknown')}")
            console.print(f"[cyan]Status:[/cyan] {log.get('status', 'unknown')}")
            console.print(f"[cyan]Space ID:[/cyan] {log.get('space_id', 'none')}")
            console.print(f"[cyan]Created:[/cyan] {log.get('created_at', 'unknown')}")
            console.print(f"[cyan]Message:[/cyan] {log.get('message', 'none')}")

            if log.get("details"):
                console.print(f"[cyan]Details:[/cyan] {log['details']}")

            if verbose:
                console.print(f"\n[blue]📊 Full log data: {log}[/blue]")
        else:
            console.print("[yellow]⚠️ Log entry not found[/yellow]")

        # Cleanup
        client.close()

    except Exception as e:
        console.print(f"[red]❌ Failed to get log entry: {e}[/red]")
        logger.exception("Get log error")
        raise typer.Exit(1)


@app.command("delete-log")
def delete_log(
    log_id: str = typer.Argument(..., help="Log entry ID to delete"),
    no_confirm: bool = typer.Option(False, "--no-confirm", help="Skip confirmation prompt"),
    use_mcp: bool = typer.Option(True, help="Prefer MCP over direct API"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Delete specific log entry.

    Permanently removes a log entry from CORE Memory.
    Requires confirmation unless --no-confirm is used.
    """
    console.print(f"[bold red]🗑️ Deleting log entry: {log_id}[/bold red]")

    try:
        # Initialize client
        client = OncoCoreClient.from_env(prefer_mcp=use_mcp)

        if verbose:
            console.print(f"[blue]🧠 Using MCP: {use_mcp}[/blue]")

        # Confirmation prompt
        if not no_confirm:
            console.print("[yellow]⚠️ This will permanently delete the log entry[/yellow]")
            if not typer.confirm("Are you sure you want to continue?"):
                console.print("[blue]❌ Operation cancelled[/blue]")
                client.close()
                return

        console.print("[red]🗑️ Deleting log entry...[/red]")

        # Delete log entry
        result = client.delete_log_entry(log_id)

        if result.get("success", False):
            console.print("[green]✅ Log entry deleted successfully[/green]")
        else:
            console.print(
                f"[red]❌ Failed to delete log entry: {result.get('message', 'unknown error')}[/red]"
            )

        if verbose:
            console.print(f"[blue]📊 Result: {result}[/blue]")

        # Cleanup
        client.close()

    except Exception as e:
        console.print(f"[red]❌ Failed to delete log entry: {e}[/red]")
        logger.exception("Delete log error")
        raise typer.Exit(1)


@app.command("add-to-queue")
def add_to_queue(
    data: str = typer.Argument(..., help="Data to add to ingestion queue"),
    space_id: Optional[str] = typer.Option(None, help="Target space ID"),
    priority: str = typer.Option("normal", help="Priority level (low, normal, high)"),
    tags: Optional[str] = typer.Option(None, help="Comma-separated tags"),
    metadata: Optional[str] = typer.Option(None, help="JSON metadata string"),
    use_mcp: bool = typer.Option(True, help="Prefer MCP over direct API"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Add data to ingestion queue.

    Queues data for processing and ingestion into CORE Memory with
    optional priority, tags, and metadata.
    """
    console.print("[bold blue]📥 Adding data to ingestion queue[/bold blue]")

    try:
        # Initialize client
        client = OncoCoreClient.from_env(prefer_mcp=use_mcp)

        if verbose:
            console.print(f"[blue]🎯 Space ID: {space_id or 'default'}[/blue]")
            console.print(f"[blue]⚡ Priority: {priority}[/blue]")
            console.print(f"[blue]🏷️ Tags: {tags or 'none'}[/blue]")
            console.print(f"[blue]🧠 Using MCP: {use_mcp}[/blue]")

        # Parse tags
        tags_list = None
        if tags:
            tags_list = [tag.strip() for tag in tags.split(",") if tag.strip()]

        # Parse metadata
        parsed_metadata = None
        if metadata:
            try:
                import json

                parsed_metadata = json.loads(metadata)
                console.print("[blue]📋 Metadata parsed successfully[/blue]")
            except json.JSONDecodeError as e:
                console.print(f"[red]❌ Invalid metadata JSON: {e}[/red]")
                raise typer.Exit(1)

        console.print("[blue]📤 Adding to queue...[/blue]")

        # Add to queue
        result = client.add_data_to_ingestion_queue(
            data=data,
            space_id=space_id,
            priority=priority,
            tags=tags_list,
            metadata=parsed_metadata,
        )

        console.print("[green]✅ Data added to ingestion queue[/green]")

        if result.get("queue_id"):
            console.print(f"[green]🆔 Queue ID: {result['queue_id']}[/green]")

        if verbose:
            console.print(f"[blue]📊 Result: {result}[/blue]")

        # Cleanup
        client.close()

    except Exception as e:
        console.print(f"[red]❌ Failed to add to queue: {e}[/red]")
        logger.exception("Add to queue error")
        raise typer.Exit(1)


@app.command("episode-facts")
def get_episode_facts(
    episode_id: str = typer.Argument(..., help="Episode ID to get facts for"),
    limit: int = typer.Option(100, help="Maximum number of facts to retrieve"),
    offset: int = typer.Option(0, help="Number of facts to skip"),
    include_metadata: bool = typer.Option(True, help="Include metadata in response"),
    use_mcp: bool = typer.Option(True, help="Prefer MCP over direct API"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Get episode facts from CORE Memory.

    Retrieves all facts associated with a specific episode ID with
    optional pagination and metadata inclusion.
    """
    console.print(f"[bold blue]📊 Getting episode facts for: {episode_id}[/bold blue]")

    try:
        # Initialize client
        client = OncoCoreClient.from_env(prefer_mcp=use_mcp)

        if verbose:
            console.print(f"[blue]📊 Limit: {limit}, Offset: {offset}[/blue]")
            console.print(f"[blue]📋 Include metadata: {include_metadata}[/blue]")
            console.print(f"[blue]🧠 Using MCP: {use_mcp}[/blue]")

        console.print("[blue]🔍 Retrieving episode facts...[/blue]")

        # Get episode facts
        facts = client.get_episode_facts(
            episode_id=episode_id, limit=limit, offset=offset, include_metadata=include_metadata
        )

        console.print(f"[green]✅ Found {len(facts)} episode facts[/green]")

        if facts:
            # Display facts in a table
            table = Table(title=f"Episode Facts for {episode_id}")
            table.add_column("Fact", style="cyan", no_wrap=False)
            if include_metadata:
                table.add_column("Metadata", style="yellow", no_wrap=False)

            for fact in facts:
                if isinstance(fact, dict):
                    fact_content = fact.get("content", "")
                    metadata_str = ""
                    if include_metadata and fact.get("metadata"):
                        metadata_str = str(fact["metadata"])[:50] + "..."
                    table.add_row(fact_content, metadata_str)

            console.print(table)

            if verbose:
                console.print(f"\n[blue]📊 Full facts data: {facts}[/blue]")
        else:
            console.print("[yellow]⚠️ No episode facts found[/yellow]")

        # Cleanup
        client.close()

    except Exception as e:
        console.print(f"[red]❌ Failed to get episode facts: {e}[/red]")
        logger.exception("Get episode facts error")
        raise typer.Exit(1)


@app.command("ingestion-status")
def check_ingestion_status(
    run_id: Optional[str] = typer.Option(None, help="Specific run ID to check"),
    space_id: Optional[str] = typer.Option(None, help="Space ID to check status for"),
    use_mcp: bool = typer.Option(True, help="Prefer MCP over direct API"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Check ingestion status.

    Retrieves the current status of data ingestion processing,
    optionally filtered by run ID or space ID.
    """
    console.print("[bold blue]📊 Checking ingestion status[/bold blue]")

    try:
        # Initialize client
        client = OncoCoreClient.from_env(prefer_mcp=use_mcp)

        if verbose:
            console.print(f"[blue]🏃 Run ID: {run_id or 'latest'}[/blue]")
            console.print(f"[blue]🎯 Space ID: {space_id or 'all'}[/blue]")
            console.print(f"[blue]🧠 Using MCP: {use_mcp}[/blue]")

        console.print("[blue]🔍 Checking status...[/blue]")

        # Check ingestion status
        status = client.check_ingestion_status(run_id=run_id, space_id=space_id)

        console.print("[green]✅ Ingestion status retrieved[/green]")

        # Display status information
        console.print(f"[cyan]Status:[/cyan] {status.get('status', 'unknown')}")
        console.print(f"[cyan]Progress:[/cyan] {status.get('progress', 'N/A')}")
        console.print(f"[cyan]Total Items:[/cyan] {status.get('total_items', 'N/A')}")
        console.print(f"[cyan]Processed:[/cyan] {status.get('processed', 'N/A')}")
        console.print(f"[cyan]Failed:[/cyan] {status.get('failed', 'N/A')}")

        if status.get("current_run_id"):
            console.print(f"[cyan]Current Run ID:[/cyan] {status['current_run_id']}")

        if status.get("last_updated"):
            console.print(f"[cyan]Last Updated:[/cyan] {status['last_updated']}")

        if verbose:
            console.print(f"\n[blue]📊 Full status data: {status}[/blue]")

        # Cleanup
        client.close()

    except Exception as e:
        console.print(f"[red]❌ Failed to check ingestion status: {e}[/red]")
        logger.exception("Check ingestion status error")
        raise typer.Exit(1)


@app.command("space-details")
def get_space_details(
    space_id: str = typer.Argument(..., help="Space ID to get details for"),
    include_stats: bool = typer.Option(True, help="Include statistics"),
    include_metadata: bool = typer.Option(True, help="Include metadata"),
    use_mcp: bool = typer.Option(True, help="Prefer MCP over direct API"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Get detailed information about a memory space.

    Retrieves comprehensive details about a specific CORE Memory space
    including statistics and metadata.
    """
    console.print(f"[bold blue]📂 Getting space details for: {space_id}[/bold blue]")

    try:
        # Initialize client
        client = OncoCoreClient.from_env(prefer_mcp=use_mcp)

        if verbose:
            console.print(f"[blue]📊 Include stats: {include_stats}[/blue]")
            console.print(f"[blue]📋 Include metadata: {include_metadata}[/blue]")
            console.print(f"[blue]🧠 Using MCP: {use_mcp}[/blue]")

        console.print("[blue]🔍 Retrieving space details...[/blue]")

        # Get space details
        details = client.get_space_details(
            space_id=space_id, include_stats=include_stats, include_metadata=include_metadata
        )

        if details:
            console.print("[green]✅ Space details retrieved[/green]")

            # Display space information
            console.print(f"[cyan]ID:[/cyan] {details.get('id', 'unknown')}")
            console.print(f"[cyan]Name:[/cyan] {details.get('name', 'unnamed')}")
            console.print(f"[cyan]Description:[/cyan] {details.get('description', 'none')}")
            console.print(f"[cyan]Created:[/cyan] {details.get('created_at', 'unknown')}")
            console.print(f"[cyan]Updated:[/cyan] {details.get('updated_at', 'unknown')}")

            if include_stats and details.get("stats"):
                stats = details["stats"]
                console.print("\n[cyan]Statistics:[/cyan]")
                console.print(f"  Episodes: {stats.get('episode_count', 0)}")
                console.print(f"  Facts: {stats.get('fact_count', 0)}")
                console.print(f"  Storage Size: {stats.get('storage_size', 'unknown')}")

            if include_metadata and details.get("metadata"):
                console.print(f"\n[cyan]Metadata:[/cyan] {details['metadata']}")

            if verbose:
                console.print(f"\n[blue]📊 Full details: {details}[/blue]")
        else:
            console.print("[yellow]⚠️ Space not found[/yellow]")

        # Cleanup
        client.close()

    except Exception as e:
        console.print(f"[red]❌ Failed to get space details: {e}[/red]")
        logger.exception("Get space details error")
        raise typer.Exit(1)


@app.command("update-space")
def update_space(
    space_id: str = typer.Argument(..., help="Space ID to update"),
    name: Optional[str] = typer.Option(None, help="New name for the space"),
    description: Optional[str] = typer.Option(None, help="New description for the space"),
    metadata: Optional[str] = typer.Option(None, help="JSON metadata string"),
    use_mcp: bool = typer.Option(True, help="Prefer MCP over direct API"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Update properties of an existing memory space.

    Modifies the name, description, or metadata of a CORE Memory space.
    """
    console.print(f"[bold blue]📝 Updating space: {space_id}[/bold blue]")

    try:
        # Initialize client
        client = OncoCoreClient.from_env(prefer_mcp=use_mcp)

        if verbose:
            console.print(f"[blue]📝 Name: {name or 'unchanged'}[/blue]")
            console.print(f"[blue]📋 Description: {description or 'unchanged'}[/blue]")
            console.print(f"[blue]🧠 Using MCP: {use_mcp}[/blue]")

        # Parse metadata
        parsed_metadata = None
        if metadata:
            try:
                import json

                parsed_metadata = json.loads(metadata)
                console.print("[blue]📋 Metadata parsed successfully[/blue]")
            except json.JSONDecodeError as e:
                console.print(f"[red]❌ Invalid metadata JSON: {e}[/red]")
                raise typer.Exit(1)

        console.print("[blue]📤 Updating space...[/blue]")

        # Update space
        result = client.update_space(
            space_id=space_id, name=name, description=description, metadata=parsed_metadata
        )

        if result.get("success", False):
            console.print("[green]✅ Space updated successfully[/green]")
        else:
            console.print(
                f"[red]❌ Failed to update space: {result.get('message', 'unknown error')}[/red]"
            )

        if verbose:
            console.print(f"[blue]📊 Result: {result}[/blue]")

        # Cleanup
        client.close()

    except Exception as e:
        console.print(f"[red]❌ Failed to update space: {e}[/red]")
        logger.exception("Update space error")
        raise typer.Exit(1)


@app.command("delete-space")
def delete_space(
    space_id: str = typer.Argument(..., help="Space ID to delete"),
    no_confirm: bool = typer.Option(False, "--no-confirm", help="Skip confirmation prompt"),
    use_mcp: bool = typer.Option(True, help="Prefer MCP over direct API"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Delete a memory space.

    Permanently removes a CORE Memory space and all its contents.
    Requires confirmation unless --no-confirm is used.
    """
    console.print(f"[bold red]🗑️ Deleting space: {space_id}[/bold red]")

    try:
        # Initialize client
        client = OncoCoreClient.from_env(prefer_mcp=use_mcp)

        if verbose:
            console.print(f"[blue]🧠 Using MCP: {use_mcp}[/blue]")

        # Confirmation prompt
        if not no_confirm:
            console.print(
                "[yellow]⚠️ This will permanently delete the space and all its contents[/yellow]"
            )
            if not typer.confirm("Are you sure you want to continue?"):
                console.print("[blue]❌ Operation cancelled[/blue]")
                client.close()
                return

        console.print("[red]🗑️ Deleting space...[/red]")

        # Delete space
        result = client.delete_space(space_id=space_id, confirm=no_confirm)

        if result.get("success", False):
            console.print("[green]✅ Space deleted successfully[/green]")
        else:
            console.print(
                f"[red]❌ Failed to delete space: {result.get('message', 'unknown error')}[/red]"
            )

        if verbose:
            console.print(f"[blue]📊 Result: {result}[/blue]")

        # Cleanup
        client.close()

    except Exception as e:
        console.print(f"[red]❌ Failed to delete space: {e}[/red]")
        logger.exception("Delete space error")
        raise typer.Exit(1)


@app.command("register-webhook")
def register_webhook(
    url: str = typer.Argument(..., help="Webhook URL"),
    events: Optional[str] = typer.Option(None, help="Comma-separated event types"),
    secret: str = typer.Option("", help="Webhook secret for verification"),
    use_mcp: bool = typer.Option(True, help="Prefer MCP over direct API"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Register a new webhook.

    Creates a webhook endpoint for receiving CORE Memory events
    with optional event filtering and security.
    """
    console.print(f"[bold blue]🔗 Registering webhook: {url}[/bold blue]")

    try:
        # Initialize client
        client = OncoCoreClient.from_env(prefer_mcp=use_mcp)

        if verbose:
            console.print(f"[blue]📡 Events: {events or 'all'}[/blue]")
            console.print(f"[blue]🔐 Secret: {'configured' if secret else 'none'}[/blue]")
            console.print(f"[blue]🧠 Using MCP: {use_mcp}[/blue]")

        # Parse events
        events_list = None
        if events:
            events_list = [event.strip() for event in events.split(",") if event.strip()]

        console.print("[blue]📤 Registering webhook...[/blue]")

        # Register webhook
        result = client.register_webhook(url=url, events=events_list, secret=secret)

        console.print("[green]✅ Webhook registered successfully[/green]")

        if result.get("webhook_id"):
            console.print(f"[green]🆔 Webhook ID: {result['webhook_id']}[/green]")

        if verbose:
            console.print(f"[blue]📊 Result: {result}[/blue]")

        # Cleanup
        client.close()

    except Exception as e:
        console.print(f"[red]❌ Failed to register webhook: {e}[/red]")
        logger.exception("Register webhook error")
        raise typer.Exit(1)


@app.command("list-webhooks")
def list_webhooks(
    space_id: Optional[str] = typer.Option(None, help="Space ID to filter webhooks"),
    active: Optional[bool] = typer.Option(None, help="Filter by active status"),
    limit: int = typer.Option(100, help="Maximum number of webhooks to retrieve"),
    offset: int = typer.Option(0, help="Number of webhooks to skip"),
    use_mcp: bool = typer.Option(True, help="Prefer MCP over direct API"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    List all webhooks.

    Retrieves and displays all registered webhooks with optional filtering.
    """
    console.print("[bold blue]📋 Listing webhooks[/bold blue]")

    try:
        # Initialize client
        client = OncoCoreClient.from_env(prefer_mcp=use_mcp)

        if verbose:
            console.print(f"[blue]🎯 Space ID: {space_id or 'all'}[/blue]")
            console.print(f"[blue]📊 Limit: {limit}, Offset: {offset}[/blue]")
            console.print(f"[blue]🔄 Active: {active if active is not None else 'all'}[/blue]")
            console.print(f"[blue]🧠 Using MCP: {use_mcp}[/blue]")

        console.print("[blue]🔍 Retrieving webhooks...[/blue]")

        # List webhooks
        webhooks = client.list_webhooks(
            space_id=space_id, active=active, limit=limit, offset=offset
        )

        console.print(f"[green]✅ Found {len(webhooks)} webhooks[/green]")

        if webhooks:
            # Display webhooks in a table
            table = Table(title="Webhooks")
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("URL", style="green", no_wrap=False)
            table.add_column("Events", style="yellow", no_wrap=False)
            table.add_column("Active", style="magenta", justify="center")
            table.add_column("Created", style="blue")

            for webhook in webhooks:
                if isinstance(webhook, dict):
                    webhook_id = webhook.get("id", "unknown")[:16] + "..."
                    url = webhook.get("url", "")
                    events = (
                        ", ".join(webhook.get("events", [])) if webhook.get("events") else "all"
                    )
                    is_active = "✅" if webhook.get("active", False) else "❌"
                    created_at = webhook.get("created_at", "unknown")
                    table.add_row(webhook_id, url, events, is_active, created_at)

            console.print(table)

            if verbose:
                console.print(f"\n[blue]📊 Full webhooks data: {webhooks}[/blue]")
        else:
            console.print("[yellow]⚠️ No webhooks found[/yellow]")

        # Cleanup
        client.close()

    except Exception as e:
        console.print(f"[red]❌ Failed to list webhooks: {e}[/red]")
        logger.exception("List webhooks error")
        raise typer.Exit(1)


@app.command("get-webhook")
def get_webhook(
    webhook_id: str = typer.Argument(..., help="Webhook ID to retrieve"),
    use_mcp: bool = typer.Option(True, help="Prefer MCP over direct API"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Get webhook details.

    Retrieves detailed information about a specific webhook.
    """
    console.print(f"[bold blue]📋 Getting webhook: {webhook_id}[/bold blue]")

    try:
        # Initialize client
        client = OncoCoreClient.from_env(prefer_mcp=use_mcp)

        if verbose:
            console.print(f"[blue]🧠 Using MCP: {use_mcp}[/blue]")

        console.print("[blue]🔍 Retrieving webhook details...[/blue]")

        # Get webhook
        webhook = client.get_webhook(webhook_id)

        if webhook:
            console.print("[green]✅ Webhook details retrieved[/green]")

            # Display webhook information
            console.print(f"[cyan]ID:[/cyan] {webhook.get('id', 'unknown')}")
            console.print(f"[cyan]URL:[/cyan] {webhook.get('url', 'unknown')}")
            console.print(
                f"[cyan]Events:[/cyan] {', '.join(webhook.get('events', [])) if webhook.get('events') else 'all'}"
            )
            console.print(f"[cyan]Active:[/cyan] {webhook.get('active', False)}")
            console.print(f"[cyan]Created:[/cyan] {webhook.get('created_at', 'unknown')}")
            console.print(f"[cyan]Updated:[/cyan] {webhook.get('updated_at', 'unknown')}")

            if webhook.get("secret"):
                console.print("[cyan]Secret:[/cyan] configured")

            if verbose:
                console.print(f"\n[blue]📊 Full webhook data: {webhook}[/blue]")
        else:
            console.print("[yellow]⚠️ Webhook not found[/yellow]")

        # Cleanup
        client.close()

    except Exception as e:
        console.print(f"[red]❌ Failed to get webhook: {e}[/red]")
        logger.exception("Get webhook error")
        raise typer.Exit(1)


@app.command("update-webhook")
def update_webhook(
    webhook_id: str = typer.Argument(..., help="Webhook ID to update"),
    url: str = typer.Option(..., help="New webhook URL", prompt=True),
    events: str = typer.Option(..., help="Comma-separated event types", prompt=True),
    secret: str = typer.Option("", help="New webhook secret"),
    active: bool = typer.Option(True, help="Whether webhook is active"),
    use_mcp: bool = typer.Option(True, help="Prefer MCP over direct API"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Update webhook properties.

    Modifies the URL, events, secret, or active status of an existing webhook.
    """
    console.print(f"[bold blue]📝 Updating webhook: {webhook_id}[/bold blue]")

    try:
        # Initialize client
        client = OncoCoreClient.from_env(prefer_mcp=use_mcp)

        if verbose:
            console.print(f"[blue]📡 URL: {url}[/blue]")
            console.print(f"[blue]📋 Events: {events}[/blue]")
            console.print(f"[blue]🔐 Secret: {'configured' if secret else 'none'}[/blue]")
            console.print(f"[blue]🔄 Active: {active}[/blue]")
            console.print(f"[blue]🧠 Using MCP: {use_mcp}[/blue]")

        # Parse events
        events_list = [event.strip() for event in events.split(",") if event.strip()]

        console.print("[blue]📤 Updating webhook...[/blue]")

        # Update webhook
        result = client.update_webhook(
            webhook_id=webhook_id, url=url, events=events_list, secret=secret, active=active
        )

        if result.get("success", False):
            console.print("[green]✅ Webhook updated successfully[/green]")
        else:
            console.print(
                f"[red]❌ Failed to update webhook: {result.get('message', 'unknown error')}[/red]"
            )

        if verbose:
            console.print(f"[blue]📊 Result: {result}[/blue]")

        # Cleanup
        client.close()

    except Exception as e:
        console.print(f"[red]❌ Failed to update webhook: {e}[/red]")
        logger.exception("Update webhook error")
        raise typer.Exit(1)


@app.command("delete-webhook")
def delete_webhook(
    webhook_id: str = typer.Argument(..., help="Webhook ID to delete"),
    no_confirm: bool = typer.Option(False, "--no-confirm", help="Skip confirmation prompt"),
    use_mcp: bool = typer.Option(True, help="Prefer MCP over direct API"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Delete a webhook.

    Permanently removes a webhook registration.
    Requires confirmation unless --no-confirm is used.
    """
    console.print(f"[bold red]🗑️ Deleting webhook: {webhook_id}[/bold red]")

    try:
        # Initialize client
        client = OncoCoreClient.from_env(prefer_mcp=use_mcp)

        if verbose:
            console.print(f"[blue]🧠 Using MCP: {use_mcp}[/blue]")

        # Confirmation prompt
        if not no_confirm:
            console.print("[yellow]⚠️ This will permanently delete the webhook[/yellow]")
            if not typer.confirm("Are you sure you want to continue?"):
                console.print("[blue]❌ Operation cancelled[/blue]")
                client.close()
                return

        console.print("[red]🗑️ Deleting webhook...[/red]")

        # Delete webhook
        result = client.delete_webhook(webhook_id=webhook_id, confirm=no_confirm)

        if result.get("success", False):
            console.print("[green]✅ Webhook deleted successfully[/green]")
        else:
            console.print(
                f"[red]❌ Failed to delete webhook: {result.get('message', 'unknown error')}[/red]"
            )

        if verbose:
            console.print(f"[blue]📊 Result: {result}[/blue]")

        # Cleanup
        client.close()

    except Exception as e:
        console.print(f"[red]❌ Failed to delete webhook: {e}[/red]")
        logger.exception("Delete webhook error")
        raise typer.Exit(1)
