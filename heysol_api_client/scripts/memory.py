"""
Memory-related CLI commands.
"""

import click
from typing import Optional

from .common import create_client, format_json_output
from heysol import HeySolError


@click.group()
def memory():
    """Memory operations."""
    pass


@memory.command("ingest")
@click.argument("message", required=False)
@click.option("--file", type=click.Path(exists=True), help="File containing message to ingest")
@click.option("--space-id", help="Space ID")
@click.option("--session-id", help="Session ID")
@click.option("--api-key", help="HeySol API key (overrides environment variable)")
@click.option("--base-url", help="Base URL for API (overrides default)")
@click.option("--pretty", is_flag=True, help="Pretty print JSON output")
@click.option("--skip-mcp", is_flag=True, help="Skip MCP initialization")
def memory_ingest(message, file, space_id, session_id, api_key, base_url, pretty, skip_mcp):
    """Ingest data into memory."""
    if not message and not file:
        raise click.ClickException("Message or file is required")

    final_message = message
    if file:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                final_message = f.read()
        except Exception as e:
            raise click.ClickException(f"Error reading file: {e}")

    try:
        client = create_client(api_key=api_key, base_url=base_url, skip_mcp=skip_mcp)
        result = client.ingest(
            message=final_message,
            space_id=space_id,
            session_id=session_id
        )
        click.echo(format_json_output(result, pretty))
    except (HeySolError, click.ClickException) as e:
        raise click.ClickException(str(e))
    finally:
        if 'client' in locals():
            client.close()


@memory.command("search")
@click.argument("query")
@click.option("--space-id", help="Space ID")
@click.option("--limit", type=int, default=10, help="Result limit")
@click.option("--include-invalidated", is_flag=True, help="Include invalidated results")
@click.option("--api-key", help="HeySol API key (overrides environment variable)")
@click.option("--base-url", help="Base URL for API (overrides default)")
@click.option("--pretty", is_flag=True, help="Pretty print JSON output")
@click.option("--skip-mcp", is_flag=True, help="Skip MCP initialization")
def memory_search(query, space_id, limit, include_invalidated, api_key, base_url, pretty, skip_mcp):
    """Search memory."""
    try:
        client = create_client(api_key=api_key, base_url=base_url, skip_mcp=skip_mcp)
        result = client.search(
            query=query,
            space_ids=[space_id] if space_id else None,
            limit=limit,
            include_invalidated=include_invalidated
        )
        click.echo(format_json_output(result, pretty))
    except (HeySolError, click.ClickException) as e:
        raise click.ClickException(str(e))
    finally:
        if 'client' in locals():
            client.close()


@memory.command("search-graph")
@click.argument("query")
@click.option("--space-id", help="Space ID")
@click.option("--limit", type=int, default=10, help="Result limit")
@click.option("--depth", type=int, default=2, help="Graph search depth")
@click.option("--include-metadata", is_flag=True, default=True, help="Include metadata")
@click.option("--api-key", help="HeySol API key (overrides environment variable)")
@click.option("--base-url", help="Base URL for API (overrides default)")
@click.option("--pretty", is_flag=True, help="Pretty print JSON output")
@click.option("--skip-mcp", is_flag=True, help="Skip MCP initialization")
def memory_search_graph(query, space_id, limit, depth, include_metadata, api_key, base_url, pretty, skip_mcp):
    """Search knowledge graph."""
    try:
        client = create_client(api_key=api_key, base_url=base_url, skip_mcp=skip_mcp)
        result = client.search_knowledge_graph(
            query=query,
            space_id=space_id,
            limit=limit,
            depth=depth,
            include_metadata=include_metadata
        )
        click.echo(format_json_output(result, pretty))
    except (HeySolError, click.ClickException) as e:
        raise click.ClickException(str(e))
    finally:
        if 'client' in locals():
            client.close()


@memory.command("queue")
@click.argument("data", required=False)
@click.option("--file", type=click.Path(exists=True), help="File containing data to queue")
@click.option("--space-id", help="Space ID")
@click.option("--priority", type=click.Choice(["low", "normal", "high"]), default="normal", help="Priority level")
@click.option("--tags", multiple=True, help="Tags (can specify multiple)")
@click.option("--metadata", help="JSON metadata string")
@click.option("--api-key", help="HeySol API key (overrides environment variable)")
@click.option("--base-url", help="Base URL for API (overrides default)")
@click.option("--pretty", is_flag=True, help="Pretty print JSON output")
@click.option("--skip-mcp", is_flag=True, help="Skip MCP initialization")
def memory_queue(data, file, space_id, priority, tags, metadata, api_key, base_url, pretty, skip_mcp):
    """Add data to ingestion queue."""
    if not data and not file:
        raise click.ClickException("Data or file is required")

    final_data = data
    if file:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                final_data = f.read()
        except Exception as e:
            raise click.ClickException(f"Error reading file: {e}")

    parsed_metadata = None
    if metadata:
        try:
            import json
            parsed_metadata = json.loads(metadata)
        except json.JSONDecodeError as e:
            raise click.ClickException(f"Invalid JSON metadata: {e}")

    try:
        client = create_client(api_key=api_key, base_url=base_url, skip_mcp=skip_mcp)
        result = client.add_data_to_ingestion_queue(
            data=final_data,
            space_id=space_id,
            priority=priority,
            tags=list(tags) if tags else None,
            metadata=parsed_metadata
        )
        click.echo(format_json_output(result, pretty))
    except (HeySolError, click.ClickException) as e:
        raise click.ClickException(str(e))
    finally:
        if 'client' in locals():
            client.close()


@memory.command("episode")
@click.argument("episode_id")
@click.option("--limit", type=int, default=100, help="Result limit")
@click.option("--offset", type=int, default=0, help="Result offset")
@click.option("--include-metadata", is_flag=True, default=True, help="Include metadata")
@click.option("--api-key", help="HeySol API key (overrides environment variable)")
@click.option("--base-url", help="Base URL for API (overrides default)")
@click.option("--pretty", is_flag=True, help="Pretty print JSON output")
@click.option("--skip-mcp", is_flag=True, help="Skip MCP initialization")
def memory_episode(episode_id, limit, offset, include_metadata, api_key, base_url, pretty, skip_mcp):
    """Get episode facts."""
    try:
        client = create_client(api_key=api_key, base_url=base_url, skip_mcp=skip_mcp)
        result = client.get_episode_facts(
            episode_id=episode_id,
            limit=limit,
            offset=offset,
            include_metadata=include_metadata
        )
        click.echo(format_json_output(result, pretty))
    except (HeySolError, click.ClickException) as e:
        raise click.ClickException(str(e))
    finally:
        if 'client' in locals():
            client.close()


if __name__ == "__main__":
    memory()