"""
Log-related CLI commands.
"""

import click
from typing import Optional

from .common import create_client, format_json_output
from heysol import HeySolError


@click.group()
def logs():
    """Log operations."""
    pass


@logs.command("list")
@click.option("--space-id", help="Space ID")
@click.option("--source", help="Filter by source")
@click.option("--limit", type=int, default=100, help="Result limit")
@click.option("--offset", type=int, default=0, help="Result offset")
@click.option("--status", help="Filter by status")
@click.option("--start-date", help="Start date filter")
@click.option("--end-date", help="End date filter")
@click.option("--api-key", help="HeySol API key (overrides environment variable)")
@click.option("--base-url", help="Base URL for API (overrides default)")
@click.option("--pretty", is_flag=True, help="Pretty print JSON output")
@click.option("--skip-mcp", is_flag=True, help="Skip MCP initialization")
def logs_list(space_id, source, limit, offset, status, start_date, end_date, api_key, base_url, pretty, skip_mcp):
    """Get ingestion logs."""
    try:
        client = create_client(api_key=api_key, base_url=base_url, skip_mcp=skip_mcp)
        logs_list = client.get_ingestion_logs(
            space_id=space_id,
            limit=limit,
            offset=offset,
            status=status,
            start_date=start_date,
            end_date=end_date
        )

        # Filter by source if specified
        if source:
            logs_list = [log for log in logs_list if log.get('source') == source]

        click.echo(format_json_output(logs_list, pretty))
    except (HeySolError, click.ClickException) as e:
        raise click.ClickException(str(e))
    finally:
        if 'client' in locals():
            client.close()


@logs.command("delete")
@click.argument("log_id")
@click.option("--confirm", is_flag=True, help="Confirm deletion (required)")
@click.option("--api-key", help="HeySol API key (overrides environment variable)")
@click.option("--base-url", help="Base URL for API (overrides default)")
@click.option("--pretty", is_flag=True, help="Pretty print JSON output")
@click.option("--skip-mcp", is_flag=True, help="Skip MCP initialization")
def logs_delete(log_id, confirm, api_key, base_url, pretty, skip_mcp):
    """Delete a specific log entry by ID."""
    if not confirm:
        raise click.ClickException("Deletion requires --confirm flag for safety")

    try:
        client = create_client(api_key=api_key, base_url=base_url, skip_mcp=skip_mcp)
        result = client.delete_log_entry(log_id=log_id)
        click.echo(format_json_output(result, pretty))
    except (HeySolError, click.ClickException) as e:
        raise click.ClickException(str(e))
    finally:
        if 'client' in locals():
            client.close()


@logs.command("delete-by-source")
@click.argument("source")
@click.option("--space-id", help="Space ID to filter logs")
@click.option("--confirm", is_flag=True, help="Confirm deletion (required)")
@click.option("--limit", type=int, default=1000, help="Maximum logs to process")
@click.option("--api-key", help="HeySol API key (overrides environment variable)")
@click.option("--base-url", help="Base URL for API (overrides default)")
@click.option("--pretty", is_flag=True, help="Pretty print JSON output")
@click.option("--skip-mcp", is_flag=True, help="Skip MCP initialization")
def logs_delete_by_source(source, space_id, confirm, limit, api_key, base_url, pretty, skip_mcp):
    """Delete logs by source (batch operation)."""
    if not confirm:
        raise click.ClickException("Deletion requires --confirm flag for safety")

    try:
        client = create_client(api_key=api_key, base_url=base_url, skip_mcp=skip_mcp)

        # Get all logs and filter by source
        click.echo(f"Fetching logs to find source '{source}'...")
        logs_list = client.get_ingestion_logs(
            space_id=space_id,
            limit=limit,
            offset=0
        )

        # Filter logs by source
        source_logs = [log for log in logs_list if log.get('source') == source]

        if not source_logs:
            click.echo(format_json_output({"message": f"No logs found for source '{source}'", "deleted": 0}, pretty))
            return

        click.echo(f"Found {len(source_logs)} logs for source '{source}'. Deleting...")

        # Delete each log individually
        deleted_count = 0
        failed_count = 0

        for log in source_logs:
            try:
                client.delete_log_entry(log_id=log['id'])
                deleted_count += 1
                click.echo(f"Deleted log {log['id']}")
            except Exception as e:
                failed_count += 1
                click.echo(f"Failed to delete log {log['id']}: {str(e)}")

        result = {
            "source": source,
            "total_found": len(source_logs),
            "deleted": deleted_count,
            "failed": failed_count,
            "message": f"Batch deletion completed for source '{source}'"
        }

        click.echo(format_json_output(result, pretty))
    except (HeySolError, click.ClickException) as e:
        raise click.ClickException(str(e))
    finally:
        if 'client' in locals():
            client.close()


@logs.command("get")
@click.argument("log_id")
@click.option("--api-key", help="HeySol API key (overrides environment variable)")
@click.option("--base-url", help="Base URL for API (overrides default)")
@click.option("--pretty", is_flag=True, help="Pretty print JSON output")
@click.option("--skip-mcp", is_flag=True, help="Skip MCP initialization")
def logs_get(log_id, api_key, base_url, pretty, skip_mcp):
    """Get a specific log by ID."""
    try:
        client = create_client(api_key=api_key, base_url=base_url, skip_mcp=skip_mcp)
        result = client.get_specific_log(log_id=log_id)
        click.echo(format_json_output(result, pretty))
    except (HeySolError, click.ClickException) as e:
        raise click.ClickException(str(e))
    finally:
        if 'client' in locals():
            client.close()


@logs.command("get-by-source")
@click.argument("source")
@click.option("--space-id", help="Space ID to filter logs")
@click.option("--limit", type=int, default=100, help="Result limit")
@click.option("--offset", type=int, default=0, help="Result offset")
@click.option("--status", help="Filter by status")
@click.option("--start-date", help="Start date filter")
@click.option("--end-date", help="End date filter")
@click.option("--api-key", help="HeySol API key (overrides environment variable)")
@click.option("--base-url", help="Base URL for API (overrides default)")
@click.option("--pretty", is_flag=True, help="Pretty print JSON output")
@click.option("--skip-mcp", is_flag=True, help="Skip MCP initialization")
def logs_get_by_source(source, space_id, limit, offset, status, start_date, end_date, api_key, base_url, pretty, skip_mcp):
    """Get logs filtered by source."""
    try:
        client = create_client(api_key=api_key, base_url=base_url, skip_mcp=skip_mcp)
        logs_list = client.get_ingestion_logs(
            space_id=space_id,
            limit=limit,
            offset=offset,
            status=status,
            start_date=start_date,
            end_date=end_date
        )

        # Filter by source
        filtered_logs = [log for log in logs_list if log.get('source') == source]

        click.echo(format_json_output(filtered_logs, pretty))
    except (HeySolError, click.ClickException) as e:
        raise click.ClickException(str(e))
    finally:
        if 'client' in locals():
            client.close()


@logs.command("sources")
@click.option("--space-id", help="Space ID to filter logs")
@click.option("--limit", type=int, default=1000, help="Result limit for source extraction")
@click.option("--offset", type=int, default=0, help="Result offset")
@click.option("--status", help="Filter by status")
@click.option("--start-date", help="Start date filter")
@click.option("--end-date", help="End date filter")
@click.option("--api-key", help="HeySol API key (overrides environment variable)")
@click.option("--base-url", help="Base URL for API (overrides default)")
@click.option("--pretty", is_flag=True, help="Pretty print JSON output")
@click.option("--skip-mcp", is_flag=True, help="Skip MCP initialization")
def logs_sources(space_id, limit, offset, status, start_date, end_date, api_key, base_url, pretty, skip_mcp):
    """List unique sources from memory logs."""
    try:
        client = create_client(api_key=api_key, base_url=base_url, skip_mcp=skip_mcp)
        logs_list = client.get_ingestion_logs(
            space_id=space_id,
            limit=limit,
            offset=offset,
            status=status,
            start_date=start_date,
            end_date=end_date
        )

        # Extract unique sources
        sources = set()
        for log in logs_list:
            if 'source' in log:
                sources.add(log['source'])

        unique_sources = sorted(list(sources))
        result = {"sources": unique_sources, "count": len(unique_sources)}

        click.echo(format_json_output(result, pretty))
    except (HeySolError, click.ClickException) as e:
        raise click.ClickException(str(e))
    finally:
        if 'client' in locals():
            client.close()


if __name__ == "__main__":
    logs()