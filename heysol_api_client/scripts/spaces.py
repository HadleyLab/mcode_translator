"""
Space-related CLI commands.
"""

import click
from typing import Optional

from .common import create_client, format_json_output
from heysol import HeySolError


@click.group()
def spaces():
    """Memory space operations."""
    pass


@spaces.command("list")
@click.option("--api-key", help="HeySol API key (overrides environment variable)")
@click.option("--base-url", help="Base URL for API (overrides default)")
@click.option("--pretty", is_flag=True, help="Pretty print JSON output")
@click.option("--skip-mcp", is_flag=True, help="Skip MCP initialization")
def spaces_list(api_key, base_url, pretty, skip_mcp):
    """List available spaces."""
    try:
        client = create_client(api_key=api_key, base_url=base_url, skip_mcp=skip_mcp)
        spaces_list = client.get_spaces()
        click.echo(format_json_output(spaces_list, pretty))
    except (HeySolError, click.ClickException) as e:
        raise click.ClickException(str(e))
    finally:
        if 'client' in locals():
            client.close()


@spaces.command("create")
@click.argument("name")
@click.option("--description", help="Space description")
@click.option("--api-key", help="HeySol API key (overrides environment variable)")
@click.option("--base-url", help="Base URL for API (overrides default)")
@click.option("--pretty", is_flag=True, help="Pretty print JSON output")
@click.option("--skip-mcp", is_flag=True, help="Skip MCP initialization")
def spaces_create(name, description, api_key, base_url, pretty, skip_mcp):
    """Create a new space."""
    try:
        client = create_client(api_key=api_key, base_url=base_url, skip_mcp=skip_mcp)
        space_id = client.create_space(name, description or "")
        result = {"space_id": space_id, "name": name, "description": description}
        click.echo(format_json_output(result, pretty))
    except (HeySolError, click.ClickException) as e:
        raise click.ClickException(str(e))
    finally:
        if 'client' in locals():
            client.close()


@spaces.command("get")
@click.argument("space_id")
@click.option("--include-stats/--no-include-stats", default=True, help="Include statistics")
@click.option("--include-metadata/--no-include-metadata", default=True, help="Include metadata")
@click.option("--api-key", help="HeySol API key (overrides environment variable)")
@click.option("--base-url", help="Base URL for API (overrides default)")
@click.option("--pretty", is_flag=True, help="Pretty print JSON output")
@click.option("--skip-mcp", is_flag=True, help="Skip MCP initialization")
def spaces_get(space_id, include_stats, include_metadata, api_key, base_url, pretty, skip_mcp):
    """Get space details."""
    try:
        client = create_client(api_key=api_key, base_url=base_url, skip_mcp=skip_mcp)
        details = client.get_space_details(space_id, include_stats, include_metadata)
        click.echo(format_json_output(details, pretty))
    except (HeySolError, click.ClickException) as e:
        raise click.ClickException(str(e))
    finally:
        if 'client' in locals():
            client.close()


@spaces.command("update")
@click.argument("space_id")
@click.option("--name", help="New space name")
@click.option("--description", help="New space description")
@click.option("--metadata", help="JSON metadata string")
@click.option("--api-key", help="HeySol API key (overrides environment variable)")
@click.option("--base-url", help="Base URL for API (overrides default)")
@click.option("--pretty", is_flag=True, help="Pretty print JSON output")
@click.option("--skip-mcp", is_flag=True, help="Skip MCP initialization")
def spaces_update(space_id, name, description, metadata, api_key, base_url, pretty, skip_mcp):
    """Update space properties."""
    parsed_metadata = None
    if metadata:
        try:
            import json
            parsed_metadata = json.loads(metadata)
        except json.JSONDecodeError as e:
            raise click.ClickException(f"Invalid JSON metadata: {e}")

    try:
        client = create_client(api_key=api_key, base_url=base_url, skip_mcp=skip_mcp)
        result = client.update_space(
            space_id=space_id,
            name=name,
            description=description,
            metadata=parsed_metadata
        )
        click.echo(format_json_output(result, pretty))
    except (HeySolError, click.ClickException) as e:
        raise click.ClickException(str(e))
    finally:
        if 'client' in locals():
            client.close()


@spaces.command("bulk-ops")
@click.argument("intent")
@click.option("--space-id", help="Target space ID")
@click.option("--statement-ids", multiple=True, help="Statement IDs (can specify multiple)")
@click.option("--space-ids", multiple=True, help="Space IDs (can specify multiple)")
@click.option("--api-key", help="HeySol API key (overrides environment variable)")
@click.option("--base-url", help="Base URL for API (overrides default)")
@click.option("--pretty", is_flag=True, help="Pretty print JSON output")
@click.option("--skip-mcp", is_flag=True, help="Skip MCP initialization")
def spaces_bulk_ops(intent, space_id, statement_ids, space_ids, api_key, base_url, pretty, skip_mcp):
    """Perform bulk operations on spaces."""
    try:
        client = create_client(api_key=api_key, base_url=base_url, skip_mcp=skip_mcp)
        result = client.bulk_space_operations(
            intent=intent,
            space_id=space_id,
            statement_ids=list(statement_ids) if statement_ids else None,
            space_ids=list(space_ids) if space_ids else None
        )
        click.echo(format_json_output(result, pretty))
    except (HeySolError, click.ClickException) as e:
        raise click.ClickException(str(e))
    finally:
        if 'client' in locals():
            client.close()


@spaces.command("delete")
@click.argument("space_id")
@click.option("--confirm", is_flag=True, help="Confirm deletion (required)")
@click.option("--api-key", help="HeySol API key (overrides environment variable)")
@click.option("--base-url", help="Base URL for API (overrides default)")
@click.option("--pretty", is_flag=True, help="Pretty print JSON output")
@click.option("--skip-mcp", is_flag=True, help="Skip MCP initialization")
def spaces_delete(space_id, confirm, api_key, base_url, pretty, skip_mcp):
    """Delete a space."""
    if not confirm:
        raise click.ClickException("Space deletion requires --confirm flag for safety")

    try:
        client = create_client(api_key=api_key, base_url=base_url, skip_mcp=skip_mcp)
        result = client.delete_space(space_id=space_id, confirm=confirm)
        click.echo(format_json_output(result, pretty))
    except (HeySolError, click.ClickException) as e:
        raise click.ClickException(str(e))
    finally:
        if 'client' in locals():
            client.close()


if __name__ == "__main__":
    spaces()