"""
Webhook-related CLI commands.
"""

import click
from typing import Optional

from .common import create_client, format_json_output
from heysol import HeySolError


@click.group()
def webhooks():
    """Webhook operations."""
    pass


@webhooks.command("create")
@click.argument("url")
@click.option("--secret", required=True, help="Webhook secret")
@click.option("--events", multiple=True, help="Events to subscribe to (can specify multiple)")
@click.option("--api-key", help="HeySol API key (overrides environment variable)")
@click.option("--base-url", help="Base URL for API (overrides default)")
@click.option("--pretty", is_flag=True, help="Pretty print JSON output")
@click.option("--skip-mcp", is_flag=True, help="Skip MCP initialization")
def webhooks_create(url, secret, events, api_key, base_url, pretty, skip_mcp):
    """Create a new webhook."""
    try:
        client = create_client(api_key=api_key, base_url=base_url, skip_mcp=skip_mcp)
        result = client.register_webhook(
            url=url,
            events=list(events) if events else None,
            secret=secret
        )
        click.echo(format_json_output(result, pretty))
    except (HeySolError, click.ClickException) as e:
        raise click.ClickException(str(e))
    finally:
        if 'client' in locals():
            client.close()


@webhooks.command("get")
@click.argument("webhook_id")
@click.option("--api-key", help="HeySol API key (overrides environment variable)")
@click.option("--base-url", help="Base URL for API (overrides default)")
@click.option("--pretty", is_flag=True, help="Pretty print JSON output")
@click.option("--skip-mcp", is_flag=True, help="Skip MCP initialization")
def webhooks_get(webhook_id, api_key, base_url, pretty, skip_mcp):
    """Get webhook details."""
    try:
        client = create_client(api_key=api_key, base_url=base_url, skip_mcp=skip_mcp)
        result = client.get_webhook(webhook_id=webhook_id)
        click.echo(format_json_output(result, pretty))
    except (HeySolError, click.ClickException) as e:
        raise click.ClickException(str(e))
    finally:
        if 'client' in locals():
            client.close()


@webhooks.command("list")
@click.option("--space-id", help="Space ID")
@click.option("--active", type=bool, help="Filter by active status")
@click.option("--limit", type=int, default=100, help="Result limit")
@click.option("--offset", type=int, default=0, help="Result offset")
@click.option("--api-key", help="HeySol API key (overrides environment variable)")
@click.option("--base-url", help="Base URL for API (overrides default)")
@click.option("--pretty", is_flag=True, help="Pretty print JSON output")
@click.option("--skip-mcp", is_flag=True, help="Skip MCP initialization")
def webhooks_list(space_id, active, limit, offset, api_key, base_url, pretty, skip_mcp):
    """List webhooks."""
    try:
        client = create_client(api_key=api_key, base_url=base_url, skip_mcp=skip_mcp)
        result = client.list_webhooks(
            space_id=space_id,
            active=active,
            limit=limit,
            offset=offset
        )
        click.echo(format_json_output(result, pretty))
    except (HeySolError, click.ClickException) as e:
        raise click.ClickException(str(e))
    finally:
        if 'client' in locals():
            client.close()


@webhooks.command("update")
@click.argument("webhook_id")
@click.argument("url")
@click.option("--events", multiple=True, required=True, help="Events to subscribe to (can specify multiple)")
@click.option("--secret", required=True, help="Webhook secret")
@click.option("--active/--inactive", default=True, help="Webhook active status")
@click.option("--api-key", help="HeySol API key (overrides environment variable)")
@click.option("--base-url", help="Base URL for API (overrides default)")
@click.option("--pretty", is_flag=True, help="Pretty print JSON output")
@click.option("--skip-mcp", is_flag=True, help="Skip MCP initialization")
def webhooks_update(webhook_id, url, events, secret, active, api_key, base_url, pretty, skip_mcp):
    """Update webhook properties."""
    try:
        client = create_client(api_key=api_key, base_url=base_url, skip_mcp=skip_mcp)
        result = client.update_webhook(
            webhook_id=webhook_id,
            url=url,
            events=list(events),
            secret=secret,
            active=active
        )
        click.echo(format_json_output(result, pretty))
    except (HeySolError, click.ClickException) as e:
        raise click.ClickException(str(e))
    finally:
        if 'client' in locals():
            client.close()


@webhooks.command("delete")
@click.argument("webhook_id")
@click.option("--confirm", is_flag=True, help="Confirm deletion (required)")
@click.option("--api-key", help="HeySol API key (overrides environment variable)")
@click.option("--base-url", help="Base URL for API (overrides default)")
@click.option("--pretty", is_flag=True, help="Pretty print JSON output")
@click.option("--skip-mcp", is_flag=True, help="Skip MCP initialization")
def webhooks_delete(webhook_id, confirm, api_key, base_url, pretty, skip_mcp):
    """Delete a webhook."""
    if not confirm:
        raise click.ClickException("Webhook deletion requires --confirm flag for safety")

    try:
        client = create_client(api_key=api_key, base_url=base_url, skip_mcp=skip_mcp)
        result = client.delete_webhook(webhook_id=webhook_id, confirm=confirm)
        click.echo(format_json_output(result, pretty))
    except (HeySolError, click.ClickException) as e:
        raise click.ClickException(str(e))
    finally:
        if 'client' in locals():
            client.close()


if __name__ == "__main__":
    webhooks()