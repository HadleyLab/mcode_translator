"""
MCP tools-related CLI commands.
"""

import click

from .common import create_client, format_json_output
from heysol import HeySolError


@click.group()
def tools():
    """MCP tools operations."""
    pass


@tools.command("list")
@click.option("--api-key", help="HeySol API key (overrides environment variable)")
@click.option("--base-url", help="Base URL for API (overrides default)")
@click.option("--pretty", is_flag=True, help="Pretty print JSON output")
@click.option("--skip-mcp", is_flag=True, help="Skip MCP initialization")
def tools_list(api_key, base_url, pretty, skip_mcp):
    """List MCP tools."""
    try:
        client = create_client(api_key=api_key, base_url=base_url, skip_mcp=skip_mcp)
        tools_list = client.get_available_tools()
        click.echo(format_json_output(tools_list, pretty))
    except (HeySolError, click.ClickException) as e:
        raise click.ClickException(str(e))
    finally:
        if 'client' in locals():
            client.close()


if __name__ == "__main__":
    tools()