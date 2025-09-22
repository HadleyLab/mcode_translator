"""
Profile-related CLI commands.
"""

import click

from .common import create_client, format_json_output
from heysol import HeySolError


@click.group()
def profile():
    """User profile operations."""
    pass


@profile.command("get")
@click.option("--api-key", help="HeySol API key (overrides environment variable)")
@click.option("--base-url", help="Base URL for API (overrides default)")
@click.option("--pretty", is_flag=True, help="Pretty print JSON output")
@click.option("--skip-mcp", is_flag=True, help="Skip MCP initialization")
def profile_get(api_key, base_url, pretty, skip_mcp):
    """Get user profile."""
    try:
        client = create_client(api_key=api_key, base_url=base_url, skip_mcp=skip_mcp)
        profile = client.get_user_profile()
        click.echo(format_json_output(profile, pretty))
    except (HeySolError, click.ClickException) as e:
        raise click.ClickException(str(e))
    finally:
        if 'client' in locals():
            client.close()


if __name__ == "__main__":
    profile()