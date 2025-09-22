"""
OAuth2-related CLI commands.
"""

import click
from typing import Optional

from .common import create_client, format_json_output
from heysol import HeySolError


@click.group()
def oauth2():
    """OAuth2 operations."""
    pass


@oauth2.command("authorize")
@click.option("--redirect-uri", help="Redirect URI")
@click.option("--scope", help="OAuth2 scope")
@click.option("--api-key", help="HeySol API key (overrides environment variable)")
@click.option("--base-url", help="Base URL for API (overrides default)")
@click.option("--pretty", is_flag=True, help="Pretty print JSON output")
@click.option("--skip-mcp", is_flag=True, help="Skip MCP initialization")
def oauth2_authorize(redirect_uri, scope, api_key, base_url, pretty, skip_mcp):
    """OAuth2 authorization endpoint."""
    try:
        client = create_client(api_key=api_key, base_url=base_url, skip_mcp=skip_mcp)
        # This would typically redirect to OAuth2 authorization URL
        # For now, we'll show the authorization URL
        auth_url = f"{client.base_url}/oauth2/authorize"
        if redirect_uri:
            auth_url += f"?redirect_uri={redirect_uri}"
        if scope:
            auth_url += f"&scope={scope}"

        result = {
            "authorization_url": auth_url,
            "redirect_uri": redirect_uri,
            "scope": scope
        }
        click.echo(format_json_output(result, pretty))
    except (HeySolError, click.ClickException) as e:
        raise click.ClickException(str(e))
    finally:
        if 'client' in locals():
            client.close()


@oauth2.command("token")
@click.option("--grant-type", default="authorization_code", help="OAuth2 grant type")
@click.option("--code", help="Authorization code")
@click.option("--redirect-uri", help="Redirect URI")
@click.option("--client-id", help="OAuth2 client ID")
@click.option("--client-secret", help="OAuth2 client secret")
@click.option("--api-key", help="HeySol API key (overrides environment variable)")
@click.option("--base-url", help="Base URL for API (overrides default)")
@click.option("--pretty", is_flag=True, help="Pretty print JSON output")
@click.option("--skip-mcp", is_flag=True, help="Skip MCP initialization")
def oauth2_token(grant_type, code, redirect_uri, client_id, client_secret, api_key, base_url, pretty, skip_mcp):
    """OAuth2 token endpoint."""
    try:
        client = create_client(api_key=api_key, base_url=base_url, skip_mcp=skip_mcp)

        # Prepare token request data
        token_data = {
            "grant_type": grant_type,
            "client_id": client_id,
            "client_secret": client_secret
        }

        if code:
            token_data["code"] = code
        if redirect_uri:
            token_data["redirect_uri"] = redirect_uri

        # Make token request
        token_url = f"{client.base_url}/oauth2/token"
        import requests
        response = requests.post(
            token_url,
            data=token_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        response.raise_for_status()
        result = response.json()
        click.echo(format_json_output(result, pretty))
    except (HeySolError, click.ClickException) as e:
        raise click.ClickException(str(e))
    finally:
        if 'client' in locals():
            client.close()


@oauth2.command("userinfo")
@click.option("--access-token", help="Access token")
@click.option("--api-key", help="HeySol API key (overrides environment variable)")
@click.option("--base-url", help="Base URL for API (overrides default)")
@click.option("--pretty", is_flag=True, help="Pretty print JSON output")
@click.option("--skip-mcp", is_flag=True, help="Skip MCP initialization")
def oauth2_userinfo(access_token, api_key, base_url, pretty, skip_mcp):
    """OAuth2 user info endpoint."""
    try:
        client = create_client(api_key=api_key, base_url=base_url, skip_mcp=skip_mcp)

        headers = {}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"

        userinfo_url = f"{client.base_url}/oauth2/userinfo"
        import requests
        response = requests.get(userinfo_url, headers=headers)
        response.raise_for_status()
        result = response.json()
        click.echo(format_json_output(result, pretty))
    except (HeySolError, click.ClickException) as e:
        raise click.ClickException(str(e))
    finally:
        if 'client' in locals():
            client.close()


@oauth2.command("introspect")
@click.argument("token")
@click.option("--token-type", default="access_token", help="Token type")
@click.option("--client-id", help="OAuth2 client ID")
@click.option("--client-secret", help="OAuth2 client secret")
@click.option("--api-key", help="HeySol API key (overrides environment variable)")
@click.option("--base-url", help="Base URL for API (overrides default)")
@click.option("--pretty", is_flag=True, help="Pretty print JSON output")
@click.option("--skip-mcp", is_flag=True, help="Skip MCP initialization")
def oauth2_introspect(token, token_type, client_id, client_secret, api_key, base_url, pretty, skip_mcp):
    """OAuth2 token introspection endpoint."""
    try:
        client = create_client(api_key=api_key, base_url=base_url, skip_mcp=skip_mcp)

        # Prepare introspection request data
        introspect_data = {
            "token": token,
            "token_type_hint": token_type,
            "client_id": client_id,
            "client_secret": client_secret
        }

        # Make introspection request
        introspect_url = f"{client.base_url}/oauth2/introspect"
        import requests
        response = requests.post(
            introspect_url,
            data=introspect_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        response.raise_for_status()
        result = response.json()
        click.echo(format_json_output(result, pretty))
    except (HeySolError, click.ClickException) as e:
        raise click.ClickException(str(e))
    finally:
        if 'client' in locals():
            client.close()


if __name__ == "__main__":
    oauth2()