import os

#!/usr/bin/env python3
"""
mCODE Translator CLI - Pure Typer Backend

A comprehensive command-line interface for mCODE translation, processing,
and memory operations using HeySol API client integration.
"""

from pathlib import Path
import sys
from typing import Optional

import typer

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv not available, continue without it

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Subcommand imports - import after path setup
from .commands import config, memory, patients, trials, matching

app = typer.Typer(
    name="mcode-translator",
    help="mCODE Translator CLI - Clinical data processing with HeySol memory integration",
    add_completion=True,
    rich_markup_mode="rich",
)


# Global state for subcommands
class GlobalState:
    api_key: Optional[str] = None
    user: Optional[str] = None
    base_url: Optional[str] = None
    source: Optional[str] = None
    skip_mcp: bool = False


state = GlobalState()


@app.callback()
def cli_callback(
    ctx: typer.Context,
    api_key: Optional[str] = typer.Option(
        None, help="HeySol API key (overrides environment variable)"
    ),
    user: Optional[str] = typer.Option(
        None, help="User instance name from registry (alternative to --api-key)"
    ),
    base_url: Optional[str] = typer.Option(None, help="Base URL for API (overrides default)"),
    source: Optional[str] = typer.Option(None, help="Source identifier (overrides default)"),
    skip_mcp: bool = typer.Option(False, help="Skip MCP initialization"),
) -> None:
    """mCODE Translator CLI - Clinical Data Processing with Persistent Memory

    Authentication: Use --api-key, --user (registry), or HEYSOL_API_KEY env var.
    Get API key from: https://core.heysol.ai/settings/api
    """
    # Use the reorganized configuration system
    try:
        from config.heysol_config import McodeHeySolConfig

        # Create configuration based on what's provided
        if api_key or user:
            # Explicit authentication provided
            config = McodeHeySolConfig.with_authentication(
                api_key=api_key, user=user, base_url=base_url
            )
        else:
            # No explicit auth - use environment variables
            config = McodeHeySolConfig.from_env()

        # Store in global state for subcommands
        state.api_key = config.get_api_key()
        state.user = config.user
        state.base_url = config.get_base_url()
        state.source = source
        state.skip_mcp = skip_mcp
        ctx.obj = state

    except ImportError:
        # Fallback to original logic if config system unavailable
        resolved_api_key = api_key or os.getenv("HEYSOL_API_KEY")
        resolved_user = user
        resolved_base_url = base_url or "https://core.heysol.ai/api/v1"

        # For help commands, don't require credentials
        if "--help" in sys.argv or "help" in sys.argv:
            resolved_api_key = resolved_api_key or "dummy-key-for-help"

        # Validate authentication
        if not resolved_api_key and not resolved_user:
            typer.echo(
                "No authentication provided. Use --api-key, --user, or set HEYSOL_API_KEY environment variable.",
                err=True,
            )
            raise typer.Exit(1)

        # Check for explicit conflicts between command line flags
        if api_key is not None and user is not None:
            typer.echo(
                "Both --api-key and --user provided. Please use only one authentication method.",
                err=True,
            )
            raise typer.Exit(1)

        # If API key from env var and user flag provided, prioritize user
        if resolved_api_key and resolved_user and api_key is None:
            resolved_api_key = None

        # Store in global state for subcommands
        state.api_key = resolved_api_key
        state.user = resolved_user
        state.base_url = resolved_base_url
        state.source = source
        state.skip_mcp = skip_mcp
        ctx.obj = state


# Add command groups with detailed descriptions
if patients and hasattr(patients, "app"):
    app.add_typer(patients.app, name="patients", help="Patient data processing operations")
if trials and hasattr(trials, "app"):
    app.add_typer(trials.app, name="trials", help="Clinical trial data processing operations")
if matching and hasattr(matching, "app"):
    app.add_typer(matching.app, name="matching", help="Patient-trial matching operations")
if memory and hasattr(memory, "app"):
    app.add_typer(memory.app, name="memory", help="CORE Memory operations and management")
if config and hasattr(config, "app"):
    app.add_typer(config.app, name="config", help="Configuration management and validation")


@app.command("version")
def version():
    """Show version information."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    console = Console()
    version_text = Text(f"mCODE Translator CLI v{__version__}", style="bold blue")
    console.print(Panel(version_text, title="Version", border_style="blue"))


@app.command("status")
def status():
    """Check system status and connectivity."""
    from rich.console import Console

    console = Console()
    console.print("[bold blue]🔍 Checking mCODE Translator Status...[/bold blue]")

    try:
        # Check HeySol API connectivity
        from heysol import HeySolClient

        # Get configuration for authentication
        from config.heysol_config import get_config

        config = get_config()

        # Update config with current state
        if state.api_key:
            config.api_key = state.api_key
        if state.user:
            config.user = state.user

        # Get authentication credentials
        api_key = config.get_api_key()
        user = config.user

        if not api_key and not user:
            console.print("[red]❌ No authentication configured[/red]")
            console.print(
                "[yellow]💡 Use --api-key, --user, or set HEYSOL_API_KEY environment variable[/yellow]"
            )
            raise typer.Exit(1)

        # Initialize client based on authentication method
        if api_key:
            client = HeySolClient(api_key=api_key)
        elif user:
            # Try to resolve user registry authentication
            console.print(f"[blue]🔑 Using registry authentication for user: {user}[/blue]")
            resolved_api_key = config.resolve_user_authentication()
            if resolved_api_key:
                client = HeySolClient(api_key=resolved_api_key)
                console.print(
                    f"[green]✅ Registry authentication successful for user: {user}[/green]"
                )
            else:
                console.print(f"[red]❌ Could not resolve API key for user: {user}[/red]")
                console.print("[yellow]💡 Check that the user exists in the registry[/yellow]")
                raise typer.Exit(1)

        # Test basic connectivity
        spaces = client.get_spaces()
        mcp_available = client.is_mcp_available()

        console.print("[green]✅ HeySol API client initialized successfully[/green]")
        console.print(f"[blue]📊 Available memory spaces: {len(spaces)}[/blue]")
        console.print(
            f"[blue]🧠 MCP integration: {'Available' if mcp_available else 'API Only'}[/blue]"
        )

        # Check configuration
        from config.heysol_config import get_config

        config = get_config()
        console.print("[green]✅ Configuration loaded successfully[/green]")
        console.print(f"[blue]🎯 Base URL: {config.get_base_url()}[/blue]")

        # Check memory components
        from services.heysol_client import OncoCoreClient
        from storage.mcode_memory_storage import OncoCoreMemory

        memory_client = OncoCoreClient.from_env()
        memory_storage = OncoCoreMemory()

        console.print("[green]✅ Memory components initialized successfully[/green]")

        # Show memory stats
        stats = memory_storage.get_memory_stats()
        console.print(
            f"[blue]💾 Patients space: {stats.patients_space.get('name', 'Not found') if stats.patients_space else 'Not found'}[/blue]"
        )
        console.print(
            f"[blue]🧪 Trials space: {stats.trials_space.get('name', 'Not found') if stats.trials_space else 'Not found'}[/blue]"
        )

        console.print("\n[bold green]🎉 All systems operational![/bold green]")

        # Cleanup
        client.close()
        memory_client.close()

    except ImportError as e:
        console.print(f"[red]❌ Import error: {e}[/red]")
        console.print("[yellow]💡 Install dependencies: pip install -e .[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Status check failed: {e}[/red]")
        raise typer.Exit(1)


@app.command("doctor")
def doctor():
    """Run comprehensive system diagnostics."""
    from rich.console import Console

    console = Console()
    console.print("[bold blue]🩺 Running mCODE Translator Diagnostics...[/bold blue]")

    diagnostics = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": sys.platform,
        "dependencies": {},
        "configuration": {},
        "connectivity": {},
        "memory": {},
    }

    # Check Python version
    if sys.version_info >= (3, 10):
        console.print("[green]✅ Python version: {diagnostics['python_version']}[/green]")
    else:
        console.print(
            f"[red]❌ Python version {diagnostics['python_version']} - requires Python 3.10+[/red]"
        )

    # Check dependencies
    required_deps = [
        "typer",
        "rich",
        "pydantic",
        "requests",
        "python-dotenv",
        "heysol",
        "pandas",
        "scipy",
        "openai",
    ]

    for dep in required_deps:
        try:
            __import__(dep)
            diagnostics["dependencies"][dep] = "✅"
            console.print(f"[green]✅ {dep} available[/green]")
        except ImportError:
            diagnostics["dependencies"][dep] = "❌"
            console.print(f"[red]❌ {dep} missing[/red]")

    # Check configuration
    try:
        from config.heysol_config import get_config

        get_config()
        diagnostics["configuration"]["heysol_config"] = "✅"
        console.print("[green]✅ HeySol configuration loaded[/green]")
    except Exception as e:
        diagnostics["configuration"]["heysol_config"] = f"❌ {e}"
        console.print(f"[red]❌ HeySol configuration failed: {e}[/red]")

    # Check API connectivity
    try:
        from heysol import HeySolClient

        api_key = state.api_key
        if api_key:
            client = HeySolClient(api_key=api_key)
            spaces = client.get_spaces()
            diagnostics["connectivity"]["heysol_api"] = f"✅ {len(spaces)} spaces"
            console.print(
                f"[green]✅ HeySol API connected - {len(spaces)} spaces available[/green]"
            )
            client.close()
        else:
            diagnostics["connectivity"]["heysol_api"] = "❌ No API key"
            console.print("[red]❌ HeySol API key not configured[/red]")
    except Exception as e:
        diagnostics["connectivity"]["heysol_api"] = f"❌ {e}"
        console.print(f"[red]❌ HeySol API connection failed: {e}[/red]")

    # Check memory components
    try:
        from services.heysol_client import OncoCoreClient
        from storage.mcode_memory_storage import OncoCoreMemory

        memory_client = OncoCoreClient.from_env()
        memory_storage = OncoCoreMemory()

        stats = memory_storage.get_memory_stats()
        diagnostics["memory"]["onco_core"] = "✅"
        diagnostics["memory"]["storage"] = f"✅ {stats.total_spaces} spaces"
        console.print("[green]✅ Memory components operational[/green]")
        console.print(f"[blue]📊 Memory spaces: {stats.total_spaces}[/blue]")

        memory_client.close()

    except Exception as e:
        diagnostics["memory"]["components"] = f"❌ {e}"
        console.print(f"[red]❌ Memory components failed: {e}[/red]")

    # Summary
    all_good = all(
        "❌" not in str(status) for category in diagnostics.values() for status in category.values()
    )

    if all_good:
        console.print("\n[bold green]🎉 All diagnostics passed![/bold green]")
        console.print("[green]🚀 System ready for mCODE translation operations[/green]")
    else:
        console.print("\n[yellow]⚠️ Some diagnostics failed - check configuration[/yellow]")
        console.print(
            "[blue]💡 Run 'mcode-translator config check' for detailed config validation[/blue]"
        )


__version__ = "2.0.0"


def main() -> None:
    app()


if __name__ == "__main__":
    main()
