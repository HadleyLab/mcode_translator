"""
Configuration Commands

Commands for managing mCODE Translator configuration including
validation, setup, and environment management.
"""

import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.logging_config import get_logger

console = Console()
logger = get_logger(__name__)

app = typer.Typer(
    name="config",
    help="Configuration management and validation",
    add_completion=True,
)


@app.command("check")
def check_config(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed configuration information"),
):
    """
    Check and validate current configuration.

    Performs comprehensive validation of all configuration files,
    environment variables, and system requirements.
    """
    console.print("[bold blue]🔍 Checking mCODE Translator Configuration[/bold blue]")

    config_status = {
        "environment": {},
        "files": {},
        "apis": {},
        "memory": {},
    }

    # Check environment variables
    console.print("[blue]🔧 Checking environment variables...[/blue]")

    required_env_vars = [
        "HEYSOL_API_KEY",
        "OPENAI_API_KEY",
    ]

    optional_env_vars = [
        "MCODE_CACHE_ENABLED",
        "MCODE_DEFAULT_MODEL",
        "MCODE_STRICT_MODE",
    ]

    for var in required_env_vars:
        value = os.getenv(var)
        if value and len(value.strip()) > 10:  # Basic validation
            config_status["environment"][var] = "✅"
            if verbose:
                console.print(f"[green]✅ {var}: configured[/green]")
        else:
            config_status["environment"][var] = "❌"
            console.print(f"[red]❌ {var}: missing or invalid[/red]")

    for var in optional_env_vars:
        value = os.getenv(var)
        if value:
            config_status["environment"][var] = "✅"
            if verbose:
                console.print(f"[blue]ℹ️ {var}: {value}[/blue]")
        else:
            config_status["environment"][var] = "⚠️"
            if verbose:
                console.print(f"[yellow]⚠️ {var}: not set (using defaults)[/yellow]")

    # Check configuration files
    console.print("[blue]📁 Checking configuration files...[/blue]")

    config_files = [
        "src/config/heysol_config.py",
        "src/config/apis_config.json",
        "src/config/core_memory_config.json",
        "src/config/llms_config.json",
        "src/config/logging_config.json",
        "src/config/patterns_config.json",
        "src/config/validation_config.json",
    ]

    for config_file in config_files:
        file_path = Path(config_file)
        if file_path.exists():
            config_status["files"][config_file] = "✅"
            if verbose:
                console.print(f"[green]✅ {config_file}: exists[/green]")
        else:
            config_status["files"][config_file] = "❌"
            console.print(f"[red]❌ {config_file}: missing[/red]")

    # Check API configurations
    console.print("[blue]🔗 Checking API configurations...[/blue]")

    try:
        from utils.config import Config
        config = Config()

        # Check LLM configurations
        llm_configs = config.get_all_llm_configs()
        config_status["apis"]["llm_configs"] = f"✅ {len(llm_configs)} models"
        console.print(f"[green]✅ LLM configurations: {len(llm_configs)} models available[/green]")

        # Test API key retrieval
        try:
            api_key = config.get_api_key("gpt-4")
            if api_key:
                config_status["apis"]["api_keys"] = "✅"
                console.print("[green]✅ API keys: configured[/green]")
            else:
                config_status["apis"]["api_keys"] = "❌"
                console.print("[red]❌ API keys: missing[/red]")
        except Exception as e:
            config_status["apis"]["api_keys"] = f"❌ {e}"
            console.print(f"[red]❌ API keys: {e}[/red]")

    except Exception as e:
        config_status["apis"]["config_loading"] = f"❌ {e}"
        console.print(f"[red]❌ Configuration loading failed: {e}[/red]")

    # Check memory configuration
    console.print("[blue]🧠 Checking CORE Memory configuration...[/blue]")

    try:
        core_memory_config = config.get_core_memory_config()
        config_status["memory"]["core_config"] = "✅"
        console.print("[green]✅ CORE Memory configuration loaded[/green]")

        api_base_url = core_memory_config.get("core_memory", {}).get("api_base_url")
        if api_base_url:
            config_status["memory"]["api_url"] = f"✅ {api_base_url}"
            if verbose:
                console.print(f"[blue]📡 API URL: {api_base_url}[/blue]")
        else:
            config_status["memory"]["api_url"] = "❌"
            console.print("[red]❌ CORE Memory API URL not configured[/red]")

    except Exception as e:
        config_status["memory"]["config"] = f"❌ {e}"
        console.print(f"[red]❌ CORE Memory configuration failed: {e}[/red]")

    # Summary
    all_good = all(
        status in ["✅", "⚠️"] or not status.startswith("❌")
        for category in config_status.values()
        for status in category.values()
    )

    if all_good:
        console.print("\n[bold green]🎉 Configuration check passed![/bold green]")
        console.print("[green]🚀 System is ready for mCODE translation operations[/green]")
    else:
        console.print("\n[yellow]⚠️ Configuration issues detected[/yellow]")
        console.print("[blue]💡 Run 'mcode-translator config setup' to configure missing settings[/blue]")

    return config_status


@app.command("setup")
def setup_config(
    interactive: bool = typer.Option(True, help="Run interactive setup"),
    force: bool = typer.Option(False, help="Overwrite existing configuration"),
):
    """
    Set up mCODE Translator configuration.

    Guides through the configuration process for API keys, settings,
    and environment setup.
    """
    console.print("[bold blue]⚙️ mCODE Translator Configuration Setup[/bold blue]")

    if interactive:
        console.print("[yellow]This will guide you through configuring the mCODE Translator.[/yellow]")
        console.print("[yellow]You'll need API keys for HeySol and OpenAI.[/yellow]")

        # Check for existing configuration
        if not force:
            try:
                from utils.config import Config
                config = Config()
                console.print("[blue]Existing configuration detected.[/blue]")
                proceed = typer.confirm("Overwrite existing configuration?", default=False)
                if not proceed:
                    console.print("[green]Setup cancelled - keeping existing configuration[/green]")
                    return
            except:
                pass

        # HeySol API Key setup
        console.print("\n[bold cyan]1. HeySol API Configuration[/bold cyan]")
        console.print("Get your API key from: https://core.heysol.ai/settings/api")

        heysol_key = typer.prompt("Enter HeySol API Key", hide_input=True)
        if heysol_key:
            os.environ["HEYSOL_API_KEY"] = heysol_key
            console.print("[green]✅ HeySol API key configured[/green]")
        else:
            console.print("[red]❌ HeySol API key is required[/red]")
            raise typer.Exit(1)

        # OpenAI API Key setup
        console.print("\n[bold cyan]2. OpenAI API Configuration[/bold cyan]")
        console.print("Get your API key from: https://platform.openai.com/api-keys")

        openai_key = typer.prompt("Enter OpenAI API Key", hide_input=True)
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
            console.print("[green]✅ OpenAI API key configured[/green]")
        else:
            console.print("[red]❌ OpenAI API key is required[/red]")
            raise typer.Exit(1)

        # Optional settings
        console.print("\n[bold cyan]3. Optional Settings[/bold cyan]")

        default_model = typer.prompt("Default LLM model", default="gpt-4")
        os.environ["MCODE_DEFAULT_MODEL"] = default_model

        strict_mode = typer.confirm("Enable strict validation mode?", default=True)
        os.environ["MCODE_STRICT_MODE"] = str(strict_mode).lower()

        cache_enabled = typer.confirm("Enable caching?", default=True)
        os.environ["MCODE_CACHE_ENABLED"] = str(cache_enabled).lower()

        console.print("\n[green]✅ Configuration setup completed![/green]")
        console.print("[blue]💡 Restart your shell or run 'source ~/.bashrc' to apply environment variables[/blue]")
        console.print("[blue]💡 Or run 'mcode-translator config check' to verify configuration[/blue]")

    else:
        console.print("[red]Non-interactive setup not yet implemented[/red]")
        console.print("[blue]Use --interactive flag for guided setup[/blue]")


@app.command("show")
def show_config(
    category: str = typer.Option("all", help="Configuration category to show (all, env, apis, memory)"),
):
    """
    Display current configuration settings.

    Shows detailed configuration information organized by category.
    """
    console.print(f"[bold blue]📋 mCODE Translator Configuration - {category.upper()}[/bold blue]")

    try:
        from utils.config import Config
        config = Config()

        if category in ["all", "env"]:
            console.print("\n[cyan]Environment Variables:[/cyan]")
            env_vars = {
                "HEYSOL_API_KEY": os.getenv("HEYSOL_API_KEY", "not set"),
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "not set")[:20] + "..." if os.getenv("OPENAI_API_KEY") else "not set",
                "MCODE_DEFAULT_MODEL": os.getenv("MCODE_DEFAULT_MODEL", "gpt-4"),
                "MCODE_STRICT_MODE": os.getenv("MCODE_STRICT_MODE", "true"),
                "MCODE_CACHE_ENABLED": os.getenv("MCODE_CACHE_ENABLED", "true"),
            }

            table = Table()
            table.add_column("Variable", style="cyan")
            table.add_column("Value", style="green")

            for var, value in env_vars.items():
                table.add_row(var, value)

            console.print(table)

        if category in ["all", "apis"]:
            console.print("\n[cyan]API Configurations:[/cyan]")
            try:
                llm_configs = config.get_all_llm_configs()
                console.print(f"Available LLM models: {len(llm_configs)}")

                for model_key, model_config in llm_configs.items():
                    console.print(f"  • {model_key}: {model_config.name}")
            except Exception as e:
                console.print(f"[red]Error loading LLM configs: {e}[/red]")

        if category in ["all", "memory"]:
            console.print("\n[cyan]CORE Memory Configuration:[/cyan]")
            try:
                memory_config = config.get_core_memory_config()
                core_memory = memory_config.get("core_memory", {})

                console.print(f"API Base URL: {core_memory.get('api_base_url', 'not set')}")
                console.print(f"Source: {core_memory.get('source', 'not set')}")
                console.print(f"Timeout: {core_memory.get('timeout_seconds', 'not set')}s")
                console.print(f"Batch Size: {core_memory.get('storage_settings', {}).get('batch_size', 'not set')}")

            except Exception as e:
                console.print(f"[red]Error loading memory config: {e}[/red]")

    except Exception as e:
        console.print(f"[red]❌ Failed to load configuration: {e}[/red]")
        logger.exception("Configuration display error")
        raise typer.Exit(1)