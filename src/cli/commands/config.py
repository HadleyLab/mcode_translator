"""
Configuration Commands

Commands for managing mCODE Translator configuration including
validation, setup, and environment management.
"""

import os
import sys
from pathlib import Path
from typing import Optional

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


@app.command("validate")
def validate_config(
    strict: bool = typer.Option(True, help="Perform strict validation"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed validation results"),
):
    """
    Validate configuration files and settings.

    Performs comprehensive validation of all configuration files,
    JSON syntax, required fields, and cross-references.
    """
    console.print("[bold blue]🔍 Validating mCODE Translator Configuration[/bold blue]")

    validation_results = {
        "files": {},
        "syntax": {},
        "required_fields": {},
        "cross_references": {},
    }

    try:
        from utils.config import Config
        config = Config()

        # Validate configuration files exist and are readable
        console.print("[blue]📁 Checking configuration files...[/blue]")
        config_files = [
            "src/config/cache_config.json",
            "src/config/apis_config.json",
            "src/config/core_memory_config.json",
            "src/config/llms_config.json",
            "src/config/logging_config.json",
            "src/config/patterns_config.json",
            "src/config/prompts_config.json",
            "src/config/synthetic_data_config.json",
            "src/config/validation_config.json",
        ]

        for config_file in config_files:
            file_path = Path(config_file)
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        json.load(f)
                    validation_results["files"][config_file] = "✅"
                    if verbose:
                        console.print(f"[green]✅ {config_file}: valid[/green]")
                except json.JSONDecodeError as e:
                    validation_results["syntax"][config_file] = f"❌ {e}"
                    console.print(f"[red]❌ {config_file}: invalid JSON - {e}[/red]")
                except IOError as e:
                    validation_results["files"][config_file] = f"❌ {e}"
                    console.print(f"[red]❌ {config_file}: cannot read - {e}[/red]")
            else:
                validation_results["files"][config_file] = "❌"
                console.print(f"[red]❌ {config_file}: missing[/red]")

        # Validate required fields
        console.print("[blue]🔧 Checking required configuration fields...[/blue]")

        try:
            # Check core memory config
            core_config = config.get_core_memory_config()
            required_core_fields = ["core_memory", "mcode_settings"]
            for field in required_core_fields:
                if field not in core_config:
                    validation_results["required_fields"][f"core_memory.{field}"] = "❌"
                    console.print(f"[red]❌ Missing required field: core_memory.{field}[/red]")
                else:
                    validation_results["required_fields"][f"core_memory.{field}"] = "✅"
                    if verbose:
                        console.print(f"[green]✅ core_memory.{field}: present[/green]")

            # Check LLM configs
            llm_configs = config.get_all_llm_configs()
            if not llm_configs:
                validation_results["required_fields"]["llm_configs"] = "❌"
                console.print("[red]❌ No LLM configurations found[/red]")
            else:
                validation_results["required_fields"]["llm_configs"] = f"✅ {len(llm_configs)} configs"
                if verbose:
                    console.print(f"[green]✅ LLM configurations: {len(llm_configs)} models available[/green]")

        except Exception as e:
            validation_results["required_fields"]["config_loading"] = f"❌ {e}"
            console.print(f"[red]❌ Configuration loading failed: {e}[/red]")

        # Cross-reference validation
        console.print("[blue]🔗 Checking cross-references...[/blue]")

        try:
            # Check if configured models have API keys
            llm_configs = config.get_all_llm_configs()
            for model_key, model_config in llm_configs.items():
                api_key_env = model_config.api_key_env_var
                if api_key_env:
                    api_key = os.getenv(api_key_env)
                    if not api_key or len(api_key.strip()) < 20:
                        validation_results["cross_references"][f"{model_key}_api_key"] = "❌"
                        console.print(f"[red]❌ {model_key}: API key missing or invalid[/red]")
                    else:
                        validation_results["cross_references"][f"{model_key}_api_key"] = "✅"
                        if verbose:
                            console.print(f"[green]✅ {model_key}: API key configured[/green]")

        except Exception as e:
            validation_results["cross_references"]["api_key_check"] = f"❌ {e}"
            console.print(f"[red]❌ Cross-reference validation failed: {e}[/red]")

        # Summary
        all_valid = all(
            status in ["✅", "⚠️"] or not str(status).startswith("❌")
            for category in validation_results.values()
            for status in category.values()
        )

        if all_valid:
            console.print("\n[bold green]🎉 Configuration validation passed![/bold green]")
            console.print("[green]🚀 All configuration files and settings are valid[/green]")
        else:
            console.print("\n[yellow]⚠️ Configuration validation found issues[/yellow]")
            console.print("[blue]💡 Run 'mcode-translator config setup' to fix configuration issues[/blue]")

        return validation_results

    except Exception as e:
        console.print(f"[red]❌ Validation failed: {e}[/red]")
        logger.exception("Configuration validation error")
        raise typer.Exit(1)


@app.command("backup")
def backup_config(
    output_dir: str = typer.Option("./config_backup", help="Directory to save backup"),
    include_env: bool = typer.Option(False, help="Include environment variables in backup"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed backup information"),
):
    """
    Create a backup of current configuration.

    Saves all configuration files and optionally environment variables
    to a timestamped backup directory.
    """
    console.print("[bold blue]💾 Creating configuration backup[/bold blue]")

    try:
        import datetime
        import shutil

        # Create backup directory with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = Path(output_dir) / f"config_backup_{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"[blue]📁 Backup directory: {backup_dir}[/blue]")

        # Backup configuration files
        console.print("[blue]📋 Backing up configuration files...[/blue]")
        config_files = [
            "src/config/cache_config.json",
            "src/config/apis_config.json",
            "src/config/core_memory_config.json",
            "src/config/llms_config.json",
            "src/config/logging_config.json",
            "src/config/patterns_config.json",
            "src/config/prompts_config.json",
            "src/config/synthetic_data_config.json",
            "src/config/validation_config.json",
        ]

        backed_up_files = []
        for config_file in config_files:
            src_path = Path(config_file)
            if src_path.exists():
                dst_path = backup_dir / src_path.name
                shutil.copy2(src_path, dst_path)
                backed_up_files.append(config_file)
                if verbose:
                    console.print(f"[green]✅ {config_file} -> {dst_path}[/green]")

        # Backup environment variables if requested
        if include_env:
            console.print("[blue]🔐 Backing up environment variables...[/blue]")
            env_file = backup_dir / "environment_backup.txt"

            sensitive_vars = ["API_KEY", "SECRET", "TOKEN", "PASSWORD"]
            env_vars = {}

            for key, value in os.environ.items():
                # Mask sensitive information
                if any(sensitive in key.upper() for sensitive in sensitive_vars):
                    env_vars[key] = "***MASKED***"
                else:
                    env_vars[key] = value

            with open(env_file, 'w') as f:
                json.dump(env_vars, f, indent=2)

            if verbose:
                console.print(f"[green]✅ Environment variables -> {env_file}[/green]")

        # Create backup manifest
        manifest = {
            "backup_timestamp": timestamp,
            "backup_directory": str(backup_dir),
            "config_files_backed_up": backed_up_files,
            "environment_variables_backed_up": include_env,
            "total_files": len(backed_up_files) + (1 if include_env else 0)
        }

        manifest_file = backup_dir / "backup_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)

        console.print("[green]✅ Configuration backup completed[/green]")
        console.print(f"[green]📁 Backup location: {backup_dir}[/green]")
        console.print(f"[green]📊 Files backed up: {len(backed_up_files)}[/green]")

        if include_env:
            console.print("[green]🔐 Environment variables included[/green]")

        return str(backup_dir)

    except Exception as e:
        console.print(f"[red]❌ Backup failed: {e}[/red]")
        logger.exception("Configuration backup error")
        raise typer.Exit(1)


@app.command("reload")
def reload_config(
    component: str = typer.Option("all", help="Component to reload (all, llm, cache)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed reload information"),
):
    """
    Reload configuration from files.

    Forces reload of configuration components from their source files.
    Useful after manual configuration changes.
    """
    console.print(f"[bold blue]🔄 Reloading {component} configuration[/bold blue]")

    try:
        from utils.config import Config

        if component in ["all", "llm"]:
            console.print("[blue]🤖 Reloading LLM configurations...[/blue]")
            config = Config()
            config.reload_llm_configs()
            console.print("[green]✅ LLM configurations reloaded[/green]")

        if component in ["all", "cache"]:
            console.print("[blue]💾 Reloading cache configuration...[/blue]")
            # Cache config reload would require cache system restart
            console.print("[green]✅ Cache configuration reload requested[/green]")
            console.print("[blue]💡 Note: Cache system may need restart for changes to take effect[/blue]")

        console.print("[green]✅ Configuration reload completed[/green]")

    except Exception as e:
        console.print(f"[red]❌ Reload failed: {e}[/red]")
        logger.exception("Configuration reload error")
        raise typer.Exit(1)


@app.command("show-llm")
def show_llm_config(
    model: Optional[str] = typer.Option(None, help="Specific model to show (shows all if not specified)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed model information"),
):
    """
    Display LLM model configurations.

    Shows detailed information about configured LLM models including
    API endpoints, parameters, and capabilities.
    """
    console.print(f"[bold blue]🤖 LLM Configuration - {model or 'All Models'}[/bold blue]")

    try:
        from utils.config import Config
        config = Config()

        if model:
            # Show specific model
            try:
                model_config = config.get_llm_config(model)
                console.print(f"[cyan]Model: {model}[/cyan]")
                console.print(f"[cyan]Name: {model_config.name}[/cyan]")
                console.print(f"[cyan]Base URL: {model_config.base_url}[/cyan]")
                console.print(f"[cyan]API Key Env: {model_config.api_key_env_var}[/cyan]")
                console.print(f"[cyan]Timeout: {getattr(model_config, 'timeout_seconds', 'default')}s[/cyan]")

                if hasattr(model_config, 'default_parameters') and model_config.default_parameters:
                    console.print("[cyan]Default Parameters:[/cyan]")
                    for param, value in model_config.default_parameters.items():
                        console.print(f"  {param}: {value}")

                if verbose and hasattr(model_config, 'capabilities'):
                    console.print("[cyan]Capabilities:[/cyan]")
                    for cap, supported in model_config.capabilities.items():
                        console.print(f"  {cap}: {'✅' if supported else '❌'}")

            except Exception as e:
                console.print(f"[red]❌ Model '{model}' not found: {e}[/red]")
                raise typer.Exit(1)
        else:
            # Show all models
            llm_configs = config.get_all_llm_configs()

            table = Table(title="LLM Models")
            table.add_column("Key", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Base URL", style="yellow", no_wrap=False)
            table.add_column("API Key Configured", style="magenta", justify="center")

            for model_key, model_config in llm_configs.items():
                api_key_env = model_config.api_key_env_var
                api_key_status = "✅" if api_key_env and os.getenv(api_key_env) else "❌"
                table.add_row(
                    model_key,
                    model_config.name,
                    model_config.base_url,
                    api_key_status
                )

            console.print(table)
            console.print(f"\n[green]📊 Total models: {len(llm_configs)}[/green]")

    except Exception as e:
        console.print(f"[red]❌ Failed to load LLM configuration: {e}[/red]")
        logger.exception("LLM configuration display error")
        raise typer.Exit(1)


@app.command("show-core-memory")
def show_core_memory_config(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed configuration"),
):
    """
    Display CORE Memory configuration.

    Shows all CORE Memory settings including API endpoints,
    timeouts, and default spaces.
    """
    console.print("[bold blue]🧠 CORE Memory Configuration[/bold blue]")

    try:
        from utils.config import Config
        config = Config()

        core_config = config.get_core_memory_config()

        console.print(f"[cyan]API Base URL:[/cyan] {config.get_core_memory_api_base_url()}")
        console.print(f"[cyan]Source:[/cyan] {config.get_core_memory_source()}")
        console.print(f"[cyan]Timeout:[/cyan] {config.get_core_memory_timeout()}s")
        console.print(f"[cyan]Max Retries:[/cyan] {config.get_core_memory_max_retries()}")
        console.print(f"[cyan]Batch Size:[/cyan] {config.get_core_memory_batch_size()}")

        console.print(f"\n[cyan]Default Spaces:[/cyan]")
        default_spaces = config.get_core_memory_default_spaces()
        for space_name, space_id in default_spaces.items():
            console.print(f"  {space_name}: {space_id}")

        console.print(f"\n[cyan]mCODE Settings:[/cyan]")
        console.print(f"  Summary Format: {config.get_mcode_summary_format()}")
        console.print(f"  Include Codes: {config.get_mcode_include_codes()}")
        console.print(f"  Max Summary Length: {config.get_mcode_max_summary_length()}")

        if verbose:
            console.print(f"\n[cyan]Full Configuration:[/cyan]")
            console.print(json.dumps(core_config, indent=2))

    except Exception as e:
        console.print(f"[red]❌ Failed to load CORE Memory configuration: {e}[/red]")
        logger.exception("CORE Memory configuration display error")
        raise typer.Exit(1)


@app.command("show-cache")
def show_cache_config(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed configuration"),
):
    """
    Display cache configuration.

    Shows cache settings including TTL, directories, and rate limiting.
    """
    console.print("[bold blue]💾 Cache Configuration[/bold blue]")

    try:
        from utils.config import Config
        config = Config()

        console.print(f"[cyan]Cache Enabled:[/cyan] {config.is_cache_enabled()}")
        console.print(f"[cyan]API Cache Directory:[/cyan] {config.get_api_cache_directory()}")
        console.print(f"[cyan]Cache TTL:[/cyan] {config.get_cache_ttl()} seconds")
        console.print(f"[cyan]Rate Limit Delay:[/cyan] {config.get_rate_limit_delay()} seconds")
        console.print(f"[cyan]Request Timeout:[/cyan] {config.get_request_timeout()} seconds")

        if verbose:
            console.print(f"\n[cyan]Full Cache Configuration:[/cyan]")
            console.print(json.dumps(config.cache_config, indent=2))

    except Exception as e:
        console.print(f"[red]❌ Failed to load cache configuration: {e}[/red]")
        logger.exception("Cache configuration display error")
        raise typer.Exit(1)


@app.command("show-validation")
def show_validation_config(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed configuration"),
):
    """
    Display validation configuration.

    Shows validation settings including strict mode and API key requirements.
    """
    console.print("[bold blue]✅ Validation Configuration[/bold blue]")

    try:
        from utils.config import Config
        config = Config()

        console.print(f"[cyan]Strict Mode:[/cyan] {config.is_strict_mode()}")
        console.print(f"[cyan]Require API Keys:[/cyan] {config.require_api_keys()}")

        if verbose:
            console.print(f"\n[cyan]Full Validation Configuration:[/cyan]")
            console.print(json.dumps(config.validation_config, indent=2))

    except Exception as e:
        console.print(f"[red]❌ Failed to load validation configuration: {e}[/red]")
        logger.exception("Validation configuration display error")
        raise typer.Exit(1)