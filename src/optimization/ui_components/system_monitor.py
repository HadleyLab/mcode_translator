"""
System Status Monitoring Components for Modern Optimization UI
"""

import psutil
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

class SystemMonitor:
    """Monitor system resources and status"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        from src.utils.config import Config
        self.cache_dir = Path(cache_dir) if cache_dir else Path(Config().get_api_cache_directory())
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current system resource usage"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_mb': psutil.virtual_memory().used / (1024 * 1024),
            'memory_total_mb': psutil.virtual_memory().total / (1024 * 1024),
            'disk_percent': psutil.disk_usage('/').percent
        }
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get cache status information"""
        if not self.cache_dir.exists():
            return {
                'enabled': False,
                'size_mb': 0,
                'file_count': 0
            }
        
        # Calculate cache size and file count
        total_size = 0
        file_count = 0
        
        for file_path in self.cache_dir.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1
        
        return {
            'enabled': True,
            'size_mb': total_size / (1024 * 1024),
            'file_count': file_count
        }
    
    def clear_cache(self) -> None:
        """Clear the cache directory"""
        if self.cache_dir.exists():
            for file_path in self.cache_dir.rglob('*'):
                if file_path.is_file():
                    file_path.unlink()
    
    def get_configuration_status(self) -> Dict[str, Any]:
        """Get configuration file status"""
        config_files = {
            'prompts_config': 'prompts/prompts_config.json',
            'models_config': 'models/models_config.json',
            'main_config': 'config.json'
        }
        
        status = {}
        for name, path in config_files.items():
            file_path = Path(path)
            status[name] = {
                'exists': file_path.exists(),
                'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat() if file_path.exists() else None,
                'size_bytes': file_path.stat().st_size if file_path.exists() else 0
            }
        
        return status