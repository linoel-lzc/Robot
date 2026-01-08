import os
from pathlib import Path
import importlib.resources

def get_resource_path(filename: str, resource_type: str = None) -> Path:
    """
    Resolve the path to a resource file.
    
    Priority:
    1. Exact path (if filename is a path and exists)
    2. Current working directory (if filename is just a name)
    3. Package resources (robotarm.resources.{resource_type})
    
    Args:
        filename: The name or path of the file.
        resource_type: 'configs', 'layouts', or 'scores'. Required if looking in package resources.
        
    Returns:
        Path object to the file.
        
    Raises:
        FileNotFoundError: If the file cannot be found.
    """
    # 1. Check if it's an exact path or in CWD
    path = Path(filename)
    if path.exists():
        return path
    
    # 2. Check if it's in CWD under the specific directory (e.g. ./configs/filename)
    if resource_type:
        cwd_path = Path.cwd() / resource_type / filename
        if cwd_path.exists():
            return cwd_path

    # 3. Check package resources
    if resource_type:
        try:
            # For Python 3.9+ we can use files()
            # Assuming the structure is robotarm/resources/<resource_type>/<filename>
            # We map resource_type 'configs' to package 'robotarm.resources.configs'
            package_path = importlib.resources.files(f'robotarm.resources.{resource_type}') / filename
            if package_path.is_file():
                return Path(package_path)
        except (ImportError, ModuleNotFoundError, AttributeError):
            # Fallback for older python or if structure is different? 
            # But we are setting up the structure now, so this should work.
            pass
            
    raise FileNotFoundError(f"Resource '{filename}' not found in local path or package resources.")

def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path

