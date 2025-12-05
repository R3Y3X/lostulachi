"""
Utilidades compartidas para carga de configuración
"""

import os
from pathlib import Path
from dotenv import load_dotenv

def load_config():
    """
    Carga la configuración desde variables de entorno.
    
    Returns:
        dict: Diccionario con las configuraciones
    """
    root_dir = Path(__file__).parent.parent.parent
    env_path = root_dir / '.env'
    
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    
    return {
        'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY'),
        'ROOT_DIR': root_dir
    }

