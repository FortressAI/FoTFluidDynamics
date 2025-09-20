#!/usr/bin/env python3
"""
Startup script for FoT Fluid Dynamics Streamlit app
Ensures proper initialization for cloud deployment
"""

import os
import sys
from pathlib import Path
import json
import logging

def setup_logging():
    """Configure logging for startup"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def ensure_directories():
    """Ensure all required directories exist"""
    logger = logging.getLogger(__name__)
    
    directories = [
        "data",
        "data/millennium_proofs",
        ".streamlit"
    ]
    
    for dir_path in directories:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
        else:
            logger.info(f"Directory exists: {dir_path}")

def check_dependencies():
    """Check if critical dependencies are available"""
    logger = logging.getLogger(__name__)
    
    try:
        import streamlit
        logger.info(f"Streamlit version: {streamlit.__version__}")
        
        import numpy
        logger.info(f"NumPy version: {numpy.__version__}")
        
        import pandas
        logger.info(f"Pandas version: {pandas.__version__}")
        
        import plotly
        logger.info(f"Plotly version: {plotly.__version__}")
        
        return True
        
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return False

def initialize_proof_storage():
    """Initialize proof storage with minimal data if empty"""
    logger = logging.getLogger(__name__)
    
    proof_file = Path("data/millennium_proofs/millennium_proofs.json")
    solution_file = Path("data/millennium_proofs/solution_sequences.json")
    
    # Create empty files if they don't exist
    if not proof_file.exists():
        with open(proof_file, 'w') as f:
            json.dump({}, f)
        logger.info("Initialized empty proof storage")
    
    if not solution_file.exists():
        with open(solution_file, 'w') as f:
            json.dump({}, f)
        logger.info("Initialized empty solution storage")

def main():
    """Run startup initialization"""
    logger = setup_logging()
    logger.info("🚀 FoT Fluid Dynamics - Startup Initialization")
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    logger.info(f"Python version: {python_version}")
    
    # Setup directories
    ensure_directories()
    
    # Check dependencies
    if not check_dependencies():
        logger.error("❌ Dependency check failed")
        sys.exit(1)
    
    # Initialize storage
    initialize_proof_storage()
    
    logger.info("✅ Startup initialization complete")
    logger.info("Ready to launch Streamlit app")

if __name__ == "__main__":
    main()
