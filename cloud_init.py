#!/usr/bin/env python3
"""
Cloud initialization script for Streamlit Cloud deployment
Ensures proper setup before the main app starts
"""

import os
import sys
import json
import logging
from pathlib import Path

def setup_cloud_environment():
    """Setup cloud environment for FoT app"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🌥️ Initializing FoT app for Streamlit Cloud")
    
    # Detect cloud environment
    is_cloud = (
        os.environ.get('STREAMLIT_SHARING_MODE') or 
        '/mount/src/' in os.getcwd() or 
        '/home/adminuser/' in os.path.expanduser('~')
    )
    
    if is_cloud:
        logger.info("✅ Cloud environment detected")
    else:
        logger.info("🏠 Local environment detected")
    
    # Ensure data directories exist
    try:
        data_dir = Path("data/millennium_proofs")
        data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Data directory ready: {data_dir}")
        
        # Create minimal proof storage if it doesn't exist
        proof_file = data_dir / "millennium_proofs.json"
        if not proof_file.exists():
            with open(proof_file, 'w') as f:
                json.dump({}, f)
            logger.info("📄 Initialized empty proof storage")
            
        solution_file = data_dir / "solution_sequences.json"
        if not solution_file.exists():
            with open(solution_file, 'w') as f:
                json.dump({}, f)
            logger.info("📄 Initialized empty solution storage")
            
    except Exception as e:
        logger.warning(f"⚠️ Could not setup data directory: {e}")
    
    # Check Python version compatibility
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    logger.info(f"🐍 Python version: {python_version}")
    
    if sys.version_info.major < 3 or sys.version_info.minor < 8:
        logger.warning("⚠️ Python version may cause compatibility issues")
    
    # Test critical imports
    try:
        import streamlit
        import pandas
        import numpy
        import plotly
        logger.info("✅ Critical dependencies available")
    except ImportError as e:
        logger.error(f"❌ Missing critical dependency: {e}")
        return False
    
    logger.info("🚀 Cloud initialization complete")
    return True

if __name__ == "__main__":
    success = setup_cloud_environment()
    sys.exit(0 if success else 1)
