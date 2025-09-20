"""
FoT Core Package - Field of Truth vQbit Framework
"""

__version__ = "1.0.0"
__author__ = "Rick Gillespie"
__email__ = "bliztafree@gmail.com"

# Make core modules available for import
try:
    from .vqbit_engine import VQbitEngine, VQbitState
    from .navier_stokes_engine import NavierStokesEngine
    from .millennium_solver import MillenniumSolver
    
    __all__ = [
        'VQbitEngine',
        'VQbitState', 
        'NavierStokesEngine',
        'MillenniumSolver'
    ]
except ImportError as e:
    # Graceful fallback for cloud environments
    import logging
    logging.warning(f"Some core modules not available: {e}")
    __all__ = []
