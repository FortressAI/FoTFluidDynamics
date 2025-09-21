# API Reference

**Complete API Documentation for the Field of Truth vQbit Framework**

---

## Overview

The FoT Framework provides a comprehensive API for solving the Navier-Stokes Millennium Prize Problem. This reference covers all public classes, methods, and functions available to users and developers.

---

## Core Modules

### `core.vqbit_engine`

#### `VQbitEngine`

Main engine for quantum virtue-coherence operations.

```python
class VQbitEngine:
    """
    8096-dimensional quantum state engine with virtue operators
    """
    
    def __init__(self, vqbit_dimension: int = 8096, virtue_weights: List[float] = None):
        """
        Initialize vQbit engine
        
        Args:
            vqbit_dimension: Dimension of quantum Hilbert space (default: 8096)
            virtue_weights: Weights for [Justice, Temperance, Prudence, Fortitude]
        """
    
    def initialize(self) -> None:
        """Initialize virtue operators and quantum space"""
    
    def create_vqbit_state(self, data: np.ndarray, virtue_type: VirtueType) -> VQbitState:
        """
        Create quantum state from classical data
        
        Args:
            data: Classical data to encode (velocity field, vorticity, etc.)
            virtue_type: Primary virtue for state initialization
            
        Returns:
            VQbitState: Quantum state representation
        """
    
    def evolve_vqbit_state(self, state: VQbitState, time_step: float) -> VQbitState:
        """
        Evolve quantum state using virtue-guided dynamics
        
        Args:
            state: Current quantum state
            time_step: Time evolution step
            
        Returns:
            VQbitState: Evolved quantum state
        """
    
    def compute_virtue_coherence(self, state: VQbitState) -> float:
        """
        Compute virtue coherence score
        
        Args:
            state: Quantum state to analyze
            
        Returns:
            float: Coherence score [0, 1]
        """
    
    def extract_classical_data(self, state: VQbitState) -> np.ndarray:
        """
        Extract classical data from quantum state
        
        Args:
            state: Quantum state
            
        Returns:
            np.ndarray: Classical field data
        """
```

#### `VQbitState`

Quantum state representation in 8096-dimensional Hilbert space.

```python
class VQbitState:
    """
    Quantum state in virtue-coherence framework
    """
    
    def __init__(self, amplitudes: np.ndarray, virtue_scores: Dict[VirtueType, float]):
        """
        Initialize quantum state
        
        Args:
            amplitudes: Complex amplitudes in quantum basis
            virtue_scores: Virtue scores for each cardinal virtue
        """
    
    @property
    def norm(self) -> float:
        """State normalization (should be 1.0)"""
    
    @property 
    def coherence(self) -> float:
        """Overall virtue coherence"""
    
    @property
    def entanglement_entropy(self) -> float:
        """von Neumann entropy of entanglement"""
    
    def normalize(self) -> 'VQbitState':
        """Return normalized state"""
    
    def inner_product(self, other: 'VQbitState') -> complex:
        """Compute inner product with another state"""
    
    def expectation_value(self, operator: np.ndarray) -> float:
        """Compute expectation value of operator"""
```

#### `VirtueType`

Enumeration of cardinal virtues.

```python
class VirtueType(Enum):
    """Cardinal virtues as quantum operators"""
    JUSTICE = "justice"           # Conservation enforcement
    TEMPERANCE = "temperance"     # Vorticity moderation  
    PRUDENCE = "prudence"         # Long-term stability
    FORTITUDE = "fortitude"       # Robustness
```

---

### `core.navier_stokes_engine`

#### `NavierStokesEngine`

Classical PDE solver with quantum virtue guidance.

```python
class NavierStokesEngine:
    """
    Navier-Stokes equation solver with virtue enhancement
    """
    
    def __init__(self, spatial_dimension: int = 3, viscosity: float = 0.01):
        """
        Initialize Navier-Stokes solver
        
        Args:
            spatial_dimension: Spatial dimensions (2 or 3)
            viscosity: Kinematic viscosity parameter
        """
    
    def initialize(self, vqbit_engine: VQbitEngine) -> None:
        """Initialize with vQbit engine coupling"""
    
    def solve_millennium_problem(self, problem_id: str) -> Solution:
        """
        Solve specific Millennium Prize problem instance
        
        Args:
            problem_id: Unique problem identifier
            
        Returns:
            Solution: Complete solution with verification data
        """
    
    def compute_vorticity_evolution(self, velocity_field: np.ndarray) -> np.ndarray:
        """
        Compute vorticity evolution under virtue guidance
        
        Args:
            velocity_field: Current velocity field
            
        Returns:
            np.ndarray: Evolved vorticity field
        """
    
    def check_regularity_criteria(self, solution: Solution) -> Dict[str, bool]:
        """
        Check all regularity criteria for solution
        
        Args:
            solution: Solution to verify
            
        Returns:
            Dict[str, bool]: Verification results for each criterion
        """
```

#### `Solution`

Solution representation with verification data.

```python
class Solution:
    """
    Complete solution representation
    """
    
    def __init__(self, velocity: np.ndarray, pressure: np.ndarray, 
                 time_points: np.ndarray, virtue_evolution: List[float]):
        """
        Initialize solution
        
        Args:
            velocity: Velocity field evolution
            pressure: Pressure field evolution  
            time_points: Time discretization
            virtue_evolution: Virtue coherence over time
        """
    
    @property
    def is_valid(self) -> bool:
        """Check if solution is valid"""
    
    @property
    def max_velocity_norm(self) -> float:
        """Maximum velocity magnitude"""
    
    @property
    def energy_evolution(self) -> np.ndarray:
        """Kinetic energy over time"""
    
    def verify_conservation_laws(self) -> Dict[str, float]:
        """Verify conservation law compliance"""
    
    def compute_regularity_metrics(self) -> Dict[str, float]:
        """Compute solution regularity metrics"""
```

---

### `core.millennium_solver`

#### `MillenniumSolver`

Specialized solver for Clay Institute problem.

```python
class MillenniumSolver:
    """
    Clay Institute Millennium Prize Problem solver
    """
    
    def __init__(self):
        """Initialize millennium solver"""
    
    def initialize(self, vqbit_engine: VQbitEngine, ns_engine: NavierStokesEngine) -> None:
        """Initialize with component engines"""
    
    def solve_millennium_problem(self, problem_id: str) -> MillenniumProof:
        """
        Solve Millennium Prize Problem instance
        
        Args:
            problem_id: Problem instance identifier
            
        Returns:
            MillenniumProof: Complete proof with verification
        """
    
    def verify_millennium_conditions(self, solution: Solution) -> MillenniumProof:
        """
        Verify all four Millennium Prize conditions
        
        Args:
            solution: Solution to verify
            
        Returns:
            MillenniumProof: Verification results
        """
    
    def generate_proof_certificate(self, proof: MillenniumProof) -> ProofCertificate:
        """
        Generate formal proof certificate
        
        Args:
            proof: Verified millennium proof
            
        Returns:
            ProofCertificate: Clay Institute compatible certificate
        """
```

#### `MillenniumProof`

Complete proof verification results.

```python
class MillenniumProof:
    """
    Verification results for Millennium Prize conditions
    """
    
    def __init__(self, problem_id: str, solution: Solution):
        """
        Initialize proof verification
        
        Args:
            problem_id: Problem instance identifier
            solution: Solution to verify
        """
    
    @property
    def global_existence_verified(self) -> bool:
        """Global existence verification result"""
    
    @property  
    def uniqueness_verified(self) -> bool:
        """Uniqueness verification result"""
    
    @property
    def smoothness_verified(self) -> bool:
        """Smoothness verification result"""
    
    @property
    def energy_conservation_verified(self) -> bool:
        """Energy conservation verification result"""
    
    @property
    def overall_confidence(self) -> float:
        """Overall proof confidence [0, 1]"""
    
    def generate_verification_report(self) -> str:
        """Generate detailed verification report"""
    
    def export_to_latex(self) -> str:
        """Export proof to LaTeX format"""
```

#### `ProofCertificate`

Formal proof certificate for submission.

```python
class ProofCertificate:
    """
    Formal proof certificate for Clay Institute submission
    """
    
    def __init__(self, proof: MillenniumProof, author: str, institution: str):
        """
        Initialize proof certificate
        
        Args:
            proof: Verified millennium proof
            author: Principal author name
            institution: Affiliated institution
        """
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert certificate to dictionary"""
    
    def to_json(self) -> str:
        """Convert certificate to JSON string"""
    
    def export_pdf(self, filename: str) -> None:
        """Export certificate to PDF"""
    
    def validate_for_submission(self) -> List[str]:
        """Validate certificate for Clay Institute submission"""
```

---

## Utility Functions

### `utils.data_processing`

```python
def load_initial_conditions(filename: str) -> np.ndarray:
    """Load initial conditions from file"""

def save_solution(solution: Solution, filename: str) -> None:
    """Save solution to file"""

def visualize_velocity_field(velocity: np.ndarray, title: str = None) -> None:
    """Create velocity field visualization"""

def compute_energy_spectrum(velocity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute kinetic energy spectrum"""
```

### `utils.validation`

```python
def validate_initial_conditions(u0: np.ndarray) -> List[str]:
    """Validate initial conditions for Navier-Stokes"""

def check_divergence_free(velocity: np.ndarray, tolerance: float = 1e-12) -> bool:
    """Check if velocity field is divergence-free"""

def verify_smoothness(field: np.ndarray, order: int = 2) -> bool:
    """Verify field smoothness to specified order"""

def compute_reynolds_number(velocity: np.ndarray, length_scale: float, viscosity: float) -> float:
    """Compute Reynolds number for flow"""
```

---

## Configuration Classes

### `Config`

Global configuration management.

```python
class Config:
    """
    Global configuration for FoT framework
    """
    
    @classmethod
    def load_from_file(cls, filename: str) -> 'Config':
        """Load configuration from JSON file"""
    
    @classmethod
    def get_default(cls) -> 'Config':
        """Get default configuration"""
    
    def save_to_file(self, filename: str) -> None:
        """Save configuration to file"""
    
    def update(self, **kwargs) -> None:
        """Update configuration parameters"""
```

### `VQbitConfig`

vQbit engine specific configuration.

```python
class VQbitConfig:
    """Configuration for vQbit engine"""
    
    def __init__(self):
        self.dimension = 8096
        self.virtue_weights = [0.25, 0.25, 0.25, 0.25]
        self.coherence_threshold = 0.8
        self.max_iterations = 10000
        self.convergence_tolerance = 1e-12
        self.use_gpu = False
        self.num_threads = None  # Auto-detect
```

---

## Error Handling

### Custom Exceptions

```python
class FoTException(Exception):
    """Base exception for FoT framework"""
    pass

class VQbitException(FoTException):
    """vQbit engine related exceptions"""
    pass

class NavierStokesException(FoTException):
    """Navier-Stokes solver exceptions"""
    pass

class MillenniumException(FoTException):
    """Millennium solver exceptions"""
    pass

class VirtueCoherenceLoss(VQbitException):
    """Raised when virtue coherence drops below threshold"""
    pass

class ConvergenceError(FoTException):
    """Raised when solver fails to converge"""
    pass

class ValidationError(FoTException):
    """Raised when input validation fails"""
    pass
```

---

## Examples

### Basic Usage

```python
from core.vqbit_engine import VQbitEngine, VirtueType
from core.navier_stokes_engine import NavierStokesEngine  
from core.millennium_solver import MillenniumSolver

# Initialize components
vqbit_engine = VQbitEngine(vqbit_dimension=8096)
vqbit_engine.initialize()

ns_engine = NavierStokesEngine(spatial_dimension=3, viscosity=0.01)
ns_engine.initialize(vqbit_engine)

millennium_solver = MillenniumSolver()
millennium_solver.initialize(vqbit_engine, ns_engine)

# Solve Millennium Prize Problem
problem_id = "millennium_re1000_L1.0"
proof = millennium_solver.solve_millennium_problem(problem_id)

# Generate certificate
certificate = millennium_solver.generate_proof_certificate(proof)
print(f"Proof confidence: {proof.overall_confidence:.2%}")
```

### Advanced Usage

```python
# Custom virtue weights
virtue_weights = [0.3, 0.2, 0.3, 0.2]  # Emphasize Justice and Prudence
vqbit_engine = VQbitEngine(virtue_weights=virtue_weights)

# Custom initial conditions
import numpy as np
u0 = np.random.randn(3, 64, 64, 64)  # Random initial velocity
u0 = make_divergence_free(u0)  # Ensure incompressibility

# Create vQbit state
state = vqbit_engine.create_vqbit_state(u0, VirtueType.JUSTICE)

# Manual evolution
dt = 1e-4
for t in np.linspace(0, 1, 10000):
    state = vqbit_engine.evolve_vqbit_state(state, dt)
    coherence = vqbit_engine.compute_virtue_coherence(state)
    
    if coherence < 0.5:
        print(f"Warning: Low coherence at t={t}")
        break

# Extract final solution
final_velocity = vqbit_engine.extract_classical_data(state)
```

---

## Performance Optimization

### Memory Management

```python
# For systems with limited memory
vqbit_engine = VQbitEngine(vqbit_dimension=1024)  # Reduced dimension

# Enable memory monitoring
import psutil
memory_usage = psutil.virtual_memory().percent
if memory_usage > 80:
    print("Warning: High memory usage")
```

### Parallel Processing

```python
# Enable multi-threading
import os
os.environ['OMP_NUM_THREADS'] = '8'  # Use 8 cores

# GPU acceleration (if available)
vqbit_engine = VQbitEngine(use_gpu=True)
```

### Caching

```python
# Enable result caching
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_solve_problem(problem_id):
    return millennium_solver.solve_millennium_problem(problem_id)
```

---

## Testing

### Unit Tests

```python
import pytest
from core.vqbit_engine import VQbitEngine

def test_vqbit_initialization():
    """Test vQbit engine initialization"""
    engine = VQbitEngine(vqbit_dimension=128)
    engine.initialize()
    assert engine.virtue_operators is not None
    assert len(engine.virtue_operators) == 4

def test_virtue_coherence():
    """Test virtue coherence computation"""
    engine = VQbitEngine(vqbit_dimension=128)
    engine.initialize()
    
    # Create test state
    amplitudes = np.random.randn(128) + 1j * np.random.randn(128)
    amplitudes /= np.linalg.norm(amplitudes)
    
    state = VQbitState(amplitudes, {})
    coherence = engine.compute_virtue_coherence(state)
    
    assert 0 <= coherence <= 1
```

### Integration Tests

```python
def test_millennium_solver_integration():
    """Test complete millennium solver workflow"""
    # Initialize all components
    vqbit_engine = VQbitEngine(vqbit_dimension=256)
    vqbit_engine.initialize()
    
    ns_engine = NavierStokesEngine()
    ns_engine.initialize(vqbit_engine)
    
    millennium_solver = MillenniumSolver()
    millennium_solver.initialize(vqbit_engine, ns_engine)
    
    # Solve simple problem
    problem_id = "test_re100_L0.5"
    proof = millennium_solver.solve_millennium_problem(problem_id)
    
    # Verify results
    assert proof.global_existence_verified
    assert proof.uniqueness_verified
    assert proof.smoothness_verified
    assert proof.energy_conservation_verified
    assert proof.overall_confidence > 0.9
```

---

## API Versioning

Current API version: **2.1.0**

### Version History
- **2.1.0**: Added entropy control theory integration
- **2.0.0**: Complete vQbit framework implementation
- **1.5.0**: Millennium solver integration
- **1.0.0**: Initial release with basic vQbit engine

### Backwards Compatibility
The API maintains backwards compatibility within major versions. Deprecated features are marked and will be removed in the next major version.

---

## Rate Limits and Quotas

### Local Usage
No limits for local installation.

### Cloud API (Future)
- **Free Tier**: 100 requests/day
- **Research Tier**: 1000 requests/day  
- **Enterprise Tier**: Unlimited

---

**Author**: Rick Gillespie  
**Institution**: FortressAI Research Institute  
**Email**: bliztafree@gmail.com  
**Last Updated**: September 2025  
**API Version**: 2.1.0
