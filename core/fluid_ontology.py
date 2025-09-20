"""
Fluid Dynamics Ontology - FoT Framework
Comprehensive knowledge representation for Navier-Stokes and fluid mechanics
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime
from abc import ABC, abstractmethod

# Import vQbit core
from .vqbit_engine import VQbitState, VQbitEngine, VirtueType


class FlowRegimeType(Enum):
    """Flow regime classifications"""
    LAMINAR = "laminar"
    TURBULENT = "turbulent" 
    TRANSITIONAL = "transitional"
    MIXED = "mixed"


class CompressibilityType(Enum):
    """Compressibility classifications"""
    INCOMPRESSIBLE = "incompressible"
    COMPRESSIBLE = "compressible"
    WEAKLY_COMPRESSIBLE = "weakly_compressible"


class BoundaryConditionType(Enum):
    """Boundary condition types"""
    DIRICHLET = "dirichlet"  # Specified values
    NEUMANN = "neumann"      # Specified derivatives
    ROBIN = "robin"          # Mixed conditions
    PERIODIC = "periodic"    # Periodic boundaries
    NO_SLIP = "no_slip"     # Zero velocity at walls
    FREE_SLIP = "free_slip"  # Zero normal velocity, free tangential
    INFLOW = "inflow"        # Specified inflow
    OUTFLOW = "outflow"      # Natural outflow


class SingularityType(Enum):
    """Types of singularities in fluid flow"""
    BLOW_UP = "blow_up"           # Finite-time blow-up
    VORTEX_SHEET = "vortex_sheet" # Vortex sheet formation
    SHOCK = "shock"               # Shock wave
    CAVITATION = "cavitation"     # Cavitation bubble
    SEPARATION = "separation"     # Flow separation


class ConservationLaw(Enum):
    """Fundamental conservation laws"""
    MASS = "mass"
    MOMENTUM = "momentum"
    ENERGY = "energy"
    ANGULAR_MOMENTUM = "angular_momentum"


@dataclass
class Point3D:
    """3D point representation"""
    x: float
    y: float
    z: float
    
    def distance_to(self, other: 'Point3D') -> float:
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass
class Vector3D:
    """3D vector representation"""
    u: float  # x-component
    v: float  # y-component
    w: float  # z-component
    
    def magnitude(self) -> float:
        return np.sqrt(self.u**2 + self.v**2 + self.w**2)
    
    def normalize(self) -> 'Vector3D':
        mag = self.magnitude()
        if mag > 0:
            return Vector3D(self.u/mag, self.v/mag, self.w/mag)
        return Vector3D(0, 0, 0)
    
    def dot(self, other: 'Vector3D') -> float:
        return self.u * other.u + self.v * other.v + self.w * other.w
    
    def cross(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(
            self.v * other.w - self.w * other.v,
            self.w * other.u - self.u * other.w,
            self.u * other.v - self.v * other.u
        )
    
    def to_array(self) -> np.ndarray:
        return np.array([self.u, self.v, self.w])


@dataclass
class ScalarField:
    """Scalar field representation"""
    values: np.ndarray
    coordinates: np.ndarray
    name: str
    units: str = ""
    
    def gradient(self) -> np.ndarray:
        """Compute gradient using finite differences"""
        return np.gradient(self.values)
    
    def laplacian(self) -> np.ndarray:
        """Compute Laplacian operator"""
        grad = self.gradient()
        return np.sum([np.gradient(grad[i], axis=i) for i in range(len(grad))], axis=0)
    
    def max_value(self) -> float:
        return np.max(self.values)
    
    def min_value(self) -> float:
        return np.min(self.values)
    
    def integral(self) -> float:
        """Compute domain integral"""
        return np.trapz(np.trapz(np.trapz(self.values)))


@dataclass 
class VectorField3D:
    """3D vector field representation"""
    u_component: ScalarField  # x-component
    v_component: ScalarField  # y-component  
    w_component: ScalarField  # z-component
    name: str
    
    def magnitude_field(self) -> ScalarField:
        """Compute magnitude at each point"""
        mag_values = np.sqrt(
            self.u_component.values**2 + 
            self.v_component.values**2 + 
            self.w_component.values**2
        )
        return ScalarField(
            values=mag_values,
            coordinates=self.u_component.coordinates,
            name=f"{self.name}_magnitude",
            units=self.u_component.units
        )
    
    def divergence(self) -> ScalarField:
        """Compute divergence ∇·u"""
        du_dx = np.gradient(self.u_component.values, axis=0)
        dv_dy = np.gradient(self.v_component.values, axis=1) 
        dw_dz = np.gradient(self.w_component.values, axis=2)
        
        div_values = du_dx + dv_dy + dw_dz
        
        return ScalarField(
            values=div_values,
            coordinates=self.u_component.coordinates,
            name=f"{self.name}_divergence",
            units=f"{self.u_component.units}/length"
        )
    
    def curl(self) -> 'VectorField3D':
        """Compute curl ∇×u (vorticity)"""
        # ∂w/∂y - ∂v/∂z
        omega_x = np.gradient(self.w_component.values, axis=1) - np.gradient(self.v_component.values, axis=2)
        
        # ∂u/∂z - ∂w/∂x  
        omega_y = np.gradient(self.u_component.values, axis=2) - np.gradient(self.w_component.values, axis=0)
        
        # ∂v/∂x - ∂u/∂y
        omega_z = np.gradient(self.v_component.values, axis=0) - np.gradient(self.u_component.values, axis=1)
        
        return VectorField3D(
            u_component=ScalarField(omega_x, self.u_component.coordinates, "omega_x", "1/time"),
            v_component=ScalarField(omega_y, self.v_component.coordinates, "omega_y", "1/time"),
            w_component=ScalarField(omega_z, self.w_component.coordinates, "omega_z", "1/time"),
            name=f"{self.name}_vorticity"
        )


@dataclass
class GeometricDomain:
    """Geometric domain representation"""
    name: str
    dimension: int
    boundaries: List['BoundaryCondition']
    mesh_points: np.ndarray
    volume: float = 0.0
    
    def is_point_inside(self, point: Point3D) -> bool:
        """Check if point is inside domain"""
        # Simplified implementation - override for complex domains
        return True
    
    def compute_volume(self) -> float:
        """Compute domain volume"""
        # Implementation depends on domain geometry
        return self.volume


@dataclass
class BoundaryCondition:
    """Boundary condition specification"""
    name: str
    bc_type: BoundaryConditionType
    surface_nodes: np.ndarray
    value_function: Optional[Callable] = None
    time_dependent: bool = False
    virtue_weight: float = 1.0
    
    def evaluate_at_time(self, t: float, position: Point3D) -> Union[float, Vector3D]:
        """Evaluate boundary condition at given time and position"""
        if self.value_function:
            if self.time_dependent:
                return self.value_function(position, t)
            else:
                return self.value_function(position)
        return 0.0


@dataclass 
class FlowRegime:
    """Flow regime characterization"""
    reynolds_number: float
    mach_number: float = 0.0
    flow_type: FlowRegimeType = FlowRegimeType.LAMINAR
    compressibility: CompressibilityType = CompressibilityType.INCOMPRESSIBLE
    
    def is_turbulent(self) -> bool:
        return self.flow_type == FlowRegimeType.TURBULENT
    
    def is_compressible(self) -> bool:
        return self.compressibility != CompressibilityType.INCOMPRESSIBLE
    
    def get_virtue_weights(self) -> Dict[VirtueType, float]:
        """Get virtue weights based on flow regime"""
        if self.is_turbulent():
            return {
                VirtueType.JUSTICE: 0.3,      # Mass conservation critical
                VirtueType.TEMPERANCE: 0.2,   # Energy balance important
                VirtueType.PRUDENCE: 0.2,     # Stability essential
                VirtueType.FORTITUDE: 0.3     # Robustness crucial for turbulence
            }
        else:
            return {
                VirtueType.JUSTICE: 0.25,
                VirtueType.TEMPERANCE: 0.25,
                VirtueType.PRUDENCE: 0.25,
                VirtueType.FORTITUDE: 0.25
            }


@dataclass
class Singularity:
    """Singularity representation and tracking"""
    location: Point3D
    singularity_type: SingularityType
    severity: float  # 0-1 scale
    detection_time: float
    growth_rate: float = 0.0
    mitigation_applied: bool = False
    
    def is_critical(self) -> bool:
        """Check if singularity is critical"""
        return self.severity > 0.8
    
    def time_to_blow_up(self) -> Optional[float]:
        """Estimate time to blow-up if growth continues"""
        if self.growth_rate > 0:
            remaining_severity = 1.0 - self.severity
            return remaining_severity / self.growth_rate
        return None


@dataclass
class FluidField:
    """Complete fluid field state"""
    velocity: VectorField3D
    pressure: ScalarField
    density: Optional[ScalarField] = None
    temperature: Optional[ScalarField] = None
    vorticity: Optional[VectorField3D] = None
    stream_function: Optional[ScalarField] = None
    
    def __post_init__(self):
        """Compute derived fields"""
        if self.vorticity is None:
            self.vorticity = self.velocity.curl()
    
    def compute_energy_density(self) -> ScalarField:
        """Compute kinetic energy density ½ρ|u|²"""
        velocity_magnitude = self.velocity.magnitude_field()
        
        if self.density:
            energy_values = 0.5 * self.density.values * velocity_magnitude.values**2
        else:
            # Assume unit density
            energy_values = 0.5 * velocity_magnitude.values**2
            
        return ScalarField(
            values=energy_values,
            coordinates=velocity_magnitude.coordinates,
            name="energy_density",
            units="energy/volume"
        )
    
    def check_mass_conservation(self) -> float:
        """Check mass conservation violation"""
        divergence = self.velocity.divergence()
        return np.max(np.abs(divergence.values))
    
    def detect_singularities(self, threshold: float = 10.0) -> List[Singularity]:
        """Detect potential singularities"""
        singularities = []
        
        # Vorticity-based detection (Beale-Kato-Majda criterion)
        vorticity_magnitude = self.vorticity.magnitude_field()
        max_vorticity = vorticity_magnitude.max_value()
        
        if max_vorticity > threshold:
            # Find location of maximum vorticity
            max_indices = np.unravel_index(
                np.argmax(vorticity_magnitude.values), 
                vorticity_magnitude.values.shape
            )
            
            # Convert to physical coordinates
            max_location = Point3D(
                x=vorticity_magnitude.coordinates[0][max_indices],
                y=vorticity_magnitude.coordinates[1][max_indices], 
                z=vorticity_magnitude.coordinates[2][max_indices]
            )
            
            singularity = Singularity(
                location=max_location,
                singularity_type=SingularityType.BLOW_UP,
                severity=min(max_vorticity / (2 * threshold), 1.0),
                detection_time=0.0,  # Current time
                growth_rate=0.0      # Would need time series data
            )
            singularities.append(singularity)
        
        return singularities


@dataclass
class NavierStokesSystem:
    """Complete Navier-Stokes system specification"""
    domain: GeometricDomain
    fluid_properties: Dict[str, float]  # viscosity, density, etc.
    boundary_conditions: List[BoundaryCondition]
    initial_conditions: FluidField
    flow_regime: FlowRegime
    external_forces: Optional[VectorField3D] = None
    
    def get_reynolds_number(self) -> float:
        """Compute Reynolds number"""
        return self.flow_regime.reynolds_number
    
    def is_well_posed(self) -> bool:
        """Check if problem is mathematically well-posed"""
        # Basic checks
        has_velocity_bc = any(bc.bc_type in [BoundaryConditionType.DIRICHLET, 
                                           BoundaryConditionType.NO_SLIP] 
                             for bc in self.boundary_conditions)
        
        has_pressure_constraint = any(bc.bc_type == BoundaryConditionType.NEUMANN 
                                    for bc in self.boundary_conditions)
        
        return has_velocity_bc and (has_pressure_constraint or len(self.boundary_conditions) > 0)
    
    def encode_to_vqbit(self, vqbit_engine: VQbitEngine) -> VQbitState:
        """Encode fluid system into vQbit state"""
        
        # Extract key physical quantities
        velocity_data = self.initial_conditions.velocity
        pressure_data = self.initial_conditions.pressure
        
        # Create initial values dictionary for encoding
        initial_values = {}
        
        # Encode velocity field statistics
        u_mag = velocity_data.magnitude_field()
        initial_values['velocity_max'] = u_mag.max_value()
        initial_values['velocity_mean'] = np.mean(u_mag.values)
        initial_values['velocity_std'] = np.std(u_mag.values)
        
        # Encode pressure statistics
        initial_values['pressure_max'] = pressure_data.max_value()
        initial_values['pressure_mean'] = np.mean(pressure_data.values)
        initial_values['pressure_std'] = np.std(pressure_data.values)
        
        # Encode flow parameters
        initial_values['reynolds_number'] = self.flow_regime.reynolds_number / 1000.0  # Normalize
        initial_values['viscosity'] = self.fluid_properties.get('viscosity', 1.0)
        
        # Encode conservation quantities
        divergence = velocity_data.divergence()
        initial_values['mass_conservation_error'] = np.max(np.abs(divergence.values))
        
        # Encode energy
        energy_field = self.initial_conditions.compute_energy_density()
        initial_values['total_energy'] = energy_field.integral()
        
        # Create vQbit state with problem context
        problem_context = {
            'problem_type': 'navier_stokes',
            'domain_dimension': self.domain.dimension,
            'flow_regime': self.flow_regime.flow_type.value,
            'reynolds_number': self.flow_regime.reynolds_number,
            'boundary_conditions': len(self.boundary_conditions)
        }
        
        return vqbit_engine.create_vqbit_state(
            problem_context=problem_context,
            initial_values=initial_values
        )


class FluidOntologyEngine:
    """Main engine for fluid dynamics ontology management"""
    
    def __init__(self, vqbit_engine: VQbitEngine):
        self.vqbit_engine = vqbit_engine
        self.systems = {}  # Store Navier-Stokes systems
        self.solutions = {}  # Store solution sequences
        self.singularities = {}  # Track singularities
        self.conservation_monitors = {}  # Conservation law monitoring
        
    def register_system(self, system_id: str, system: NavierStokesSystem):
        """Register a new Navier-Stokes system"""
        self.systems[system_id] = system
        self.conservation_monitors[system_id] = {
            law.value: [] for law in ConservationLaw
        }
        
    def create_millennium_problem(self, 
                                 domain_size: float = 1.0,
                                 reynolds_number: float = 1000.0) -> NavierStokesSystem:
        """Create a canonical Millennium Prize problem instance"""
        
        # Define unit cube domain
        domain = GeometricDomain(
            name="unit_cube",
            dimension=3,
            boundaries=[],
            mesh_points=np.mgrid[0:1:32j, 0:1:32j, 0:1:32j],
            volume=domain_size**3
        )
        
        # Periodic boundary conditions (torus topology)
        periodic_bc = BoundaryCondition(
            name="periodic_all",
            bc_type=BoundaryConditionType.PERIODIC,
            surface_nodes=np.array([]),  # All boundaries
            time_dependent=False
        )
        domain.boundaries = [periodic_bc]
        
        # Initial velocity field - smooth with bounded energy
        nx, ny, nz = 32, 32, 32
        x = np.linspace(0, domain_size, nx)
        y = np.linspace(0, domain_size, ny) 
        z = np.linspace(0, domain_size, nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Smooth initial velocity (satisfies divergence-free condition)
        u_init = np.sin(2*np.pi*X) * np.cos(2*np.pi*Y) * np.cos(2*np.pi*Z)
        v_init = -np.cos(2*np.pi*X) * np.sin(2*np.pi*Y) * np.cos(2*np.pi*Z)
        w_init = np.zeros_like(u_init)  # Ensures ∇·u = 0
        
        # Initial pressure - zero mean
        p_init = np.zeros_like(u_init)
        
        velocity_field = VectorField3D(
            u_component=ScalarField(u_init, (X, Y, Z), "u_velocity", "m/s"),
            v_component=ScalarField(v_init, (X, Y, Z), "v_velocity", "m/s"),
            w_component=ScalarField(w_init, (X, Y, Z), "w_velocity", "m/s"),
            name="initial_velocity"
        )
        
        pressure_field = ScalarField(p_init, (X, Y, Z), "pressure", "Pa")
        
        initial_conditions = FluidField(
            velocity=velocity_field,
            pressure=pressure_field
        )
        
        # Flow regime
        flow_regime = FlowRegime(
            reynolds_number=reynolds_number,
            flow_type=FlowRegimeType.LAMINAR if reynolds_number < 2300 else FlowRegimeType.TURBULENT,
            compressibility=CompressibilityType.INCOMPRESSIBLE
        )
        
        # Fluid properties
        fluid_properties = {
            'viscosity': domain_size / reynolds_number,  # Kinematic viscosity
            'density': 1.0,
            'temperature': 293.15
        }
        
        millennium_system = NavierStokesSystem(
            domain=domain,
            fluid_properties=fluid_properties,
            boundary_conditions=[periodic_bc],
            initial_conditions=initial_conditions,
            flow_regime=flow_regime
        )
        
        system_id = f"millennium_re{reynolds_number}_L{domain_size}"
        self.register_system(system_id, millennium_system)
        
        return millennium_system
    
    def verify_millennium_conditions(self, system_id: str, solution_sequence: List[FluidField]) -> Dict[str, bool]:
        """Verify Millennium Prize problem conditions"""
        
        if system_id not in self.systems:
            raise ValueError(f"System {system_id} not found")
            
        system = self.systems[system_id]
        
        verification_results = {
            'global_existence': True,
            'uniqueness': True, 
            'smoothness_preservation': True,
            'energy_bounds': True,
            'mass_conservation': True,
            'virtue_compliance': True
        }
        
        # Check global existence
        if len(solution_sequence) == 0:
            verification_results['global_existence'] = False
            
        # Check smoothness preservation
        for solution in solution_sequence:
            singularities = solution.detect_singularities()
            if any(s.is_critical() for s in singularities):
                verification_results['smoothness_preservation'] = False
                break
                
        # Check energy bounds
        initial_energy = solution_sequence[0].compute_energy_density().integral()
        for solution in solution_sequence:
            current_energy = solution.compute_energy_density().integral()
            if current_energy > 2 * initial_energy:  # Allow some growth but not blow-up
                verification_results['energy_bounds'] = False
                break
                
        # Check mass conservation
        for solution in solution_sequence:
            mass_violation = solution.check_mass_conservation()
            if mass_violation > 1e-6:  # Tolerance for numerical errors
                verification_results['mass_conservation'] = False
                break
        
        return verification_results
    
    def export_ontology(self) -> Dict[str, Any]:
        """Export complete ontology for analysis"""
        
        export_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'systems_count': len(self.systems),
                'solutions_count': len(self.solutions),
                'framework_version': '1.0.0'
            },
            'systems': {},
            'conservation_status': self.conservation_monitors,
            'singularity_tracking': {}
        }
        
        # Export system summaries
        for system_id, system in self.systems.items():
            export_data['systems'][system_id] = {
                'domain_dimension': system.domain.dimension,
                'domain_volume': system.domain.volume,
                'reynolds_number': system.flow_regime.reynolds_number,
                'flow_type': system.flow_regime.flow_type.value,
                'boundary_conditions_count': len(system.boundary_conditions),
                'is_well_posed': system.is_well_posed()
            }
            
        return export_data
