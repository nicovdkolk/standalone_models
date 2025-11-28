"""
Stand-alone powering and propeller models for Shipyard.

This module contains stand-alone implementations of:
- Diesel-electric and PTI/PTO powering architectures (PowerChain, DieselEngine)
- Propeller models (Wageningen B-series, custom Kt/Kq curves)

All classes are designed to accept a Physics-like object for physical constants
and are ready for future integration into the main Shipyard codebase.
"""

import math
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Physics:
    """
    Simple physics constants dataclass for testing and stand-alone use.

    This provides the same interface as Shipyard's Physics class.
    """
    g: float = 9.81  # Gravity [m/s²]
    rho_w: float = 1025.0  # Water density [kg/m³]
    rho_air: float = 1.225  # Air density [kg/m³]
    nu_w: float = 1.05e-6  # Water kinematic viscosity [m²/s]
    nu_air: float = 1.5e-5  # Air kinematic viscosity [m²/s]

    def rotate_z(self, xyz, angle):
        """Rotate point around z-axis by angle (radians)."""
        rot_matrix = [
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1]
        ]
        return [
            rot_matrix[0][0] * xyz[0] + rot_matrix[0][1] * xyz[1] + rot_matrix[0][2] * xyz[2],
            rot_matrix[1][0] * xyz[0] + rot_matrix[1][1] * xyz[1] + rot_matrix[1][2] * xyz[2],
            rot_matrix[2][0] * xyz[0] + rot_matrix[2][1] * xyz[1] + rot_matrix[2][2] * xyz[2]
        ]


# Import main classes for easy access
from .diesel_engine import DieselEngine
from .powering import Powering
from .wageningen_b import WageningenB
from .custom_curves import CustomCurves
from .propeller import Propeller

__all__ = [
    'Physics',
    'DieselEngine',
    'Powering',
    'WageningenB',
    'CustomCurves',
    'Propeller'
]
