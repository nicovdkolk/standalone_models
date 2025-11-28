from typing import Union, Dict
from scipy.interpolate import interp1d


# Fuel type definitions with Lower Heating Value (LHV) and density
# Based on typical marine fuel properties
FUEL_PROPERTIES = {
    "MGO": {
        "lhv": 42.7,  # MJ/kg (representative value from 42-43.5 range)
        "density": 850.0  # kg/m³ (representative value from 820-890 range)
    },
    "HFO": {
        "lhv": 40.0,  # MJ/kg (representative value from 39-41 range)
        "density": 990.0  # kg/m³ (representative value from 975-1010 range)
    },
    "LNG": {
        "lhv": 49.0,  # MJ/kg (representative value from 48-50 range)
        "density": 450.0  # kg/m³ (representative value from 410-500 range)
    },
    "Methanol": {
        "lhv": 20.5,  # MJ/kg (representative value from 20-21 range)
        "density": 790.0  # kg/m³
    },
    "Ammonia": {
        "lhv": 18.5,  # MJ/kg (representative value from 18-19 range)
        "density": 680.0  # kg/m³ (representative value)
    }
}


class DieselEngine:
    """
    Diesel engine model for fuel consumption calculation at a given load percentage.
    MCR (maximum continuous rating) is mandatory.
    CSR (continuous service rating) is optional.
    CSR defaults to:
    - 91% of MCR (high speed, <2000 kW)
    - 85% of MCR (medium speed, 2000-18000 kW)
    - 65% of MCR (low speed, >18000 kW)

    SFOC (specific fuel oil consumption) is optional, may be a dict or a float:
    - Dict with keys:
        - "%MCR": List of load percentages [0-1.0]
        - "SFOC": List of SFOC values [kg/kWh]
    - Float value for SFOC [kg/kWh] at CSR

    If a dict with keys "%MCR" and "SFOC" is provided, it is used as the SFOC values and curve.
    If a float value for SFOC is provided, it is applied with default SFOC curves for high speed,
    medium speed and low speed engines.

    If no SFOC is provided, default SFOC values and curves are used for high speed, medium speed
    and low speed engines.
    - High speed (<2000 kW): 0.190 kg/kWh at 91% MCR
    - Medium speed (2000-18000 kW): 0.180 kg/kWh at 85% MCR
    - Low speed (>18000 kW): 0.160 kg/kWh at 65% MCR
    """

    # Configure engine load limits and default SFOC/CSR values by speed class.

    DEFAULT_SFOC_HIGH_SPEED = 0.190
    DEFAULT_SFOC_MEDIUM_SPEED = 0.180
    DEFAULT_SFOC_LOW_SPEED = 0.160

    DEFAULT_CSR_HIGH_SPEED = 0.91
    DEFAULT_CSR_MEDIUM_SPEED = 0.85
    DEFAULT_CSR_LOW_SPEED = 0.65

    def __init__(
            self,
            sfoc: Union[Dict, float] = None,
            mcr: float = None,
            csr: float = None,
            fuel_type: str = "MGO",
            fuel_lhv: float = None,
            fuel_density: float = None,
    ):
        # Determine CSR and SFOC curve configuration based on engine rating.
        self.mcr = mcr #this is the only mandatory parameter
        if csr is not None:
            self.csr = csr
        elif mcr is not None and mcr < 2000:
            self.csr = self.DEFAULT_CSR_HIGH_SPEED * mcr
        elif mcr is not None and mcr < 18000:
            self.csr = self.DEFAULT_CSR_MEDIUM_SPEED * mcr
        elif mcr is not None:
            self.csr = self.DEFAULT_CSR_LOW_SPEED * mcr
        else:
            # If mcr is None, csr will also be None
            self.csr = None

        self.sfoc_interpolator = None

        if not isinstance(sfoc, dict):
            if mcr is None:
                # Cannot determine engine type without MCR
                raise ValueError("MCR is required for DieselEngine")
            if mcr < 2000:  #high speed
                if not isinstance(sfoc, float):
                    sfoc_value = self.DEFAULT_SFOC_HIGH_SPEED
                else:
                    sfoc_value = sfoc
                self.sfoc = {
                    "%MCR": [0.16, 0.23, 0.32, 0.55, 0.91, 0.944],
                    "SFOC": [round(x * sfoc_value, 3) for x in [1.345, 1.26, 1.2, 1.09, 1.0, 1.044]]
                }
            elif 2000 <= mcr <= 18000:  #medium speed
                if not isinstance(sfoc, float):
                    sfoc_value = self.DEFAULT_SFOC_MEDIUM_SPEED
                else:
                    sfoc_value = sfoc
                self.sfoc = {
                    "%MCR": [0.15, 0.35, 0.5, 0.65, 0.7, 0.75, 0.85, 1.0],
                    "SFOC": [round(x * sfoc_value, 3) for x in [1.086, 1.037, 1.009, 1.005, 1.0, 1.0, 1.0, 1.025]]
                }
            else:  # mcr > 18000  #low speed
                if not isinstance(sfoc, float):
                    sfoc_value = self.DEFAULT_SFOC_LOW_SPEED
                else:
                    sfoc_value = sfoc
                self.sfoc = {
                    "%MCR": [0.35, 0.5, 0.65, 0.85, 1.0],
                    "SFOC": [round(x * sfoc_value, 3) for x in [1.043, 1.016, 1.0, 1.004, 1.023]]
                }
        else:
            self.sfoc = sfoc

        self.sfoc_interpolator = interp1d(
            self.sfoc["%MCR"],
            self.sfoc["SFOC"], kind='cubic',
            fill_value=(self.sfoc["SFOC"][0], self.sfoc["SFOC"][-1]), bounds_error=False)

        # Handle fuel properties
        # Validate fuel_type
        if fuel_type not in FUEL_PROPERTIES:
            raise ValueError(f"Unknown fuel_type '{fuel_type}'. Valid options: {list(FUEL_PROPERTIES.keys())}")
        
        # Get default fuel properties from fuel_type
        default_props = FUEL_PROPERTIES[fuel_type]
        
        # Use provided values or defaults, with validation
        self.fuel_lhv = fuel_lhv if fuel_lhv is not None else default_props["lhv"]
        self.fuel_density = fuel_density if fuel_density is not None else default_props["density"]
        self.fuel_type = fuel_type
        
        # Validate fuel properties
        if self.fuel_lhv <= 0:
            raise ValueError("fuel_lhv must be positive")
        if self.fuel_density <= 0:
            raise ValueError("fuel_density must be positive")


    def sfoc_at_load(self, percentage_mcr: float) -> float:
        # Return SFOC by interpolating along the %MCR curve.
        return self.sfoc_interpolator(percentage_mcr)

    def engine_efficiency(self, percentage_mcr: float = None, brake_power: float = None) -> float:
        """
        Calculate engine efficiency η_eng at given load.
        
        η_eng = 3.6 / (SFOC · H_L) = 3600 / (SFOC · H_L · 1000)
        
        where:
        - SFOC is in kg/kWh
        - H_L (LHV) is in MJ/kg
        - The factor accounts for unit conversion: 1 kWh = 3.6 MJ, 1 MJ/s = 1000 kW
        - Result is dimensionless (0-1)
        
        Derivation:
        - P_B [kW] = η_eng · ṁ_f [kg/s] · H_L [MJ/kg] · 1000 [kW/(MJ/s)]
        - SFOC = ṁ_f [kg/h] / P_B [kW] = (ṁ_f [kg/s] · 3600) / P_B [kW]
        - Rearranging: ṁ_f = SFOC · P_B / 3600
        - Substituting: P_B = η_eng · (SFOC · P_B / 3600) · H_L · 1000
        - Simplifying: η_eng = 3600 / (SFOC · H_L · 1000) = 3.6 / (SFOC · H_L)
        
        Either percentage_mcr or brake_power must be provided.
        If brake_power is provided, percentage_mcr is calculated from it.
        
        Parameters:
        -----------
        percentage_mcr : float, optional
            Load as percentage of MCR [0-1.0]
        brake_power : float, optional
            Brake power [kW] (used to calculate percentage_mcr if provided)
            
        Returns:
        --------
        float
            Engine efficiency η_eng [dimensionless, 0-1]
        """
        if percentage_mcr is None:
            if brake_power is None:
                raise ValueError("Either percentage_mcr or brake_power must be provided")
            if self.mcr is None:
                raise ValueError("MCR must be set to calculate efficiency from brake_power")
            percentage_mcr = brake_power / self.mcr
        
        sfoc = self.sfoc_at_load(percentage_mcr)  # kg/kWh
        # η_eng = 3.6 / (SFOC · H_L)
        # Accounts for: SFOC [kg/kWh], H_L [MJ/kg], and unit conversion (1 kWh = 3.6 MJ, 1 MJ/s = 1000 kW)
        efficiency = 3.6 / (sfoc * self.fuel_lhv)
        return efficiency


    def fuel_consumption(self, brake_power: float) -> float:
        """
        Calculate fuel consumption for given brake power P_B.
        
        Uses SFOC (Specific Fuel Oil Consumption) directly:
        ṁ_f [kg/h] = SFOC [kg/kWh] · P_B [kW]
        
        Parameters:
        -----------
        brake_power : float
            Brake power P_B [kW] - mechanical power at engine crankshaft

        Returns:
        --------
        float
            Fuel consumption [kg/h]
        """
        # Convert brake power and interpolated SFOC into hourly fuel use.
        return self.sfoc_at_load(brake_power / self.mcr) * brake_power

    def fuel_consumption_mass_flow(self, brake_power: float) -> float:
        """
        Calculate fuel mass flow rate for given brake power P_B.
        
        Parameters:
        -----------
        brake_power : float
            Brake power P_B [kWm] - mechanical power at engine crankshaft

        Returns:
        --------
        float
            Fuel mass flow rate [kg/s]
        """
        # Convert hourly consumption to mass flow rate
        return self.fuel_consumption(brake_power) / 3600.0

    def fuel_consumption_volumetric(self, brake_power: float) -> float:
        """
        Calculate volumetric fuel flow rate for given brake power P_B.
        
        Parameters:
        -----------
        brake_power : float
            Brake power P_B [kWm] - mechanical power at engine crankshaft

        Returns:
        --------
        float
            Volumetric fuel flow rate [m³/h]
        """
        # Volumetric flow = mass flow / density
        mass_flow_kg_per_h = self.fuel_consumption(brake_power)
        return mass_flow_kg_per_h / self.fuel_density

    def chemical_energy_rate(self, brake_power: float, unit: str = "kW") -> float:
        """
        Calculate chemical energy rate Ė_fuel for given brake power P_B.
        
        Ė_fuel = ṁ_f · H_L
        
        where:
        - ṁ_f is fuel mass flow rate [kg/s]
        - H_L is lower heating value [MJ/kg]
        - Result is in MJ/s or kW (1 MJ/s = 1000 kW)
        
        Parameters:
        -----------
        brake_power : float
            Brake power P_B [kWm] - mechanical power at engine crankshaft
        unit : str, optional
            Output unit: "kW" (default) or "MJ/s"
            
        Returns:
        --------
        float
            Chemical energy rate [kW] or [MJ/s] depending on unit parameter
        """
        mass_flow_kg_per_s = self.fuel_consumption_mass_flow(brake_power)
        energy_rate_mj_per_s = mass_flow_kg_per_s * self.fuel_lhv
        
        if unit == "kW":
            # Convert MJ/s to kW: 1 MJ/s = 1000 kW
            return energy_rate_mj_per_s * 1000.0
        elif unit == "MJ/s":
            return energy_rate_mj_per_s
        else:
            raise ValueError(f"Unknown unit '{unit}'. Valid options: 'kW', 'MJ/s'")


    def is_valid_load(self, brake_power: float) -> bool:
        """
        Check if power load is within valid range (15% to 100% of MCR).
        To be used for dealing with out of engine range issue.

        Parameters:
        -----------
        brake_power : float
            Brake power [kW]

        Returns:
        --------
        bool
            True if load is valid
        """
        # Confirm requested brake power sits within typical operational band.
        return 0.15 * self.mcr <= brake_power <= self.mcr