from typing import Union, Dict
from scipy.interpolate import interp1d


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


    def sfoc_at_load(self, percentage_mcr: float) -> float:
        # Return SFOC by interpolating along the %MCR curve.
        return self.sfoc_interpolator(percentage_mcr)


    def fuel_consumption(self, brake_power: float) -> float:
        """
        Calculate fuel consumption for given brake power.
        Fuel consumption is limited to positive values.

        Parameters:
        -----------
        brake_power : float
            Brake power [kW]

        Returns:
        --------
        float
            Fuel consumption [kg/h]
        """
        # Convert brake power and interpolated SFOC into hourly fuel use.
        return max(0, self.sfoc_at_load(brake_power / self.mcr) * brake_power)


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
        return 0.15 <= brake_power <= self.mcr