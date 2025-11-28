from typing import Union, Dict
import math
from .diesel_engine import DieselEngine


class Powering:
    """
    Models power flow from engine and/or genset brake power through various transmission stages to
    delivered shaft power and grid load kWe, for diesel-direct (DD), diesel-electric (DE), and
    PTI/PTO (PTI/PTO) architectures. DD mode is the default.
    
    Handles backwards and forwards engine models:
    Backwards: Shaft power and grid load kWe >> engine brake power and/or genset brake power
    Forward: Engine and/or genset brake power >> Shaft power and grid load kWe
    
    Uses DieselEngine class for main engine and genset fuel consumption calculations.
    """

    def __init__(
        self,

        diesel_electric: bool = False, #DE mode is False by default
        pti_pto: bool = False, #PTI/PTO mode is False by default

        hotel_load: float = None,
        n_shaftlines: float = None,  #for future use, uses n_prop for now

        me_mcr: float = None,  # main engine MCR [kW], mandatory for DD mode

        me_csr: float = None,  # main engine CSR [kW], optional
        me_sfoc: Union[Dict, float, None] = None,  # main engine SFOC curve / single value, optional

        n_gensets: int = 0,  # number of gensets, mandatory for DE mode
        genset_mcr: float = None,  # genset MCR [kW], mandatory for DE mode

        genset_csr: float = None,  # genset CSR [kW], optional
        genset_sfoc: Union[Dict, float, None] = None,  # genset SFOC curve / single value, optional

        eta_gearbox: float = None,  # Gearbox efficiency
        eta_shaft: float = None,   # Shaft efficiency
        eta_Tr: float = None,  # Transmission efficiency (legacy)
        eta_Gen: float = None,  # Generator efficiency (legacy)
        eta_generator: float = None,  # Generator efficiency
        pti_pto_kw: float = None,  # PTI/PTO power [kW], mandatory for PTI/PTO mode

        eta_converters: float = None,  # Converter efficiency
        eta_electric_motor: float = None,  # Electric motor efficiency
        eta_pti_pto: float = None,  # PTI/PTO efficiency
    ):

        self.diesel_electric = diesel_electric
        self.pti_pto = pti_pto
        self.shaftlines = n_shaftlines
        self.hotel_load = hotel_load
        
        # Only create main engine if me_mcr is provided (not needed in DE mode)
        if me_mcr is not None:
            self.main_engine = DieselEngine(me_sfoc, me_mcr, me_csr)
        else:
            self.main_engine = None
        
        # Only create gensets if genset_mcr is provided
        if genset_mcr is not None and n_gensets > 0:
            self.gensets = [DieselEngine(genset_sfoc, genset_mcr, genset_csr) for _ in range(n_gensets)]
        else:
            self.gensets = []
     
        self.eta_converters = eta_converters
        self.eta_gearbox = eta_gearbox if eta_gearbox is not None else 1
        self.eta_shaft = eta_shaft if eta_shaft is not None else 0.99
        # For legacy compatibility: prefer eta_Tr, fallback to eta_shaft * eta_gearbox
        self.eta_Tr = eta_Tr if eta_Tr is not None else self.eta_shaft * self.eta_gearbox
        self.eta_pti_pto = eta_pti_pto if eta_pti_pto is not None else 0.93
        self.eta_electric_motor = eta_electric_motor if eta_electric_motor is not None else 0.95
        # For legacy compatibility: prefer eta_generator, fallback to eta_Gen, then default to 0.96
        if eta_generator is not None:
            self.eta_generator = eta_generator
        elif eta_Gen is not None:
            self.eta_generator = eta_Gen
        else:
            self.eta_generator = 0.96

        # Store parameters for validation
        self.n_gensets = n_gensets
        self.genset_mcr = genset_mcr
        self.pti_pto_kw = pti_pto_kw
        self.me_mcr = me_mcr  # Store for validation

        # Validate configuration after initialization
        self._validate_configuration()

    def _validate_configuration(self):
        """Validate configuration after initialization."""
        if self.diesel_electric and self.pti_pto:
            raise ValueError("Cannot enable both diesel_electric and pti_pto modes")

        if self.diesel_electric:
            if self.genset_mcr is None or self.n_gensets is None or self.n_gensets == 0:
                raise ValueError("DE mode requires genset_mcr and n_gensets > 0")
        
        if self.pti_pto and self.pti_pto_kw is None:
            raise ValueError("PTI/PTO mode requires pti_pto_kw")

        if not self.diesel_electric and self.me_mcr is None:
            raise ValueError("DD mode requires main_engine MCR")


    def shaft_power_from_delivered_power(self, delivered_power: float) -> float:
        return delivered_power / self.eta_shaft


    def consumers(self, EST_power_consumption: float) -> float:
        """consumers on electrical grid kWe
        considering hotel load and EST power consumption"""
        return EST_power_consumption + self.hotel_load


    def grid_load_and_brake_power_from_consumers_and_shaft_power(self, consumers: float, shaft_power: float) -> tuple[float, float]:
        """
        Calculate grid load and brake power from consumers and shaft power.
        If PTI/PTO mode, the additional grid load is limited by the PTI/PTO power rating kWe.

        Units of grid load and consumers are kWe (electrical).
        Units of shaft power and brake power are kWm (mechanical).
        These are tranformed by eta_pti_pto.

        Parameters:
        -----------
        consumers : float
            Other consumers on electrical grid [kWe]
        shaft_power : float
            Shaft power [kWm]
            
        Returns:
        --------
        tuple[float, float]
            (grid_load [kWe], brake_power [kWm])
        """

        grid_load = consumers
        power_brake = 0.0
        
        if self.pti_pto:
            if self.main_engine is None:
                raise ValueError("PTI/PTO mode requires main_engine (me_mcr must be provided)")
            
            required_brake_power_m = shaft_power / self.eta_gearbox  # kWm
            required_brake_power_e = min(self.pti_pto_kw, grid_load) / (self.eta_converters * self.eta_pti_pto)  # kWm, limited by PTI/PTO power
            
            if required_brake_power_m + required_brake_power_e < self.main_engine.csr:  # PTO mode, no genset, enough power from main engine
                grid_load = 0
                power_brake = required_brake_power_m + required_brake_power_e  # kWm

            else:  # genset engaged, PTI/PTO mode, not enough power from main engine
                grid_load += (required_brake_power_m - self.main_engine.csr) / (self.eta_converters * self.eta_pti_pto) # kWe, additional power needed from genset
                power_brake = self.main_engine.csr  # kWm

        elif self.diesel_electric:
                        
            grid_load += shaft_power / (self.eta_gearbox * self.eta_electric_motor * self.eta_converters)  # kWe
            power_brake = 0  # kWm (no main engine in DE mode)
        
        else:  # Diesel-direct mode
            
            grid_load = grid_load / self.eta_converters  # kWe
            power_brake = shaft_power / self.eta_gearbox  # kWm
        
        return grid_load, power_brake


    def grid_load(self, consumers: float, shaft_power: float) -> float:
        grid_load, _ = self.grid_load_and_brake_power_from_consumers_and_shaft_power(consumers, shaft_power)
        return grid_load


    def power_brake(self, consumers: float, shaft_power: float) -> float:
        _, power_brake = self.grid_load_and_brake_power_from_consumers_and_shaft_power(consumers, shaft_power)
        return power_brake


    def main_engine_fc(self, power_brake: float) -> float:
        """Calculate main engine fuel consumption in kg/h"""
        if self.main_engine is None:
            return 0.0
        return self.main_engine.fuel_consumption(power_brake)


    def n_gensets_active(self, grid_load: float) -> int:
        """number of gensets active, limited to the total number of gensets"""
        if len(self.gensets) == 0:
            return 0
        return int(min(math.ceil(grid_load / self.eta_generator / self.gensets[0].csr), self.n_gensets))


    def aux_power_per_genset(self, grid_load: float, n_gensets_active: int) -> float:
        """auxilliary power per genset kWm"""
        return grid_load / self.eta_generator / n_gensets_active


    def genset_fc(self, aux_power_per_genset: float, n_gensets_active: int):
        """Calculate genset fuel consumption in kg/h.
        aux_power_per_genset is the auxiliary power per genset kWm.
        n_gensets_active is the number of gensets active.
        """
        return sum(self.gensets[i].fuel_consumption(aux_power_per_genset) for i in range(n_gensets_active))
    

    def total_fc(self, main_engine_fc: float, genset_fc: float) -> float:
        """Calculate total fuel consumption in kg/h"""
        return main_engine_fc + genset_fc