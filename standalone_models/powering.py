import math
from typing import Dict, Union

from .diesel_engine import DieselEngine


class Powering:
    """Map engine and genset power to shaft or grid loads for DD, DE, and PTI/PTO setups."""

    # Encapsulate drivetrain configuration and associated fuel/efficiency calculations.

    def __init__(
        self,

        diesel_electric: bool = False, #DE mode is False by default
        pti_pto: bool = False, #PTI/PTO mode is False by default

        hotel_load: float = None,
        n_shaftlines: float = None,  #for future use, uses n_prop for now

        me_mcr: float = None,  # main engine MCR [kW], mandatory for DD mode

        me_csr: float = None,  # main engine CSR [kW], optional
        me_sfoc: Union[Dict, float, None] = None,  # main engine SFOC curve / single value, optional
        me_fuel_type: str = "MGO",  # main engine fuel type
        me_fuel_lhv: float = None,  # main engine fuel LHV [MJ/kg], optional
        me_fuel_density: float = None,  # main engine fuel density [kg/m³], optional

        n_gensets: int = 0,  # number of gensets, mandatory for DE mode
        genset_mcr: float = None,  # genset MCR [kW], mandatory for DE mode

        genset_csr: float = None,  # genset CSR [kW], optional
        genset_sfoc: Union[Dict, float, None] = None,  # genset SFOC curve / single value, optional
        genset_fuel_type: str = "MGO",  # genset fuel type
        genset_fuel_lhv: float = None,  # genset fuel LHV [MJ/kg], optional
        genset_fuel_density: float = None,  # genset fuel density [kg/m³], optional

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
        # Initialize propulsion chain components and efficiencies for configured architecture.
        self.diesel_electric = diesel_electric
        self.pti_pto = pti_pto
        self.shaftlines = n_shaftlines
        self.hotel_load = 0.0 if hotel_load is None else hotel_load
        
        # Only create main engine if me_mcr is provided (not needed in DE mode)
        if me_mcr is not None:
            self.main_engine = DieselEngine(
                me_sfoc, me_mcr, me_csr,
                fuel_type=me_fuel_type,
                fuel_lhv=me_fuel_lhv,
                fuel_density=me_fuel_density
            )
        else:
            self.main_engine = None
        
        # Only create gensets if genset_mcr is provided
        if genset_mcr is not None and n_gensets > 0:
            self.gensets = [
                DieselEngine(
                    genset_sfoc, genset_mcr, genset_csr,
                    fuel_type=genset_fuel_type,
                    fuel_lhv=genset_fuel_lhv,
                    fuel_density=genset_fuel_density
                ) for _ in range(n_gensets)
            ]
        else:
            self.gensets = []
     
        # Provide robust defaults for efficiencies to avoid None propagation in calculations.
        self.eta_converters = eta_converters if eta_converters is not None else 0.98
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
        # Ensure mutually exclusive modes and required parameters are set.
        if self.diesel_electric and self.pti_pto:
            raise ValueError("Cannot enable both diesel_electric and pti_pto modes")

        if self.diesel_electric:
            if self.genset_mcr is None or self.n_gensets is None or self.n_gensets == 0:
                raise ValueError("DE mode requires genset_mcr and n_gensets > 0")
        
        if self.pti_pto and self.pti_pto_kw is None:
            raise ValueError("PTI/PTO mode requires pti_pto_kw")

        if not self.diesel_electric and self.me_mcr is None:
            raise ValueError("DD mode requires main_engine MCR")

        # Guard against non-physical efficiencies that would distort power flow calculations.
        for name, value in [
            ("eta_converters", self.eta_converters),
            ("eta_gearbox", self.eta_gearbox),
            ("eta_shaft", self.eta_shaft),
            ("eta_generator", self.eta_generator),
            ("eta_pti_pto", self.eta_pti_pto),
            ("eta_electric_motor", self.eta_electric_motor),
        ]:
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be within (0, 1]; received {value}")


    def shaft_power_from_delivered_power(self, delivered_power: float) -> float:
        """
        Calculate shaft power P_S from delivered power P_D.
        
        P_D = η_shaft · P_S (or P_S = P_D / η_shaft)
        
        Parameters:
        -----------
        delivered_power : float
            Delivered power P_D [kW]
            
        Returns:
        --------
        float
            Shaft power P_S [kW]
        """
        # Convert delivered shaft power back to mechanical shaft requirement.
        # P_S = P_D / η_shaft
        return delivered_power / self.eta_shaft


    def consumers(self, EST_power_consumption: float) -> float:
        """consumers on electrical grid kWe
        considering hotel load and EST power consumption"""
        # Combine hotel load with electrical side-consumer demand.
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

        # Resolve electrical and mechanical demand paths for the configured propulsion mode.
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
        # Helper: return only main-engine brake power portion.
        _, power_brake = self.grid_load_and_brake_power_from_consumers_and_shaft_power(consumers, shaft_power)
        return power_brake


    def main_engine_fc(self, power_brake: float) -> float:
        """Calculate main engine fuel consumption in kg/h"""
        if self.main_engine is None:
            return 0.0
        # Delegate fuel burn to the DieselEngine instance.
        return self.main_engine.fuel_consumption(power_brake)


    def n_gensets_active(self, grid_load: float) -> int:
        """number of gensets active"""
        if len(self.gensets) == 0:
            return 0
        # Round up required genset count.
        return int(math.ceil(grid_load / self.eta_generator / self.gensets[0].csr))


    def aux_power_per_genset(self, grid_load: float, n_gensets_active: int) -> float:
        """auxilliary power per genset kWm"""
        # Split electrical demand evenly across active gensets.
        return grid_load / self.eta_generator / n_gensets_active


    def genset_fc(self, aux_power_per_genset: float, n_gensets_active: int):
        """Calculate genset fuel consumption in kg/h.
        aux_power_per_genset is the auxiliary power per genset kWm.
        n_gensets_active is the number of gensets active.
        Handles overflow capacity beyond installed gensets by using the last genset's characteristics.
        """
        if len(self.gensets) == 0 or n_gensets_active == 0:
            return 0.0
        
        # Use installed gensets, overflow uses last genset's characteristics
        installed_count = min(n_gensets_active, len(self.gensets))
        overflow_count = max(0, n_gensets_active - len(self.gensets))
        
        return (sum(self.gensets[i].fuel_consumption(aux_power_per_genset) for i in range(installed_count)) +
                self.gensets[-1].fuel_consumption(aux_power_per_genset) * overflow_count)


    def total_fc(self, main_engine_fc: float, genset_fc: float) -> float:
        """Calculate total fuel consumption in kg/h"""
        # Combine main engine and genset fuel consumption into fleet total.
        return main_engine_fc + genset_fc

    def main_engine_fc_volumetric(self, power_brake: float) -> float:
        """
        Calculate main engine volumetric fuel consumption.
        
        Parameters:
        -----------
        power_brake : float
            Main engine brake power [kWm]
            
        Returns:
        --------
        float
            Main engine volumetric fuel consumption [m³/h]
        """
        if self.main_engine is None:
            return 0.0
        return self.main_engine.fuel_consumption_volumetric(power_brake)

    def genset_fc_volumetric(self, aux_power_per_genset: float, n_gensets_active: int) -> float:
        """
        Calculate genset volumetric fuel consumption.
        
        Parameters:
        -----------
        aux_power_per_genset : float
            Auxiliary power per genset [kWm]
        n_gensets_active : int
            Number of active gensets
            
        Returns:
        --------
        float
            Total genset volumetric fuel consumption [m³/h]
        Handles overflow capacity beyond installed gensets by using the last genset's characteristics.
        """
        if len(self.gensets) == 0 or n_gensets_active == 0:
            return 0.0
        
        # Use installed gensets, overflow uses last genset's characteristics
        installed_count = min(n_gensets_active, len(self.gensets))
        overflow_count = max(0, n_gensets_active - len(self.gensets))
        
        return (sum(self.gensets[i].fuel_consumption_volumetric(aux_power_per_genset) for i in range(installed_count)) +
                self.gensets[-1].fuel_consumption_volumetric(aux_power_per_genset) * overflow_count)

    def total_fc_volumetric(self, main_engine_fc_volumetric: float, genset_fc_volumetric: float) -> float:
        """
        Calculate total volumetric fuel consumption.
        
        Parameters:
        -----------
        main_engine_fc_volumetric : float
            Main engine volumetric fuel consumption [m³/h]
        genset_fc_volumetric : float
            Genset volumetric fuel consumption [m³/h]
            
        Returns:
        --------
        float
            Total volumetric fuel consumption [m³/h]
        """
        return main_engine_fc_volumetric + genset_fc_volumetric

    def main_engine_efficiency(self, power_brake: float) -> float:
        """
        Calculate main engine efficiency at current load.
        
        Parameters:
        -----------
        power_brake : float
            Main engine brake power [kWm]
            
        Returns:
        --------
        float
            Main engine efficiency η_eng [dimensionless, 0-1]
        """
        if self.main_engine is None:
            return 0.0
        return self.main_engine.engine_efficiency(brake_power=power_brake)

    def genset_efficiency(self, aux_power_per_genset: float) -> float:
        """
        Calculate genset efficiency at current load.
        
        Parameters:
        -----------
        aux_power_per_genset : float
            Auxiliary power per genset [kWm]
            
        Returns:
        --------
        float
            Genset efficiency η_eng [dimensionless, 0-1]
            Returns efficiency of first genset (assumes all gensets are identical)
        """
        if len(self.gensets) == 0:
            return 0.0
        return self.gensets[0].engine_efficiency(brake_power=aux_power_per_genset)

    def main_engine_chemical_energy_rate(self, power_brake: float, unit: str = "kW") -> float:
        """
        Calculate main engine chemical energy rate.
        
        Parameters:
        -----------
        power_brake : float
            Main engine brake power [kWm]
        unit : str, optional
            Output unit: "kW" (default) or "MJ/s"
            
        Returns:
        --------
        float
            Main engine chemical energy rate [kW] or [MJ/s]
        """
        if self.main_engine is None:
            return 0.0
        return self.main_engine.chemical_energy_rate(power_brake, unit=unit)

    def genset_chemical_energy_rate(self, aux_power_per_genset: float, n_gensets_active: int, unit: str = "kW") -> float:
        """
        Calculate total genset chemical energy rate.
        
        Parameters:
        -----------
        aux_power_per_genset : float
            Auxiliary power per genset [kWm]
        n_gensets_active : int
            Number of active gensets
        unit : str, optional
            Output unit: "kW" (default) or "MJ/s"
            
        Returns:
        --------
        float
            Total genset chemical energy rate [kW] or [MJ/s]
        Handles overflow capacity beyond installed gensets by using the last genset's characteristics.
        """
        if len(self.gensets) == 0 or n_gensets_active == 0:
            return 0.0
        
        # Use installed gensets, overflow uses last genset's characteristics
        installed_count = min(n_gensets_active, len(self.gensets))
        overflow_count = max(0, n_gensets_active - len(self.gensets))

        installed_energy = sum(
            self.gensets[i].chemical_energy_rate(aux_power_per_genset, unit=unit)
            for i in range(installed_count)
        )
        overflow_energy = self.gensets[-1].chemical_energy_rate(
            aux_power_per_genset, unit=unit
        ) * overflow_count

        return installed_energy + overflow_energy
