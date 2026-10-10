"""Real PyBaMM SPM ageing with thermal, SEI and irreversible plating physics."""
from __future__ import annotations

import os
os.environ["PYBAMM_DISABLE_TELEMETRY"] = "true"
import numpy as np

from specrsi.benchmarks.science import pybamm_score
from .common import InvalidCandidate,exact_keys,vector


def simulate(candidate: dict,c: dict) -> dict:
    exact_keys(candidate,("currents_c","soc_breakpoints"))
    currents = vector(candidate["currents_c"],c["stage_count"],*c["current_bounds_c"],"currents_c")
    socs = vector(candidate["soc_breakpoints"],c["stage_count"],c["initial_soc"],c["target_soc"],"soc_breakpoints")
    if any(a>=b for a,b in zip([c["initial_soc"]]+socs[:-1],socs)) or abs(socs[-1]-c["target_soc"])>1e-9:
        raise InvalidCandidate("SOC breakpoints must increase and end at target SOC")
    ideal_time = sum((b-a)/current*3600 for a,b,current in zip([c["initial_soc"]]+socs[:-1],socs,currents))
    if ideal_time>c["max_charge_seconds"]:
        raise InvalidCandidate("Ideal coulombic charging time already exceeds 600 seconds")
    import pybamm
    pybamm.set_logging_level("ERROR")
    rows=[]
    for ambient in c["temperatures_c"]:
        params=pybamm.ParameterValues(c["parameter_set"])
        params.update({"Ambient temperature [K]":ambient+273.15,"Initial temperature [K]":ambient+273.15,
                       "Upper voltage cut-off [V]":c["voltage_upper_v"],"Lower voltage cut-off [V]":c["voltage_lower_v"]})
        x0,x1,_,_=pybamm.lithium_ion.get_min_max_stoichiometries(params)
        model=pybamm.lithium_ion.SPM(options=c["options"])
        # Explicit electrode-stoichiometric SOC, independent of candidate-reported values.
        model.variables["Registered SOC"]=(model.variables["Negative electrode stoichiometry"]-x0)/(x1-x0)
        def soc_event(target):
            return pybamm.step.CustomTermination(f"Registered SOC {target}",lambda v:target-v["Registered SOC"])
        steps=[]
        for current,target in zip(currents,socs):
            steps.append(pybamm.step.c_rate(-current,duration=c["max_charge_seconds"],termination=[soc_event(target)],period="5 seconds"))
        steps.extend([
            pybamm.step.rest(c["rest_seconds"]),
            pybamm.step.c_rate(-1/3,duration="4 hours",termination="4.2 V"),
            pybamm.step.voltage(4.2,duration="2 hours",termination="C/50"),
            pybamm.step.c_rate(c["discharge_c"],duration="2 hours",termination=f'{c["voltage_lower_v"]} V',period="10 seconds"),
            pybamm.step.c_rate(-0.5,duration="1 hour",termination=[soc_event(c["initial_soc"])],period="10 seconds"),
            pybamm.step.rest(c["rest_seconds"]),
        ])
        experiment=pybamm.Experiment([tuple(steps)]*c["cycles"])
        mesh=c["solver"]["mesh_points"]
        solver=pybamm.IDAKLUSolver(rtol=c["solver"]["rtol"],atol=c["solver"]["atol"],options={"num_threads":1})
        sim=pybamm.Simulation(model,experiment=experiment,parameter_values=params,solver=solver,
                              var_pts={"x_n":mesh,"x_s":mesh,"x_p":mesh,"r_n":mesh,"r_p":mesh})
        try:
            solution=sim.solve(initial_soc=c["initial_soc"])
        except (pybamm.SolverError,ValueError) as exc:
            raise InvalidCandidate(f"PyBaMM integration failed: {exc}") from exc
        if len(solution.cycles)!=c["cycles"]:
            raise InvalidCandidate("PyBaMM did not complete all 30 cycles")
        capacities=[]; charge_times=[]; min_overpotential=float("inf"); max_temperature=-float("inf")
        for cycle in solution.cycles:
            if len(cycle.steps)!=len(steps):
                raise InvalidCandidate("PyBaMM cycle ended before all protocol steps")
            charge=cycle.steps[:len(currents)]
            charge_time=sum(float(s.t[-1]-s.t[0]) for s in charge)
            final_soc=float(charge[-1]["Registered SOC"].entries[-1])
            if charge_time>c["max_charge_seconds"]+1e-3 or final_soc<c["target_soc"]-1e-4:
                raise InvalidCandidate("Did not reach 60% SOC within 600 seconds")
            charge_times.append(charge_time)
            for step in charge:
                eta=float(np.min(step["X-averaged negative electrode lithium plating reaction overpotential [V]"].entries))
                min_overpotential=min(min_overpotential,eta)
            temperature=float(np.max(cycle["Volume-averaged cell temperature [K]"].entries))-273.15
            max_temperature=max(max_temperature,temperature)
            reference=cycle.steps[len(currents)+3]
            discharge=reference["Discharge capacity [A.h]"].entries
            capacities.append(float(discharge[-1]-discharge[0]))
        if max_temperature>c["temperature_limit_c"] or min_overpotential<c["plating_overpotential_min_v"]:
            raise InvalidCandidate(f"Battery safety constraint: Tmax={max_temperature:.4g} C, plating eta min={min_overpotential:.4g} V")
        if not np.all(np.isfinite(capacities+charge_times+[max_temperature,min_overpotential])) or min(capacities)<=0:
            raise InvalidCandidate("Nonfinite or nonpositive battery reference observations")
        predictor=c["predictor"]
        start=predictor["fit_start_cycle"]-1
        slope,intercept=np.polyfit(np.arange(1,c["cycles"]+1)[start:],capacities[start:],1)
        capacity0=capacities[0]
        fade=max(-float(slope),predictor["min_slope_ah_per_cycle"])
        lifetime=min(predictor["max_predicted_cycles"],max(0.,(float(intercept)-predictor["end_of_life_fraction"]*capacity0)/fade))
        rows.append({"ambient_temperature_c":ambient,"cycles_completed":len(capacities),"reference_capacities_ah":capacities,
                     "charge_times_seconds":charge_times,"max_temperature_c":max_temperature,"min_plating_overpotential_v":min_overpotential,
                     "fitted_capacity_slope_ah_per_cycle":float(slope),"predicted_end_of_life_cycles":lifetime})
    result=pybamm_score([r["predicted_end_of_life_cycles"] for r in rows],valid=True,temperatures_c=c["temperatures_c"])
    return {"native_score":result.score,"native_unit":result.metrics["native_unit"],"temperature_runs":rows,
            "simulator":f"PyBaMM {pybamm.__version__} SPM + lumped thermal + SEI + irreversible plating",
            "predictor_scope":"registered new linear capacity-fade extrapolation; not authors' undisclosed early-life predictor"}
