"""Shared feasible numeric edit catalog for SpecRSI and native optimizers."""
from .executor import mutate


def parameter_catalog(domain, parameters, effective_parameters, rng, size=12):
    """One global edit, then stable case edits in the supplied case order."""
    if size < 2 or not effective_parameters:
        raise ValueError('A numeric catalog needs >=2 choices and registered cases')
    choices=[]
    for i in range(size):
        index=-1 if i==0 else (i-1)%len(effective_parameters)
        previous=parameters if index<0 else effective_parameters[index]
        changed=mutate(domain,previous,rng)
        choices.append({'task_index':index,'parameters':changed,
                        'changes':{k:v for k,v in changed.items() if v!=previous[k]}})
    return choices
