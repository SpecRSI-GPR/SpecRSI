import random
from types import SimpleNamespace
from specrsi.core.models import thaw
from specrsi.science.baselines import make_backend
from specrsi.science.executor import make_candidate,make_scope,selected_parameters
from specrsi.science.proposals import parameter_catalog
from specrsi.learning.science_bridge import HFParameterDrafter


def test_specrsi_and_native_optimizers_share_numeric_catalog():
    parent=make_candidate('reactor');scope=make_scope('reactor',case_ids=(0,1))
    draft=object.__new__(HFParameterDrafter);draft.catalog_size=12
    expected=parameter_catalog('reactor',thaw(parent.artifacts['parameters']),
        [selected_parameters(parent,t,scope) for t in scope.tasks],random.Random(42))
    assert draft.catalog(parent,SimpleNamespace(scope=scope),random.Random(42))==expected


def test_baseline_routes_have_fixed_features_and_transfer_to_native_audit():
    bundle=make_backend({'science_domain':'reactor','science_case_ids':[0,1],
                         'science_proposal_space':'global_and_case_routes'})
    backend=bundle['backend'];parent=bundle['initial']
    backend.client=object()
    backend._generate_json=lambda *args:({'edit':1},13)
    proposal=backend.propose(parent,{},42,1000);child=proposal.value
    assert proposal.tokens==13
    assert set(thaw(child.payload)['scenario_overrides'])=={'case:0'}
    before=bundle['features'](parent);after=bundle['features'](child)
    assert before.keys()==after.keys() and before!=after
    assert all(before[k]==after[k] for k in before if k.startswith('case_1_'))
    for role in ('selection','audit'):
        initial=backend.evaluate(parent,bundle[role],0,role,1000)
        edited=backend.evaluate(child,bundle[role],0,role,1000)
        assert initial.complete and edited.complete
        assert initial.items[1].score==edited.items[1].score
        assert initial.items[0].trace!=edited.items[0].trace
