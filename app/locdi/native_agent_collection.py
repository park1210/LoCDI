"""Five prespecified Agent bindings for Native TRAINING, never a new Router.

Collection is a controlled intervention: it deliberately runs each registered
specialist. No Utility admission or adaptive performance is claimed.
"""
from copy import deepcopy
from contextlib import contextmanager
from unittest.mock import patch
from app.core.agent_registry import AgentRegistry
from app.core.hospital_llm_policy import HospitalLLMPolicy
from app.locdi.common_menu_backend import make_common_backend
from app.locdi.run_simulation import _SOCKET_LOCK
from app.locdi.evidence_subset_wire import encode,decode,VERSION


@contextmanager
def fixed_training_agent(department):
    original=HospitalLLMPolicy._decision
    class Fixed(AgentRegistry):
        def select_specialty(self,observation,memory,candidates,*,router,seed=0):
            if department not in candidates:raise ValueError('PRESPECIFIED_AGENT_NOT_BOUND')
            row=super().select_specialty(observation,memory,candidates,router=router,seed=seed)
            row.update(selected_specialty=department,decision_status='selected',status='selected',
                rationale='Prespecified five-Agent Native training intervention; not an adaptive selection.',
                router='FIXED_TRAINING_AGENT_NOT_ADAPTIVE_UTILITY')
            return row
    def decide(policy,runtime,role,department_arg=''):
        saved=(policy.agent_registry,policy.routing_mode,policy.primary_router_mode)
        if role!='er':raise ValueError('UNPLANNED_CONSULT_ROLE_DURING_COLLECTION')
        policy.agent_registry=Fixed(saved[0].snapshot())
        policy.routing_mode='llm';policy.primary_router_mode='adaptive'
        try:return original(policy,runtime,role,department_arg)
        finally:policy.agent_registry,policy.routing_mode,policy.primary_router_mode=saved
    with _SOCKET_LOCK,patch.object(HospitalLLMPolicy,'_decision',new=decide),\
            patch('app.locdi.evidence_selection_wire.encode',new=encode),\
            patch('app.locdi.evidence_selection_wire.decode',new=decode),\
            patch('app.locdi.evidence_selection_wire.VERSION',new=VERSION):
        yield


def collect_one_backend(case,*,department,**kwargs):
    """A fresh Native/Scheduler episode, Top5, VoI off, fixed Specialty role.

Features use the established selective-memory engine path. The resulting
episode is explicitly TRAINING, not a B2 or B3 evaluation result.
"""
    from app.locdi.current_ablation_runner import features_for
    with fixed_training_agent(department):
        backend=make_common_backend({'TRAIN':deepcopy(case)},**kwargs)
        result=backend({'case_ref':'TRAIN'},features_for('B2'))
    result['collection_contract']='FIVE_PRESPECIFIED_SPECIALTIES_NATIVE_EPISODES.v1'
    result['B2_evaluation_claimed']=False
    return result
