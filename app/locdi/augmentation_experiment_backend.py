"""Paired D0/D2 overlay after unchanged Source validation; no formula changes."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from app.locdi.common_menu_backend import make_common_backend
from app.locdi.m5_common_source import CommonM5SourceView
from app.locdi.original50_augmentation import initial_preview,validate_augmentation,SCOPE
from app.locdi.run_simulation import _SOCKET_LOCK
from app.locdi.llm_setup import digest
from app.locdi.routing_collector import atomic_json
from app.locdi.evidence_subset_wire import encode as subset_encode, decode as subset_decode, VERSION as EVIDENCE_WIRE

POLICY='ORIGINAL50_LITERATURE_SUBJECTIVE_INITIAL_OVERLAY.v1'

def make_paired_backend(cases,*,dataset_view,augmented_rows,catalog,**kwargs):
    if dataset_view not in ('D0','D2'):raise ValueError('PAIRED_DATASET_VIEW')
    authority=kwargs['authorization']
    if (authority.get('paired_scope')!='ORIGINAL50_D0_D2_DEVELOPMENT_ONLY' or
        authority.get('dataset_views')!=['D0','D2'] or authority.get('accept_label_conditioned_development') is not True or
        authority.get('accept_augmentation_initial_time_policy') is not True):raise ValueError('NEW_PAIRED_OVERLAY_AUTHORITY_REQUIRED')
    config=kwargs['config'];root=Path(kwargs['checkpoint_root']);counter=[0]
    originals={r['patient_id']:r for r in cases.values()};rows_by_patient={cases[ref]['patient_id']:row for ref,row in augmented_rows.items()}
    class Overlay(CommonM5SourceView):
        def __init__(self,binding,authorization):
            super().__init__(binding,authorization);self.case=originals[binding['patient_id']];self.row=rows_by_patient[binding['patient_id']]
            validate_augmentation(self.case,self.row,catalog)
        def metadata(self):
            return {**super().metadata(),'dataset_overlay':{'policy':POLICY,'view':'D2','scope':SCOPE,
                'row_fingerprint':self.row['row_fingerprint'],'synthetic_available_minute':0,
                'synthetic_objective_evidence':False,'source_grants_created':0,'clinical_reviewed':False}}
        def apply(self,adapted,*,features,purpose):
            runtime,audit=super().apply(adapted,features=features,purpose=purpose)
            if digest(adapted.agent_visible['current_initial_state'])!=digest(self.case['agent_visible']['current_initial_state']):raise ValueError('ORIGINAL_INITIAL_STATE_CHANGED')
            state=initial_preview(self.case,self.row,catalog,view='D2',accept_development_proxy=True)
            # Only subjective initial fields. No objective findings, history,
            # Source events, test results, Memory records or action menu changes.
            runtime['current_initial_state']=deepcopy(state)
            runtime['doctor_visible_info']=deepcopy(state['doctor_visible_info'])
            runtime['patient_visible_info']=deepcopy(state['patient_visible_info'])
            return runtime,audit
    backend=make_common_backend(cases,**kwargs)
    def execute(case,features):
        counter[0]+=1;ref=case['case_ref'];original=cases[ref];row=augmented_rows[ref]
        validate_augmentation(original,row,catalog)
        state=initial_preview(original,row,catalog,view=dataset_view,accept_development_proxy=dataset_view=='D2')
        folder=root/('Episode%04d'%counter[0])
        with _SOCKET_LOCK, patch('app.locdi.evidence_selection_wire.encode', new=subset_encode), patch('app.locdi.evidence_selection_wire.decode', new=subset_decode), patch('app.locdi.evidence_selection_wire.VERSION', new=EVIDENCE_WIRE):
            try:
                if dataset_view=='D2':
                    with patch('app.locdi.common_menu_backend.CommonM5SourceView',new=Overlay):result=backend(case,features)
                else:result=backend(case,features)
            finally:
                if folder.exists():atomic_json(folder/'dataset_view_audit.json',{'view':dataset_view,'policy':POLICY,
                    'original_case_fingerprint':digest(original),'actual_initial_state':state,'actual_initial_fingerprint':digest(state),
                    'augmentation_row_fingerprint':row['row_fingerprint'] if dataset_view=='D2' else None,
                    'source_reference_in_model':False,'synthetic_available_minute':0 if dataset_view=='D2' else None,
                    'synthetic_objective_evidence':False,'clinical_reviewed':False,'evidence_wire':EVIDENCE_WIRE,
                    'scope':SCOPE if dataset_view=='D2' else 'ORIGINAL50_MATCHED_DEVELOPMENT'})
        result['dataset_view']=dataset_view
        result['initial_state_fingerprint']=digest(state)
        return result
    return execute
