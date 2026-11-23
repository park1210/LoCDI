"""Separate opt-in factory; legacy path delegates unchanged to existing backend."""
from contextvars import ContextVar
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch
from app.core.longitudinal_memory import PatientMemoryStore
from app.core.hospital_llm_policy import HospitalLLMPolicy
from app.core.clinical_observation import doctor_observation
from app.locdi.common_menu_backend import make_common_backend,CommonMenuNativeClient
from app.locdi.medical_knowledge import MedicalKnowledgeIndex,historical_labels
from app.locdi.diagnostic_memory_rag import (DiagnosticMemoryRetriever,public_projection,knowledge_projection,
    MEMORY_MODES,KNOWLEDGE_MODES,historical_series)
from app.locdi.routing_collector import atomic_json
from app.locdi.llm_setup import digest

ACTIVE_RUNTIME=ContextVar('memory_rag_actual_decision_runtime',default=None)

def make_memory_rag_backend(cases,*,memory_mode='legacy_top5',knowledge_mode='none',source_map=None,corpus=None,**kwargs):
    if memory_mode not in MEMORY_MODES or knowledge_mode not in KNOWLEDGE_MODES:raise ValueError('MEMORY_RAG_MODE')
    # No patches, extra payload keys, metadata or modified selection on default path.
    if memory_mode=='legacy_top5' and knowledge_mode=='none':return make_common_backend(cases,**kwargs)
    if kwargs.get('knowledge_corpus') is not None or kwargs.get('memory_time_labels') or kwargs.get('rule_based'):raise ValueError('SEPARATE_FACTORS_REQUIRED')
    index=MedicalKnowledgeIndex(corpus) if knowledge_mode!='none' else None
    retriever=DiagnosticMemoryRetriever(source_map) if memory_mode=='diagnostic_retrieval' else None
    base_retrieve=PatientMemoryStore.retrieve;core_decision=HospitalLLMPolicy._decision
    configs=kwargs['config'];bindings=kwargs['bindings'];root=Path(kwargs['checkpoint_root']);episode=[0]
    backend=make_common_backend(cases,**kwargs)
    def execute_case(case,features):
        if features.get('routing_mode')!='er_only' or features.get('voi_enabled'):raise ValueError('MEMORY_RAG_B2_ROUTING_UNCHANGED')
        episode[0]+=1;folder=root/('Episode%04d'%episode[0]);binding=bindings[case['case_ref']];traces=[];knowledge_traces=[]
        def retrieve(store,identity,query,mode='selective',top_k=5,*,max_chars=None,embeddings=None):
            legacy=base_retrieve(store,identity,query,mode,top_k,max_chars=max_chars,embeddings=embeddings)
            if mode!='selective':return legacy
            rt=ACTIVE_RUNTIME.get();projection,qt=(public_projection(doctor_observation(rt),binding,rt) if rt is not None else ({},{'query':'','query_fingerprint':digest(''),'visibility_status':'NO_BOUND_DECISION_YET'}))
            pool=base_retrieve(store,identity,query,'full_history')['records']
            result,trace=retriever.retrieve(legacy,pool,identity,qt['query'],top_k=top_k,max_chars=max_chars)
            trace.update(query_source=qt,minute=rt.current_time_min if rt is not None else None,
                historical_series=historical_series(result['records'],source_map))
            traces.append(trace);return result
        def decision(policy,runtime,role,department=''):
            token=ACTIVE_RUNTIME.set(runtime)
            try:return core_decision(policy,runtime,role,department)
            finally:ACTIVE_RUNTIME.reset(token)
        class FlagsClient(CommonMenuNativeClient):
            def chat_json(self,system_prompt,user_prompt,model=None,*,schema=None):
                payload=json.loads(user_prompt)
                if memory_mode=='historical_display':payload['memory_scope_annotations']=historical_labels(payload['observation'])
                if knowledge_mode=='common_pack':payload['external_medical_knowledge']=index.common()
                elif knowledge_mode=='adaptive_bm25':
                    packet,qt=knowledge_projection(index,payload['observation'],binding,runtime=ACTIVE_RUNTIME.get())
                    payload['external_medical_knowledge']=packet;knowledge_traces.append({'request':self.request_count+1,'packet':packet,'query_source':qt})
                return super().chat_json(system_prompt,json.dumps(payload,ensure_ascii=False,sort_keys=True),model,schema=schema)
        with ExitStack() as stack:
            stack.enter_context(patch('app.locdi.common_menu_backend.CommonMenuNativeClient',new=FlagsClient))
            stack.enter_context(patch.object(HospitalLLMPolicy,'_decision',new=decision))
            if retriever is not None:stack.enter_context(patch.object(PatientMemoryStore,'retrieve',new=retrieve))
            try:result=backend(case,features)
            finally:
                if folder.exists():atomic_json(folder/'memory_rag_audit.json',{'memory_mode':memory_mode,'knowledge_mode':knowledge_mode,
                    'memory_retrievals':traces,'knowledge_queries':knowledge_traces,'knowledge_is_patient_evidence':False})
        result['memory_rag']={'memory_mode':memory_mode,'knowledge_mode':knowledge_mode,'corpus_fingerprint':index.fingerprint if index else None,
            'knowledge_queries':len(knowledge_traces),'clinical_evaluation':'NOT_EVALUATED'}
        return result
    return execute_case
