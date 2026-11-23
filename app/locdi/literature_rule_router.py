"""Development referral-signal prototype; not a diagnosis or emergency triage.

Equal priorities; conflict/no evidence => abstain. No Source Reference or
fitted utility is used. Specialty assignment is an explicit researcher mapping.
"""
import json,re,math
from app.core.agent_registry import AgentRegistry
from app.locdi.medical_knowledge import assert_public
from app.locdi.common_runtime_recovery import released_measurement_view
VERSION='literature_rule_based.v1'
RULES={
 'R_ENDO_A1C':{'department':'endocrinology','priority':10,'source_ids':['S02'],'basis':'A1c >=6.5 percent is a referral signal; not confirmed type2 diabetes'},
 'R_RENAL_EGFR':{'department':'nephrology','priority':10,'source_ids':['S07','S15'],'basis':'eGFR <60 prompts renal assessment; chronicity is not assumed'},
 'R_CARDIO_SYMPTOM':{'department':'cardiology','priority':10,'source_ids':['S01','S11'],'basis':'Explicit current chest pain/palpitations; not a diagnosis or acuity triage'},
 'R_PULMO_SYMPTOM':{'department':'pulmonology','priority':10,'source_ids':['S09','S13'],'basis':'Explicit current wheeze/shortness of breath'},
 'R_NEURO_SYMPTOM':{'department':'neurology','priority':10,'source_ids':['S03'],'basis':'Explicit current memory loss/confusion requires assessment'},
}
def select_rule(observed,binding,candidates):
    assert_public(observed);hits=[]
    evidence=list(observed.get('objective_findings',[]))+[h.get('answer','') for h in observed.get('history',[]) if h.get('answer_status')!='unknown']
    p=released_measurement_view({'observation':observed,'evidence':evidence},binding)
    for m in p['released_measurements']:
        try:v=float(m['value'])
        except (ValueError,TypeError):continue
        if not math.isfinite(v):continue
        rule=None
        if m['kind']=='Test' and m['code']=='4548-4' and m['unit']=='%' and v>=6.5:rule='R_ENDO_A1C'
        if m['kind']=='Test' and m['code']=='33914-3' and m['unit']=='mL/min/{1.73_m2}' and v<60:rule='R_RENAL_EGFR'
        if rule:hits.append({'rule_id':rule,'basis':'CURRENT_RELEASED_MEASUREMENT','evidence_index':m['evidence_index']})
    text_fields=[observed.get('chief_complaint',''),*observed.get('current_symptoms',[])]
    for index,text in enumerate(text_fields):
        if re.search(r'\b(no|not|denies|without|history of|family history)\b',str(text),re.I):continue
        for pattern,rule in [('chest pain|palpitations','R_CARDIO_SYMPTOM'),('wheez(?:e|ing)|shortness of breath','R_PULMO_SYMPTOM'),('memory loss|confusion','R_NEURO_SYMPTOM')]:
            if re.search(pattern,str(text),re.I):hits.append({'rule_id':rule,'basis':'CURRENT_SYMPTOM_TEXT','field_ordinal':index})
    hits=[h for h in hits if RULES[h['rule_id']]['department'] in candidates]
    departments=sorted({RULES[h['rule_id']]['department'] for h in hits})
    status='selected' if len(departments)==1 else 'abstained'
    return {'router':VERSION,'mode':'rule_based','status':'available' if status=='selected' else status,
        'decision_status':status,'selected_specialty':departments[0] if status=='selected' else None,
        'candidate_departments':sorted(candidates),'allowed_departments':sorted(candidates),
        'rule_hits':hits,'rationale':'unique literature-derived referral signal' if status=='selected' else 'conflicting specialties' if departments else 'insufficient current signals',
        'priority_policy':'equal priority; abstain on conflict','clinical_routing_validated':False,
        'specialty_scores':{},'utility_estimates':{},'score_semantics':'NO_UTILITY_RULE_BASELINE'}

class LiteratureRuleRegistry(AgentRegistry):
    def __init__(self,manifest,utility_artifact=None,*,source_binding):
        super().__init__(manifest,utility_artifact);self.source_binding=source_binding
    def select_specialty(self,current_state,retrieved_memory,departments,*,router='rule_similarity',seed=0):
        return select_rule(current_state,self.source_binding,departments)
