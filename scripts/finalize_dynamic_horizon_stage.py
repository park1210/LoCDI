"""Materialize C4-19 stage results without overwriting its failed reader attempt.

The successful complete execution remains independently stored in replay_002.
Stage v1 collects its verified public results plus existing C3 evidence context.
"""
import csv
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.dataset_freeze import load,sha
from app.locdi.llm_setup import digest
from app.locdi.dynamic_horizon_evaluation import complete_identity_recall
from scripts.enrich_full_cohort_sources import write


def csv_write(path,records):
    with path.open('x',encoding='utf-8',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(records[0]));w.writeheader();w.writerows(records)


def finalize():
    stage=ROOT/'outputs/research_evaluation/c4_19_v1';private=ROOT/'outputs/private/research_evaluation/c4_19_v1'
    execution=ROOT/'outputs/research_evaluation/c4_19_replay_002';ep=ROOT/'outputs/private/research_evaluation/c4_19_replay_002'
    if (stage/'execution_reference.json').exists() or (stage/'output_manifest.json').exists():raise ValueError('STAGE_FINALIZATION_ALREADY_USED')
    result=load(execution/'verification.json');tests=load(execution/'test_results.json')
    if result['status']!='PASS' or result['decisions']!=263 or tests['passed']!=247 or tests['total']!=247:raise ValueError('COMPLETE_VERIFIED_EXECUTION_REQUIRED')
    if load(stage/'dynamic_horizon_policy_spec.json')!=load(execution/'dynamic_horizon_policy_spec.json'):raise ValueError('POLICY_CHANGED_BETWEEN_ATTEMPTS')
    # Preserve the first completed regression under an explicit historical name.
    old=stage/'test_results.json';archive=stage/'test_results_before_reader_fix.json'
    if not old.resolve().is_relative_to(ROOT.resolve()) or archive.exists():raise ValueError('SAFE_UNUSED_ARCHIVE_REQUIRED')
    old.rename(archive)
    copies={}
    for p in execution.iterdir():
        if not p.is_file():continue
        if p.name in {'dynamic_horizon_policy_spec.json','consolidated_results_table.csv','research_claim_evidence_matrix.csv'}:continue
        target=stage/('evaluation_protocol_successful_execution.json' if p.name=='evaluation_protocol.json' else p.name)
        with target.open('xb') as stream:stream.write(p.read_bytes())
        if sha(target)!=sha(p):raise ValueError('PUBLIC_MATERIALIZATION_HASH')
        copies[target.relative_to(ROOT).as_posix()]={'source':p.relative_to(ROOT).as_posix(),'sha256':sha(p)}
    c3path=ROOT/'outputs/research_evaluation/c3_16_v1/reports/comparison_analysis.json'
    c3=load(c3path);a,c=c3['conditions']['A'],c3['conditions']['C']
    with (execution/'consolidated_results_table.csv').open(encoding='utf-8',newline='') as stream:table=list(csv.DictReader(stream))
    recall=complete_identity_recall(load(ROOT/'outputs/research_evaluation/c4_10_v1/csv_export_aligned/retrieval_evaluation.json'))
    for row in table:
        if row['area']=='Memory' and row['metric']=='micro recall':
            row['value']=str(recall['value'])
            row['denominator']=str(recall['numerator'])+'/'+str(recall['denominator'])+';'+str(recall['complete_queries'])+' reference-complete queries'
    table += [
        {'area':'External AgentClinic','evidence':'C3-16 stored A/C, no new execution','metric':'repeated action proposal rate A/C','value':str(a['repeated_proposal_rate'])+'/'+str(c['repeated_proposal_rate']),'denominator':str(a['repeated_proposals'])+'/'+str(a['information_proposals'])+' vs '+str(c['repeated_proposals'])+'/'+str(c['information_proposals']),'scope':'Exposed5 scenarios, not Memory/Routing/VoI performance'},
        {'area':'External AgentClinic','evidence':'C3-16 stored A/C','metric':'exact lexical match A/C','value':str(a['exact_match_count'])+'/'+str(c['exact_match_count']),'denominator':'5 cases per condition','scope':'Not clinical equivalence; historical generation31, new generation0'}]
    csv_write(stage/'consolidated_results_table.csv',table)
    with (execution/'research_claim_evidence_matrix.csv').open(encoding='utf-8',newline='') as stream:claims=list(csv.DictReader(stream))
    claims.append({'claim':'AgentClinic state-aware action control','status':'IMPLEMENTED_AND_EMPIRICALLY_EVALUATED','evidence':'C3-16 stored5paired cases, lexical match0/5 each','boundary':'Separate exposed external task, not clinical or LoCDI core benefit'})
    csv_write(stage/'research_claim_evidence_matrix.csv',claims)
    write(stage/'consolidation_provenance.json',{'source_execution_tables':{p:sha(execution/p) for p in ['consolidated_results_table.csv','research_claim_evidence_matrix.csv']},
        'additional_C3_context':{'path':c3path.relative_to(ROOT).as_posix(),'sha256':sha(c3path)},
        'C3_result_not_merged_into_VoI_denominators':True})
    write(stage/'execution_reference.json',{'canonical_successful_execution':'c4_19_replay_002',
        'public_execution_path':execution.relative_to(ROOT).as_posix(),'private_execution_path':ep.relative_to(ROOT).as_posix(),
        'source_trace_sha256':sha(ep/'dynamic_execution_trace.jsonl'),'execution_verification_sha256':sha(execution/'verification.json'),
        'final_test_results_sha256':sha(execution/'test_results.json'),'materialized_public_files':copies,
        'first_attempt':{'namespace':'c4_19_v1','status':'FAILED_BEFORE_FIRST_ACCEPTED_DECISION','record':(private/'first_execution_failure.json').relative_to(ROOT).as_posix()},
        'successful_attempt_decisions':263,'failed_and_successful_decisions_merged':False,
        'policy_threshold_changed':False,'existing_completed_namespace_overwritten':False})
    print('STAGE_C4_19_V1_MATERIALIZED_WITH_FAILURE_PRESERVED',flush=True)


if __name__=='__main__':finalize()
