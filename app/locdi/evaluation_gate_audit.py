"""Read-only audit of the pinned real collection, with NO review promotion."""
from copy import deepcopy
import hashlib

from app.locdi.llm_setup import ROOT, digest
from app.locdi.routing_collector import PRIVATE_ROOT, load_preparation
from app.locdi.specialty_collection_analysis import _safe_json
from app.locdi.dataset_loader import load_cases
from app.locdi.routing_training_setup import HOLDOUT, REGISTRY
from validation.routing_correctness_gate import empty_review_results, validate_evaluation_gate


def actual_gate_inputs():
    bundle = load_preparation('phase24b_candidates_v2')
    run = PRIVATE_ROOT/'phase24b_collect_001'
    root = ROOT/'outputs/private/routing_collection_review/phase24c_review_v2'
    packet = _safe_json(root/'review_packet.json'); manifest = _safe_json(root/'manifest.json')
    files = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in run.glob('*.json')}
    if (manifest.get('complete') is not True or digest(packet) != manifest.get('packet_fingerprint')
            or files != packet['source_file_sha256'] or packet['preparation_fingerprint'] != bundle['fingerprint']
            or packet['run_id'] != run.name or _safe_json(REGISTRY) != bundle['registry']):
        raise ValueError('BASELINE_CHANGED')
    phase_d = ROOT/'outputs/private/unresolved_analysis/phase24d_analysis_v1'
    dm = _safe_json(phase_d/'manifest.json'); da = _safe_json(phase_d/'coverage_and_response_analysis.json')
    dt = _safe_json(phase_d/'independent_review_template.json')
    if (dm.get('complete') is not True or dm['analysis_fingerprint'] != digest(da) or dm['template_fingerprint'] != digest(dt)
            or da['baseline_file_hashes'] != files or da['phase24c_packet_fingerprint'] != digest(packet)):
        raise ValueError('PHASE_D_ANALYSIS_CHANGED')
    references = _safe_json(PRIVATE_ROOT/'phase24b_candidates_v2/reference_candidates.json')
    refmap = {r['case_ref']:r for r in references}
    original_fingerprints = {item['case_ref']: {a['source_outcome']['response_ref']: a['source_outcome']['response_fingerprint']
        for a in item['agent_reviews']} for item in packet['items']}
    sources, contexts = [], []
    for c in bundle['candidates']:
        d = c['decision']
        sources.append({'source_ref': d['source_ref'], 'purpose': 'routing_training',
            **{k:deepcopy(d[k]) for k in ('patient_id','visit_id','decision_id','minute','observation')},
            'source_fingerprint': c['source_fingerprint'], 'visit_started_at': c['visit_started_at'],
            'visibility': {p:r['available_minute'] for p,r in c['visibility'].items()}, 'review': {'status':'NOT_REVIEWED'}})
        contexts.append({'case_ref': c['case_ref'], **{k:d[k] for k in ('patient_id','visit_id','decision_id')},
            'encounter_started_at': c['visit_started_at'], 'source_fingerprint': c['source_fingerprint'],
            'visibility': deepcopy(c['visibility']), 'reference_candidates': deepcopy(refmap[c['case_ref']]),
            'diagnosis_catalog': deepcopy(bundle['catalog']), 'original_response_fingerprints': original_fingerprints[c['case_ref']]})
    data = {'schema_version':'locdi.routing_collection.v1', 'purpose':'routing_training', 'registry':bundle['registry'],
            'split_manifest':bundle['split_manifest'], 'decisions':[c['decision'] for c in bundle['candidates']],
            'outcomes':[_safe_json(p) for p in sorted(run.glob('Candidate*.json'))], 'judgments':[]}
    options = dict(review_results=empty_review_results(), decision_contexts=contexts, registry_manifest=bundle['registry'],
        source_catalog=sources, label_catalog=[], holdout_patient_ids=[r['patient_id'] for r in load_cases(HOLDOUT)],
        dataset_fingerprint=hashlib.sha256(HOLDOUT.read_bytes()).hexdigest())
    return data, options


def audit_actual_gate():
    data, options = actual_gate_inputs()
    result = validate_evaluation_gate(data, **options)
    result.update(run_id='phase24b_collect_001', preparation='phase24b_candidates_v2',
                  baseline_outcome_count=len(data['outcomes']), clinical_metrics='NOT_EVALUATED')
    return result
