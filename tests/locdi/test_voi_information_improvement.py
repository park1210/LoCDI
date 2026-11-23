"""TEST_ONLY source, representation, bounded estimation and reveal contracts."""
from copy import deepcopy
import math
from pathlib import Path
import unittest
from unittest.mock import patch
from app.locdi.voi_information_improvement import *
from app.locdi.voi_source_action_inventory import project,action_key,unique_reference,select_first
from app.locdi.dynamic_horizon import policy_spec
from app.core.voi_planning import plan_voi
from scripts.evaluate_voi_information_improvement import baseline_view
from test_routing_voi_readiness import test_only_voi

K=action_key('Test','TEST_ONLY_X','u');L=action_key('Test','TEST_ONLY_Y','u')


def visits():
    return [{'patient_id':'TEST_ONLY_P'+str(i),'visit_id':'TEST_ONLY_V'+str(i),'diagnosis':'TEST_ONLY_A' if i<6 else 'TEST_ONLY_B',
        'values':{K:{'value':i,'data_type':'numeric','available_minute':1,'event_id':'TEST_ONLY_E'+str(i)},
                  L:{'value':i%3,'data_type':'numeric','available_minute':2,'event_id':'TEST_ONLY_L'+str(i)}}} for i in range(12)]


def source():
    start='2026-01-01T00:00:00+00:00';raw={'PATIENT':'TEST_ONLY_P','ENCOUNTER':'TEST_ONLY_V','CODE':'TEST_ONLY_X','UNITS':'u','VALUE':'7','TYPE':'numeric'}
    e={'event_type':'observations','action_type_candidate':'Test','patient_id':'TEST_ONLY_P','encounter_id':'TEST_ONLY_V','timestamp_upper_epoch_ms':1767225660000,
        'timestamp':'2026-01-01T00:01:00+00:00','target':{'code':'TEST_ONLY_X','system':None},'event_id':'TEST_ONLY_E',
        'source_metadata':{'source_file':'TEST_ONLY_CSV','source_row':2,'source_record':raw}}
    return {'case_id':'TEST_ONLY_C','patient_id':'TEST_ONLY_P','index_encounter':{'encounter_id':'TEST_ONLY_V','start_time':start},'source_events':[e],'symptom_sources':[]}


class SourceContracts(unittest.TestCase):
    def test_source_raw_identity(self):self.assertEqual(project(source())['values'][K]['value'],7.)
    def test_source_time_proxy_explicit(self):self.assertEqual(project(source())['values'][K]['availability_basis'],'RECORDED_TIME_PROXY')
    def test_system_not_invented(self):self.assertIn('SYSTEM_UNSPECIFIED',K)
    def test_missing_outcome_not_generated(self):
        s=source();s['source_events']=[];self.assertEqual(project(s)['values'],{})
    def test_patient_mismatch_blocked(self):
        s=source();s['source_events'][0]['source_metadata']['source_record']['PATIENT']='TEST_ONLY_OTHER'
        with self.assertRaisesRegex(ValueError,'IDENTITY'):project(s)
    def test_visit_mismatch_blocked(self):
        s=source();s['source_events'][0]['encounter_id']='TEST_ONLY_OTHER'
        with self.assertRaisesRegex(ValueError,'IDENTITY'):project(s)
    def test_post_explicit_diagnosis_blocked(self):
        s=source();s['diagnosis_confirmed_at']='2026-01-01T00:00:30+00:00';self.assertEqual(project(s)['values'],{})
    def test_future_window_blocked(self):
        s=source();s['source_events'][0]['timestamp_upper_epoch_ms']+=45*60000;self.assertEqual(project(s)['values'],{})
    def test_unknown_time_blocked(self):
        s=source();s['source_events'][0]['timestamp_upper_epoch_ms']=None;self.assertEqual(project(s)['values'],{})
    def test_condition_not_action(self):
        s=source();s['source_events'][0]['event_type']='conditions';self.assertEqual(project(s)['values'],{})
    def test_categorical_separate(self):
        s=source();s['source_events'][0]['source_metadata']['source_record']['VALUE']='TEST_ONLY_CATEGORY'
        self.assertEqual(project(s)['values'][K]['data_type'],'categorical')
    def test_duplicate_identical_first(self):
        s=source();s['source_events']*=2;self.assertEqual(len(project(s)['values']),1)
    def test_conflicting_first_not_chosen(self):
        s=source();e=deepcopy(s['source_events'][0]);e['source_metadata']['source_record']['VALUE']='9';s['source_events'].append(e)
        self.assertEqual(project(s)['values'],{})
    def test_reference_single_only(self):
        s=source();s['index_conditions']=[{'source_record':{'PATIENT':'TEST_ONLY_P','ENCOUNTER':'TEST_ONLY_V','DESCRIPTION':'A'}}]
        self.assertEqual(unique_reference(s,{'A','B'}),'A')
    def test_ambiguous_reference_not_selected(self):
        s=source();s['index_conditions']=[{'source_record':{'PATIENT':'TEST_ONLY_P','ENCOUNTER':'TEST_ONLY_V','DESCRIPTION':d}} for d in ['A','B']]
        self.assertIsNone(unique_reference(s,{'A','B'}))


class EncodingAndEstimation(unittest.TestCase):
    def test_quantile_linear_r7(self):self.assertEqual(quantile([0,3,6,9],1/3),3)
    def test_binary_equal_low(self):self.assertEqual(bucket(5,[5]),'0')
    def test_ternary_equal_cuts_retained(self):
        self.assertEqual(bucket(5,[5,5]),'0');self.assertEqual(bucket(6,[5,5]),'2')
    def test_train_only_quantile(self):self.assertEqual(thresholds(visits()[:3],[K],'ternary')[K],[2/3,4/3])
    def test_degenerate_ternary_schema(self):
        vs=visits()
        for v in vs:v['values'][K]['value']=1
        m,r=estimate(test_only_voi(),vs,[K],'ternary',[]);self.assertEqual(len(m['actions'][0]['outcomes']),3)
        self.assertEqual({x['outcome_id'] for x in r},{'0'});InformationOutcomes(m)
    def test_missing_not_negative(self):
        vs=visits();vs[0]['values'].pop(L);m,r=estimate(test_only_voi(),vs,[K,L],'binary',[])
        aid=m['provenance']['action_key_mapping'][L];self.assertEqual(sum(x['action_id']==aid for x in r),11)
    def test_prior_still_unique_visits(self):
        m,r=estimate(test_only_voi(),visits(),[K,L],'binary',[]);self.assertEqual(m['provenance']['training_visits'],12);self.assertEqual(m['prior']['TEST_ONLY_A'],.5)
    def test_train_eval_patient_overlap_blocked(self):
        with self.assertRaisesRegex(ValueError,'VALIDATION_LEAKAGE'):estimate(test_only_voi(),visits(),[K],'binary',['TEST_ONLY_P1'])
    def test_alpha_one(self):
        m,r=estimate(test_only_voi(),visits(),[K],'binary',[]);self.assertEqual(m['provenance']['smoothing_alpha'],1)
    def test_action_cap_preserved(self):
        with self.assertRaisesRegex(ValueError,'ACTION_CAP8'):estimate(test_only_voi(),visits(),[str(i) for i in range(9)],'binary',[])
    def test_m6_frozen_m4_bounds(self):
        tr=visits();m,_=estimate(test_only_voi(),tr,[K],'binary',[]);b=m['provenance']['binning_thresholds'];extra=deepcopy(tr[0]);extra['patient_id']='TEST_ONLY_NEW';extra['visit_id']='TEST_ONLY_NEW';extra['values'][K]['value']=1000
        n,_=estimate(test_only_voi(),tr+[extra],[K],'binary',[],fixed_bounds=b);self.assertEqual(n['provenance']['binning_thresholds'],b)
    def test_no_template_mutation(self):
        m=test_only_voi();before=deepcopy(m);estimate(m,visits(),[K],'binary',[]);self.assertEqual(m,before)
    def test_baseline_probabilities_unchanged(self):
        old=test_only_voi();legacy=compact(['Test','TEST_ONLY_X','u']);old['provenance']['binning_thresholds']={legacy:5};before=deepcopy(old)
        with patch('app.locdi.voi_information_improvement.fit_outcomes',side_effect=AssertionError('NO_M1_REFIT')):v=baseline_view(old,[K],{legacy:'TEST_ONLY_T'})
        self.assertEqual(old,before);self.assertEqual(v['actions'][0]['outcomes'][0]['likelihood'],old['actions'][0]['outcomes'][0]['likelihood'])
    def test_posterior_normalized(self):
        m,_=estimate(test_only_voi(),visits(),[K,L],'binary',[]);b,u=discrete_belief(m,visits()[0]['values']);self.assertAlmostEqual(sum(b.values()),1)
    def test_truth_not_initialization(self):
        m,_=estimate(test_only_voi(),visits(),[K],'binary',[]);v=deepcopy(visits()[0]);v['patient_id']='TEST_ONLY_HELD';w=deepcopy(v);w['diagnosis']='TEST_ONLY_B'
        self.assertEqual(score_results(m,[v])[0]['belief'],score_results(m,[w])[0]['belief'])
    def test_fixed_budget25(self):
        b=EstimationBudget()
        for name in ['M2','M3','M4','M5','M6']:
            for fold in range(5):b.reserve(name,fold)
        self.assertEqual(b.count,25)
        with self.assertRaises(ValueError):b.reserve('M2',0)
    def test_no_duplicate_fold_refit(self):
        b=EstimationBudget();b.reserve('M2',0)
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):b.reserve('M2',0)
    def test_m1_not_refittable_budget(self):
        with self.assertRaisesRegex(ValueError,'ESTIMATION_BINDING'):EstimationBudget().reserve('M1',0)
    def test_continuous_probe_separated(self):
        m=gaussian_estimate(visits(),[K],{'TEST_ONLY_A':.5,'TEST_ONLY_B':.5},[]);self.assertFalse(m['core_discrete_compatible'])
    def test_gaussian_train_only(self):
        m=gaussian_estimate(visits()[:6],[K],{'TEST_ONLY_A':.5,'TEST_ONLY_B':.5},[]);self.assertEqual(m['stats'][K]['pooled_mean'],2.5)
    def test_gaussian_shared_variance(self):
        m=gaussian_estimate(visits(),[K],{'TEST_ONLY_A':.5,'TEST_ONLY_B':.5},[]);self.assertGreater(m['stats'][K]['shared_variance'],0);self.assertNotIn('class_variances',m['stats'][K])
    def test_gaussian_sparse_fallback(self):
        m=gaussian_estimate(visits()[:3],[K],{'TEST_ONLY_A':.5,'TEST_ONLY_B':.5},[]);self.assertEqual(m['stats'][K]['class_means']['TEST_ONLY_B'],1)
    def test_gaussian_constant_floor(self):
        vs=visits()
        for v in vs:v['values'][K]['value']=0
        m=gaussian_estimate(vs,[K],{'TEST_ONLY_A':.5,'TEST_ONLY_B':.5},[]);self.assertEqual(m['stats'][K]['shared_variance'],1e-6)
    def test_gaussian_log_underflow_no_clipping(self):
        m=gaussian_estimate(visits(),[K],{'TEST_ONLY_A':.5,'TEST_ONLY_B':.5},[]);v=deepcopy(visits()[0]);v['patient_id']='TEST_ONLY_HELD';v['values'][K]['value']=1e6
        r=score_results(m,[v],'gaussian')[0];self.assertTrue(math.isfinite(r['score']['nll']));self.assertFalse(r['score']['zero_probability'])
    def test_gaussian_valid_probability(self):
        m=gaussian_estimate(visits(),[K],{'TEST_ONLY_A':.5,'TEST_ONLY_B':.5},[]);b=gaussian_belief(m,visits()[0]['values']);self.assertAlmostEqual(sum(b.values()),1)
    def test_gaussian_validation_blocks(self):
        with self.assertRaisesRegex(ValueError,'VALIDATION_LEAKAGE'):gaussian_estimate(visits(),[K],{'TEST_ONLY_A':.5,'TEST_ONLY_B':.5},['TEST_ONLY_P0'])
    def test_evaluation_denominator_mismatch(self):
        with self.assertRaisesRegex(ValueError,'DENOMINATOR'):paired([], [{'patient_id':'P','visit_id':'V','score':{}}],10)
    def test_paired_constant_difference(self):
        left=[{'patient_id':str(i),'visit_id':str(i),'score':{'nll':2.,'brier':1.,'top1':0.,'top3':0.}} for i in range(3)];right=deepcopy(left)
        for r in right:r['score']['nll']=1.
        d=paired(left,right,100);self.assertEqual(d['patient_macro']['nll']['ci95'],[-1.,-1.])
    def test_patient_macro_not_visit(self):
        sc=probability_score({'A':.8,'B':.2},'A');bad=probability_score({'A':.2,'B':.8},'A')
        rows=[{'patient_id':p,'score':s,'prior_score':sc,'used_keys':[]} for p,s in [('P1',sc),('P1',sc),('P2',bad)]]
        self.assertNotEqual(aggregate(rows)['patient_macro']['nll'],aggregate(rows)['visit_macro']['nll'])
    def test_no_horizon_tuning(self):self.assertEqual(policy_spec()['thresholds'],{'low':1/3,'high':2/3})
    def test_core_12_action_limit(self):
        a=test_only_voi()['actions'][0]
        with self.assertRaisesRegex(ValueError,'Too many'):plan_voi(test_only_voi()['prior'],[dict(a,id=str(i)) for i in range(13)],test_only_voi()['terminal_utility'])
    def test_eight_action_planner_h3_under_limit(self):
        m=test_only_voi();aa=[dict(m['actions'][0],id='TEST_ONLY_'+str(i)) for i in range(8)];p=plan_voi(m['prior'],aa,m['terminal_utility'],horizon=3)
        self.assertLessEqual(p['evaluated_states'],20000)
    def test_no_core_file_mutation(self):
        import hashlib
        p=Path(__file__).resolve().parents[2]/'app/core/diagnostic_belief.py';h=hashlib.sha256(p.read_bytes()).hexdigest();estimate(test_only_voi(),visits(),[K],'binary',[]);self.assertEqual(h,hashlib.sha256(p.read_bytes()).hexdigest())


class SelectionAndReveal(unittest.TestCase):
    def train(self):
        ks=[action_key('Test','TEST_ONLY_'+str(j),'u') for j in range(6)]
        vs=[{'patient_id':'TEST_ONLY_P'+str(i),'visit_id':'TEST_ONLY_V'+str(i),'diagnosis':'TEST_ONLY_D'+str(i%15),
            'values':{k:{'value':(i*17+j*j)%29 if j<5 else i%15,'data_type':'numeric'} for j,k in enumerate(ks)}} for i in range(60)]
        return ks,vs,{'TEST_ONLY_D'+str(i):1/15 for i in range(15)}
    def test_action_selection_train_only_deterministic(self):
        ks,v,p=self.train();a=select_actions(v,ks[:5],ks,p);b=select_actions(deepcopy(v),ks[:5],ks,p);self.assertEqual(a,b)
    def test_extra_source_support_rule(self):
        ks,v,p=self.train();selected,s,l=select_actions(v,ks[:5],ks,p);self.assertIn(ks[-1],selected)
    def test_too_sparse_extra_not_selected(self):
        ks,v,p=self.train()
        for x in v[5:]:x['values'].pop(ks[-1])
        self.assertEqual(select_actions(v,ks[:5],ks,p)[0],ks[:5])
    def test_redundant_extra_not_selected(self):
        ks,v,p=self.train()
        for x in v:x['values'][ks[-1]]['value']=x['values'][ks[0]]['value']
        self.assertEqual(select_actions(v,ks[:5],ks,p)[0],ks[:5])
    def test_no_categorical_extra(self):
        ks,v,p=self.train()
        for x in v:x['values'][ks[-1]].update(data_type='categorical',value='TEST_ONLY_CATEGORY')
        self.assertEqual(select_actions(v,ks[:5],ks,p)[0],ks[:5])
    def test_missing_values_not_filled(self):
        ks,v,p=self.train();v[0]['values'].pop(ks[-1]);select_actions(v,ks[:5],ks,p);self.assertNotIn(ks[-1],v[0]['values'])
    def panel(self,missing=False):
        m,_=estimate(test_only_voi(),visits(),[K,L],'binary',[]);v=deepcopy(visits()[0])
        if missing:v['values'].pop(K)
        return SourceOutcomePanel(m,v),m
    def test_reveal_requires_selection(self):
        p,m=self.panel();aid=p.menu()[0]
        with self.assertRaisesRegex(ValueError,'BEFORE_SELECTION'):p.acquire(aid,selected=None,budget_remaining=1)
    def test_reveal_actual_source(self):
        p,m=self.panel();aid=p.menu()[0];r=p.acquire(aid,selected=aid,budget_remaining=1);self.assertEqual(r['source_event_id'],'TEST_ONLY_E0');self.assertNotIn(aid,p.menu())
    def test_reveal_missing_source_blocks(self):
        p,m=self.panel(True);aid=m['provenance']['action_key_mapping'][K]
        with self.assertRaisesRegex(ValueError,'MISSING_SOURCE'):p.acquire(aid,selected=aid,budget_remaining=1)
    def test_reveal_budget_blocks(self):
        p,m=self.panel();aid=p.menu()[0]
        with self.assertRaisesRegex(ValueError,'BUDGET'):p.acquire(aid,selected=aid,budget_remaining=0)
    def test_repeat_blocks(self):
        p,m=self.panel();aid=p.menu()[0];p.acquire(aid,selected=aid,budget_remaining=2)
        with self.assertRaisesRegex(ValueError,'REPEATED'):p.acquire(aid,selected=aid,budget_remaining=1)
    def test_unknown_action_blocks(self):
        p,m=self.panel()
        with self.assertRaisesRegex(ValueError,'UNKNOWN_ACTION'):p.acquire('UNKNOWN',selected='UNKNOWN',budget_remaining=1)


if __name__=='__main__':unittest.main()
