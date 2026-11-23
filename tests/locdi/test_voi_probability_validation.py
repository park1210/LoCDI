"""TEST_ONLY probability/identity contracts, no clinical labels or network."""
from copy import deepcopy
import math
from pathlib import Path
import unittest
from app.locdi.voi_probability_validation import *
from scripts.evaluate_voi_probability_model import evaluate_visits, aggregate
from app.locdi.dynamic_horizon import policy_spec
from test_routing_voi_readiness import test_only_voi


KEY='TEST_ONLY_KEY'


def fixture():
    m=test_only_voi()
    for i,o in enumerate(m['actions'][0]['outcomes']):o['id']=BINS[i];o['supports_diagnoses']=[]
    m['provenance']['binning_thresholds']={KEY:5}
    return m


def visits():
    return [{'patient_id':'TEST_ONLY_P'+str(i//2),'visit_id':'TEST_ONLY_V'+str(i),'case_id':'TEST_ONLY_C'+str(i),
        'diagnosis':'TEST_ONLY_A' if i<6 else 'TEST_ONLY_B','values':{KEY:{'value':i,'event_id':'TEST_ONLY_E'+str(i)}}} for i in range(12)]


class ProbabilityContracts(unittest.TestCase):
    def test_group_patients_never_split(self):
        vs=visits();f=grouped_folds(vs,5)
        self.assertEqual(len(f),6);self.assertEqual(set(f.values()),set(range(5)))
        for v in vs:self.assertEqual(f[v['patient_id']],f['TEST_ONLY_P'+str(int(v['visit_id'].split('V')[-1])//2)])
    def test_split_deterministic(self):self.assertEqual(grouped_folds(visits(),5),grouped_folds(visits()[::-1],5))
    def test_duplicate_visit_rejected(self):
        vs=visits()
        with self.assertRaisesRegex(ValueError,'DUPLICATE_VISIT'):grouped_folds(vs+[vs[0]])
    def test_small_group_rejected(self):
        with self.assertRaisesRegex(ValueError,'TOO_FEW'):grouped_folds(visits()[:2],5)
    def test_fold_range(self):
        with self.assertRaises(ValueError):grouped_folds(visits(),1)
    def test_train_only_median(self):
        tr=visits()[:4];a,rr=fit_fold(fixture(),tr,{KEY:'TEST_ONLY_T'},['TEST_ONLY_HELD'])
        self.assertEqual(a['provenance']['binning_thresholds'][KEY],1.5)
        self.assertEqual(len(rr),4)
    def test_train_only_prior(self):
        a,_=fit_fold(fixture(),visits()[:4],{KEY:'TEST_ONLY_T'},[])
        self.assertEqual(a['prior']['TEST_ONLY_A'],5/6);self.assertEqual(a['prior']['TEST_ONLY_B'],1/6)
    def test_train_only_likelihood(self):
        a,_=fit_fold(fixture(),visits()[:4],{KEY:'TEST_ONLY_T'},[])
        self.assertEqual(a['actions'][0]['outcomes'][0]['likelihood']['TEST_ONLY_A'],.5)
    def test_validation_patient_blocks_fit(self):
        with self.assertRaisesRegex(ValueError,'VALIDATION_LEAKAGE'):fit_fold(fixture(),visits(),{KEY:'TEST_ONLY_T'},['TEST_ONLY_P0'])
    def test_oov_train_rejected(self):
        vs=visits();vs[0]['diagnosis']='TEST_ONLY_OOV'
        with self.assertRaisesRegex(ValueError,'OOV_TRAIN'):fit_fold(fixture(),vs,{KEY:'TEST_ONLY_T'},[])
    def test_zero_support_smoothing_is_explicit(self):
        a,_=fit_fold(fixture(),visits()[:4],{KEY:'TEST_ONLY_T'},[])
        self.assertEqual(a['provenance']['empirical_support']['TEST_ONLY_T']['TEST_ONLY_B'],0)
        self.assertEqual(a['actions'][0]['outcomes'][0]['likelihood']['TEST_ONLY_B'],.5)
        self.assertIn('not_strict',a['provenance']['coverage_policy'])
    def test_original_model_unchanged(self):
        m=fixture();before=deepcopy(m);fit_fold(m,visits(),{KEY:'TEST_ONLY_T'},[]);self.assertEqual(m,before)
    def test_validation_values_do_not_affect_estimate(self):
        tr=visits()[:8];a,_=fit_fold(fixture(),tr,{KEY:'TEST_ONLY_T'},['TEST_ONLY_HELD'])
        held=visits()[-1];held['values'][KEY]['value']=1e9
        b,_=fit_fold(fixture(),tr,{KEY:'TEST_ONLY_T'},['TEST_ONLY_HELD']);self.assertEqual(a,b)
    def test_encode_equal_median_low(self):self.assertEqual(encode(5,5),BINS[0])
    def test_encode_above_high(self):self.assertEqual(encode(5.1,5),BINS[1])
    def test_nonfinite_rejected(self):
        for v in [float('nan'),float('inf')]:
            with self.assertRaises(ValueError):encode(v,0)
    def test_nll_and_brier(self):
        s=probability_score({'A':.8,'B':.2},'A');self.assertAlmostEqual(s['nll'],-math.log(.8));self.assertAlmostEqual(s['brier'],.08)
    def test_top3_and_lexical_tie(self):
        s=probability_score({'B':.25,'A':.25,'C':.25,'D':.25},'D');self.assertEqual(s['top1'],0);self.assertEqual(s['top3'],0)
    def test_zero_probability_not_clipped(self):
        s=probability_score({'A':1.,'B':0.},'B');r=summary([s]);self.assertIsNone(r['nll']);self.assertEqual(r['nll_status'],'POSITIVE_INFINITY')
    def test_zero_denominator(self):self.assertEqual(summary([])['status'],'NOT_APPLICABLE');self.assertIsNone(calibration([])['ece'])
    def test_oov_reference_rejected(self):
        with self.assertRaisesRegex(ValueError,'OOV_REFERENCE'):probability_score({'A':.5,'B':.5},'C')
    def test_distribution_rejected(self):
        with self.assertRaises(ValueError):probability_score({'A':.8,'B':.8},'A')
    def test_calibration_bin_denominator(self):
        ss=[probability_score({'A':.5,'B':.5},'A'),probability_score({'A':1.,'B':0.},'A')]
        c=calibration(ss);self.assertEqual(sum(x['count'] for x in c['bins']),2);self.assertEqual(c['bins'][9]['count'],1);self.assertAlmostEqual(c['ece'],.25)
    def test_calibration_boundary(self):
        c=calibration([probability_score({'A':.9,'B':.1},'A')]);self.assertEqual(c['bins'][9]['count'],1)
    def test_posterior_normalization(self):
        m=fixture();b,p=apply_outcome(m['prior'],m['actions'][0],BINS[0]);self.assertAlmostEqual(sum(b.values()),1);self.assertAlmostEqual(b['TEST_ONLY_A'],.9);self.assertAlmostEqual(sum(p.values()),1)
    def test_unknown_outcome_rejected(self):
        with self.assertRaisesRegex(ValueError,'UNKNOWN_OUTCOME'):apply_outcome(fixture()['prior'],fixture()['actions'][0],'UNKNOWN')
    def test_duplicate_same_action_once(self):
        e={'action_id':'TEST_ONLY_T','outcome_id':BINS[0],'minute':1,'event_index':0};b,s=trajectory(fixture(),[e,e]);self.assertEqual(len(s),1);self.assertAlmostEqual(b['TEST_ONLY_A'],.9)
    def test_conflicting_repeat_rejected(self):
        es=[{'action_id':'TEST_ONLY_T','outcome_id':x,'minute':i,'event_index':i} for i,x in enumerate(BINS)]
        with self.assertRaisesRegex(ValueError,'CONFLICTING'):trajectory(fixture(),es)
    def test_future_observation_rejected(self):
        with self.assertRaisesRegex(ValueError,'FUTURE'):unique_observations([{'minute':2,'event_index':0}],1)
    def test_same_minute_event_order(self):
        es=[{'action_id':a,'outcome_id':'x','minute':1,'event_index':i} for a,i in [('B',2),('A',1)]]
        self.assertEqual(unique_observations(es,1)[0]['action_id'],'A')
    def test_zero_observations_preserve_prior(self):self.assertEqual(trajectory(fixture(),[]),(fixture()['prior'],[]))
    def test_horizon_threshold_unchanged(self):
        self.assertEqual(policy_spec()['thresholds'],{'low':1/3,'high':2/3})
    def test_horizon_zero_remaining(self):self.assertEqual(horizon(fixture()['prior'],fixture(),['TEST_ONLY_T'])['eligibility_status'],'TERMINAL_ONLY')
    def test_independent_baseline_train_counts(self):
        a,rr=fit_fold(fixture(),visits(),{KEY:'TEST_ONLY_T'},[]);ind=independent_outcomes(a,rr);self.assertEqual(sum(ind['TEST_ONLY_T'].values()),1)
    def test_evaluation_train_overlap_rejected(self):
        a,rr=fit_fold(fixture(),visits(),{KEY:'TEST_ONLY_T'},[])
        with self.assertRaisesRegex(ValueError,'EVALUATION_PATIENT'):evaluate_visits(a,visits(),{KEY:'TEST_ONLY_T'},rr)
    def test_oov_eval_retained(self):
        v=visits()[0];v['diagnosis']='TEST_ONLY_OOV';r=evaluate_visits(fixture(),[v],{KEY:'TEST_ONLY_T'},[]);self.assertEqual(r[0]['status'],'NOT_EVALUATED_OOV')
    def test_unordered_no_false_timeline(self):
        a,rr=fit_fold(fixture(),visits(),{KEY:'TEST_ONLY_T'},[]);v=visits()[0];v['patient_id']='TEST_ONLY_NEW'
        r=evaluate_visits(a,[v],{KEY:'TEST_ONLY_T'},rr);self.assertEqual([s['step'] for s in r[0]['progression']],[0,'unordered_final'])
    def test_patient_macro_differs_visit_macro(self):
        from scripts.evaluate_voi_probability_model import macro
        good=probability_score({'A':.9,'B':.1},'A');bad=probability_score({'A':.1,'B':.9},'A')
        rows=[{'patient_id':'P1','final':good}]*3+[{'patient_id':'P2','final':bad}]
        r=macro(rows,'final');self.assertNotEqual(r['patient_macro']['nll'],r['visit_macro']['nll'])
    def test_missing_action_not_fabricated(self):
        v=visits()[0];v['values']={};m=fixture();m['provenance']['training_patients']=[]
        r=evaluate_visits(m,[v],{KEY:'TEST_ONLY_T'},[]);self.assertEqual(r[0]['outcomes'],[]);self.assertEqual(r[0]['prior'],r[0]['final'])
    def test_source_numeric_identity(self):
        e={'action':'Test','evidence':compact({'records':[{'event_type':'observations','code':'TEST_ONLY','recorded_evidence':{'units':'u','value':'7'}}]})}
        self.assertEqual(numeric(e),(compact(['Test','TEST_ONLY','u']),7.))
    def test_ambiguous_numeric_not_inferred(self):self.assertIsNone(numeric({'action':'Ask','evidence':compact({'facts':[]})}))
    def test_source_late_values_excluded(self):
        e={'event_id':'TEST_ONLY','action':'Ask','available_minute':46,'evidence':compact({'facts':[{'code':'X','units':'u','value':1}]})}
        self.assertEqual(first_values([e],[compact(['Ask','X','u'])],45),{})
    def test_tied_conflicting_source_not_selected(self):
        es=[{'event_id':str(i),'action':'Ask','available_minute':0,'evidence':compact({'facts':[{'code':'X','units':'u','value':i}]})} for i in [1,2]]
        self.assertEqual(first_values(es,[compact(['Ask','X','u'])]),{})
    def test_discriminability_internal(self):
        d=discriminability(fixture());self.assertAlmostEqual(d['actions'][0]['tv_mean'],.8);self.assertGreater(d['actions'][0]['prior_expected_information_gain_nats'],0)


if __name__=='__main__':unittest.main()
