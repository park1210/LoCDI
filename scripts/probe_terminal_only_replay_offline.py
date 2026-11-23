"""Network-denied Core replay and terminal-contract regressions; no LLM."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime,timedelta,timezone
from unittest.mock import patch
import json,sys,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_terminal_only_replay as runner
from scripts.probe_common_menu_offline import TestOnlyTransport,Reply
from app.locdi.memory_citation_wire import _decision_nodes
from app.locdi.terminal_only_recovery import terminal_request,VERSION
from app.locdi.current_ablation_runner import GenerationBudget,_BUDGET
from app.locdi.original_source_inventory import read
from app.locdi.routing_collector import atomic_json


class TerminalFake(TestOnlyTransport):
    def __init__(self,config,invalid=False):super().__init__(config);self.invalid=invalid
    def post(self,url,*,json):
        self.posts.append(url);p=__import__('json').loads(json['messages'][1]['content']);self.payloads.append(p)
        assert p['terminal_request_contract']==VERSION and p['allowed_actions']==['diagnose']
        nodes=list(_decision_nodes(json['format']))
        assert all(n['properties']['action']['enum']==['diagnose'] for n in nodes)
        node=next(n for n in nodes if 'unresolved' in n['properties']['certainty']['enum'])
        answer={'action':'diagnose','target':'Unresolved','certainty':'unresolved',
                'department':p['department_catalog'][0],'rationale':'TEST_ONLY insufficient evidence.',
                'assessment':'TEST_ONLY not a clinical judgment.', 'evidence_subset_code':'NONE',
                'memory_subset_code':next(c for c in node['properties']['memory_subset_code']['enum'] if set(c)=={'0'})}
        if self.invalid:answer['evidence_subset_code']='999'
        return Reply({'model':self.config['model'],'done':True,'done_reason':'stop','prompt_eval_count':1,
                      'eval_count':1,'message':{'content':__import__('json').dumps(answer)}})


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pub,cls.pr,cls.plan,cls.bp,cls.parent=runner.load()
        cls.tasks=read(cls.pr/'tasks.json');cls.config=read(cls.bp/'llm_config.json')
        cls.output=cls.pr/'TEST_ONLY'
        cls.output.mkdir(exist_ok=False)
        cls.results=[]
    def setUp(self):
        self.patches=[patch('socket.socket.connect',side_effect=AssertionError('NETWORK_FORBIDDEN_TEST_ONLY')),
                      patch('socket.create_connection',side_effect=AssertionError('NETWORK_FORBIDDEN_TEST_ONLY'))]
        for p in self.patches:p.start();self.addCleanup(p.stop)
    def auth(self):
        a=read(self.pr/'approval_template.json');now=datetime.now(timezone.utc)
        for k in a:
            if k.startswith('accept_') or k in ('operator_approved','no_other_ollama_work'):a[k]=True
        a.update(operator_ref='TEST_ONLY_NOT_REAL_APPROVAL',user_statement='TEST_ONLY_NETWORK_DENIED',
                 recorded_at=now.isoformat(),expires_at=(now+timedelta(hours=1)).isoformat())
        return a
    def test_ten_original_stop_snapshots_through_original_core(self):
        for task in self.tasks:
            with self.subTest(panel=task['panel_ordinal']):
                fake=TerminalFake(self.config);budget=GenerationBudget(1,1);token=_BUDGET.set(budget)
                try:
                    r=runner.replay_one(task,p=self.plan,bp=self.bp,parent_pr=self.parent,authorization=self.auth(),
                        folder=self.output/('Panel%02d'%task['panel_ordinal']),factory=lambda _:fake)
                    self.assertEqual(budget.used,1);self.assertEqual(len(fake.posts),1)
                    self.assertEqual(r['prediction']['certainty'],'unresolved')
                    self.assertTrue(r['terminal_core_revalidated'])
                    self.assertEqual(r['cached_responses'],task['stored_requests'])
                    self.results.append({'panel_ordinal':task['panel_ordinal'],'status':'PASS',
                                         'cached':r['cached_responses'],'terminal_fake_calls':1,'actual_generation':0})
                finally:_BUDGET.reset(token)
    def test_invalid_citation_rejected_without_retry(self):
        fake=TerminalFake(self.config,invalid=True);budget=GenerationBudget(1,1);token=_BUDGET.set(budget)
        try:
            with self.assertRaises(ValueError):runner.replay_one(self.tasks[0],p=self.plan,bp=self.bp,parent_pr=self.parent,
                authorization=self.auth(),folder=self.output/'invalid_citation',factory=lambda _:fake)
            self.assertEqual(len(fake.posts),1);self.assertEqual(budget.used,1)
        finally:_BUDGET.reset(token)
    def test_exhausted_budget_prevents_terminal_post(self):
        fake=TerminalFake(self.config);budget=GenerationBudget(1,1);budget.used=1;token=_BUDGET.set(budget)
        try:
            with self.assertRaises(ValueError):runner.replay_one(self.tasks[0],p=self.plan,bp=self.bp,parent_pr=self.parent,
                authorization=self.auth(),folder=self.output/'budget_exhausted',factory=lambda _:fake)
            self.assertEqual(len(fake.posts),0)
        finally:_BUDGET.reset(token)
    def test_prefix_wire_mutation_is_blocked(self):
        c=runner.PrefixReplayTransport(self.tasks[0]['source_episode'],{},lambda _:self.fail('NO_TRANSPORT'))
        wire=deepcopy(c.before[0]['request']);wire['options']['seed']=43
        with self.assertRaisesRegex(ValueError,'PREFIX_WIRE_DIFFERENCE'):c.post('unused',json=wire)
        self.assertEqual(c.new_posts,0)
    def test_core_terminal_permission_is_required(self):
        with self.assertRaisesRegex(ValueError,'CORE_TERMINAL_BRANCH'):
            terminal_request({'allowed_actions':['request_test']},{'properties':{'action':{'enum':['request_test']}}})
    def test_narrowing_preserves_facts_and_supported_constraints(self):
        p={'allowed_actions':['request_test','diagnose'],'candidate_actions':[{'action':'request_test'},{'action':'diagnose'}],
           'observation':{'elapsed_minutes':12},'evidence':['TEST_ONLY'],'supported_diagnosis_eligible_indices':[0]}
        branch={'properties':{'action':{'enum':['diagnose']},'target':{'enum':['Unresolved']},'certainty':{'enum':['unresolved']}}}
        original=deepcopy(p);new,s=terminal_request(p,{'anyOf':[{'properties':{'action':{'enum':['request_test']}}},branch]})
        self.assertEqual(p,original)
        for k in ('observation','evidence','supported_diagnosis_eligible_indices'):self.assertEqual(new[k],p[k])
        self.assertEqual(s,branch);self.assertEqual(new['allowed_actions'],['diagnose'])


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'status':'PASS' if result.wasSuccessful() else 'FAIL','test_methods':result.testsRun,
            'failures':len(result.failures),'errors':len(result.errors),'real_stop_replays':getattr(Contracts,'results',[]),
            'actual_LLM_generation':0,'fitting':0,'network':'DENIED',
            'plan_fingerprint':getattr(Contracts,'plan',{}).get('fingerprint')}
    atomic_json(runner.PUB/'TEST_ONLY_validation.json',report)
    raise SystemExit(0 if result.wasSuccessful() else 1)
