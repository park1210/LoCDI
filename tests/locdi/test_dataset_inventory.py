"""TEST_ONLY D1 audit contracts. No generation, fitting or clinical verdicts."""
from copy import deepcopy
import ast
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.locdi.dataset_inventory import (MISSING, value_state, coverage, checked_csv_links,
    refs_linked, analyze_locdi, inventory_entry, audit_episode_orders)
from app.locdi.dataset_loader import load_cases
from app.locdi.llm_setup import ROOT, digest
from scripts.audit_dataset_inventory import save_inventory, main


class DatasetInventoryContracts(unittest.TestCase):
    def test_missing_null_and_empty_are_distinct(self):
        self.assertEqual([value_state(v) for v in (MISSING,None,' ',[],{},0,False)],
                         ['MISSING','NULL','EMPTY_STRING','EMPTY_ARRAY','EMPTY_OBJECT','POPULATED','POPULATED'])

    def test_zero_denominator_not_zero_percent(self):
        result=coverage('TEST_ONLY',[])
        self.assertEqual(result['populated_percent'],'NOT_APPLICABLE')
        self.assertEqual(result['source_linked_cases'],'UNKNOWN')

    def test_source_proxy_time_and_review_separate_not_summed(self):
        r=coverage('TEST_ONLY',[None,'TEST_ONLY',[]],linked=[False,True,False],temporal=[False,True,False],
                   proxy=[False,True,False],reviewed=[False,False,False])
        self.assertEqual((r['populated_cases'],r['empty_cases']),(1,2))
        self.assertEqual(r['source_linked_cases'],1)
        self.assertEqual(r['development_proxy_cases'],1)
        self.assertEqual(r['clinically_reviewed_cases'],0)

    def test_actual_csv_content_required_and_path_escape_not_read(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT/'tests')) as directory:
            root=Path(directory);(root/'test.csv').write_text('FIELD\nTEST_ONLY\n',encoding='utf-8')
            ref={'source_file':'test.csv','source_row':2,'source_record':{'FIELD':'TEST_ONLY'}}
            outside={'source_file':'../outside.csv','source_row':2,'source_record':{'FIELD':'TEST_ONLY'}}
            with patch('app.locdi.dataset_inventory.SYNTHEA',root):
                links,missing=checked_csv_links([{'TEST_ONLY':[ref,outside]}])
            self.assertTrue(links['test.csv',2]);self.assertFalse(links['../outside.csv',2])
            self.assertEqual(missing,['../outside.csv'])
            self.assertTrue(refs_linked(ref,links))
            ref['source_record']['FIELD']='TEST_ONLY_TAMPER'
            with patch('app.locdi.dataset_inventory.SYNTHEA',root):
                changed,_=checked_csv_links([ref])
            self.assertFalse(refs_linked(ref,changed))

    def test_not_found_has_no_invented_size_or_hash(self):
        r=inventory_entry(ROOT/'tests/TEST_ONLY_DOES_NOT_EXIST.json',name='TEST_ONLY')
        self.assertEqual(r['file_status'],'NOT_FOUND')
        self.assertIsNone(r['bytes'])
        self.assertEqual(r['unique_patients'],'UNKNOWN')

    def test_saved_log_ordering_distinguishes_linkage_and_time(self):
        doc={'workflow':{'events':[
            {'event_type':'test_ordered','target_ref':'TEST_ONLY_T','minute':5},
            {'event_type':'result_received','target_ref':'TEST_ONLY_T','minute':4},
            {'event_type':'result_received','target_ref':'TEST_ONLY_MISSING','minute':8}]}}
        r=audit_episode_orders([('TEST_ONLY',doc)])['TEST_ONLY']
        self.assertEqual(r['before_order_violations'],1);self.assertEqual(r['unlinked_results'],1)
        doc['workflow']['events'][-1]['target_ref']='TEST_ONLY_T'
        doc['workflow']['events'][1]['minute']=6
        r=audit_episode_orders([('TEST_ONLY',doc)])['TEST_ONLY']
        self.assertEqual(r['before_order_violations']+r['unlinked_results'],0)

    def test_readonly_real_case_aggregation_and_private_ids_omitted(self):
        # Existing row reused read-only; no synthetic clinical evidence created.
        case=load_cases(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl')[0]
        before=digest(case);r=analyze_locdi([case])
        self.assertEqual(before,digest(case))
        self.assertEqual(r['release_boundary_pass_cases'],1)
        self.assertEqual(r['counts']['memory_time_or_identity_violations'],0)
        self.assertEqual(r['counts']['actual_release_observed_cases'],0)
        self.assertEqual(r['clinical_evaluation_usable_cases'],0)
        for value in (case['patient_id'],case['case_id'],case['agent_visible']['index_encounter_id']):
            self.assertNotIn(value,json.dumps(r))

    def test_saved_aggregate_csv_json_no_overwrite(self):
        report={'locdi':{'coverage':[coverage('TEST_ONLY',[None,'TEST_ONLY'])]}}
        with tempfile.TemporaryDirectory(dir=str(ROOT/'tests')) as directory:
            root=Path(directory);(root/'outputs/reports').mkdir(parents=True)
            with patch('scripts.audit_dataset_inventory.ROOT',root):
                p,c=save_inventory(report,'TEST_ONLY')
                before=(p.read_bytes(),c.read_bytes())
                self.assertEqual(json.loads(p.read_text(encoding='utf-8')),report)
                with self.assertRaises(FileExistsError):save_inventory(report,'TEST_ONLY')
                with self.assertRaises(ValueError):save_inventory(report,'../escape')
                self.assertEqual(before,(p.read_bytes(),c.read_bytes()))

    def test_cli_default_has_no_write_and_sanitizes_errors(self):
        stub={'locdi':{'case_count':0},'inventory':[],'actual_routing_gate':{'training_readiness':'BLOCKED'}}
        with patch('scripts.audit_dataset_inventory.audit_inventory',return_value=stub),patch('scripts.audit_dataset_inventory.save_inventory') as save,patch('builtins.print'):
            self.assertEqual(main([]),0);save.assert_not_called()
        with patch('scripts.audit_dataset_inventory.audit_inventory',side_effect=ValueError('TEST_ONLY_PRIVATE')),patch('builtins.print') as output:
            self.assertEqual(main([]),2);self.assertNotIn('TEST_ONLY_PRIVATE',str(output.call_args))

    def test_audit_has_no_generation_training_or_probability_estimation_call(self):
        tree=ast.parse((ROOT/'app/locdi/dataset_inventory.py').read_text(encoding='utf-8'))
        calls={n.func.id if isinstance(n.func,ast.Name) else n.func.attr if isinstance(n.func,ast.Attribute) else '' for n in ast.walk(tree) if isinstance(n,ast.Call)}
        self.assertFalse(calls & {'fit_utility','chat_json','run_case','run_cases','posterior','plan_voi','run_integrated'})


if __name__=='__main__':unittest.main()
