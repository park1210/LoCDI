"""TEST_ONLY symbols and offline inspection; no external grading or models."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from app.locdi.llm_setup import ROOT,digest
from app.locdi.external_benchmark_contract import validate_healthbench,separate_healthbench,inventory_healthbench
from app.locdi.reference_readiness import empty_reference_form,validate_blinded_packet,source_path,json_pointer
from app.locdi.dataset_suitability import sha
from app.locdi.source_replay import atomic_rows


def fixture():
    return {'prompt_id':'TEST_ONLY_SAMPLE','prompt':[{'role':'user','content':'TEST_ONLY_INPUT'}],
        'rubrics':[{'criterion':'TEST_ONLY_HIDDEN_RUBRIC','points':2,'tags':['TEST_ONLY']}],
        'ideal_completions_data':{'ideal_completion':'TEST_ONLY_HIDDEN_REFERENCE'},'canary':'TEST_ONLY_HIDDEN_CANARY'}


class D4ReferenceAcquisitionContracts(unittest.TestCase):
    def test_actual_acquired_schema_hash_all_rows(self):
        manifests=list((ROOT/'datasets/external/healthbench').glob('*/dataset_manifest_canonical_v2.json'))
        self.assertEqual(len(manifests),1);m=json.loads(manifests[0].read_text(encoding='utf-8'))
        self.assertEqual(digest({k:v for k,v in m.items() if k!='manifest_fingerprint'}),m['manifest_fingerprint'])
        for entry in m['files'].values():self.assertEqual(sha(ROOT/entry['path']),entry['sha256'])
        data=ROOT/m['files']['2025-05-07-06-14-12_oss_eval.jsonl']['path'];before=sha(data)
        inventory=inventory_healthbench(data);self.assertEqual(inventory,m['inventory']);self.assertEqual(before,sha(data))
        self.assertEqual(inventory['sample_count'],5000);self.assertFalse(m['independent_test_approved'])

    def test_required_fields_and_message_scope_rejected(self):
        for key in ('prompt_id','prompt','rubrics'):
            row=fixture();del row[key]
            with self.assertRaises(ValueError):validate_healthbench(row)
        row=fixture();row['prompt'][0]['rubric']='TEST_ONLY_LEAK'
        with self.assertRaisesRegex(ValueError,'MESSAGE_SCOPE'):validate_healthbench(row)

    def test_reference_and_rubric_never_in_agent_preview(self):
        row=fixture();before=deepcopy(row);preview=separate_healthbench(row)
        self.assertEqual(row,before);self.assertEqual(preview['agent_input'],{'messages':row['prompt']})
        text=json.dumps(preview['agent_input'])
        for secret in ('TEST_ONLY_HIDDEN_RUBRIC','TEST_ONLY_HIDDEN_REFERENCE','TEST_ONLY_HIDDEN_CANARY'):self.assertNotIn(secret,text)
        self.assertEqual(preview['private_evaluation']['rubrics'],row['rubrics'])

    def test_test_split_not_relabelled_or_used_for_training(self):
        for purpose in ('training','utility_fit','research_evaluation'):
            with self.assertRaises(ValueError):separate_healthbench(fixture(),purpose=purpose)
        for split in ('train','validation','test'):
            with self.assertRaises(ValueError):separate_healthbench(fixture(),assigned_split=split)
        self.assertEqual(separate_healthbench(fixture())['original_split'],'test')

    def test_negative_points_preserved_not_diagnosis_booleans(self):
        row=fixture();row['rubrics'].append({'criterion':'TEST_ONLY_NEGATIVE','points':-3,'tags':[]})
        result=separate_healthbench(row)
        self.assertEqual(result['private_evaluation']['rubrics'][-1]['points'],-3)
        self.assertIsNone(result['clinical_diagnosis_gold']);self.assertIsNone(result['routing_correctness'])

    def test_unknown_review_never_promoted(self):
        preview=separate_healthbench(fixture())
        self.assertEqual(preview['clinical_diagnosis_reference_status'],'NOT_REVIEWED')
        self.assertIsNone(preview['reviewer']);self.assertIsNone(preview['reviewed_at'])
        packet={'review_form':empty_reference_form(),'agent_responses_hidden':True,'clinical_gold_approved':False}
        self.assertTrue(validate_blinded_packet(packet))
        for key,value in [('review_status','REVIEWED'),('reviewer','TEST_ONLY_FAKE_REVIEWER'),('clinical_verdict',True)]:
            altered=deepcopy(packet);altered['review_form'][key]=value
            with self.assertRaises(ValueError):validate_blinded_packet(altered)
        altered=deepcopy(packet);altered['clinical_gold_approved']=True
        with self.assertRaises(ValueError):validate_blinded_packet(altered)

    def test_blind_reference_rejects_agent_responses_at_any_depth(self):
        for key in ('agent_reviews','agent_response','llm_audit','evaluation','gold'):
            p={'review_form':empty_reference_form(),'agent_responses_hidden':True,'nested':{key:[]}}
            with self.assertRaisesRegex(ValueError,'REVIEW_NOT_BLINDED'):validate_blinded_packet(p)

    def test_public_inventory_never_outputs_sample_or_patient_identity(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT/'tests')) as directory:
            p=Path(directory)/'TEST_ONLY.jsonl';row=fixture();row['patient_id']='TEST_ONLY_PATIENT';p.write_text(json.dumps(row)+'\n',encoding='utf-8')
            text=json.dumps(inventory_healthbench(p));self.assertNotIn('TEST_ONLY_PATIENT',text);self.assertNotIn('TEST_ONLY_SAMPLE',text)
            self.assertNotIn('TEST_ONLY_HIDDEN_RUBRIC',text)

    def test_saved_internal_packets_blinded_blank_and_immutable(self):
        root=ROOT/'outputs/private/reference_review/D4_reference_v1'
        manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        packets=list(root.glob('*case*.json'))+[root/'routing_Candidate010_blinded.json']
        self.assertEqual(len(packets),80)
        for p in packets:
            before=sha(p);self.assertTrue(validate_blinded_packet(json.loads(p.read_text(encoding='utf-8'))));self.assertEqual(before,sha(p))
        for entry in manifest['files'].values():self.assertEqual(sha(ROOT/entry['path']),entry['sha256'])

    def test_source_pointer_scope_and_atomic_no_overwrite(self):
        self.assertEqual(json_pointer({'a/b':{'~':[{'value':'TEST_ONLY'}]}},'/a~1b/~0/0/value'),'TEST_ONLY')
        with self.assertRaises(ValueError):source_path('../TEST_ONLY')
        with tempfile.TemporaryDirectory(dir=str(ROOT/'tests')) as directory:
            root=Path(directory);p=atomic_rows(root,'TEST_ONLY.json',{'status':'NOT_REVIEWED'},jsonl=False);before=p.read_bytes()
            with self.assertRaises(FileExistsError):atomic_rows(root,'TEST_ONLY.json',{},jsonl=False)
            self.assertEqual(p.read_bytes(),before)


if __name__=='__main__':unittest.main()
