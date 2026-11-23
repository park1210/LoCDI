"""TEST_ONLY post-failure exporter regression. No requests or clinical review."""
from copy import deepcopy
import unittest
from app.locdi.episode_exporter import _configuration,_fingerprint


def metadata():
    config={'memory_mode':'none','routing_mode':'er_only','primary_router':'rule_similarity',
        'action_selector':'specialist_heuristic','memory_source':'fixed','top_k':None,
        'minute_limit':120,'routing_seed':0,'max_retries':0,'max_memory_chars':None,
        'voi_enabled':False,'accept_recorded_release_assumption':True,
        'dependency_fingerprints':{k:None for k in ('registry_manifest','routing_utility','voi_model',
            'embedding_artifact','diagnosis_catalog','diagnosis_departments')}}
    return {'configuration':config,'configuration_fingerprint':_fingerprint(config)}


def source_config():
    return {'policy':'SYNTHEA_SEQUENTIAL_RECORDED_TIME_PROXY.v1','fingerprint':'a'*64,'test_only':True,
        'turnaround_minutes':5,'clinical_release_approved':False,'v1_grants_created':0,
        'candidate_wire_projection':'lossless clinical data; action,target pairs unchanged'}


class SequentialExportContractTests(unittest.TestCase):
    def test_original_config_unchanged(self):
        m=metadata();self.assertEqual(_configuration(m),m['configuration'])
    def test_explicit_optional_metadata_preserved(self):
        m=metadata();m['configuration']['sequential_source_view']=source_config()
        m['configuration_fingerprint']=_fingerprint(m['configuration'])
        before=deepcopy(m);self.assertEqual(_configuration(m),m['configuration']);self.assertEqual(m,before)
    def test_tampered_fingerprint_still_rejected(self):
        m=metadata();m['configuration']['sequential_source_view']=source_config()
        with self.assertRaises(ValueError):_configuration(m)
    def test_clinical_promotion_rejected(self):
        m=metadata();v=source_config();v['clinical_release_approved']=True
        m['configuration']['sequential_source_view']=v;m['configuration_fingerprint']=_fingerprint(m['configuration'])
        with self.assertRaises(ValueError):_configuration(m)
    def test_unknown_config_fields_rejected(self):
        m=metadata();m['configuration']['unknown']='TEST_ONLY';m['configuration_fingerprint']=_fingerprint(m['configuration'])
        with self.assertRaises(ValueError):_configuration(m)
    def test_unknown_metadata_fields_rejected(self):
        m=metadata();v=source_config();v['grant']='TEST_ONLY_FAKE'
        m['configuration']['sequential_source_view']=v;m['configuration_fingerprint']=_fingerprint(m['configuration'])
        with self.assertRaises(ValueError):_configuration(m)


if __name__=='__main__':unittest.main()
