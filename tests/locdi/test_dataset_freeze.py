"""TEST_ONLY local files and metadata; never patient generation or grading."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from app.locdi import dataset_freeze as f


class DatasetFreezeContracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=str(f.ROOT/'outputs'));self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        original=f.load(f.ROOT/f.CANDIDATE);self.candidate=deepcopy(original)
        for b in self.candidate['bundles']:
            name='TEST_ONLY/'+b['bundle']+'.txt';p=self.root/name;p.parent.mkdir(exist_ok=True)
            p.write_text('TEST_ONLY '+b['bundle'],encoding='utf-8')
            entries=[{'path':name,'sha256':f.sha(p),'bytes':p.stat().st_size,'private':False}]
            b.update(files=entries,file_count=1,files_fingerprint=f.digest(entries))
        self.candidate.pop('manifest_fingerprint');self.candidate['manifest_fingerprint']=f.digest(self.candidate)
        self.expected=self.candidate['manifest_fingerprint']
        for name in f.INPUTS+(f.BASELINE,f.REGRESSION)+f.TOOLS:
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text('{}',encoding='utf-8')
        self.save_candidate()

    def save_candidate(self):
        p=self.root/f.CANDIDATE;p.write_text(json.dumps(self.candidate),encoding='utf-8')
        (self.root/f.SUMMARY).write_text(json.dumps({'manifest_fingerprint':self.candidate['manifest_fingerprint'],
            'private_manifest_sha256':f.sha(p)}),encoding='utf-8')

    def docs(self):
        return f.documents(self.candidate,{'D3':{'strict_source':'STRICT_SOURCE_ACTUAL_AVAILABILITY_UNVERIFIED'}},
            {'TEST_ONLY':'UNCHANGED'},{'TEST_ONLY':[]},{'TEST_ONLY':True},{},root=self.root,created_at='2026-10-30T00:00:00+00:00')

    def test_candidate_fingerprint_mismatch(self):
        with self.assertRaisesRegex(f.FreezeError,'CANDIDATE_FINGERPRINT_MISMATCH'):f.candidate(self.root,expected_fp='0'*64)

    def test_valid_candidate_and_public_link(self):
        _,checked=f.candidate(self.root,expected_fp=self.expected);self.assertEqual(checked['files'],6)
        (self.root/f.SUMMARY).write_text('{}',encoding='utf-8')
        with self.assertRaises((f.FreezeError,KeyError)):f.candidate(self.root,expected_fp=self.expected)

    def test_file_hash_mismatch(self):
        (self.root/'TEST_ONLY/A.txt').write_text('TEST_ONLY CHANGED',encoding='utf-8')
        with self.assertRaisesRegex(f.FreezeError,'FILE_HASH_MISMATCH'):f.candidate(self.root,expected_fp=self.expected)

    def test_missing_source(self):
        (self.root/'TEST_ONLY/A.txt').unlink()
        with self.assertRaisesRegex(f.FreezeError,'MISSING_SOURCE'):f.candidate(self.root,expected_fp=self.expected)

    def test_duplicate_file_entry(self):
        bundles=deepcopy(self.candidate['bundles']);b=bundles[0]
        b['files']*=2;b['file_count']=2;b['files_fingerprint']=f.digest(b['files'])
        with self.assertRaisesRegex(f.FreezeError,'DUPLICATE_FILE_ENTRY'):f.entries_check(self.root,bundles)

    def test_path_traversal_and_windows_absolute_path(self):
        for name in ('../escape','C:/private','/absolute','a/../private','a\\b','a//b','./file'):
            with self.subTest(name=name),self.assertRaisesRegex(f.FreezeError,'PATH_TRAVERSAL'):f.safe_path(self.root,name)

    def test_same_content_different_paths_are_reported_not_dropped(self):
        bundles=deepcopy(self.candidate['bundles']);entry=deepcopy(bundles[0]['files'][0]);entry['path']='TEST_ONLY/copy.txt'
        (self.root/entry['path']).write_bytes((self.root/'TEST_ONLY/A.txt').read_bytes())
        b=bundles[0];b['files'].append(entry);b['file_count']=2;b['files_fingerprint']=f.digest(b['files'])
        checked=f.entries_check(self.root,bundles)
        self.assertEqual(checked['files'],7);self.assertEqual(len(checked['duplicate_content_groups']),1)

    def test_public_private_leakage_and_no_private_paths(self):
        public,_=self.docs();f.public_boundary(public)
        for value in ({'patient_id':'TEST_ONLY'}, {'x':'outputs/private/TEST_ONLY'}, {'rubrics':[]}):
            with self.assertRaisesRegex(f.FreezeError,'PUBLIC_PRIVATE_LEAKAGE'):f.public_boundary(value)

    def test_existing_validation_projection_keeps_rubric_counts_public(self):
        # TEST_ONLY projection test over the existing saved D6 audit; not a
        # replacement for the live source validation in the freeze/verifier.
        data=f.load(f.ROOT/'outputs/reports/D6_dataset_validation_v1.json')
        splits=f.load(f.ROOT/'outputs/reports/D6_split_and_exposure_v1.json')
        with patch('app.locdi.dataset_release_readiness.validate_all',return_value=data),patch(
                'app.locdi.dataset_release_readiness.split_audit',return_value=splits):
            summary,_=f.validate_data()
        f.public_boundary(summary)
        self.assertEqual(summary['healthbench']['rubric_count'],57237)
        self.assertNotIn('rubrics',summary['healthbench'])

    def test_manifest_hashes_no_circular_dependency(self):
        public,private=self.docs();f.verify_pair(public,private,self.root)
        self.assertNotIn('public_manifest_fingerprint',private)
        private['public_manifest_fingerprint']=public['manifest_fingerprint']
        with self.assertRaisesRegex(f.FreezeError,'CIRCULAR_HASH_DEPENDENCY'):f.verify_pair(public,private,self.root)

    def test_unknown_license_is_not_promoted(self):
        public,_=self.docs();self.assertEqual(public['license_access_status']['A'],'UNKNOWN')
        public['license_access_status']['A']='APPROVED_REDISTRIBUTION'
        with self.assertRaisesRegex(f.FreezeError,'UNKNOWN_LICENSE_PROMOTED'):f.policy(public)

    def test_unreviewed_reference_and_clinical_readiness_not_promoted(self):
        public,_=self.docs();p=deepcopy(public);p['clinical_review_status']='REVIEWED'
        with self.assertRaisesRegex(f.FreezeError,'UNREVIEWED_REFERENCE_PROMOTED'):f.policy(p)
        p=deepcopy(public);p['evaluation_readiness']['CLINICAL_RESEARCH_EVALUATION_READY']=True
        with self.assertRaisesRegex(f.FreezeError,'READINESS_PROMOTION_FORBIDDEN'):f.policy(p)

    def test_split_memberships_cannot_change(self):
        public,private=self.docs()
        with self.assertRaisesRegex(f.FreezeError,'SPLIT_OR_EXPOSURE_CHANGED'):
            f.snapshot_check(public,private,public['dataset_validation_status'],{'TEST_ONLY':'CHANGED'},private['private_memberships'])

    def test_development_exposure_and_strict_source_stay_blocked(self):
        public,_=self.docs();p=deepcopy(public);p['development_exposure_summary']['healthbench_excluded_pilot_count']=0
        with self.assertRaisesRegex(f.FreezeError,'DEVELOPMENT_EXPOSURE_CHANGED'):f.policy(p)
        p=deepcopy(public);p['dataset_validation_status']['D3']['strict_source']='READY'
        with self.assertRaisesRegex(f.FreezeError,'VISIBILITY_POLICY_CHANGED'):f.policy(p)

    def test_post_freeze_source_change_detected(self):
        public,private=self.docs();f.verify_pair(public,private,self.root)
        (self.root/'TEST_ONLY/B.txt').write_text('TEST_ONLY post-freeze change',encoding='utf-8')
        with self.assertRaisesRegex(f.FreezeError,'FILE_HASH_MISMATCH'):f.verify_pair(public,private,self.root)

    def test_no_clobber_preserves_complete_manifests(self):
        public,private=self.docs();f.write_release(public,private,self.root)
        saved=(self.root/f.PUBLIC).read_bytes()
        with self.assertRaisesRegex(f.FreezeError,'RELEASE_ID_ALREADY_EXISTS'):f.write_release(public,private,self.root)
        self.assertEqual((self.root/f.PUBLIC).read_bytes(),saved)

    def test_storage_failure_leaves_incomplete_release_not_success(self):
        public,private=self.docs();write=f.atomic_rows
        def fail_second(root,name,value,**kwargs):
            if str(root).endswith('locdi_dataset_v1.0') and 'outputs' not in str(root.relative_to(self.root)):
                raise OSError('TEST_ONLY storage failure')
            return write(root,name,value,**kwargs)
        with patch.object(f,'atomic_rows',side_effect=fail_second):
            with self.assertRaisesRegex(f.FreezeError,'FREEZE_STORAGE_FAILED_INCOMPLETE_RELEASE'):f.write_release(public,private,self.root)
        self.assertFalse((self.root/f.PUBLIC).exists());self.assertTrue((self.root/f.PRIVATE).exists())
        with self.assertRaisesRegex(f.FreezeError,'MISSING_SOURCE'):f.load(self.root/f.PUBLIC)

    def test_deterministic_manifest_and_link_fingerprint(self):
        a,b=self.docs();c,d=self.docs();self.assertEqual(a,c);self.assertEqual(b,d)
        self.assertEqual(f.digest(a),f.digest(c));a['private_manifest_fingerprint']='0'*64
        with self.assertRaisesRegex(f.FreezeError,'PUBLIC_MANIFEST_FINGERPRINT_MISMATCH'):f.verify_pair(a,b,self.root)

    def test_original_protection_and_strict_json(self):
        p='TEST_ONLY/A.txt';self.assertEqual(f.protected_check(self.root,{p:f.sha(self.root/p)}),1)
        with self.assertRaisesRegex(f.FreezeError,'ORIGINAL_FILE_CHANGED'):f.protected_check(self.root,{p:'0'*64})
        path=self.root/'bad.json';path.write_text('{"x":1,"x":2}',encoding='utf-8')
        with self.assertRaisesRegex(f.FreezeError,'DUPLICATE_JSON_KEY'):f.load(path)

    def test_real_existing_candidate_185_files_stays_valid(self):
        c,checked=f.candidate();self.assertEqual(c['manifest_fingerprint'],f.EXPECTED);self.assertEqual(checked['files'],185)

    def test_default_plan_does_not_write_and_no_clinical_calls(self):
        from scripts.freeze_d7_dataset import main
        from contextlib import redirect_stdout
        import io
        with patch('scripts.freeze_d7_dataset.freeze',return_value={'status':'DRY_RUN'}) as freeze,redirect_stdout(io.StringIO()):
            self.assertEqual(main([]),0)
        freeze.assert_called_once_with(write=False)


if __name__=='__main__':unittest.main()
