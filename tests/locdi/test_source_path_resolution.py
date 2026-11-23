"""TEST_ONLY source relocation, manifest and row-integrity regressions."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from app.locdi import source_paths
from app.locdi.m5_common_source import verify_csv


class SourcePathResolution(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'locdi';self.synthea=Path(self.tmp.name)/'001_synthea-docker'
        self.root.mkdir();self.synthea.mkdir();self.name=source_paths.COHORT+'observations.csv'
        self.csv=self.synthea/self.name;self.csv.parent.mkdir(parents=True)
        self.raw={'PATIENT':'TEST_ONLY_PATIENT','ENCOUNTER':'TEST_ONLY_VISIT','VALUE':'1','UNITS':'TEST_ONLY'}
        with self.csv.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(self.raw));w.writeheader();w.writerow(self.raw)
        self.manifest=self.synthea/'raw_manifest.json'
        self.manifest.write_text(json.dumps({'output/cohort_1000_json_v1/csv/observations.csv':source_paths.sha_file(self.csv)}))
        for name,value in (('ROOT',self.root),('SYNTHEA',self.synthea),('MANIFEST',self.manifest)):
            p=patch.object(source_paths,name,value);p.start();self.addCleanup(p.stop)
        source_paths._checked_hash.cache_clear()

    def test_exact_sibling_and_unchanged_reference(self):
        ref={'source_metadata':{'source_file':self.name,'source_row':2,'source_record':self.raw}}
        before=deepcopy(ref);verify_csv([ref])
        self.assertEqual(ref,before);self.assertEqual(source_paths.resolve_source_file(self.name),self.csv.resolve())

    def test_wrong_bytes_rejected_even_with_same_filename(self):
        self.csv.write_text('WRONG_COHORT',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'HASH_MISMATCH'):source_paths.resolve_source_file(self.name)

    def test_no_other_cohort_fallback(self):
        self.csv.rename(self.csv.with_name('different_observations.csv'))
        with self.assertRaises(FileNotFoundError):source_paths.resolve_source_file(self.name)

    def test_row_mismatch_still_rejected(self):
        bad={**self.raw,'VALUE':'2'}
        with self.assertRaisesRegex(ValueError,'RAW_CSV_ROW_MISMATCH'):
            verify_csv([{'source_metadata':{'source_file':self.name,'source_row':2,'source_record':bad}}])

    def test_path_escape_rejected(self):
        with self.assertRaisesRegex(ValueError,'UNSAFE'):source_paths.resolve_source_file('../observations.csv')


if __name__=='__main__':unittest.main()
