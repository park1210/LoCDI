"""Saved 50-case candidate dataset only. No model calls or performance scoring."""
from pathlib import Path
import csv,json,unittest
from app.locdi.original_source_inventory import original_cases,read
from app.locdi.original50_augmentation import validate_augmentation,directly_discloses_label,SCOPE
from app.locdi.literature_augmentation_catalog import evidence_catalog
from app.core.longitudinal_memory import timestamp
ROOT=Path(__file__).resolve().parents[2]
PR=ROOT/'outputs/private/research_evaluation/ORIGINAL50_LITERATURE_AUGMENTATION_v1'
PUB=ROOT/'outputs/research_evaluation/ORIGINAL50_LITERATURE_AUGMENTATION_v1'

class ActualAugmentedDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases=original_cases();cls.catalog=evidence_catalog()
        cls.rows=[json.loads(s) for s in (PR/'AUGMENTED_CASES.jsonl').read_text(encoding='utf-8').splitlines()]
    def test_all_fifty_one_to_one_original_links_and_row_hashes(self):
        self.assertEqual([r['case_ordinal'] for r in self.rows],list(range(1,51)))
        for c,r in zip(self.cases,self.rows):self.assertEqual(validate_augmentation(c,r,self.catalog)['structural_status'],'PASS')
    def test_original_measurements_and_fixed_memory_are_not_changed(self):
        self.assertEqual(sum(r['original_history']['fixed_memory_records'] for r in self.rows),72590)
        self.assertEqual(sum(len(r['original_findings']) for r in self.rows),2509)
        for c,r in zip(self.cases,self.rows):self.assertEqual(r['original_findings'],[e for e in c['environment']['source_event_store'] if e['event_type']=='observations'])
    def test_literature_outputs_are_label_conditioned_not_clinically_reviewed(self):
        self.assertTrue(all(r['evaluation_scope']==SCOPE and not r['clinical_reviewed'] and not r['research_evaluation_ready'] and r['clinical_diagnosability']=='UNREVIEWED' for r in self.rows))
    def test_no_fabricated_test_values_or_unknown_literature_features(self):
        self.assertEqual(sum(len(r['literature_informed_synthetic']['synthetic_test_results']) for r in self.rows),0)
        for r in self.rows:
            for s in r['literature_informed_synthetic']['symptoms']:self.assertEqual(s['text'],self.catalog['features'][s['feature_id']]['text'])
    def test_current_and_past_recovery_have_cutoff_and_no_release_grants(self):
        self.assertEqual(sum(len(r['source_recovered_findings']) for r in self.rows),325)
        self.assertEqual(sum(f['scope']=='CURRENT' for r in self.rows for f in r['source_recovered_findings']),1)
        for c,r in zip(self.cases,self.rows):
            cutoff=timestamp(c['agent_visible']['index_encounter_start']).timestamp()+45*60
            for f in r['source_recovered_findings']:
                self.assertLessEqual(timestamp(f['recorded_at']).timestamp(),cutoff);self.assertFalse(f['execution_release_approved'])
    def test_validated_preview_keeps_original_information_and_private_reference_out(self):
        for c,r in zip(self.cases,self.rows):
            p=read(PR/'validated_previews'/('Patient%02d.json'%r['case_ordinal']))
            self.assertEqual(p['D0'],c['agent_visible']['current_initial_state'])
            self.assertFalse(p['runtime_admitted']);self.assertFalse(directly_discloses_label(json.dumps(p['D2_PREVIEW_ONLY'])))
            self.assertNotIn('source_reference',p['D2_PREVIEW_ONLY']);self.assertNotIn('candidate_tests',json.dumps(p['D2_PREVIEW_ONLY']))
            self.assertTrue(set(p['D0']['patient_visible_info']['associated_symptoms'])<=set(p['D2_PREVIEW_ONLY']['patient_visible_info']['associated_symptoms']))
    def test_real_native_m5_mapping_has_no_invented_actions(self):
        report=read(PUB/'actual_dataset_validation.json')
        self.assertEqual(report['status'],'PASS');self.assertEqual(report['counts'].get('candidate_mapping_native_mismatch',0),0)
        self.assertEqual(report['counts']['native_bound_recommendations'],16)
    def test_main_document_fifty_and_future_rejections_recorded(self):
        text=(PR/'ORIGINAL50_AUGMENTATION_SUMMARY.md').read_text(encoding='utf-8')
        self.assertEqual(sum(line.startswith('## Patient ') for line in text.splitlines()),50)
        counts=read(PUB/'augmentation_summary.json')['rejected_reasons'];self.assertEqual(counts['FUTURE_SYMPTOM_EVENT'],4853)
        self.assertEqual(len({r['literature_informed_synthetic']['chief_complaint']['text'] for r in self.rows}),50)
    def test_validated_provenance_pointers_and_counts(self):
        from scripts.validate_augmentation_provenance import pointer
        with (PR/'AUGMENTATION_PROVENANCE_VALIDATED.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),532)
        for row in rows:
            if row['item_type']=='LITERATURE_INFORMED_SYNTHETIC':
                from app.locdi.llm_setup import digest
                self.assertEqual(digest(pointer(self.rows[int(row['case_ordinal'])-1],row['source_pointer'])),row['source_sha256'])
        self.assertEqual(read(PUB/'provenance_validation.json')['source_event_pointers_resolved'],325)
if __name__=='__main__':unittest.main()
