"""Read-only saved audit checks, not a clinical evaluation or model run."""
from pathlib import Path
import csv,json,re,unittest
from app.locdi.original_source_inventory import original_cases,eligible
ROOT=Path(__file__).resolve().parents[2]
PR=ROOT/'outputs/private/research_evaluation/ORIGINAL50_PATIENT_DIAGNOSTIC_AUDIT_v1'
PUB=ROOT/'outputs/research_evaluation/ORIGINAL50_PATIENT_DIAGNOSTIC_AUDIT_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
class Original50Audit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cases=original_cases();cls.summary=read(PUB/'summary.json')
    def test_exact_fifty_and_no_authored_synthetic_cases(self):
        self.assertEqual(self.summary['patients_processed'],50);self.assertEqual(self.summary['synthetic_pilot_patients_included'],0)
    def test_all_fixed_memory_preserved_and_source_verified(self):
        self.assertEqual(sum(len(v['records']) for c in self.cases for v in c['agent_visible']['past_visits']),72590)
        self.assertEqual(self.summary['source_verified_memory_records'],72590)
    def test_all_memory_patient_visit_and_cutoff_boundaries(self):
        self.assertTrue(all(eligible(r,c['agent_visible']['longitudinal']) for c in self.cases for v in c['agent_visible']['past_visits'] for r in v['records']))
    def test_main_readable_document_has_fifty_sections_and_all_memory(self):
        text=(PR/'ORIGINAL50_PATIENT_DIAGNOSTIC_SUMMARY.md').read_text(encoding='utf-8')
        self.assertEqual(re.findall(r'^## Patient (\d+)$',text,re.M),['%02d'%i for i in range(1,51)])
        self.assertEqual(text.count('; Memory='),72590)
    def test_required_private_files_and_patient_csv_fifty(self):
        for n in ('ORIGINAL50_PATIENT_MATRIX.csv','ORIGINAL50_DISEASE_DISTRIBUTION.csv','ORIGINAL50_TEST_RESULTS.csv',
                  'ORIGINAL50_SYMPTOM_AUDIT.csv','B2_MEMORY_RETRIEVAL_AUDIT.md','B2_REGRESSION_CASE_COMPARISON.csv','DATA_QUALITY_FINDINGS.md','README.md'):
            self.assertTrue((PR/n).is_file(),n)
        with (PR/'ORIGINAL50_PATIENT_MATRIX.csv').open(encoding='utf-8-sig',newline='') as f:self.assertEqual(len(list(csv.DictReader(f))),50)
    def test_public_contains_no_actual_patient_or_encounter_identifiers(self):
        text='\n'.join(p.read_text(encoding='utf-8') for p in PUB.glob('*.json'))
        for c in self.cases:
            self.assertNotIn(c['patient_id'],text);self.assertNotIn(c['agent_visible']['index_encounter_id'],text)
    def test_zero_generation_and_no_clinical_approval(self):
        self.assertEqual(self.summary['actual_LLM_generation'],0);self.assertEqual(self.summary['fitting'],0)
        self.assertFalse(self.summary['memory_algorithm_changed']);self.assertEqual(self.summary['clinical_accuracy'],'NOT_EVALUATED')
    def test_false_and_none_not_conflated_in_trace_csv(self):
        with (PR/'B2_REGRESSION_CASE_COMPARISON.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),10);self.assertEqual(sum(r['CORRECT_TO_INCORRECT']=='True' for r in rows),2)
        self.assertTrue(all(r['same_contract']=='False' for r in rows))
if __name__=='__main__':unittest.main()
