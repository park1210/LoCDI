"""TEST_ONLY nonclinical source alignment and complete-reference metric contracts."""
from copy import deepcopy
import unittest
from scripts.evaluate_measurement_identity import align, native_index, identity_label, metrics, URI, export_value


def fixture(code='8480-6', system='LOINC', display='TEST_ONLY', unit='mm[Hg]'):
    return {'record':{'encounters':[{'uuid':'TEST_VISIT','start':0,'observations':[
        {'codes':[{'system':system,'code':code,'display':display}],'start':1000,'value':120.04,
         'unit':unit,'uuid':'TEST_RESOURCE','type':code}]}]}}


class MeasurementIdentityTests(unittest.TestCase):
    def aligned(self, **kwargs):
        return align(native_index(fixture(**kwargs)),'TEST_VISIT','8480-6','1970-01-01T00:00:01Z','120.0','mm[Hg]')

    def test_explicit_native_coding(self):
        self.assertEqual(self.aligned()['system_uri'],URI)

    def test_pattern_does_not_infer_namespace(self):
        self.assertEqual(self.aligned(system='')['status'],'UNKNOWN')

    def test_other_namespace_not_promoted(self):
        self.assertEqual(self.aligned(system='SNOMED-CT')['status'],'UNKNOWN')

    def test_display_not_label_operand(self):
        a=self.aligned(display='DIFFERENT_TEST_ONLY_NAME')
        self.assertEqual(identity_label('observations',a,{(URI,'8480-6'):'mm[Hg]'}),
                         ('POSITIVE','EXPLICIT_SOURCE_CODE_AND_RECORDED_UNIT_IDENTITY'))

    def test_unit_no_automatic_conversion(self):
        a=self.aligned(); a['unit']='kPa'
        self.assertEqual(identity_label('observations',a,{(URI,'8480-6'):'mm[Hg]'})[0],'UNKNOWN')

    def test_different_code_negative_only_exact_task(self):
        self.assertEqual(identity_label('observations',self.aligned(),{(URI,'8462-4'):'mm[Hg]'})[0],'NEGATIVE')

    def test_structural_failure_blocks_even_nonmeasurement(self):
        self.assertEqual(identity_label('conditions',{}, {},False)[0],'UNKNOWN')

    def test_query_incomplete_blocks_negative(self):
        self.assertEqual(identity_label('conditions',{}, {},query_complete=False)[0],'UNKNOWN')

    def test_nonmeasurement_is_not_clinical_irrelevance(self):
        label,reason=identity_label('conditions',{}, {})
        self.assertEqual(label,'NEGATIVE');self.assertIn('MEASUREMENT_TASK',reason)

    def test_wrong_visit(self):
        a=align(native_index(fixture()),'OTHER','8480-6','1970-01-01T00:00:01Z','120.0','mm[Hg]')
        self.assertEqual(a['status'],'UNKNOWN')

    def test_wrong_time(self):
        a=align(native_index(fixture()),'TEST_VISIT','8480-6','1970-01-01T00:00:02Z','120.0','mm[Hg]')
        self.assertEqual(a['status'],'UNKNOWN')

    def test_wrong_value(self):
        a=align(native_index(fixture()),'TEST_VISIT','8480-6','1970-01-01T00:00:01Z','120.1','mm[Hg]')
        self.assertEqual(a['status'],'UNKNOWN')

    def test_ambiguous_native_resources_block(self):
        d=fixture();d['record']['encounters'][0]['observations'].append(deepcopy(d['record']['encounters'][0]['observations'][0]))
        a=align(native_index(d),'TEST_VISIT','8480-6','1970-01-01T00:00:01Z','120.0','mm[Hg]')
        self.assertEqual(a['reason'],'AMBIGUOUS_NATIVE_JOIN')

    def test_components_recursed_without_parent_invention(self):
        d=fixture(); o=d['record']['encounters'][0]['observations'][0]
        d['record']['encounters'][0]['observations']=[{'observations':[o]}]
        self.assertEqual(len(native_index(d)),1)

    def test_numeric_export_no_clinical_tolerance(self):
        self.assertEqual(export_value(120.04),'120.0');self.assertEqual(export_value(120.05),'120.1')

    def test_documented_csv_clean_not_synonym_normalization(self):
        self.assertEqual(export_value(' A,B\r\nC\rD\nE '),'A B C D E')
        self.assertEqual(export_value('Alpha  Beta'),'Alpha  Beta')

    def test_coded_value_export_preserves_display_without_guess(self):
        self.assertEqual(export_value({'code':'TEST_ONLY','display':'A,B'}),'A B')
        self.assertIsNone(export_value({'code':'TEST_ONLY'}))

    def test_unknown_blocks_full_recall(self):
        self.assertIsNone(metrics({'a':'POSITIVE','b':'UNKNOWN'},['a'])['recall'])

    def test_complete_full_pool_recall(self):
        r=metrics({'a':'POSITIVE','b':'POSITIVE','c':'NEGATIVE'},['a','c'])
        self.assertEqual(r['recall'],.5);self.assertEqual(r['precision'],.5)

    def test_no_positive_is_not_zero_recall(self):
        self.assertIsNone(metrics({'a':'NEGATIVE'},['a'])['recall'])

    def test_off_has_no_precision_denominator(self):
        r=metrics({'a':'POSITIVE'},[])
        self.assertIsNone(r['precision']);self.assertEqual(r['recall'],0)

    def test_duplicate_results_rejected(self):
        with self.assertRaisesRegex(ValueError,'BOUNDARY'):metrics({'a':'POSITIVE'},['a','a'])

    def test_out_of_pool_rejected(self):
        with self.assertRaisesRegex(ValueError,'BOUNDARY'):metrics({'a':'POSITIVE'},['b'])

    def test_labels_not_mutated(self):
        d=fixture();saved=deepcopy(d)
        self.assertEqual(native_index(d),native_index(d));self.assertEqual(d,saved)

    def test_false_unknown_distinct(self):
        with self.assertRaises(ValueError):metrics({'a':False},[])


if __name__=='__main__':unittest.main()
