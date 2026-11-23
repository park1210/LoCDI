"""TEST_ONLY official interface protections and explicit free-text universe."""
from copy import deepcopy
import unittest
from app.locdi.current_ablation_runner import make_contract, validate_contract
from experiments.run_locdi_ablation import run_ablation, AblationConfigurationError
from test_current_ablation_runner import fixture


class Interface(unittest.TestCase):
    def test_explicit_existing_free_text_not_source_labels(self):
        c,p,a=fixture()
        common=deepcopy(c['common'])
        common['diagnosis_catalog']={'mode':'existing_free_text','validator_sha256':'a'*64,'source_label_injected':False}
        validate_contract(make_contract(common,p,purpose='TEST_ONLY'),p)

    def test_source_label_injection_flag_blocked(self):
        c,p,a=fixture();common=deepcopy(c['common'])
        common['diagnosis_catalog']={'mode':'existing_free_text','validator_sha256':'a'*64,'source_label_injected':True}
        with self.assertRaises(ValueError):validate_contract(make_contract(common,p,purpose='TEST_ONLY'),p)

    def test_unchanged_official_not_ready_config_blocked(self):
        import json
        from pathlib import Path
        for path in sorted(Path('experiments/configs').glob('B[0-4].json')):
            config=json.loads(path.read_text(encoding='utf-8'))
            with self.assertRaises(AblationConfigurationError):run_ablation(config)

    def test_ready_alone_never_bypasses_execution_contract(self):
        from app.locdi.current_ablation_runner import features_for
        common={k:'TEST_ONLY' for k in ['model','model_version','decoding','minute_limit','agent_registry',
            'diagnosis_candidates','evaluation','action_set','patient_split','repeats','fixed_memory']}
        common['memory_source']='fixed'
        config={'condition':'B0','features':features_for('B0'),'ready':True,'common':common}
        with self.assertRaisesRegex(AblationConfigurationError,'Pinned'):run_ablation(config)


if __name__=='__main__':unittest.main()
