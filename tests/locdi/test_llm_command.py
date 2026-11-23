"""Approval boundary tests: never run the CLI's real execution branch."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import unittest
from unittest.mock import patch
from scripts.run_development_llm_case import main


class LLMCommandContracts(unittest.TestCase):
    def test_default_command_is_dry_run_and_never_constructs_transport(self):
        output = io.StringIO()
        with patch('app.locdi.ollama_adapter._backend', side_effect=AssertionError('No transport')) as backend, redirect_stdout(output):
            self.assertEqual(main([]), 0)
        backend.assert_not_called()
        result = json.loads(output.getvalue())
        self.assertEqual(result['status'], 'dry_run')
        self.assertEqual(result['request_count'], 0)
        self.assertFalse(result['real_model_execution'])

    def test_execute_without_both_acknowledgements_is_rejected_before_loading(self):
        for args in (['--execute'], ['--execute', '--allow-llm'], ['--execute', '--accept-release-assumption']):
            with self.subTest(args=args), patch('scripts.run_development_llm_case.load_setup') as load, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(args)
                self.assertEqual(error.exception.code, 2)
                load.assert_not_called()
