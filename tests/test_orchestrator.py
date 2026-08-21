import unittest

import run


class OrchestratorCliTests(unittest.TestCase):
    def test_build_parser_has_pipeline_mode(self):
        parser = run.build_parser()
        self.assertIsNotNone(parser)
        self.assertIn("pipeline", parser._subparsers._group_actions[0].choices)

    def test_main_accepts_dashboard_mode(self):
        result = run.main(["dashboard", "--help"])
        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
