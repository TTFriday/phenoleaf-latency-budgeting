"""Synthetic tests for shared-desktop bookkeeping; no real GPU call or process stop."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from jrtip_pilot import gpu_start_policy


class GPUBookkeepingTests(unittest.TestCase):
    def test_shared_contexts_require_explicit_mode(self):
        with self.assertRaises(RuntimeError):
            gpu_start_policy({'compute_query_ok': True, 'contention': True}, False)

    def test_explicit_shared_mode_does_not_claim_exclusivity(self):
        policy = gpu_start_policy({'compute_query_ok': True, 'contention': True}, True)
        self.assertTrue(policy['other_gpu_contexts_observed'])
        self.assertFalse(policy['exclusive_platform_guaranteed'])

    def test_unavailable_query_still_refused(self):
        with self.assertRaises(RuntimeError):
            gpu_start_policy({'compute_query_ok': False, 'contention': False}, True)

    def test_empty_query_is_not_proof_of_exclusive_hardware(self):
        self.assertFalse(gpu_start_policy({'compute_query_ok': True, 'contention': False})['exclusive_platform_guaranteed'])


if __name__ == '__main__':
    unittest.main()
