"""Unit tests for reasoning token counting order vs completion truncation."""

import unittest
from array import array
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

THINK_END_ID = 99
STOP_ID = 200


def _make_reasoning_req(*, output_ids=None, finished_len=None) -> Req:
    req = Req.__new__(Req)
    req._is_reasoning_over = False
    req.reasoning_tokens = 0
    req.output_ids = array("q", output_ids or [])
    req.finished_len = finished_len
    req.require_reasoning = True
    req.sampling_params = SimpleNamespace(
        max_new_tokens=128,
        ignore_eos=False,
        stop_token_ids=[],
        stop_strs=[],
        stop_regex_strs=[],
    )
    req.tokenizer = None
    req.eos_token_ids = set()
    req.vocab_size = 200000
    req.grammar = None
    req.to_finish = None
    req.finished_reason = None
    return req


def _make_processor(think_end_id=THINK_END_ID) -> SchedulerBatchResultProcessor:
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=False,
        enable_overlap_mlx=False,
        server_args=SimpleNamespace(enable_metrics=False),
        model_config=SimpleNamespace(think_end_id=think_end_id),
        token_to_kv_pool_allocator=None,
        tree_cache=None,
        hisparse_coordinator=None,
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=SimpleNamespace(),
        draft_worker=None,
        model_worker=SimpleNamespace(on_verify_complete_cpu=lambda *a, **k: None),
        logprob_result_processor=None,
        output_streamer=SimpleNamespace(),
        abort_request=lambda *a, **k: None,
    )


class TestReasoningTokenCountOrder(CustomTestCase):
    def test_finished_len_excludes_overshoot_from_completion(self):
        req = _make_reasoning_req(output_ids=[1, 2, 3, 4, 5], finished_len=3)
        self.assertEqual(len(req.output_ids_through_stop), 3)
        self.assertLess(len(req.output_ids_through_stop), len(req.output_ids))

    def test_count_after_finish_only_includes_effective_tokens(self):
        """Simulate spec overshoot: batch adds 4 tokens but stop truncates to 3 total."""
        processor = _make_processor()
        req = _make_reasoning_req(output_ids=[10, 20])
        req.output_ids.extend([30, STOP_ID, 50, 60])
        req.finished_len = 3
        new_accept_len = 4

        processor._update_reasoning_tokens_after_finish(req, new_accept_len)

        # Old order would count all 4 new tokens (=4). Effective slice is [30] only.
        self.assertEqual(req.reasoning_tokens, 1)
        self.assertLessEqual(req.reasoning_tokens, len(req.output_ids_through_stop))

    def test_old_order_would_overcount(self):
        processor = _make_processor()
        req = _make_reasoning_req(output_ids=[10, 20])
        req.output_ids.extend([30, STOP_ID, 50, 60])
        req.finished_len = 3
        new_accept_len = 4

        processor._maybe_update_reasoning_tokens(req, [30, STOP_ID, 50, 60])

        self.assertEqual(req.reasoning_tokens, 4)
        self.assertGreater(req.reasoning_tokens, len(req.output_ids_through_stop))

    def test_think_end_still_counted_in_effective_slice(self):
        processor = _make_processor()
        req = _make_reasoning_req()
        req.output_ids.extend([10, 20, THINK_END_ID, 30])
        req.finished_len = 3
        new_accept_len = 4

        processor._update_reasoning_tokens_after_finish(req, new_accept_len)

        self.assertEqual(req.reasoning_tokens, 3)
        self.assertTrue(req._is_reasoning_over)
        self.assertEqual(req.reasoning_tokens, len(req.output_ids_through_stop))

    def test_no_reasoning_update_when_batch_fully_after_stop(self):
        processor = _make_processor()
        req = _make_reasoning_req(output_ids=[1, 2, 3], finished_len=3)
        req.output_ids.extend([4, 5])
        new_accept_len = 2

        processor._update_reasoning_tokens_after_finish(req, new_accept_len)

        self.assertEqual(req.reasoning_tokens, 0)


if __name__ == "__main__":
    unittest.main()
