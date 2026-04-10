# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import tempfile
import unittest

import torch
from torch.testing._internal.common_utils import TestCase


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestKernelAnnotationsE2E(TestCase):
    """E2E test: trace → annotate → insert annotations → cudagraph capture → profile → check trace."""

    def test_profiler_trace_has_component_annotations(self):
        """After the full pipeline (make_fx → insert_kernel_annotations →
        cudagraph → profile), the profiler trace should contain ``component``
        fields on graphed kernel events."""
        from torch.cuda._annotate_cuda_graph_trace import annotate_trace
        from torch.cuda._graph_annotations import _is_tools_id_unavailable
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch.fx.traceback import annotate_fn, preserve_node_meta

        from torchtitan.experiments.graph_trainer.cudagraph import (
            CUDAGraphWrapper,
            cudagraph_teardown,
            get_cudagraph_annotations,
        )
        from torchtitan.experiments.graph_trainer.passes import (
            insert_kernel_annotations_pass,
        )

        if _is_tools_id_unavailable():
            self.skipTest("cudaGraphNodeGetToolsId not available")

        # Simple model with annotated submodules.
        class FFN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 16)

            def forward(self, x):
                return torch.relu(self.linear(x))

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.LayerNorm(16)
                self.ffn = FFN()

            def forward(self, x):
                return self.ffn(self.norm(x))

        model = Model().cuda()

        # Annotate module components (like annotate_module_components does).
        model.norm.forward = annotate_fn({"component": "norm"})(model.norm.forward)
        model.ffn.forward = annotate_fn({"component": "ffn"})(model.ffn.forward)
        model.ffn.linear.forward = annotate_fn({"component": "ffn.linear"})(
            model.ffn.linear.forward
        )

        x = torch.randn(4, 16, device="cuda")

        # 1. Trace with make_fx (preserves custom metadata).
        with preserve_node_meta():
            gm = make_fx(model)(x)

        # Verify custom metadata survived tracing.
        components_in_graph = set()
        for node in gm.graph.nodes:
            comp = (node.meta.get("custom") or {}).get("component")
            if comp:
                components_in_graph.add(comp)
        self.assertIn("norm", components_in_graph)
        self.assertIn("ffn", components_in_graph)

        # 2. Apply passes.
        insert_kernel_annotations_pass(gm)

        # 3. Wrap with CUDAGraph.
        wrapper = CUDAGraphWrapper(gm.forward, [x], static_input_indices=())

        # 4. Warmup + capture + replay.
        wrapper(x)  # warmup
        wrapper(x)  # capture
        wrapper(x)  # replay

        # 5. Check annotations were captured.
        annotations = get_cudagraph_annotations()
        self.assertGreater(len(annotations), 0, "No annotations captured")

        # Verify at least some annotations have expected component values.
        all_components = set()
        for ann_list in annotations.values():
            for ann in ann_list:
                if isinstance(ann, dict) and "component" in ann:
                    all_components.add(ann["component"])
        self.assertIn("norm", all_components)
        self.assertIn("ffn", all_components)

        # 6. Profile and check the trace.
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        ) as prof:
            wrapper(x)
            torch.cuda.synchronize()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            trace_path = f.name
        prof.export_chrome_trace(trace_path)

        with open(trace_path) as f:
            trace = json.load(f)

        count = annotate_trace(trace, annotations)
        self.assertGreater(count, 0, "annotate_trace matched 0 events")

        # Verify component fields appear on graphed kernel events.
        components_in_trace = set()
        for e in trace["traceEvents"]:
            args = e.get("args", {})
            if args.get("graph node id", 0) != 0 and "component" in args:
                components_in_trace.add(args["component"])

        self.assertIn("norm", components_in_trace)
        self.assertIn("ffn", components_in_trace)

        # Cleanup.
        os.unlink(trace_path)
        cudagraph_teardown()


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
