# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

import torch
from torch._functorch.aot_autograd import aot_compile_joint_with_descriptors
from torch._guards import tracing
from torch._inductor.fx_passes.bucketing import (
    is_all_gather_into_tensor as is_all_gather,
)
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import TestCase
from torch.utils.checkpoint import checkpoint, CheckpointPolicy

from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.common_utils import _AC_REGION_ID
from torchtitan.experiments.graph_trainer.graph_utils import export_joint
from torchtitan.experiments.graph_trainer.passes import (
    apply_sac_pass,
    reassign_to_pg_pass,
)
from torchtitan.experiments.graph_trainer.simple_fsdp import data_parallel
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module, ModuleList


class ToyModel(Module):
    """A small toy model with multiple linear layers and activation
    checkpointing so that the backward graph recomputes the forward
    all-gathers."""

    def __init__(self, dim=16, n_layers=3):
        super().__init__()

        def _make_linear():
            cfg = Linear.Config(in_features=dim, out_features=dim, bias=True)
            return cfg.build()

        self.layers = ModuleList([_make_linear() for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = checkpoint(
                lambda m, inp: torch.relu(m(inp)),
                layer,
                x,
                use_reentrant=False,
            )
        return x


class TestReassignToPgPass(FSDPTest):
    """Integration tests: toy model + simple_fsdp + export_joint + reassign_to_pg_pass."""

    def _setup(self):
        """Set up ParallelDims and device mesh for FSDP."""
        self.parallel_dims = ParallelDims(
            dp_shard=-1,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=self.world_size,
        )

    def _make_fsdp_model(self, dim=16, n_layers=3):
        """Create a toy model and apply simple_fsdp data_parallel."""
        model = ToyModel(dim, n_layers).cuda()
        fsdp_mesh = self.parallel_dims.get_mesh("fsdp")
        model = data_parallel(model, device_mesh=fsdp_mesh, mode="fully_shard")
        return model

    def _get_fsdp_pg_name(self):
        """Get the FSDP process group name from the mesh."""
        fsdp_mesh = self.parallel_dims.get_mesh("fsdp")
        return fsdp_mesh.get_group().group_name

    def _export_and_get_bw_graph(self, model, inputs):
        """Export the joint graph and capture the backward graph via
        aot_compile_joint_with_descriptors with a custom bw_compiler."""
        joint_with_descriptors, tracing_context = export_joint(model, (inputs,))

        captured_bw_gm = {}

        def capture_bw_compiler(gm, example_inputs):
            captured_bw_gm["gm"] = gm
            captured_bw_gm["example_inputs"] = example_inputs
            return gm

        with tracing(tracing_context):
            aot_compile_joint_with_descriptors(
                joint_with_descriptors,
                bw_compiler=capture_bw_compiler,
            )

        return captured_bw_gm["gm"], captured_bw_gm["example_inputs"]

    def _count_ag_nodes_with_pg(self, gm, pg_name):
        """Count all-gather nodes in the graph that use the given PG name."""
        count = 0
        for node in gm.graph.nodes:
            if is_all_gather(node) and node.args[2] == pg_name:
                count += 1
        return count

    def _count_all_ag_nodes(self, gm):
        """Count all all-gather nodes in the graph regardless of PG."""
        count = 0
        for node in gm.graph.nodes:
            if is_all_gather(node):
                count += 1
        return count

    def test_reassign_rewrites_ag_nodes(self):
        """Apply reassign_to_pg_pass on the real backward graph and verify
        that all-gather nodes are rewritten to the target PG."""
        self._setup()
        model = self._make_fsdp_model()
        inputs = torch.randn(4, 16).cuda()
        fsdp_pg_name = self._get_fsdp_pg_name()
        target_pg_name = "test_target_pg"

        bw_gm, bw_example_inputs = self._export_and_get_bw_graph(model, inputs)

        # Before: all AG nodes should use the FSDP PG
        ag_before = self._count_ag_nodes_with_pg(bw_gm, fsdp_pg_name)
        self.assertGreater(ag_before, 0, "Expected AG nodes with FSDP PG name")

        # Apply the pass
        reassign_to_pg_pass(
            bw_gm,
            bw_example_inputs,
            source_pg_name=fsdp_pg_name,
            target_pg_name=target_pg_name,
        )

        # After: AG nodes should use the target PG
        ag_with_old = self._count_ag_nodes_with_pg(bw_gm, fsdp_pg_name)
        ag_with_new = self._count_ag_nodes_with_pg(bw_gm, target_pg_name)

        self.assertEqual(ag_with_old, 0, "No AG nodes should still use the old PG")
        self.assertEqual(
            ag_with_new, ag_before, "All AG nodes should now use the target PG"
        )

    def test_reassign_preserves_total_ag_count(self):
        """The pass should not add or remove AG nodes, only rewrite PG names."""
        self._setup()
        model = self._make_fsdp_model()
        inputs = torch.randn(4, 16).cuda()
        fsdp_pg_name = self._get_fsdp_pg_name()

        bw_gm, bw_example_inputs = self._export_and_get_bw_graph(model, inputs)

        total_before = self._count_all_ag_nodes(bw_gm)
        reassign_to_pg_pass(
            bw_gm,
            bw_example_inputs,
            source_pg_name=fsdp_pg_name,
            target_pg_name="new_pg",
        )
        total_after = self._count_all_ag_nodes(bw_gm)

        self.assertEqual(total_before, total_after)

    def test_reassign_with_non_matching_pg_is_noop(self):
        """If the source PG name doesn't match any AG node, nothing changes."""
        self._setup()
        model = self._make_fsdp_model()
        inputs = torch.randn(4, 16).cuda()
        fsdp_pg_name = self._get_fsdp_pg_name()

        bw_gm, bw_example_inputs = self._export_and_get_bw_graph(model, inputs)

        ag_before = self._count_ag_nodes_with_pg(bw_gm, fsdp_pg_name)

        # Use a non-matching source PG name
        reassign_to_pg_pass(
            bw_gm,
            bw_example_inputs,
            source_pg_name="nonexistent_pg",
            target_pg_name="target_pg",
        )

        # FSDP AG nodes should be unchanged
        ag_after = self._count_ag_nodes_with_pg(bw_gm, fsdp_pg_name)
        self.assertEqual(ag_before, ag_after)

    def test_reassign_with_extra_pg(self):
        """Test the production-like flow: create an extra FSDP PG and
        reassign AG nodes to it."""
        self._setup()
        model = self._make_fsdp_model()
        inputs = torch.randn(4, 16).cuda()
        fsdp_pg_name = self._get_fsdp_pg_name()

        # Create an extra PG mirroring the FSDP topology
        from torchtitan.experiments.graph_trainer.common_utils import (
            create_extra_fsdp_pg,
            get_extra_fsdp_pg_name,
        )

        create_extra_fsdp_pg(self.parallel_dims)
        extra_pg_name = get_extra_fsdp_pg_name(fsdp_pg_name)

        bw_gm, bw_example_inputs = self._export_and_get_bw_graph(model, inputs)

        ag_before = self._count_ag_nodes_with_pg(bw_gm, fsdp_pg_name)
        self.assertGreater(ag_before, 0)

        # Reassign to the real extra PG
        reassign_to_pg_pass(
            bw_gm,
            bw_example_inputs,
            source_pg_name=fsdp_pg_name,
            target_pg_name=extra_pg_name,
        )

        ag_old = self._count_ag_nodes_with_pg(bw_gm, fsdp_pg_name)
        ag_new = self._count_ag_nodes_with_pg(bw_gm, extra_pg_name)

        self.assertEqual(ag_old, 0)
        self.assertEqual(ag_new, ag_before)


class TestApplySACPass(TestCase):
    """Unit tests for the apply_sac_pass joint graph pass."""

    def _build_gm(self, op_targets):
        """Build a GraphModule with a chain of call_function nodes.

        Each op in op_targets becomes a call_function node. The graph
        structure is: placeholder(x), placeholder(y) -> op1 -> op2 -> ... -> output.
        """
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.placeholder("y")
        last = x
        for i, target in enumerate(op_targets):
            if target is operator.getitem:
                last = graph.call_function(target, args=(last, 0))
            else:
                last = graph.call_function(target, args=(last, y))
                # If the next op is getitem, wrap in a tuple so getitem has
                # a proper tuple/list input.
                if i + 1 < len(op_targets) and op_targets[i + 1] is operator.getitem:
                    _make_tuple = lambda x: (x, x)
                    last = graph.call_function(_make_tuple, args=(last,))
        graph.output(last)
        return torch.fx.GraphModule(torch.nn.Module(), graph)

    def _get_call_function_nodes(self, gm):
        """Return all call_function nodes from the graph."""
        return [n for n in gm.graph.nodes if n.op == "call_function"]

    def test_non_save_ops_marked_recompute(self):
        """Ops not in the save list should be marked PREFER_RECOMPUTE."""
        gm = self._build_gm(
            [
                torch.ops.aten.add.Tensor,
                torch.ops.aten.relu.default,
            ]
        )
        apply_sac_pass(gm)
        for node in self._get_call_function_nodes(gm):
            self.assertEqual(node.meta["recompute"], CheckpointPolicy.PREFER_RECOMPUTE)

    def test_save_ops_marked_must_save(self):
        """Non-mm ops in the save list should be marked MUST_SAVE."""
        custom_save = {torch.ops.aten.add.Tensor}
        gm = self._build_gm([torch.ops.aten.add.Tensor])
        apply_sac_pass(gm, op_list_to_save=custom_save)
        nodes = self._get_call_function_nodes(gm)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].meta["recompute"], CheckpointPolicy.MUST_SAVE)

    def test_getitem_propagates_parent_tags(self):
        """operator.getitem nodes should inherit the parent's recompute tag and ac_graph_id."""
        gm = self._build_gm(
            [
                torch.ops.aten.add.Tensor,
                operator.getitem,
                torch.ops.aten.relu.default,
            ]
        )
        nodes = self._get_call_function_nodes(gm)
        # nodes: [add, make_tuple, getitem, relu]
        # make_tuple is the tuple-returning parent of getitem
        self.assertEqual(nodes[0].target, torch.ops.aten.add.Tensor)
        self.assertEqual(nodes[2].target, operator.getitem)

        # Set ac_region_id on the tuple-returning parent (the direct parent of getitem)
        nodes[1].meta["custom"] = {_AC_REGION_ID: 3}

        apply_sac_pass(gm)

        tuple_node = nodes[1]
        getitem_node = nodes[2]
        self.assertEqual(getitem_node.meta["recompute"], tuple_node.meta["recompute"])
        self.assertEqual(tuple_node.meta["ac_graph_id"], 3)
        self.assertEqual(getitem_node.meta["ac_graph_id"], 3)

    def test_wait_tensor_propagates_parent_tags(self):
        """wait_tensor nodes should inherit the parent's recompute tag and ac_graph_id."""
        custom_save = {torch.ops._c10d_functional.reduce_scatter_tensor.default}
        gm = self._build_gm(
            [
                torch.ops._c10d_functional.reduce_scatter_tensor.default,
                torch.ops._c10d_functional.wait_tensor.default,
            ]
        )
        nodes = self._get_call_function_nodes(gm)
        nodes[0].meta["custom"] = {_AC_REGION_ID: 3}

        apply_sac_pass(gm, op_list_to_save=custom_save)

        rs_node = nodes[0]
        wait_node = nodes[1]
        self.assertEqual(rs_node.meta["recompute"], CheckpointPolicy.MUST_SAVE)
        self.assertEqual(wait_node.meta["recompute"], CheckpointPolicy.MUST_SAVE)
        self.assertEqual(rs_node.meta["ac_graph_id"], 3)
        self.assertEqual(wait_node.meta["ac_graph_id"], 3)

    def test_ac_graph_id_defaults_to_zero(self):
        """Nodes without ac_region_id annotation should have ac_graph_id = 0."""
        gm = self._build_gm(
            [
                torch.ops.aten.add.Tensor,
                torch.ops.aten.mm.default,
                torch.ops.aten.relu.default,
            ]
        )
        apply_sac_pass(gm)
        for node in self._get_call_function_nodes(gm):
            if node.target is not operator.getitem:
                self.assertEqual(node.meta["ac_graph_id"], 0)

    def test_ac_graph_id_from_annotation(self):
        """Nodes with _AC_REGION_ID_KEY in custom metadata should use that as ac_graph_id."""
        gm = self._build_gm(
            [
                torch.ops.aten.add.Tensor,
                torch.ops.aten.relu.default,
            ]
        )
        nodes = self._get_call_function_nodes(gm)
        # Simulate annotate_fn setting custom metadata on different nodes
        nodes[0].meta["custom"] = {_AC_REGION_ID: 1}
        nodes[1].meta["custom"] = {_AC_REGION_ID: 2}

        apply_sac_pass(gm)

        self.assertEqual(nodes[0].meta["ac_graph_id"], 1)
        self.assertEqual(nodes[1].meta["ac_graph_id"], 2)

    def test_custom_op_list_to_save(self):
        """A custom op_list_to_save should override the defaults."""
        custom_save = {torch.ops.aten.relu.default}
        gm = self._build_gm(
            [
                torch.ops.aten.add.Tensor,
                torch.ops.aten.relu.default,
            ]
        )
        apply_sac_pass(gm, op_list_to_save=custom_save)
        policies = {
            n.target: n.meta["recompute"] for n in self._get_call_function_nodes(gm)
        }
        self.assertEqual(
            policies[torch.ops.aten.add.Tensor], CheckpointPolicy.PREFER_RECOMPUTE
        )
        self.assertEqual(
            policies[torch.ops.aten.relu.default], CheckpointPolicy.MUST_SAVE
        )

    def test_mixed_mm_and_save_ops(self):
        """Graph with both mm and other save ops are annotated correctly."""
        custom_save = {torch.ops.aten.mm.default, torch.ops.aten.max.default}
        gm = self._build_gm(
            [
                torch.ops.aten.mm.default,  # 1st mm -> MUST_SAVE
                torch.ops.aten.max.default,  # in save list -> MUST_SAVE
                torch.ops.aten.mm.default,  # 2nd mm -> PREFER_RECOMPUTE
                torch.ops.aten.add.Tensor,  # not in save list -> PREFER_RECOMPUTE
                torch.ops.aten.mm.default,  # 3rd mm -> MUST_SAVE
            ]
        )
        apply_sac_pass(gm, op_list_to_save=custom_save)
        nodes = self._get_call_function_nodes(gm)
        expected = [
            (torch.ops.aten.mm.default, CheckpointPolicy.MUST_SAVE),
            (torch.ops.aten.max.default, CheckpointPolicy.MUST_SAVE),
            (torch.ops.aten.mm.default, CheckpointPolicy.PREFER_RECOMPUTE),
            (torch.ops.aten.add.Tensor, CheckpointPolicy.PREFER_RECOMPUTE),
            (torch.ops.aten.mm.default, CheckpointPolicy.MUST_SAVE),
        ]
        self.assertEqual(len(nodes), len(expected))
        for node, (target, policy) in zip(nodes, expected):
            self.assertEqual(node.target, target)
            self.assertEqual(node.meta["recompute"], policy, f"node {node.name}")


class TestAnnotateModuleComponents(TestCase):
    """Unit tests for annotate_module_components and insert_kernel_annotations_pass."""

    def _get_components(self, model, example_input):
        """Trace the model with make_fx and return the set of component annotations."""
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch.fx.traceback import preserve_node_meta

        with preserve_node_meta():
            gm = make_fx(model)(example_input)
        components = set()
        for node in gm.graph.nodes:
            comp = (node.meta.get("custom") or {}).get("component")
            if comp:
                components.add(comp)
        return components

    def test_annotate_module_components_sets_custom_metadata(self):
        """annotate_module_components wraps each child module's forward
        so that traced FX nodes carry component metadata."""
        from torchtitan.experiments.graph_trainer.common_utils import (
            annotate_module_components,
        )

        class Inner(torch.nn.Module):
            def forward(self, x):
                return x + 1

        class Outer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = Inner()
                self.b = Inner()

            def forward(self, x):
                return self.b(self.a(x))

        model = Outer()
        annotate_module_components(model)
        components = self._get_components(model, torch.randn(4))

        self.assertIn("a", components)
        self.assertIn("b", components)

    def test_annotate_module_components_nested_paths(self):
        """Nested modules get dot-separated paths."""
        from torchtitan.experiments.graph_trainer.common_utils import (
            annotate_module_components,
        )

        class Child(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.linear(x)

        class Parent(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.child = Child()

            def forward(self, x):
                return self.child(x)

        model = Parent()
        annotate_module_components(model)
        components = self._get_components(model, torch.randn(4, 4))

        # child.linear is the innermost module, so ops get that annotation.
        # The parent "child" has no ops of its own outside the linear call.
        self.assertIn("child.linear", components)

    def test_insert_kernel_annotations_pass_inserts_calls(self):
        """The pass should insert _mark_kernels_enter/exit calls at
        component boundaries."""
        from torchtitan.experiments.graph_trainer.passes import (
            _mark_kernels_enter,
            _mark_kernels_exit,
            insert_kernel_annotations_pass,
        )

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        # Two nodes with component "attn", one with "ffn"
        n1 = graph.call_function(torch.relu, (x,))
        n1.meta["custom"] = {"component": "attn"}
        n2 = graph.call_function(torch.sigmoid, (n1,))
        n2.meta["custom"] = {"component": "attn"}
        n3 = graph.call_function(torch.tanh, (n2,))
        n3.meta["custom"] = {"component": "ffn"}
        graph.output(n3)

        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        insert_kernel_annotations_pass(gm)

        targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
        self.assertIn(_mark_kernels_enter, targets)
        self.assertIn(_mark_kernels_exit, targets)

        # Count: 2 scopes (attn, ffn) = 2 enters + 2 exits
        enters = [t for t in targets if t is _mark_kernels_enter]
        exits = [t for t in targets if t is _mark_kernels_exit]
        self.assertEqual(len(enters), 2)
        self.assertEqual(len(exits), 2)

    def test_insert_kernel_annotations_pass_noop_without_metadata(self):
        """The pass should not insert anything when no custom metadata exists."""
        from torchtitan.experiments.graph_trainer.passes import (
            _mark_kernels_enter,
            insert_kernel_annotations_pass,
        )

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        n1 = graph.call_function(torch.relu, (x,))
        graph.output(n1)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        insert_kernel_annotations_pass(gm)

        targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
        self.assertNotIn(_mark_kernels_enter, targets)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
