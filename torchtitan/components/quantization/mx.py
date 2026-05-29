# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field, fields
from importlib.util import find_spec
from typing import Literal

from torchtitan.components.quantization import QuantizationConverter
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.common.nn_modules import Linear
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability, has_rocm_capability

from .utils import swap_token_dispatcher


class MXFP8Linear(Linear):
    """Linear that applies MXFP8 quantization in its constructor."""

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        """Drop-in replacement for Linear.Config that builds MXFP8Linear."""

        _recipe_name: str = "mxfp8_rceil"

    def __init__(self, config: Config):
        super().__init__(config)
        from torchao.prototype.moe_training.config import (
            MXFP8TrainingOpConfig,
            MXFP8TrainingRecipe,
        )
        from torchao.quantization.quant_api import quantize_

        recipe = MXFP8TrainingRecipe(config._recipe_name)
        mxfp8_op_config = MXFP8TrainingOpConfig.from_recipe(recipe)
        quantize_(self, config=mxfp8_op_config)


class MXFP8LinearConverter(QuantizationConverter):
    """Apply MXFP8 quantization to modules matching FQNs (e.g. Flux blocks)."""

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        recipe_name: Literal["mxfp8_rceil"] = "mxfp8_rceil"
        """
        Quantization recipe name for grouped GEMMs. Options: ["mxfp8_rceil"]

        - mxfp8_rceil: MXFP8 dynamic quantization with RCEIL rounding mode when computing the e8m0 scale factors.
        """

        fqns: list[str] = field(default_factory=list)
        """
        *Prototype feature, performance optimization still in progress*
        Comma-separated list of fully qualified names of MoE modules to apply MXFP8 dynamic quantization
        on grouped GEMM operations.
        This is a prototype feature that requires the torchao nightly build.
        """

    def __init__(self, config: Config):
        self.config = config

        if find_spec("torchao") is None:
            raise ImportError(
                "torchao is not installed. Please install it to use MXFP8 linear layers."
            )

        # Can be removed if we enable the emulated versions
        assert has_cuda_capability(
            10, 0
        ), "MXFP8 is only supported on SM100 or later architectures"

        if not self.config.model_compile_enabled:
            logger.warning(
                "torch.compile enablement is required for highest performance "
                "of MXFP8 dynamic quantization."
            )

    def convert(self, model_config) -> None:
        fqns = self.config.fqns
        for fqn, config, parent, attr in model_config.traverse(Linear.Config):
            if not fqns or any(target_fqn in fqn for target_fqn in fqns):
                new_config = MXFP8Linear.Config(
                    in_features=config.in_features,
                    out_features=config.out_features,
                    bias=config.bias,
                    param_init=config.param_init,
                    _recipe_name=self.config.recipe_name,
                )
                if isinstance(parent, list):
                    parent[attr] = new_config
                else:
                    setattr(parent, attr, new_config)

        logger.info(
            f"Converted modules to use dynamic {self.config.recipe_name} "
            "quantization for grouped_mm and linear ops"
        )


_mxfp8_experts_cache: dict[type, type] = {}


def _get_mxfp8_grouped_experts_cls(parent_cls: type) -> type:
    """Get or create an MXFP8-quantized subclass of *parent_cls*.

    Works for any ``GroupedExperts`` subclass (e.g. gpt-oss variants).
    The returned class has a proper ``_owner`` set by ``__init_subclass__``.
    """
    if parent_cls in _mxfp8_experts_cache:
        return _mxfp8_experts_cache[parent_cls]

    parent_config_cls = parent_cls.Config  # type: ignore[attr-defined]

    class MXFP8GroupedExperts(parent_cls):  # type: ignore[valid-type, misc]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc]
            recipe_name: str = "mxfp8_rceil"

        def __init__(self, config: Config):
            super().__init__(config)
            from torchao.prototype.moe_training.config import (
                MXFP8TrainingOpConfig,
                MXFP8TrainingRecipe,
            )
            from torchao.quantization.quant_api import quantize_

            recipe = MXFP8TrainingRecipe(config.recipe_name)
            mxfp8_op_config = MXFP8TrainingOpConfig.from_recipe(recipe)
            quantize_(
                self,
                config=mxfp8_op_config,
                filter_fn=lambda mod, _fqn: isinstance(mod, GroupedExperts),
            )

    MXFP8GroupedExperts.__name__ = f"MXFP8{parent_cls.__name__}"
    MXFP8GroupedExperts.__qualname__ = f"MXFP8{parent_cls.__name__}"
    _mxfp8_experts_cache[parent_cls] = MXFP8GroupedExperts
    return MXFP8GroupedExperts


class MXFP8GroupedExpertsConverter(QuantizationConverter):
    """Apply MXFP8 quantization to MoE expert grouped GEMMs."""

    # MXFP8: scaling block size is (1 x 32), so contracting dim must be divisible by 32.
    PAD_MULTIPLE = 32

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        recipe_name: Literal["mxfp8_rceil"] = "mxfp8_rceil"
        """
        Quantization recipe name for grouped GEMMs. Options: ["mxfp8_rceil"]

        - mxfp8_rceil: MXFP8 dynamic quantization with RCEIL rounding mode when computing the e8m0 scale factors.
        """

    def __init__(self, config: Config):
        self.config = config

        if find_spec("torchao") is None:
            raise ImportError(
                "torchao is not installed. Please install it to use MXFP8 MoE training."
            )

        assert has_cuda_capability(
            10, 0
        ), "MXFP8 is only supported on SM100 or later architectures"

        if not self.config.model_compile_enabled:
            logger.warning(
                "torch.compile enablement is required for highest performance "
                "of MXFP8 dynamic quantization."
            )

    def convert(self, model_config) -> None:
        for _fqn, config, parent, attr in model_config.traverse(GroupedExperts.Config):
            swap_token_dispatcher(config, self.PAD_MULTIPLE)
            base_module_cls = type(config)._owner
            quantized_cls = _get_mxfp8_grouped_experts_cls(base_module_cls)
            config_cls = quantized_cls.Config  # type: ignore[attr-defined]
            new_config = config_cls(
                **{f.name: getattr(config, f.name) for f in fields(config)},
                recipe_name=self.config.recipe_name,
            )
            if isinstance(parent, list):
                parent[attr] = new_config
            else:
                setattr(parent, attr, new_config)

        logger.info(
            f"Converted GroupedExperts to use dynamic {self.config.recipe_name} "
            "quantization for grouped_mm and linear ops"
        )


_mxfp4_experts_cache: dict[type, type] = {}


def _get_mxfp4_grouped_experts_cls(parent_cls: type) -> type:
    """Get or create an MXFP4-quantized subclass of *parent_cls*.

    Mirrors ``_get_mxfp8_grouped_experts_cls`` but dispatches the expert grouped
    GEMMs to the MXFP4 (e2m1 data + E8M0 block scales) training path. Works for
    any ``GroupedExperts`` subclass; the returned class has a proper ``_owner``
    set by ``__init_subclass__``.
    """
    if parent_cls in _mxfp4_experts_cache:
        return _mxfp4_experts_cache[parent_cls]

    parent_config_cls = parent_cls.Config  # type: ignore[attr-defined]

    class MXFP4GroupedExperts(parent_cls):  # type: ignore[valid-type, misc]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc]
            recipe_name: str = "mxfp4_rceil"
            pad_token_groups_for_grouped_mm: bool = False

        def __init__(self, config: Config):
            super().__init__(config)
            from torchao.prototype.moe_training.config import (
                MXFP4TrainingOpConfig,
                MXFP4TrainingRecipe,
            )
            from torchao.quantization.quant_api import quantize_

            recipe = MXFP4TrainingRecipe(config.recipe_name)
            mxfp4_op_config = MXFP4TrainingOpConfig.from_recipe(recipe)
            mxfp4_op_config.pad_token_groups_for_grouped_mm = (
                config.pad_token_groups_for_grouped_mm
            )
            quantize_(
                self,
                config=mxfp4_op_config,
                filter_fn=lambda mod, _fqn: isinstance(mod, GroupedExperts),
            )

    MXFP4GroupedExperts.__name__ = f"MXFP4{parent_cls.__name__}"
    MXFP4GroupedExperts.__qualname__ = f"MXFP4{parent_cls.__name__}"
    _mxfp4_experts_cache[parent_cls] = MXFP4GroupedExperts
    return MXFP4GroupedExperts


class MXFP4GroupedExpertsConverter(QuantizationConverter):
    """Apply MXFP4 quantization to MoE expert grouped GEMMs.

    MXFP4 training is currently supported only on ROCm gfx950 (MI355X) or later,
    via torchao's Triton ``tl.dot_scaled`` kernels.
    """

    # MXFP4: scaling block size is (1 x 32), so contracting dim must be divisible by 32.
    PAD_MULTIPLE = 32

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        recipe_name: Literal[
            "mxfp4_rceil", "mxfp4_rceil_wgrad_with_hp"
        ] = "mxfp4_rceil"
        """
        Quantization recipe name for grouped GEMMs. Options:
        ["mxfp4_rceil", "mxfp4_rceil_wgrad_with_hp"]

        - mxfp4_rceil: full MXFP4 fwd/dgrad/wgrad with RCEIL rounding mode when
          computing the e8m0 scale factors.
        - mxfp4_rceil_wgrad_with_hp: MXFP4 fwd/dgrad with the weight gradient kept
          in bf16 (high precision) for better numerics.
        """

        pad_token_groups_for_grouped_mm: bool = False
        """
        Whether to pad token groups to multiples of the MXFP4 scaling block (32)
        in the grouped GEMM. Set to False when using HybridEP, which already pads
        token groups as part of the all-to-all.
        """

    def __init__(self, config: Config):
        self.config = config

        if find_spec("torchao") is None:
            raise ImportError(
                "torchao is not installed. Please install it to use MXFP4 MoE training."
            )

        # MXFP4 training is currently ROCm gfx950 (MI355X) only.
        assert has_rocm_capability(
            9, 5
        ), "MXFP4 training currently requires ROCm gfx950 (MI355X) or later"

        if not self.config.model_compile_enabled:
            logger.warning(
                "torch.compile enablement is recommended for highest performance "
                "of MXFP4 dynamic quantization."
            )

    def convert(self, model_config) -> None:
        for _fqn, config, parent, attr in model_config.traverse(GroupedExperts.Config):
            swap_token_dispatcher(config, self.PAD_MULTIPLE)
            base_module_cls = type(config)._owner
            quantized_cls = _get_mxfp4_grouped_experts_cls(base_module_cls)
            config_cls = quantized_cls.Config  # type: ignore[attr-defined]
            new_config = config_cls(
                **{f.name: getattr(config, f.name) for f in fields(config)},
                recipe_name=self.config.recipe_name,
                pad_token_groups_for_grouped_mm=self.config.pad_token_groups_for_grouped_mm,
            )
            if isinstance(parent, list):
                parent[attr] = new_config
            else:
                setattr(parent, attr, new_config)

        logger.info(
            f"Converted GroupedExperts to use dynamic {self.config.recipe_name} "
            "quantization for grouped_mm ops"
        )
