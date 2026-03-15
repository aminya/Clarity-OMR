"""Minimal subset of training utilities needed for inference (DoRA loading)."""

from __future__ import annotations

from typing import Dict, List


def _prepare_model_for_dora(model, dora_config: Dict[str, object]):
    new_module_keywords = (
        "token_embedding",
        "lm_head",
        "decoder_norm",
        "contour_head",
        "deformable_attention",
        "positional_bridge",
    )

    try:
        from peft import LoraConfig, TaskType, get_peft_model
        import torch.nn as torch_nn
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "peft is required for DoRA model loading. Install with: pip install peft"
        ) from exc

    configured_targets = list(dora_config.get("target_modules", []))
    linear_targets: List[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch_nn.Linear):
            continue
        if any(marker in name for marker in new_module_keywords):
            continue
        linear_targets.append(name)
    if configured_targets:
        linear_targets = [
            name
            for name in linear_targets
            if any(name.endswith(target) or f".{target}" in name for target in configured_targets)
            or name.startswith("encoder.backbone")
        ]
    if not linear_targets:
        raise RuntimeError(
            "DoRA target-module matching failed; no linear layers matched configured targets."
        )

    peft_config = LoraConfig(
        r=int(dora_config["rank"]),
        lora_alpha=int(dora_config["alpha"]),
        lora_dropout=float(dora_config["dropout"]),
        target_modules=linear_targets,
        task_type=TaskType.FEATURE_EXTRACTION,
        use_dora=True,
        bias="none",
        modules_to_save=list(new_module_keywords),
    )
    try:
        model = get_peft_model(model, peft_config)
    except Exception as exc:
        raise RuntimeError(f"Failed to apply DoRA adapters: {exc}") from exc
    dora_applied = True

    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if "lora_" in name or any(marker in name for marker in new_module_keywords):
            parameter.requires_grad = True
    if not any(parameter.requires_grad for parameter in model.parameters()):
        for parameter in model.parameters():
            parameter.requires_grad = True

    return model, dora_applied
