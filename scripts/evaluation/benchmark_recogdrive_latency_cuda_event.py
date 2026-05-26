import argparse
import importlib.metadata
import json
import time
from pathlib import Path
from statistics import mean, median
from types import MethodType

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
from navsim.agents.recogdrive.recogdrive_backbone import (
    IMG_CONTEXT_TOKEN,
    IMG_END_TOKEN,
    IMG_START_TOKEN,
    system_message,
)
from navsim.agents.recogdrive.recogdrive_features import format_number
from navsim.agents.recogdrive.utils.conversation import get_conv_template
from navsim.agents.recogdrive.utils.internvl_preprocess import load_image
from navsim.agents.recogdrive.blocks.rope import rotate_half
from navsim.common.dataloader import SceneLoader


def cuda_time_ms(fn):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), result


def wall_time_ms(fn):
    start = time.perf_counter()
    result = fn()
    return (time.perf_counter() - start) * 1000.0, result


def summarize(values):
    values = list(values)
    values_sorted = sorted(values)
    return {
        "mean_ms": mean(values),
        "median_ms": median(values),
        "min_ms": values_sorted[0],
        "max_ms": values_sorted[-1],
        "p90_ms": values_sorted[int(0.9 * (len(values_sorted) - 1))],
    }


def get_package_version(package_name):
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def install_dit_pointwise_variant(action_head, variant):
    if variant not in {"addcmul_residual", "addcmul_pointwise"}:
        if variant != "baseline":
            raise ValueError(f"Unsupported DiT pointwise variant: {variant}")

    if hasattr(action_head.model, "set_pointwise_variant"):
        action_head.model.set_pointwise_variant(variant)
        return

    if variant == "baseline":
        return

    def modulate_addcmul(x, shift, scale):
        scale = scale.unsqueeze(1).add(1.0)
        if shift is None:
            return x * scale
        return torch.addcmul(shift.unsqueeze(1), x, scale)

    def block_forward(this, hidden_states, conditioning, encoder_hidden_states=None, rotary_embedder=None, encoder_kv=None):
        mod_params = this.adaLN_modulation(conditioning)
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = mod_params.chunk(6, dim=1)

        normed_states = this.norm1(hidden_states)
        if variant == "addcmul_pointwise":
            modulated_states = modulate_addcmul(normed_states, shift_attn, scale_attn)
        else:
            modulated_states = this.modulate(normed_states, shift_attn, scale_attn)

        attn_output = this.attn(
            modulated_states,
            encoder_hidden_states=encoder_hidden_states,
            rotary_embedder=rotary_embedder,
            encoder_kv=encoder_kv,
        )
        hidden_states = torch.addcmul(hidden_states, attn_output, gate_attn.unsqueeze(1))

        normed_states = this.norm2(hidden_states)
        if variant == "addcmul_pointwise":
            modulated_states = modulate_addcmul(normed_states, shift_ffn, scale_ffn)
        else:
            modulated_states = this.modulate(normed_states, shift_ffn, scale_ffn)

        ffn_output = this.ffn(modulated_states)
        return torch.addcmul(hidden_states, ffn_output, gate_ffn.unsqueeze(1))

    def final_forward(this, x, conditioning):
        shift, scale = this.modulation_proj(conditioning).chunk(2, dim=1)
        if variant == "addcmul_pointwise":
            x = modulate_addcmul(this.norm_final(x), shift, scale)
        else:
            x = this.modulate(this.norm_final(x), shift, scale)
        return this.linear(x)

    for block in action_head.model.transformer_blocks:
        block.forward = MethodType(block_forward, block)
    action_head.model.final_layer.forward = MethodType(final_forward, action_head.model.final_layer)


def install_dit_attention_variant(action_head, variant):
    if variant in {"baseline", "rope_slice"}:
        return
    if variant not in {"rope_slice_addcmul"}:
        raise ValueError(f"Unsupported DiT attention variant: {variant}")

    def attention_forward(this, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_embedder=None, encoder_kv=None):
        batch_size, query_length, _ = hidden_states.shape
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        is_self_attention = encoder_hidden_states is None and encoder_kv is None

        q = this.to_q(hidden_states).view(batch_size, query_length, this.num_heads, this.head_dim).transpose(1, 2)
        if encoder_kv is None:
            k = this.to_k(context).view(batch_size, -1, this.num_heads, this.head_dim).transpose(1, 2)
            v = this.to_v(context).view(batch_size, -1, this.num_heads, this.head_dim).transpose(1, 2)
            k = this.k_norm(k)
        else:
            k, v = encoder_kv

        q = this.q_norm(q)

        if rotary_embedder is not None:
            cos = rotary_embedder.cos_cached[:, :, :query_length, :].to(dtype=q.dtype)
            sin = rotary_embedder.sin_cached[:, :, :query_length, :].to(dtype=q.dtype)
            q = torch.addcmul(q * cos, rotate_half(q), sin)
            if is_self_attention:
                k = torch.addcmul(k * cos, rotate_half(k), sin)

        if hasattr(F, "scaled_dot_product_attention"):
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=this.to_out[-1].p if this.training else 0.0,
            )
        else:
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * this.scale
            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask
            attn_probs = attn_scores.softmax(dim=-1)
            attn_probs = F.dropout(attn_probs, p=this.to_out[-1].p, training=this.training)
            x = torch.matmul(attn_probs, v)

        x = x.transpose(1, 2).reshape(batch_size, query_length, -1)
        return this.to_out(x)

    for block in action_head.model.transformer_blocks:
        block.attn.forward = MethodType(attention_forward, block.attn)


SDPA_BACKENDS = {
    "flash": SDPBackend.FLASH_ATTENTION,
    "math": SDPBackend.MATH,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
}


ACTION_HEAD_DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def cast_action_head_dtype(action_head, dtype_name):
    if dtype_name == "fp32":
        return
    action_head.to(dtype=ACTION_HEAD_DTYPES[dtype_name])


def build_question(history_trajectory, high_command_one_hot):
    navigation_commands = ["turn left", "go straight", "turn right"]
    command_idx = torch.argmax(high_command_one_hot, dim=-1).item()
    command_str = navigation_commands[command_idx]
    history_str = " ".join(
        [
            f"   - t-{3-j}: ({format_number(history_trajectory[j, 0].item())}, "
            f"{format_number(history_trajectory[j, 1].item())}, "
            f"{format_number(history_trajectory[j, 2].item())})"
            for j in range(history_trajectory.shape[0])
        ]
    )
    prompt = (
        "<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n"
        "1. Visual perception from front camera view\n"
        f"2. Historical motion context (last 4 timesteps):{history_str}\n"
        f"3. Active navigation command: [{command_str.upper()}]"
    )
    output_requirements = (
        "\nOutput requirements:\n- Predict 8 future trajectory points\n"
        "- Each point format: (x:float, y:float, heading:float)\n"
        "- Use [PT, ...] to encapsulate the trajectory\n"
        "- Maintain numerical precision to 2 decimal places"
    )
    return f"{prompt}{output_requirements}"


def make_agent_input_features(agent, agent_input):
    features = {}
    for builder in agent.get_feature_builders():
        features.update(builder.compute_features(agent_input))
    return {key: value.unsqueeze(0) for key, value in features.items()}


def build_profiled_backbone_inputs(backbone, pixel_values, questions, num_patches_list):
    queries = []
    for idx, num_patches in enumerate(num_patches_list):
        question = questions[idx]
        if pixel_values is not None and "<image>" not in question:
            question = "<image>\n" + question

        template = get_conv_template("internvl2_5")
        template.system_message = system_message
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        image_tokens = (
            IMG_START_TOKEN
            + IMG_CONTEXT_TOKEN * backbone.pruned_num_image_token * num_patches
            + IMG_END_TOKEN
        )
        queries.append(query.replace("<image>", image_tokens, 1))

    backbone.tokenizer.padding_side = "left"
    if backbone.prune_keep_ratio < 1.0:
        model_inputs = backbone.tokenizer(queries, return_tensors="pt", padding=True)
    else:
        model_inputs = backbone.tokenizer(
            queries, return_tensors="pt", padding="max_length", max_length=2800
        )

    device = pixel_values.device if pixel_values.is_cuda else torch.device("cuda")
    input_ids = model_inputs["input_ids"].to(device)
    attention_mask = model_inputs["attention_mask"].to(device)
    backbone.last_input_seq_len = int(input_ids.shape[1])

    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    image_flags = torch.ones(pixel_values.size(0), dtype=torch.bool, device=device)

    return input_ids, attention_mask, position_ids, image_flags


def run_profiled_backbone(backbone, pixel_values, questions, num_patches_list):
    tokenize_wall_ms, model_inputs = wall_time_ms(
        lambda: build_profiled_backbone_inputs(backbone, pixel_values, questions, num_patches_list)
    )
    input_ids, attention_mask, position_ids, image_flags = model_inputs
    model_dtype = next(backbone.model.parameters()).dtype

    def run_vision_encoder():
        vit_embeds = backbone.model.extract_feature(pixel_values.to(model_dtype))
        return vit_embeds[image_flags]

    vision_encoder_ms, vit_embeds = cuda_time_ms(run_vision_encoder)

    if backbone.prune_keep_ratio < 1.0:
        token_select_ms, vit_embeds = cuda_time_ms(lambda: backbone._select_visual_tokens(vit_embeds))
    else:
        token_select_ms = 0.0

    def build_input_embeds():
        input_embeds = backbone.model.language_model.get_input_embeddings()(input_ids).clone()
        batch_size, seq_len, hidden_dim = input_embeds.shape
        flat_input_embeds = input_embeds.reshape(batch_size * seq_len, hidden_dim)
        flat_input_ids = input_ids.reshape(batch_size * seq_len)
        selected = flat_input_ids == backbone.img_context_token_id
        flat_vit_embeds = vit_embeds.reshape(-1, hidden_dim)

        num_selected = int(selected.sum().item())
        if num_selected != flat_vit_embeds.shape[0]:
            raise RuntimeError(
                f"IMG_CONTEXT token count ({num_selected}) does not match visual token count "
                f"({flat_vit_embeds.shape[0]})."
            )

        flat_input_embeds[selected] = flat_vit_embeds.to(flat_input_embeds.device)
        return flat_input_embeds.reshape(batch_size, seq_len, hidden_dim)

    embed_replace_ms, input_embeds = cuda_time_ms(build_input_embeds)

    language_model_ms, outputs = cuda_time_ms(
        lambda: backbone.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
        )
    )

    profile = {
        "vlm_tokenize_wall_ms": tokenize_wall_ms,
        "vision_encoder_cuda_ms": vision_encoder_ms,
        "token_select_cuda_ms": token_select_ms,
        "embed_replace_cuda_ms": embed_replace_ms,
        "language_model_cuda_ms": language_model_ms,
        "vlm_profile_cuda_ms": (
            vision_encoder_ms + token_select_ms + embed_replace_ms + language_model_ms
        ),
    }
    return profile, outputs


def run_one(agent, agent_input, image_max_num, image_backend, action_getter=None, sdpa_backend="auto"):
    feature_wall_ms, features = wall_time_ms(lambda: make_agent_input_features(agent, agent_input))

    history_trajectory = features["history_trajectory"].cuda()
    high_command_one_hot = features["high_command_one_hot"].cuda()
    status_feature = features["status_feature"].cuda()
    image_path_tensor = features["image_path_tensor"]

    image_paths = agent._decode_paths_from_tensor(image_path_tensor)
    image_wall_ms, pixel_values_list = wall_time_ms(
        lambda: [load_image(path, max_num=image_max_num, backend=image_backend) for path in image_paths]
    )
    num_patches_list = [pixel_values.shape[0] for pixel_values in pixel_values_list]

    h2d_ms, pixel_values_cat = cuda_time_ms(lambda: torch.cat(pixel_values_list, dim=0).cuda())
    questions = [build_question(history_trajectory[0], high_command_one_hot[0])]

    vlm_ms, outputs = cuda_time_ms(
        lambda: agent.backbone(pixel_values_cat, questions, num_patches_list=num_patches_list)
    )
    backbone_profile, _ = run_profiled_backbone(
        agent.backbone, pixel_values_cat, questions, num_patches_list
    )
    last_hidden_state = outputs.hidden_states[-1]

    model_dtype = next(agent.action_head.parameters()).dtype
    if last_hidden_state.ndim == 2:
        last_hidden_state = last_hidden_state.unsqueeze(0)
    last_hidden_state = last_hidden_state.to(model_dtype)

    history_trajectory_reshaped = history_trajectory.view(history_trajectory.size(0), -1)
    action_inputs = BatchFeature(
        {
            "state": torch.cat([status_feature, history_trajectory_reshaped], dim=1).to(model_dtype),
            "his_traj": history_trajectory_reshaped.to(model_dtype),
            "status_feature": status_feature.to(model_dtype),
        }
    )
    action_getter = action_getter or agent.action_head.get_action
    def run_action():
        if sdpa_backend == "auto":
            return action_getter(last_hidden_state, action_inputs)
        with sdpa_kernel(SDPA_BACKENDS[sdpa_backend]):
            return action_getter(last_hidden_state, action_inputs)

    diffusion_ms, predictions = cuda_time_ms(run_action)
    post_wall_ms, poses = wall_time_ms(lambda: predictions["pred_traj"].float().cpu().squeeze(0))

    def end_to_end_gpu_only():
        pixel_values_cat_e2e = torch.cat(pixel_values_list, dim=0).cuda()
        outputs_e2e = agent.backbone(pixel_values_cat_e2e, questions, num_patches_list=num_patches_list)
        hidden_e2e = outputs_e2e.hidden_states[-1]
        if hidden_e2e.ndim == 2:
            hidden_e2e = hidden_e2e.unsqueeze(0)
        hidden_e2e = hidden_e2e.to(model_dtype)
        if sdpa_backend == "auto":
            return action_getter(hidden_e2e, action_inputs)
        with sdpa_kernel(SDPA_BACKENDS[sdpa_backend]):
            return action_getter(hidden_e2e, action_inputs)

    e2e_gpu_ms, _ = cuda_time_ms(end_to_end_gpu_only)

    return {
        "feature_wall_ms": feature_wall_ms,
        "image_preprocess_wall_ms": image_wall_ms,
        "image_h2d_cuda_ms": h2d_ms,
        "vlm_cuda_ms": vlm_ms,
        **backbone_profile,
        "diffusion_cuda_ms": diffusion_ms,
        "e2e_gpu_cuda_ms": e2e_gpu_ms,
        "postprocess_wall_ms": post_wall_ms,
        "num_patches": sum(num_patches_list),
        "num_visual_tokens": sum(num_patches_list) * getattr(agent.backbone, "pruned_num_image_token", 256),
        "input_seq_len": getattr(agent.backbone, "last_input_seq_len", None),
        "trajectory_first_pose": poses[0].tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=["2b", "8b"], default="2b")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--prune-keep-ratio", type=float, default=1.0)
    parser.add_argument("--prune-method", type=str, default="tfps")
    parser.add_argument("--diffusion-steps", type=int, default=5)
    parser.add_argument("--image-max-num", type=int, default=12)
    parser.add_argument(
        "--image-backend",
        type=str,
        default="pil",
        choices=[
            "pil",
            "pil_draft",
            "pil_parallel",
            "pil_no_resize",
            "pil_parallel_no_resize",
            "pil_numpy",
            "pil_parallel_numpy",
            "opencv",
        ],
    )
    parser.add_argument("--compile-action-head", action="store_true")
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead")
    parser.add_argument("--fast-ddim-action", action="store_true")
    parser.add_argument(
        "--dit-pointwise-variant",
        type=str,
        default="baseline",
        choices=["baseline", "addcmul_residual", "addcmul_pointwise"],
    )
    parser.add_argument(
        "--dit-attention-variant",
        type=str,
        default="baseline",
        choices=["baseline", "rope_slice", "rope_slice_addcmul"],
    )
    parser.add_argument(
        "--sdpa-backend",
        type=str,
        default="auto",
        choices=["auto", *SDPA_BACKENDS.keys()],
        help="Force a PyTorch SDPA backend for the diffusion planner attention only.",
    )
    parser.add_argument(
        "--action-head-dtype",
        type=str,
        default="fp32",
        choices=ACTION_HEAD_DTYPES.keys(),
        help="Cast the diffusion/action head for latency ablations. The VLM dtype is unchanged.",
    )
    args = parser.parse_args()

    if args.model_size == "2b":
        checkpoint = "/data/zxz/models/ReCogDrive-2B-RL/ReCogDrive_Diffusion_Planner_2B_RL.ckpt"
        vlm_path = "/data/zxz/models/ReCogDrive-VLM-2B"
        vlm_size = "small"
    else:
        checkpoint = "/data/zxz/models/ReCogDrive-8B-RL/ReCogDrive_Diffusion_Planner_8B_RL.ckpt"
        vlm_path = "/data/zxz/models/ReCogDrive-VLM-8B"
        vlm_size = "large"

    cfg_dir = str(Path("navsim/planning/script/config/pdm_scoring").resolve())
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(
            config_name="default_run_pdm_score",
            overrides=[
                "train_test_split=navtest",
                "agent=recogdrive_agent",
                f"agent.checkpoint_path='{checkpoint}'",
                f"agent.vlm_path='{vlm_path}'",
                "agent.cache_hidden_state=False",
                "agent.cam_type=single",
                "agent.vlm_type=internvl",
                "agent.dit_type=small",
                f"agent.vlm_size={vlm_size}",
                "agent.sampling_method=ddim",
                "agent.grpo=False",
                f"agent.vlm_prune_keep_ratio={args.prune_keep_ratio}",
                f"agent.vlm_prune_method={args.prune_method}",
                f"agent.diffusion_num_inference_steps={args.diffusion_steps}",
            ],
        )

    agent: ReCogDriveAgent = instantiate(cfg.agent)
    agent.initialize()
    cast_action_head_dtype(agent.action_head, args.action_head_dtype)
    agent.eval()
    install_dit_pointwise_variant(agent.action_head, args.dit_pointwise_variant)
    install_dit_attention_variant(agent.action_head, args.dit_attention_variant)
    action_getter = agent.action_head.get_action_fast_ddim if args.fast_ddim_action else agent.action_head.get_action
    if args.compile_action_head:
        action_getter = torch.compile(
            agent.action_head.get_action,
            mode=args.compile_mode,
            fullgraph=False,
        )

    loader = SceneLoader(
        Path(cfg.navsim_log_path),
        Path(cfg.sensor_blobs_path),
        instantiate(cfg.train_test_split.scene_filter),
        agent.get_sensor_config(),
        load_image_path=True,
    )
    tokens = loader.tokens[: args.warmup + args.num_samples]
    agent_inputs = [loader.get_agent_input_from_token(token) for token in tokens]

    records = []
    with torch.inference_mode():
        for idx, agent_input in enumerate(agent_inputs):
            record = run_one(
                agent,
                agent_input,
                args.image_max_num,
                args.image_backend,
                action_getter=action_getter,
                sdpa_backend=args.sdpa_backend,
            )
            record["index"] = idx
            record["warmup"] = idx < args.warmup
            records.append(record)
            print(
                json.dumps(
                    {
                        "index": idx,
                        "warmup": record["warmup"],
                        "e2e_gpu_cuda_ms": record["e2e_gpu_cuda_ms"],
                        "vlm_cuda_ms": record["vlm_cuda_ms"],
                        "vision_encoder_cuda_ms": record["vision_encoder_cuda_ms"],
                        "token_select_cuda_ms": record["token_select_cuda_ms"],
                        "language_model_cuda_ms": record["language_model_cuda_ms"],
                        "diffusion_cuda_ms": record["diffusion_cuda_ms"],
                        "num_patches": record["num_patches"],
                        "num_visual_tokens": record["num_visual_tokens"],
                        "input_seq_len": record["input_seq_len"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    measured = [record for record in records if not record["warmup"]]
    metric_keys = [
        "feature_wall_ms",
        "image_preprocess_wall_ms",
        "image_h2d_cuda_ms",
        "vlm_cuda_ms",
        "vlm_tokenize_wall_ms",
        "vision_encoder_cuda_ms",
        "token_select_cuda_ms",
        "embed_replace_cuda_ms",
        "language_model_cuda_ms",
        "vlm_profile_cuda_ms",
        "diffusion_cuda_ms",
        "e2e_gpu_cuda_ms",
        "postprocess_wall_ms",
    ]
    summary = {
        "model_size": args.model_size,
        "num_samples": args.num_samples,
        "warmup": args.warmup,
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "flash_attn_version": get_package_version("flash-attn"),
        "prune_keep_ratio": args.prune_keep_ratio,
        "prune_method": args.prune_method,
        "diffusion_steps": args.diffusion_steps,
        "image_max_num": args.image_max_num,
        "image_backend": args.image_backend,
        "dit_pointwise_variant": args.dit_pointwise_variant,
        "dit_attention_variant": args.dit_attention_variant,
        "sdpa_backend": args.sdpa_backend,
        "action_head_dtype": args.action_head_dtype,
        "compile_action_head": args.compile_action_head,
        "compile_mode": args.compile_mode if args.compile_action_head else None,
        "fast_ddim_action": args.fast_ddim_action,
        "metrics": {key: summarize(record[key] for record in measured) for key in metric_keys},
        "num_patches": summarize(record["num_patches"] for record in measured),
        "num_visual_tokens": summarize(record["num_visual_tokens"] for record in measured),
        "input_seq_len": summarize(record["input_seq_len"] for record in measured),
        "records": records,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
