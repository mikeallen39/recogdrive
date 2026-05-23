import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from types import MethodType

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from transformers.feature_extraction_utils import BatchFeature

from benchmark_recogdrive_latency_cuda_event import (
    build_question,
    make_agent_input_features,
    summarize,
)
from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
from navsim.agents.recogdrive.utils.internvl_preprocess import load_image
from navsim.common.dataloader import SceneLoader


class CudaEventProfiler:
    def __init__(self):
        self.values = defaultdict(list)

    def time(self, name, fn):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        torch.cuda.synchronize()
        self.values[name].append(start.elapsed_time(end))
        return result

    def summary(self):
        return {key: summarize(values) for key, values in sorted(self.values.items())}


def build_agent(args):
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
    agent.eval()
    return cfg, agent


def make_diffusion_inputs(agent, agent_input, image_max_num, image_backend):
    features = make_agent_input_features(agent, agent_input)
    history_trajectory = features["history_trajectory"].cuda()
    high_command_one_hot = features["high_command_one_hot"].cuda()
    status_feature = features["status_feature"].cuda()
    image_paths = agent._decode_paths_from_tensor(features["image_path_tensor"])
    pixel_values_list = [
        load_image(path, max_num=image_max_num, backend=image_backend) for path in image_paths
    ]
    num_patches_list = [pixel_values.shape[0] for pixel_values in pixel_values_list]
    pixel_values_cat = torch.cat(pixel_values_list, dim=0).cuda()
    questions = [build_question(history_trajectory[0], high_command_one_hot[0])]

    with torch.inference_mode():
        outputs = agent.backbone(pixel_values_cat, questions, num_patches_list=num_patches_list)

    last_hidden_state = outputs.hidden_states[-1]
    if last_hidden_state.ndim == 2:
        last_hidden_state = last_hidden_state.unsqueeze(0)

    model_dtype = next(agent.action_head.parameters()).dtype
    history_trajectory_reshaped = history_trajectory.view(history_trajectory.size(0), -1)
    action_inputs = BatchFeature(
        {
            "state": torch.cat([status_feature, history_trajectory_reshaped], dim=1).to(model_dtype),
            "his_traj": history_trajectory_reshaped.to(model_dtype),
            "status_feature": status_feature.to(model_dtype),
        }
    )
    return last_hidden_state.to(model_dtype), action_inputs


def profile_get_action(planner, vl_features, action_input, profiler, use_fast_ddim=False):
    if use_fast_ddim:
        return profile_get_action_fast_ddim(planner, vl_features, action_input, profiler)

    vl_embeds = profiler.time("condition/feature_encoder", lambda: planner.feature_encoder(vl_features))
    history_embeds = profiler.time(
        "condition/history_encoder_repeat",
        lambda: planner.his_traj_encoder(action_input.his_traj.unsqueeze(1)).repeat(
            1, planner.config.action_horizon, 1
        ),
    )
    ego_embeds = profiler.time(
        "condition/ego_encoder", lambda: planner.ego_status_encoder(action_input.status_feature)
    )

    batch_size, action_dim = vl_embeds.shape[0], planner.config.action_dim
    device, dtype = vl_embeds.device, vl_embeds.dtype
    current_actions = profiler.time(
        "init/randn_actions",
        lambda: torch.randn(
            (batch_size, planner.config.action_horizon, action_dim),
            device=device,
            dtype=dtype,
        ),
    )

    eval_min_std = getattr(planner, "eval_min_sampling_denoising_std", 0.0001)
    eval_randn_clip = getattr(planner, "eval_randn_clip_value", 1.0)
    for step_idx in range(planner.ddim_steps):
        t_batch = profiler.time(
            f"step{step_idx}/make_timestep_t",
            lambda step_idx=step_idx: planner.make_timesteps(batch_size, planner.ddim_t[step_idx], device),
        )
        index_batch = profiler.time(
            f"step{step_idx}/make_timestep_index",
            lambda step_idx=step_idx: planner.make_timesteps(batch_size, step_idx, device),
        )
        mean, logvar, _ = profile_p_mean_variance(
            planner,
            current_actions,
            t_batch,
            index_batch,
            vl_embeds,
            history_embeds,
            ego_embeds,
            profiler,
            prefix=f"step{step_idx}",
            use_kv_cache=False,
        )
        std = profiler.time(f"step{step_idx}/std_exp", lambda: torch.exp(0.5 * logvar).to(dtype))
        noise = profiler.time(
            f"step{step_idx}/randn_noise_clamp",
            lambda: torch.randn_like(current_actions).clamp_(-eval_randn_clip, eval_randn_clip),
        )
        std = profiler.time(f"step{step_idx}/std_clamp", lambda: std.clamp(min=eval_min_std))
        current_actions = profiler.time(
            f"step{step_idx}/update_actions", lambda: mean + std * noise
        )

    final_clip = getattr(planner, "final_action_clip_value", 1.0)
    if final_clip is not None:
        current_actions = profiler.time(
            "final/action_clip", lambda: current_actions.clamp_(-final_clip, final_clip)
        )
    final_actions = profiler.time("final/denorm_odo", lambda: planner.denorm_odo(current_actions))
    return BatchFeature(data={"pred_traj": final_actions})


def profile_get_action_fast_ddim(planner, vl_features, action_input, profiler):
    vl_embeds = profiler.time("condition/feature_encoder", lambda: planner.feature_encoder(vl_features))
    history_embeds = profiler.time(
        "condition/history_encoder_repeat",
        lambda: planner.his_traj_encoder(action_input.his_traj.unsqueeze(1)).repeat(
            1, planner.config.action_horizon, 1
        ),
    )
    ego_embeds = profiler.time(
        "condition/ego_encoder", lambda: planner.ego_status_encoder(action_input.status_feature)
    )

    batch_size, action_dim = vl_embeds.shape[0], planner.config.action_dim
    device, dtype = vl_embeds.device, vl_embeds.dtype
    current_actions = profiler.time(
        "init/randn_actions",
        lambda: torch.randn(
            (batch_size, planner.config.action_horizon, action_dim),
            device=device,
            dtype=dtype,
        ),
    )
    vl_mean = profiler.time(
        "condition/vl_mean_repeat",
        lambda: vl_embeds.mean(1).unsqueeze(1).repeat(1, planner.config.action_horizon, 1),
    )
    pos_embedding = None
    if hasattr(planner, "position_embedding"):
        pos_embedding = profiler.time(
            "condition/action_pos_embedding",
            lambda: planner.position_embedding(torch.arange(planner.config.action_horizon, device=device)),
        )
    eta_tensor = profiler.time("condition/eta_tensor", lambda: planner.eta(current_actions).unsqueeze(1))
    kv_cache = profiler.time(
        "condition/cross_attn_kv_cache",
        lambda: planner.model.build_cross_attention_kv_cache(vl_embeds),
    )
    t_batches = [
        profiler.time(
            f"step{i}/make_timestep_t",
            lambda i=i: planner.make_timesteps(batch_size, planner.ddim_t[i], device),
        )
        for i in range(planner.ddim_steps)
    ]
    index_batches = [
        profiler.time(
            f"step{i}/make_timestep_index",
            lambda i=i: planner.make_timesteps(batch_size, i, device),
        )
        for i in range(planner.ddim_steps)
    ]

    eval_min_std = getattr(planner, "eval_min_sampling_denoising_std", 0.0001)
    eval_randn_clip = getattr(planner, "eval_randn_clip_value", 1.0)
    for step_idx in range(planner.ddim_steps):
        mean, logvar, _ = profile_p_mean_variance(
            planner,
            current_actions,
            t_batches[step_idx],
            index_batches[step_idx],
            vl_embeds,
            history_embeds,
            ego_embeds,
            profiler,
            prefix=f"step{step_idx}",
            use_kv_cache=True,
            vl_mean=vl_mean,
            pos_embedding=pos_embedding,
            eta_tensor=eta_tensor,
            kv_cache=kv_cache,
        )
        std = profiler.time(f"step{step_idx}/std_exp", lambda: torch.exp(0.5 * logvar).to(dtype))
        noise = profiler.time(
            f"step{step_idx}/randn_noise_clamp",
            lambda: torch.randn_like(current_actions).clamp_(-eval_randn_clip, eval_randn_clip),
        )
        std = profiler.time(f"step{step_idx}/std_clamp", lambda: std.clamp(min=eval_min_std))
        current_actions = profiler.time(
            f"step{step_idx}/update_actions", lambda: mean + std * noise
        )

    final_clip = getattr(planner, "final_action_clip_value", 1.0)
    if final_clip is not None:
        current_actions = profiler.time(
            "final/action_clip", lambda: current_actions.clamp_(-final_clip, final_clip)
        )
    final_actions = profiler.time("final/denorm_odo", lambda: planner.denorm_odo(current_actions))
    return BatchFeature(data={"pred_traj": final_actions})


def profile_p_mean_variance(
    planner,
    x,
    t,
    index,
    vl_features,
    history_features,
    ego_features,
    profiler,
    prefix,
    use_kv_cache,
    vl_mean=None,
    pos_embedding=None,
    eta_tensor=None,
    kv_cache=None,
):
    model_dtype = next(planner.model.parameters()).dtype
    x = profiler.time(f"{prefix}/cast_action", lambda: x.to(model_dtype))
    action_features = profiler.time(f"{prefix}/action_encoder", lambda: planner.action_encoder(x, t))
    if hasattr(planner, "position_embedding"):
        if pos_embedding is None:
            action_features = profiler.time(
                f"{prefix}/action_pos_embedding",
                lambda: action_features
                + planner.position_embedding(torch.arange(action_features.shape[1], device=x.device)),
            )
        else:
            action_features = profiler.time(
                f"{prefix}/action_pos_embedding", lambda: action_features + pos_embedding
            )
    if vl_mean is None:
        vl_mean = profiler.time(
            f"{prefix}/vl_mean_repeat",
            lambda: vl_features.mean(1).unsqueeze(1).repeat(1, planner.config.action_horizon, 1),
        )
    fused = profiler.time(
        f"{prefix}/fusion_projector",
        lambda: planner.fusion_projector(torch.cat((history_features, vl_mean, action_features), dim=2)),
    )
    if use_kv_cache:
        model_output = profiler.time(
            f"{prefix}/dit_forward",
            lambda: planner.model.forward_with_kv_cache(
                fused,
                vl_features,
                ego_features,
                t,
                kv_cache=kv_cache,
            ),
        )
    else:
        model_output = profiler.time(
            f"{prefix}/dit_forward",
            lambda: planner.model(fused, vl_features, ego_features, t),
        )
    pred_noise = profiler.time(f"{prefix}/action_decoder", lambda: planner.action_decoder(model_output))

    alpha_t = profiler.time(f"{prefix}/extract_alpha", lambda: planner.extract(planner.ddim_alphas, index, x.shape))
    sqrt_one_minus_alpha = profiler.time(
        f"{prefix}/extract_sqrt_one_minus_alpha",
        lambda: planner.extract(planner.ddim_sqrt_one_minus_alphas, index, x.shape),
    )
    x_recon = profiler.time(
        f"{prefix}/x_recon",
        lambda: (x - sqrt_one_minus_alpha * pred_noise) / (alpha_t**0.5),
    )
    clip_value = getattr(planner, "denoised_clip_value", 1.0)
    x_recon = profiler.time(
        f"{prefix}/x_recon_clip", lambda: x_recon.clamp_(-clip_value, clip_value)
    )
    alpha_prev = profiler.time(
        f"{prefix}/extract_alpha_prev", lambda: planner.extract(planner.ddim_alphas_prev, index, x.shape)
    )
    pred_noise = profiler.time(
        f"{prefix}/recompute_pred_noise",
        lambda: (x - (alpha_t**0.5) * x_recon) / sqrt_one_minus_alpha,
    )
    eps_clip = getattr(planner, "eps_clip_value", None)
    if eps_clip is not None:
        pred_noise = profiler.time(
            f"{prefix}/pred_noise_clip", lambda: pred_noise.clamp_(-eps_clip, eps_clip)
        )
    if eta_tensor is None:
        etas = profiler.time(f"{prefix}/eta_tensor", lambda: planner.eta(x).unsqueeze(1))
    else:
        etas = eta_tensor
    sigma = profiler.time(
        f"{prefix}/sigma",
        lambda: (
            etas * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)) ** 0.5
        ).clamp_(min=1e-10),
    )
    pred_dir = profiler.time(
        f"{prefix}/pred_dir",
        lambda: (1.0 - alpha_prev - sigma**2).clamp(min=0).sqrt() * pred_noise,
    )
    model_mean = profiler.time(
        f"{prefix}/model_mean", lambda: (alpha_prev**0.5) * x_recon + pred_dir
    )
    model_logvar = profiler.time(f"{prefix}/model_logvar", lambda: torch.log(sigma**2 + 1e-20))
    return model_mean, model_logvar, x_recon


def install_dit_block_profiler(model, profiler):
    def timed_forward(this, hidden_states, conditioning, encoder_hidden_states=None, rotary_embedder=None, encoder_kv=None):
        block_idx = getattr(this, "_profile_idx")
        prefix = f"dit_block{block_idx:02d}"
        mod_params = profiler.time(f"{prefix}/adaLN_modulation", lambda: this.adaLN_modulation(conditioning))
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = mod_params.chunk(6, dim=1)

        normed = profiler.time(f"{prefix}/norm1", lambda: this.norm1(hidden_states))
        modulated = profiler.time(f"{prefix}/modulate_attn", lambda: this.modulate(normed, shift_attn, scale_attn))
        attn_output = profiler.time(
            f"{prefix}/attention",
            lambda: this.attn(
                modulated,
                encoder_hidden_states=encoder_hidden_states,
                rotary_embedder=rotary_embedder,
                encoder_kv=encoder_kv,
            ),
        )
        hidden_states = profiler.time(
            f"{prefix}/attn_residual", lambda: hidden_states + gate_attn.unsqueeze(1) * attn_output
        )
        normed = profiler.time(f"{prefix}/norm2", lambda: this.norm2(hidden_states))
        modulated = profiler.time(f"{prefix}/modulate_ffn", lambda: this.modulate(normed, shift_ffn, scale_ffn))
        ffn_output = profiler.time(f"{prefix}/ffn", lambda: this.ffn(modulated))
        return profiler.time(
            f"{prefix}/ffn_residual", lambda: hidden_states + gate_ffn.unsqueeze(1) * ffn_output
        )

    for idx, block in enumerate(model.transformer_blocks):
        block._profile_idx = idx
        block.forward = MethodType(timed_forward, block)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=["2b", "8b"], default="2b")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--prune-keep-ratio", type=float, default=0.50)
    parser.add_argument("--prune-method", type=str, default="uniform")
    parser.add_argument("--diffusion-steps", type=int, default=3)
    parser.add_argument("--image-max-num", type=int, default=6)
    parser.add_argument("--image-backend", type=str, default="opencv")
    parser.add_argument("--fast-ddim-action", action="store_true")
    parser.add_argument("--profile-blocks", action="store_true")
    args = parser.parse_args()

    cfg, agent = build_agent(args)
    if args.profile_blocks:
        block_profiler = CudaEventProfiler()
        install_dit_block_profiler(agent.action_head.model, block_profiler)
    else:
        block_profiler = None

    loader = SceneLoader(
        Path(cfg.navsim_log_path),
        Path(cfg.sensor_blobs_path),
        instantiate(cfg.train_test_split.scene_filter),
        agent.get_sensor_config(),
        load_image_path=True,
    )
    tokens = loader.tokens[: args.warmup + args.num_samples]
    agent_inputs = [loader.get_agent_input_from_token(token) for token in tokens]

    profiler = CudaEventProfiler()
    records = []
    with torch.inference_mode():
        for idx, agent_input in enumerate(agent_inputs):
            if idx == args.warmup:
                profiler = CudaEventProfiler()
                if block_profiler is not None:
                    block_profiler.values.clear()
            vl_features, action_inputs = make_diffusion_inputs(
                agent, agent_input, args.image_max_num, args.image_backend
            )
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = profile_get_action(
                agent.action_head,
                vl_features,
                action_inputs,
                profiler,
                use_fast_ddim=args.fast_ddim_action,
            )
            end.record()
            torch.cuda.synchronize()
            total_ms = start.elapsed_time(end)
            records.append({"index": idx, "warmup": idx < args.warmup, "profiled_total_ms": total_ms})
            print(json.dumps(records[-1], ensure_ascii=False), flush=True)

    measured_records = [record for record in records if not record["warmup"]]
    summary = {
        "model_size": args.model_size,
        "num_samples": args.num_samples,
        "warmup": args.warmup,
        "device": torch.cuda.get_device_name(0),
        "prune_keep_ratio": args.prune_keep_ratio,
        "prune_method": args.prune_method,
        "diffusion_steps": args.diffusion_steps,
        "image_max_num": args.image_max_num,
        "image_backend": args.image_backend,
        "fast_ddim_action": args.fast_ddim_action,
        "profile_blocks": args.profile_blocks,
        "profiled_total_ms": summarize(record["profiled_total_ms"] for record in measured_records),
        "segments": {
            key: value
            for key, value in profiler.summary().items()
            if not key.startswith("dit_block") and not any(f"step{i}/dit_forward" == key for i in range(args.diffusion_steps))
        },
        "dit_forward_by_step": {
            f"step{i}/dit_forward": profiler.summary().get(f"step{i}/dit_forward")
            for i in range(args.diffusion_steps)
        },
        "dit_blocks": block_profiler.summary() if block_profiler is not None else {},
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
