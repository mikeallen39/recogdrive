import argparse
import importlib.metadata
import json
import time
from pathlib import Path
from statistics import mean, median

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
from navsim.agents.recogdrive.recogdrive_features import format_number
from navsim.agents.recogdrive.utils.internvl_preprocess import load_image
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


def run_one(agent, agent_input):
    feature_wall_ms, features = wall_time_ms(lambda: make_agent_input_features(agent, agent_input))

    history_trajectory = features["history_trajectory"].cuda()
    high_command_one_hot = features["high_command_one_hot"].cuda()
    status_feature = features["status_feature"].cuda()
    image_path_tensor = features["image_path_tensor"]

    image_paths = agent._decode_paths_from_tensor(image_path_tensor)
    image_wall_ms, pixel_values_list = wall_time_ms(lambda: [load_image(path) for path in image_paths])
    num_patches_list = [pixel_values.shape[0] for pixel_values in pixel_values_list]

    h2d_ms, pixel_values_cat = cuda_time_ms(lambda: torch.cat(pixel_values_list, dim=0).cuda())
    questions = [build_question(history_trajectory[0], high_command_one_hot[0])]

    vlm_ms, outputs = cuda_time_ms(
        lambda: agent.backbone(pixel_values_cat, questions, num_patches_list=num_patches_list)
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
    diffusion_ms, predictions = cuda_time_ms(
        lambda: agent.action_head.get_action(last_hidden_state, action_inputs)
    )
    post_wall_ms, poses = wall_time_ms(lambda: predictions["pred_traj"].float().cpu().squeeze(0))

    def end_to_end_gpu_only():
        pixel_values_cat_e2e = torch.cat(pixel_values_list, dim=0).cuda()
        outputs_e2e = agent.backbone(pixel_values_cat_e2e, questions, num_patches_list=num_patches_list)
        hidden_e2e = outputs_e2e.hidden_states[-1]
        if hidden_e2e.ndim == 2:
            hidden_e2e = hidden_e2e.unsqueeze(0)
        hidden_e2e = hidden_e2e.to(model_dtype)
        return agent.action_head.get_action(hidden_e2e, action_inputs)

    e2e_gpu_ms, _ = cuda_time_ms(end_to_end_gpu_only)

    return {
        "feature_wall_ms": feature_wall_ms,
        "image_preprocess_wall_ms": image_wall_ms,
        "image_h2d_cuda_ms": h2d_ms,
        "vlm_cuda_ms": vlm_ms,
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
            ],
        )

    agent: ReCogDriveAgent = instantiate(cfg.agent)
    agent.initialize()
    agent.eval()

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
            record = run_one(agent, agent_input)
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
