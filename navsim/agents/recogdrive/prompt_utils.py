import torch


FULL_SYSTEM_MESSAGE = """
You are a vehicle trajectory prediction model for autonomous driving. Your task is to predict the ego vehicle's 4-second trajectory based on the following inputs: multi-view images from 8 cameras, ego vehicle states (position), and discrete navigation commands. The input provides a 2-second history, and your output should ensure a safe trajectory for the next 4 seconds. Your predictions must adhere to the following metrics:
1. **No at-fault Collisions (NC)**: Avoid collisions with other objects/vehicles.
2. **Drivable Area Compliance (DAC)**: Stay within the drivable area.
3. **Time to Collision (TTC)**: Maintain a safe distance from other vehicles.
4. **Ego Progress (EP)**: Ensure the ego vehicle moves forward without being stuck.
5. **Comfort (C)**: Avoid sharp turns and sudden decelerations.
6. **Driving Direction Compliance (DDC)**: Align with the intended driving direction.
For evaluation, use the **PDM Score**, which combines these metrics: **PDM Score** = NC * DAC * (5*TTC + 5*EP + 2*C + 0*DDC) / 12.
Your predictions will be evaluated through a non-reactive 4-second simulation with an LQR controller and background actors following their recorded trajectories. The better your predictions, the higher your score.
"""

COMPACT_SYSTEM_MESSAGE = (
    "You are an autonomous driving trajectory predictor. "
    "Given the front-view image, ego motion history, and navigation command, "
    "predict a safe 4-second ego trajectory."
)

PROMPT_VARIANTS = {"full", "compact_v1"}


def format_number(n, decimal_places=2):
    return f"{n:+.{decimal_places}f}" if abs(round(n, decimal_places)) > 1e-2 else "0.0"


def get_system_message(prompt_variant: str = "full") -> str:
    if prompt_variant == "full":
        return FULL_SYSTEM_MESSAGE
    if prompt_variant == "compact_v1":
        return COMPACT_SYSTEM_MESSAGE
    raise ValueError(f"Unsupported prompt_variant: {prompt_variant}")


def build_recogdrive_question(
    history_trajectory: torch.Tensor,
    high_command_one_hot: torch.Tensor,
    prompt_variant: str = "full",
) -> str:
    if prompt_variant not in PROMPT_VARIANTS:
        raise ValueError(f"Unsupported prompt_variant: {prompt_variant}")

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

    if prompt_variant == "compact_v1":
        output_requirements = "\nOutput 8 future poses as [PT, (x,y,heading), ...]."
    else:
        output_requirements = (
            "\nOutput requirements:\n- Predict 8 future trajectory points\n"
            "- Each point format: (x:float, y:float, heading:float)\n"
            "- Use [PT, ...] to encapsulate the trajectory\n"
            "- Maintain numerical precision to 2 decimal places"
        )
    return f"{prompt}{output_requirements}"
