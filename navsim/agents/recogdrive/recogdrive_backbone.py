from typing import List, Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from .utils.conversation import get_conv_template

IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'

system_message = """
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

class RecogDriveBackbone(nn.Module):
    """
    A simplified vision-language model backbone with direct loading logic
    for different model architectures (InternVL, Qwen-VL).
    """
    def __init__(self,
                 model_type: str,
                 checkpoint_path: str,
                 device: str = "cuda",
                 prune_keep_ratio: float = 1.0,
                 prune_method: str = "tfps"):
        """
        Initializes and loads the specified model and its preprocessor/tokenizer.

        Args:
            model_type (str): The type of model to load. Supported: 'internvl', 'qwen'.
            checkpoint_path (str): The path to the model checkpoint.
            device (str): The device to load the model onto ('cuda', 'cpu').
        """
        super().__init__()

        self.model = None
        self.tokenizer = None  
        self.model_type = model_type.lower()
        self.device = device
        self.prune_keep_ratio = float(prune_keep_ratio)
        self.prune_method = prune_method.lower()
        self.last_input_seq_len = None

        print(f"Initializing backbone of type: '{self.model_type}' from path: '{checkpoint_path}'")

        if self.model_type == 'internvl':
            # --- Load InternVL Model and Tokenizer ---
            self.model = AutoModel.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                use_flash_attn=True,
                device_map=self.device
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_path,
                trust_remote_code=True,
                use_fast=False
            )
            # Load model-specific configuration
            self._configure_internvl()
            self.num_image_token = 256
            self.pruned_num_image_token = self._get_pruned_num_image_token(self.num_image_token)

        elif self.model_type == 'qwen':
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True
            )
            self.tokenizer = AutoProcessor.from_pretrained(
                checkpoint_path,
                trust_remote_code=True
            )
            
        else:
            raise ValueError(f"Unsupported model_type: '{self.model_type}'. Please choose 'internvl' or 'qwen'.")


        print(f"Backbone '{self.model_type}' loaded successfully on device '{self.device}'.")

    def _get_pruned_num_image_token(self, num_image_token: int) -> int:
        if self.prune_keep_ratio >= 1.0:
            return num_image_token
        if self.prune_keep_ratio <= 0.0:
            raise ValueError(f"prune_keep_ratio must be in (0, 1], got {self.prune_keep_ratio}")
        return max(1, int(round(num_image_token * self.prune_keep_ratio)))

    @staticmethod
    def _tfps_indices(hidden_states: torch.Tensor, keep_tokens: int) -> torch.Tensor:
        """Token farthest-point sampling using cosine distance."""
        num_tokens = hidden_states.shape[0]
        if keep_tokens >= num_tokens:
            return torch.arange(num_tokens, device=hidden_states.device)

        states = F.normalize(hidden_states.float(), p=2, dim=-1)
        distance = 1.0 - states @ states.T
        distance.fill_diagonal_(float("inf"))

        selected = torch.zeros(num_tokens, dtype=torch.bool, device=hidden_states.device)
        row_min = distance.min(dim=1).values
        first = row_min.argmax()
        selected[first] = True
        keep = [first]
        min_distance_to_selected = distance[first].clone()
        min_distance_to_selected[selected] = float("-inf")

        for _ in range(1, keep_tokens):
            next_idx = min_distance_to_selected.argmax()
            selected[next_idx] = True
            keep.append(next_idx)
            min_distance_to_selected = torch.minimum(min_distance_to_selected, distance[next_idx])
            min_distance_to_selected[selected] = float("-inf")

        return torch.stack(keep).sort().values

    def _select_visual_tokens(self, vit_embeds: torch.Tensor) -> torch.Tensor:
        if self.prune_keep_ratio >= 1.0:
            return vit_embeds

        keep_tokens = self._get_pruned_num_image_token(vit_embeds.shape[1])
        if self.prune_method in {"uniform_merge", "merge_uniform", "uniform-merge"}:
            return F.adaptive_avg_pool1d(
                vit_embeds.transpose(1, 2).contiguous(),
                keep_tokens,
            ).transpose(1, 2).contiguous()
        if self.prune_method in {"uniform", "stride"}:
            indices = torch.linspace(
                0,
                vit_embeds.shape[1] - 1,
                keep_tokens,
                device=vit_embeds.device,
            ).round().long().unique()
            if indices.numel() < keep_tokens:
                fallback = torch.arange(vit_embeds.shape[1], device=vit_embeds.device)
                indices = torch.cat([indices, fallback[~torch.isin(fallback, indices)]])[:keep_tokens]
            return vit_embeds[:, indices.sort().values, :]
        if self.prune_method not in {"tfps", "farway", "farthest"}:
            raise ValueError(f"Unsupported prune_method: {self.prune_method}")

        selected = [embeds[self._tfps_indices(embeds, keep_tokens)] for embeds in vit_embeds]
        return torch.stack(selected, dim=0)

    def _forward_internvl_with_pruning(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        image_flags: torch.Tensor,
        output_hidden_states: bool = True,
        return_dict: bool = True,
    ) -> CausalLMOutputWithPast:
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict
        model_dtype = next(self.model.parameters()).dtype

        image_flags = image_flags.squeeze(-1) if image_flags.ndim > 1 else image_flags
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids).clone()
        vit_embeds = self.model.extract_feature(pixel_values.to(model_dtype))
        vit_embeds = vit_embeds[image_flags == 1]
        vit_embeds = self._select_visual_tokens(vit_embeds)

        batch_size, seq_len, hidden_dim = input_embeds.shape
        flat_input_embeds = input_embeds.reshape(batch_size * seq_len, hidden_dim)
        flat_input_ids = input_ids.reshape(batch_size * seq_len)
        selected = flat_input_ids == self.img_context_token_id

        flat_vit_embeds = vit_embeds.reshape(-1, hidden_dim)
        num_selected = int(selected.sum().item())
        if num_selected != flat_vit_embeds.shape[0]:
            raise RuntimeError(
                f"IMG_CONTEXT token count ({num_selected}) does not match visual token count "
                f"({flat_vit_embeds.shape[0]})."
            )
        flat_input_embeds[selected] = flat_vit_embeds.to(flat_input_embeds.device)
        input_embeds = flat_input_embeds.reshape(batch_size, seq_len, hidden_dim)

        outputs = self.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return outputs
        return CausalLMOutputWithPast(
            loss=None,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _configure_internvl(self):
        """Applies specific configurations required for the InternVL model."""
        self.model.system_message = system_message
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = self.img_context_token_id
        print("InternVL model configured.")
    
    def forward(self, pixel_values: torch.Tensor, questions: List[str], num_patches_list: List[int]):
        if not self.model:
            raise RuntimeError("Backbone model has not been initialized. Call initialize() on the agent first.")
        
        model_dtype = next(self.model.parameters()).dtype

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            
            template = get_conv_template("internvl2_5")
            template.system_message = system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.pruned_num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)
        self.tokenizer.padding_side = 'left'
        if self.prune_keep_ratio < 1.0:
            model_inputs = self.tokenizer(queries, return_tensors='pt', padding=True)
        else:
            model_inputs = self.tokenizer(queries, return_tensors='pt', padding='max_length', max_length=2800)
        device = torch.device('cuda')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        self.last_input_seq_len = int(input_ids.shape[1])

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        
        num_patches = pixel_values.size(0)
        image_flags = torch.tensor([1] * num_patches, dtype=torch.long)


        if self.prune_keep_ratio < 1.0:
            return self._forward_internvl_with_pruning(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    image_flags=image_flags,
                    output_hidden_states=True,
                    return_dict=True,
            )

        return self.model(
                pixel_values=pixel_values.to(model_dtype),
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                image_flags=image_flags.squeeze(-1),
                output_hidden_states=True,
                return_dict=True,
        )

    
