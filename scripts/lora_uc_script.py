#
# Wacky fix for Composable-Diffusion with Lora (Slow and memory inefficient)
# Based on https://github.com/opparco/stable-diffusion-webui-composable-lora
#
import os, sys
import re
from pathlib import Path
from typing import Dict, List, Union
import torch
from torch import Tensor
import gradio as gr
from itertools import chain

import modules.scripts as scripts
from modules import script_callbacks, paths_internal, devices, extra_networks
from modules.processing import StableDiffusionProcessing

extensions_builtin_path = Path(paths_internal.extensions_builtin_dir)
extensions_path = Path(paths_internal.extensions_dir)
lora_path = extensions_builtin_path / "Lora"
sys.path.insert(0, str(lora_path.absolute()))
import lora as _lora

has_lycoris = False
lycoris_paths = [
    path for path in chain(extensions_path.glob("**/lycoris.py"),
                           extensions_builtin_path.glob("**/lycoris.py"))
]
if lycoris_paths:
    lycoris_path = lycoris_paths[0].parent
    sys.path.insert(0, str(lycoris_path.absolute()))
    has_lycoris = True
    import lycoris as _lycoris

re_AND = re.compile(r"\bAND\b")


def unload():
    _lora.lora_reset_cached_weight = _lora.lora_reset_cached_weight_before_uc
    _lora.lora_Linear_forward = _lora.lora_Linear_forward_before_uc
    _lora.lora_Conv2d_forward = _lora.lora_Conv2d_forward_before_uc
    # _lora.lora_MultiheadAttention_forward = _lora.lora_MultiheadAttention_forward_before_uc
    if has_lycoris:
        _lycoris.lyco_reset_cached_weight = _lycoris.lyco_reset_cached_weight_before_uc
        _lycoris.lyco_Linear_forward = _lycoris.lyco_Linear_forward_before_uc
        _lycoris.lyco_Conv2d_forward = _lycoris.lyco_Conv2d_forward_before_uc


def load():
    _lora.lora_reset_cached_weight = lora_reset_cached_weight
    _lora.lora_Linear_forward = lora_Linear_forward
    _lora.lora_Conv2d_forward = lora_Conv2d_forward
    # _lora.lora_MultiheadAttention_forward = lora_MultiheadAttention_forward
    if has_lycoris:
        _lycoris.lyco_reset_cached_weight = lyco_reset_cached_weight
        _lycoris.lyco_Linear_forward = lyco_Linear_forward
        _lycoris.lyco_Conv2d_forward = lyco_Conv2d_forward


if not hasattr(_lora, 'lora_reset_cached_weight_before_uc'):
    _lora.lora_reset_cached_weight_before_uc = _lora.lora_reset_cached_weight

if not hasattr(_lora, 'lora_Linear_forward_before_uc'):
    _lora.lora_Linear_forward_before_uc = _lora.lora_Linear_forward

if not hasattr(_lora, 'lora_Conv2d_forward_before_uc'):
    _lora.lora_Conv2d_forward_before_uc = _lora.lora_Conv2d_forward

# if not hasattr(_lora, 'lora_MultiheadAttention_forward_before_uc'):
#     _lora.lora_MultiheadAttention_forward_before_uc = _lora.lora_MultiheadAttention_forward

if has_lycoris:
    if not hasattr(_lycoris, 'lyco_reset_cached_weight_before_uc'):
        _lycoris.lyco_reset_cached_weight_before_uc = _lycoris.lyco_reset_cached_weight

    if not hasattr(_lycoris, 'lyco_Linear_forward_before_uc'):
        _lycoris.lyco_Linear_forward_before_uc = _lycoris.lyco_Linear_forward

    if not hasattr(_lycoris, 'lyco_Conv2d_forward_before_uc'):
        _lycoris.lyco_Conv2d_forward_before_uc = _lycoris.lyco_Conv2d_forward

script_callbacks.on_script_unloaded(unload)


class LoraUcScript(scripts.Script):

    def title(self):
        return "Disable Lora for UC"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        enabled = gr.Checkbox(value=False, label="Disable Lora for UC")

        self.infotext_fields = [(enabled, lambda d: "Disable Lora for UC" in d)
                                ]
        return [enabled]

    def process(self, p: StableDiffusionProcessing, enabled: bool):
        global is_enabled
        global num_batches

        is_enabled = enabled
        if has_lycoris: # Helps to override extensions-builtin LoRA load priority
            torch.nn.Linear.forward = _lycoris.lyco_Linear_forward
            torch.nn.Linear._load_from_state_dict = _lycoris.lyco_Linear_load_state_dict
            torch.nn.Conv2d.forward = _lycoris.lyco_Conv2d_forward
            torch.nn.Conv2d._load_from_state_dict = _lycoris.lyco_Conv2d_load_state_dict
            torch.nn.MultiheadAttention.forward = _lycoris.lyco_MultiheadAttention_forward
            torch.nn.MultiheadAttention._load_from_state_dict = _lycoris.lyco_MultiheadAttention_load_state_dict
        if is_enabled:
            p.extra_generation_params["Disable Lora for UC"] = True
            prompt = p.all_prompts[0]
            num_batches = p.batch_size
            load_prompt_loras(prompt)

    def process_batch(self, p, *args, **kwargs):
        reset_counters()


def reset_counters():
    global text_model_encoder_counter
    global diffusion_model_counter
    debug_layers.clear()

    # reset counter to uc head
    text_model_encoder_counter = -1
    diffusion_model_counter = 0


def load_prompt_loras(prompt: str):
    prompt_loras.clear()
    subprompts = re_AND.split(prompt)
    tmp_prompt_loras = []
    for i, subprompt in enumerate(subprompts):
        loras = {}
        _, extra_network_data = extra_networks.parse_prompt(subprompt)
        for params in extra_network_data['lora']:
            name = params.items[0]
            multiplier = float(params.items[1]) if len(
                params.items) > 1 else 1.0
            loras[name] = multiplier

        if has_lycoris:
            for params in extra_network_data['lyco']:
                name = params.items[0]
                multiplier = float(params.items[1]) if len(
                    params.items) > 1 else 1.0
                loras[name] = multiplier

        tmp_prompt_loras.append(loras)
    prompt_loras.extend(tmp_prompt_loras * num_batches)


def lora_apply_weights(self: Union[torch.nn.Conv2d, torch.nn.Linear,
                                   torch.nn.MultiheadAttention]):
    """
    Applies the currently selected set of Loras to the weights of torch layer self.
    If weights already have this particular set of loras applied, does nothing.
    If not, restores orginal weights from backup and alters weights according to loras.
    """

    lora_layer_name = getattr(self, 'lora_layer_name', None)
    if lora_layer_name is None:
        return

    current_names = getattr(self, "lora_current_names", ())
    wanted_names = tuple((x.name, x.multiplier) for x in _lora.loaded_loras)

    weights_backup = getattr(self, "lora_weights_backup", None)
    weights_adjusted = getattr(self, "lora_weights_adjusted", None)

    if weights_backup is None:
        if isinstance(self, torch.nn.MultiheadAttention):
            weights_backup = (self.in_proj_weight.to(devices.device,
                                                     copy=True),
                              self.out_proj.weight.to(devices.device,
                                                      copy=True))
        else:
            weights_backup = self.weight.to(devices.device, copy=True)

        self.lora_weights_backup = weights_backup

    if current_names != wanted_names:
        if weights_backup is not None:
            if isinstance(self, torch.nn.MultiheadAttention):
                self.in_proj_weight.copy_(weights_backup[0])
                self.out_proj.weight.copy_(weights_backup[1])
            else:
                self.weight.copy_(weights_backup)

        for lora in _lora.loaded_loras:
            module = lora.modules.get(lora_layer_name, None)
            if module is not None and hasattr(self, 'weight'):
                self.weight += _lora.lora_calc_updown(lora, module,
                                                      self.weight)
                continue

            module_q = lora.modules.get(lora_layer_name + "_q_proj", None)
            module_k = lora.modules.get(lora_layer_name + "_k_proj", None)
            module_v = lora.modules.get(lora_layer_name + "_v_proj", None)
            module_out = lora.modules.get(lora_layer_name + "_out_proj", None)

            if isinstance(
                    self, torch.nn.MultiheadAttention
            ) and module_q and module_k and module_v and module_out:
                updown_q = _lora.lora_calc_updown(lora, module_q,
                                                  self.in_proj_weight)
                updown_k = _lora.lora_calc_updown(lora, module_k,
                                                  self.in_proj_weight)
                updown_v = _lora.lora_calc_updown(lora, module_v,
                                                  self.in_proj_weight)
                updown_qkv = torch.vstack([updown_q, updown_k, updown_v])

                self.in_proj_weight += updown_qkv
                self.out_proj.weight += _lora.lora_calc_updown(
                    lora, module_out, self.out_proj.weight)
                continue

            if module is None:
                continue

            print(
                f'failed to calculate lora weights for layer {lora_layer_name}'
            )

        if isinstance(self, torch.nn.MultiheadAttention):
            weights_adjusted = (self.in_proj_weight.to(devices.device,
                                                       copy=True),
                                self.out_proj.weight.to(devices.device,
                                                        copy=True))
        else:
            weights_adjusted = self.weight.to(devices.device, copy=True)

        setattr(self, "lora_weights_adjusted", weights_adjusted)

        setattr(self, "lora_current_names", wanted_names)

    if weights_adjusted is None:
        if isinstance(self, torch.nn.MultiheadAttention):
            weights_adjusted = (self.in_proj_weight.to(devices.device,
                                                       copy=True),
                                self.out_proj.weight.to(devices.device,
                                                        copy=True))
        else:
            weights_adjusted = self.weight.to(devices.device, copy=True)

        setattr(self, "lora_weights_adjusted", weights_adjusted)


def set_adjusted_weights(self: Union[torch.nn.Conv2d, torch.nn.Linear,
                                     torch.nn.MultiheadAttention]):
    if isinstance(self, torch.nn.MultiheadAttention):
        self.in_proj_weight.data = self.lora_weights_adjusted[0]
        self.out_proj.weight.data = self.lora_weights_adjusted[1]
    else:
        self.weight.data = self.lora_weights_adjusted


def set_backup_weights(self: Union[torch.nn.Conv2d, torch.nn.Linear,
                                   torch.nn.MultiheadAttention]):
    if isinstance(self, torch.nn.MultiheadAttention):
        self.in_proj_weight.data = self.lora_weights_backup[0]
        self.out_proj.weight.data = self.lora_weights_backup[1]
    else:
        self.weight.data = self.lora_weights_backup


def lora_forward(self: Union[torch.nn.Linear, torch.nn.Conv2d,
                             torch.nn.MultiheadAttention], input: Tensor):
    global text_model_encoder_counter
    global diffusion_model_counter
    loaded_loras = len(_lora.loaded_loras)
    if has_lycoris:
        loaded_loras += len(_lycoris.loaded_lycos)

    orig_layer: Union[torch.nn.Linear, torch.nn.Conv2d,
                      torch.nn.MultiheadAttention] = None  # type: ignore
    if isinstance(self, torch.nn.Linear):
        orig_layer = torch.nn.Linear_forward_before_lora
    elif isinstance(self, torch.nn.Conv2d):
        orig_layer = torch.nn.Conv2d_forward_before_lora
    # elif isinstance(self, torch.nn.MultiheadAttention):
    #     orig_layer = torch.nn.MultiheadAttention_forward_before_lora
    else:
        raise ValueError("Unsupported layer")

    if not is_enabled or loaded_loras == 0:
        return orig_layer(self, input)

    lora_layer_name: str | None = getattr(self, 'lora_layer_name', None)
    if lora_layer_name is None:
        return orig_layer(self, input)

    # debug_layers.append(lora_layer_name)

    num_loras = loaded_loras
    if text_model_encoder_counter == -1:
        text_model_encoder_counter = len(prompt_loras) * num_loras

    num_prompts = len(prompt_loras)

    if lora_layer_name.startswith(
            "transformer_"):  # "transformer_text_model_encoder_"
        #
        if 0 <= text_model_encoder_counter // num_loras < num_prompts:
            # c
            set_adjusted_weights(self)
        else:
            # uc
            set_backup_weights(self)

        if lora_layer_name.endswith(
                "_11_mlp_fc2"):  # last lora_layer_name of text_model_encoder
            text_model_encoder_counter += 1
            # c1 c1 c2 c2 .. .. uc uc
            if text_model_encoder_counter == (num_prompts +
                                              num_batches) * num_loras:
                text_model_encoder_counter = 0

    elif lora_layer_name.startswith("diffusion_model_"):  # "diffusion_model_"
        if input.shape[0] == num_batches * num_prompts + num_batches:
            # tensor.shape[1] == uncond.shape[1]
            tensor_off = 0
            uncond_off = num_batches * num_prompts
            input_split = list(torch.split(input, 1, dim=0))
            for b in range(num_batches):
                # c
                set_adjusted_weights(self)
                for p, loras in enumerate(prompt_loras):
                    input_split[tensor_off] = orig_layer(
                        self, input_split[tensor_off])
                    tensor_off += 1
                # uc
                set_backup_weights(self)
                input_split[uncond_off] = orig_layer(self,
                                                     input_split[uncond_off])
                uncond_off += 1
            set_adjusted_weights(self)
            return torch.cat(input_split, dim=0)

        else:
            # tensor.shape[1] != uncond.shape[1]
            cur_num_prompts = input.shape[0]
            base = (diffusion_model_counter //
                    cur_num_prompts) // num_loras * cur_num_prompts
            diff_result = None

            if 0 <= base < num_prompts:
                input_split = list(torch.split(input, 1, dim=0))
                # c
                for off in range(cur_num_prompts):
                    if base + off < num_prompts:
                        set_adjusted_weights(self)
                    else:
                        set_backup_weights(self)
                    input_split[off] = orig_layer(self, input_split[off])
                diff_result = torch.cat(input_split, dim=0)
            else:
                # uc
                set_backup_weights(self)
                diff_result = orig_layer(self, input)

            if lora_layer_name.endswith(
                    "_11_1_proj_out"
            ):  # last lora_layer_name of diffusion_model
                diffusion_model_counter += cur_num_prompts
                # c1 c2 .. uc
                if diffusion_model_counter >= (num_prompts +
                                               num_batches) * num_loras:
                    diffusion_model_counter = 0
            set_adjusted_weights(self)
            return diff_result
    else:
        set_adjusted_weights(self)

    result = orig_layer(self, input)
    set_adjusted_weights(self)
    return result


def lora_reset_cached_weight(self: Union[torch.nn.Conv2d, torch.nn.Linear]):
    setattr(self, "lora_current_names", ())
    setattr(self, "lora_weights_backup", None)
    setattr(self, "lora_weights_adjusted", None)


def lora_Linear_forward(self: torch.nn.Linear, input: Tensor):
    with torch.no_grad():
        lora_apply_weights(self)
        return lora_forward(self, input)


def lora_Conv2d_forward(self: torch.nn.Conv2d, input: Tensor):
    with torch.no_grad():
        lora_apply_weights(self)
        return lora_forward(self, input)


if has_lycoris:

    def lyco_apply_weights(self: Union[torch.nn.Conv2d, torch.nn.Linear,
                                       torch.nn.MultiheadAttention]):
        """
        Applies the currently selected set of Lycos to the weights of torch layer self.
        If weights already have this particular set of lycos applied, does nothing.
        If not, restores orginal weights from backup and alters weights according to lycos.
        """

        lyco_layer_name = getattr(self, 'lyco_layer_name', None)
        if lyco_layer_name is None:
            return

        current_names = getattr(self, "lyco_current_names", ())
        lora_prev_names = getattr(self, "lora_prev_names", ())
        lora_names = getattr(self, "lora_current_names", ())
        wanted_names = tuple(
            (x.name, x.te_multiplier, x.unet_multiplier, x.dyn_dim)
            for x in _lycoris.loaded_lycos)

        weights_backup = getattr(self, "lora_weights_backup", None)
        weights_adjusted = getattr(self, "lora_weights_adjusted", None)

        if weights_backup is None:
            # print('lyco save weight')
            if isinstance(self, torch.nn.MultiheadAttention):
                weights_backup = (self.in_proj_weight.to(devices.device,
                                                         copy=True),
                                  self.out_proj.weight.to(devices.device,
                                                          copy=True))
            else:
                weights_backup = self.weight.to(devices.device, copy=True)
            self.lora_weights_backup = weights_backup

        if current_names != wanted_names or lora_prev_names != lora_names:
            if weights_backup is not None and lora_names == lora_prev_names:
                # print('lyco restore weight')
                if isinstance(self, torch.nn.MultiheadAttention):
                    self.in_proj_weight.copy_(weights_backup[0])
                    self.out_proj.weight.copy_(weights_backup[1])
                else:
                    self.weight.copy_(weights_backup)

            for lyco in _lycoris.loaded_lycos:
                module = lyco.modules.get(lyco_layer_name, None)
                multiplier = (lyco.te_multiplier
                              if 'transformer' in lyco_layer_name[:20] else
                              lyco.unet_multiplier)
                if module is not None and hasattr(self, 'weight'):
                    # print(lyco_layer_name, multiplier)
                    self.weight += _lycoris.lyco_calc_updown(
                        lyco, module, self.weight) * multiplier
                    continue

                module_q = lyco.modules.get(lyco_layer_name + "_q_proj", None)
                module_k = lyco.modules.get(lyco_layer_name + "_k_proj", None)
                module_v = lyco.modules.get(lyco_layer_name + "_v_proj", None)
                module_out = lyco.modules.get(lyco_layer_name + "_out_proj",
                                              None)

                if isinstance(
                        self, torch.nn.MultiheadAttention
                ) and module_q and module_k and module_v and module_out:
                    updown_q = _lycoris.lyco_calc_updown(
                        lyco, module_q, self.in_proj_weight)
                    updown_k = _lycoris.lyco_calc_updown(
                        lyco, module_k, self.in_proj_weight)
                    updown_v = _lycoris.lyco_calc_updown(
                        lyco, module_v, self.in_proj_weight)
                    updown_qkv = torch.vstack([updown_q, updown_k, updown_v])

                    self.in_proj_weight += updown_qkv
                    self.out_proj.weight += _lycoris.lyco_calc_updown(
                        lyco, module_out, self.out_proj.weight)
                    continue

                if module is None:
                    continue

                print(
                    f'failed to calculate lyco weights for layer {lyco_layer_name}'
                )

            if isinstance(self, torch.nn.MultiheadAttention):
                weights_adjusted = (self.in_proj_weight.to(devices.device,
                                                           copy=True),
                                    self.out_proj.weight.to(devices.device,
                                                            copy=True))
            else:
                weights_adjusted = self.weight.to(devices.device, copy=True)

            setattr(self, "lora_weights_adjusted", weights_adjusted)

            setattr(self, "lora_prev_names", lora_names)
            setattr(self, "lyco_current_names", wanted_names)

        if weights_adjusted is None:
            if isinstance(self, torch.nn.MultiheadAttention):
                weights_adjusted = (self.in_proj_weight.to(devices.device,
                                                           copy=True),
                                    self.out_proj.weight.to(devices.device,
                                                            copy=True))
            else:
                weights_adjusted = self.weight.to(devices.device, copy=True)

            setattr(self, "lora_weights_adjusted", weights_adjusted)

    def lyco_reset_cached_weight(self: Union[torch.nn.Conv2d,
                                             torch.nn.Linear]):
        setattr(self, "lyco_current_names", ())
        setattr(self, "lora_weights_backup", None)
        setattr(self, "lora_weights_adjusted", None)

    def lyco_Linear_forward(self: torch.nn.Linear, input: Tensor):
        lyco_apply_weights(self)
        return lora_forward(self, input)

    def lyco_Conv2d_forward(self: torch.nn.Conv2d, input: Tensor):
        lyco_apply_weights(self)
        return lora_forward(self, input)


# def lora_MultiheadAttention_forward(self: torch.nn.MultiheadAttention, *args, **kwargs):
#     with torch.no_grad():
#         lora_apply_weights(self)
#         return torch.nn.MultiheadAttention_forward_before_lora(self, *args, **kwargs)

is_enabled = False

num_batches: int = 0
prompt_loras: List[Dict[str, float]] = []
text_model_encoder_counter: int = -1
diffusion_model_counter: int = 0
debug_layers = []

load()
