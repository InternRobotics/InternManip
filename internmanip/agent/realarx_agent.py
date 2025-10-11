from collections import deque
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import torch

from huggingface_hub import snapshot_download

from internmanip.agent.base import BaseAgent
from internmanip.configs import AgentCfg
from internmanip.configs.dataset.data_config import DATA_CONFIG_MAP
from internmanip.dataset.embodiment_tags import EmbodimentTag
from internmanip.dataset.schema import DatasetMetadata
from internmanip.dataset.transform.base import ComposedModalityTransform
from internmanip.utils.agent_utils.io_utils import unsqueeze_dict_values, squeeze_dict_values


class RealArxAgent(BaseAgent):
    def __init__(self, config: AgentCfg):
        super().__init__(config)
        if torch.cuda.is_available():
            self.policy_model = self.policy_model.cuda()

        model_transform, observation_indices, action_indices = self.policy_model.config.transform()
        self.data_config = config.agent_settings['data_config']
        data_config_cls = DATA_CONFIG_MAP[self.data_config]
        transforms = data_config_cls.transform()
        if model_transform is not None:
            transforms.append(model_transform)
        self.transforms = ComposedModalityTransform(transforms=transforms)
        self.transforms.eval()
        self.embodiment_tag = EmbodimentTag(config.agent_settings['embodiment_tag'])
        self._load_metadata(config)

        self.pred_action_horizon = config.agent_settings.get('pred_action_horizon', 16)
        self.action_ensemble = config.agent_settings.get('action_ensemble', False)
        self.adaptive_ensemble_alpha = config.agent_settings.get('adaptive_ensemble_alpha', 0.5)
        self.ensembler_list = []


    def step(self, inputs: list[dict]) -> list[dict]:
        outputs = []
        while len(self.ensembler_list) < len(inputs):
            if self.action_ensemble:
                self.ensembler_list.append(
                    AdaptiveEnsembler(
                        pred_action_horizon=self.pred_action_horizon,
                        adaptive_ensemble_alpha=self.adaptive_ensemble_alpha,
                    )
                )
            else:
                self.ensembler_list.append(None)

        # GenManip has only one environment, so this loop will iterate only once
        for env, input in enumerate(inputs):
            if input == {}:
                outputs.append({})
                continue

            if input['robot']['step'] == 0:
                if self.action_ensemble:
                    self.ensembler_list[env].reset()
                print('instruction', input['robot']['instruction'])

            converted_input = self.convert_input(input)
            if self.action_ensemble:
                model_pred = self.inference(converted_input)
                pred_action = self.ensembler_list[env].ensemble_action(model_pred[0])
                pred_action = torch.tensor(pred_action)
            else:
                if input['robot']['step'] % self.pred_action_horizon == 0:
                    model_pred = self.inference(converted_input)
                    self.ensembler_list[env] = model_pred[0]
                pred_action = self.ensembler_list[env][input['robot']['step'] % self.pred_action_horizon]
            unnormalized_output = self.transforms.unapply({'action': pred_action})
            squeezed_output = squeeze_dict_values(unnormalized_output)
            converted_output = self.convert_output(squeezed_output, converted_input)
            outputs.append(converted_output)

        return outputs

    def reset(self):
        self.ensembler_list = []

    def inference(self, input: dict):
        unsqueezed_input = unsqueeze_dict_values(input)
        normalized_input = self.transforms(unsqueezed_input)
        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            model_pred = self.policy_model.inference(normalized_input)['action_pred'].float().cpu()
        return model_pred

    def convert_input(self, input: dict):
        if self.data_config == 'arx':
            left_arm_joint_indices = [12, 14, 16, 18, 20, 22] #todo
            right_arm_joint_indices = [13, 15, 17, 19, 21, 23]
            left_gripper_joint_indices = [24]
            right_gripper_joint_indices = [26]
            left_arm_qpos = [input['robot']['joints_state']['positions'][idx] for idx in left_arm_joint_indices]
            right_arm_qpos = [input['robot']['joints_state']['positions'][idx] for idx in right_arm_joint_indices]
            left_gripper_qpos_state = [input['robot']['joints_state']['positions'][idx] for idx in left_gripper_joint_indices]
            right_gripper_qpos_state = [input['robot']['joints_state']['positions'][idx] for idx in right_gripper_joint_indices]
            # self._debug_print_data(left_arm_qpos, title='Left Arm Qpos')
            # self._debug_print_data(left_gripper_qpos_state, title='Left Gripper Qpos State')
            # self._debug_print_data(right_arm_qpos, title='Right Arm Qpos')
            # self._debug_print_data(right_gripper_qpos_state, title='Right Gripper Qpos State')
            converted_data = {
                'video.hand_left': np.array([input['robot']['sensors']['left_camera']['rgb']]),
                'video.hand_right': np.array([input['robot']['sensors']['right_camera']['rgb']]),
                'video.head': np.array([input['robot']['sensors']['top_camera']['rgb']]),
                'state.left_joint': np.array([left_arm_qpos]),
                'state.left_gripper': np.array([left_gripper_qpos_state]),
                'state.right_joint': np.array([right_arm_qpos]),
                'state.right_gripper': np.array([right_gripper_qpos_state]),
                'annotation.human.action.task_description': [input['robot']['instruction']],
            }
        else:
            raise ValueError(f'Unsupported data config class: {self.data_config}')
        return converted_data

    def convert_output(self, output: dict, input: dict):
        if self.data_config == 'arx':
            left_arm_action = (output['action.left_joint_delta'] + input['state.left_joint'][0]).tolist()
            right_arm_action = (output['action.right_joint_delta'] + input['state.right_joint'][0]).tolist()
            left_gripper_action = output['action.left_gripper'] #todo
            right_gripper_action = output['action.right_gripper']
            #todo
            converted_data = { 
                'left_arm_action': left_arm_action,
                'right_arm_action': right_arm_action,
                'left_gripper_action': left_gripper_action,
                'right_gripper_action': right_gripper_action,
            }
        else:
            raise ValueError(f'Unsupported data config class: {self.data_config}')
        return converted_data

    def _load_metadata(self, config: AgentCfg):
        # Load metadata for normalization stats
        if Path(config.base_model_path).exists():
            metadata_path = Path(config.base_model_path) / 'experiment_cfg' / 'metadata.json'
        else:
            snapshot_path = snapshot_download(
                repo_id=config.base_model_path,
                cache_dir=config.model_kwargs['HF_cache_dir'],
                local_files_only=True,
                allow_patterns='experiment_cfg/metadata.json'
            )
            metadata_path = Path(snapshot_path) / 'experiment_cfg' / 'metadata.json'
        with open(metadata_path, 'r') as f:
            metadatas = json.load(f)

        # Get metadata for the specific embodiment
        metadata_dict = metadatas.get(self.embodiment_tag.value)
        if metadata_dict is None:
            raise ValueError(
                f'No metadata found for embodiment tag: {self.embodiment_tag.value}',
                f'make sure the metadata.json file is present at {metadata_path}',
            )
        else:
            # deserialize the ndarray
            def convert_lists_to_arrays(obj):
                if isinstance(obj, list):
                    return np.array(obj)
                if isinstance(obj, dict):
                    return {k: convert_lists_to_arrays(v) for k, v in obj.items()}
                return obj
            metadata_dict = convert_lists_to_arrays(metadata_dict)
        metadata = DatasetMetadata.model_validate(metadata_dict)
        self.transforms.set_metadata(metadata)
        self.metadata = metadata

    def _debug_print_data(self, data, title='Data Debug'):
        print(f'\n=== {title} ===')
        print(data)
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    print(f'{key}: shape={value.shape}, dtype={value.dtype}')
                elif isinstance(value, np.ndarray):
                    print(f'{key}: shape={value.shape}, dtype={value.dtype}')
                elif isinstance(value, list):
                    print(f'{key}: type=list, length={len(value)}')
                else:
                    print(f'{key}: type={type(value)}')
        else:
            if hasattr(data, 'shape'):
                print(f"shape={data.shape}, dtype={getattr(data, 'dtype', 'unknown')}")
            else:
                print(f'type={type(data)}')


class AdaptiveEnsembler:
    def __init__(self, pred_action_horizon, adaptive_ensemble_alpha=0.0):
        self.pred_action_horizon = pred_action_horizon
        self.action_history = deque(maxlen=self.pred_action_horizon)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha

    def reset(self):
        self.action_history.clear()

    def ensemble_action(self, cur_action):
        self.action_history.append(cur_action)
        num_actions = len(self.action_history)
        if cur_action.ndim == 1:
            curr_act_preds = np.stack(self.action_history)
        else:
            curr_act_preds = np.stack(
                [pred_actions[i] for (i, pred_actions) in zip(range(num_actions - 1, -1, -1), self.action_history)]
            )
        # calculate cosine similarity between the current prediction and all previous predictions
        ref = curr_act_preds[num_actions-1, :]
        previous_pred = curr_act_preds
        dot_product = np.sum(previous_pred * ref, axis=1)
        norm_previous_pred = np.linalg.norm(previous_pred, axis=1)
        norm_ref = np.linalg.norm(ref)
        cos_similarity = dot_product / (norm_previous_pred * norm_ref + 1e-7)
        # compute the weights for each prediction
        weights = np.exp(self.adaptive_ensemble_alpha * cos_similarity)
        weights = weights / weights.sum()
        # compute the weighted average across all predictions for this timestep
        cur_action = np.sum(weights[:, None] * curr_act_preds, axis=0)
        return cur_action
