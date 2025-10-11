from collections import deque
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import torch
import pandas as pd

from huggingface_hub import snapshot_download

from internmanip.agent.base import BaseAgent
from internmanip.configs import AgentCfg
from internmanip.configs.dataset.data_config import DATA_CONFIG_MAP
from internmanip.dataset.embodiment_tags import EmbodimentTag
from internmanip.dataset.schema import DatasetMetadata
from internmanip.dataset.transform.base import ComposedModalityTransform
from internmanip.utils.agent_utils.io_utils import unsqueeze_dict_values, squeeze_dict_values




class ReplayAgent(BaseAgent):
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

        modality_configs = data_config_cls.modality_config(observation_indices, action_indices)

        # data_loader
        file_path = "./data/train_real/Choose_the_5_dollars_gift_box_and_add_it_to_the_basket/data/chunk-000/episode_000000.parquet"
        df = pd.read_parquet(file_path)

        left_arm_joint = np.array(df['action.left_joint'])
        right_arm_joint = np.array(df['action.right_joint'])
        left_gripper = np.array(df['action.left_gripper'])
        right_gripper = np.array(df['action.right_gripper'])

        self.left_arm_joint = left_arm_joint
        self.right_arm_joint = right_arm_joint
        self.left_gripper = left_gripper
        self.right_gripper = right_gripper

        self.length = len(self.left_arm_joint)

    def step(self, inputs: list[dict]) -> list[dict]:
        outputs = []

        for env, input in enumerate(inputs):
            if input == {}:
                outputs.append({})
                continue
            
            step = input['robot']['step']
            if step >= self.length:
                outputs.append({})
                continue

            converted_data = { 
                'left_arm_action': self.left_arm_joint[step],
                'right_arm_action': self.right_arm_joint[step],
                'left_gripper_action': self.left_gripper[step],
                'right_gripper_action': self.right_gripper[step],
            }
            outputs.append(converted_data)

        return outputs

    def reset(self):
        self.ensembler_list = []


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
