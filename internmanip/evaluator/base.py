from typing import List, Dict, Any
from enum import Enum

from internmanip.configs.evaluator.eval_cfg import EvalCfg
from internmanip.utils.agent_utils.client import AgentClient
from internmanip.env import EnvWrapper


class EvaluatorRegistry(Enum):
    """
    Registry of evaluator subclasses.
    The key is the evaluator type.
    The value is the evaluator subclass.
    """
    SIMPLER = 'SimplerEvaluator'
    CALVIN = 'CalvinEvaluator'
    GENMANIP = 'GenmanipEvaluator'
    DUMMY = 'DummyEvaluator'

    @property
    def value(self):
        if self.name == 'SIMPLER':
            from internmanip.evaluator.simpler_evaluator import SimplerEvaluator
            return SimplerEvaluator
        elif self.name == 'CALVIN':
            from internmanip.evaluator.calvin_evaluator import CalvinEvaluator
            return CalvinEvaluator
        elif self.name == 'GENMANIP':
            from internmanip.evaluator.genmanip_evaluator import GenmanipEvaluator
            return GenmanipEvaluator
        elif self.name == 'DUMMY':
            from internmanip.evaluator.dummy_realworld_evaluator import DummyRealWorldEvaluator
            return DummyRealWorldEvaluator
        else:
            raise ValueError(f'Invalid evaluator type: {self.name}. Only {[evaluator_type.name for evaluator_type in EvaluatorRegistry]} are registered.')


class Evaluator:
    """
    Base class of all evaluators.
    """

    def __init__(self, config: EvalCfg):
        self.config = config
        if config.distributed_cfg is not None:
            self._set_distributed_device_ids()
        self.env = EnvWrapper.init(config.env)
        if config.agent.server_cfg is not None:
            self.agent = AgentClient(config.agent)
        else:
            from internmanip.agent import BaseAgent
            self.agent = BaseAgent.init(config.agent)

    @classmethod
    def _update_results(cls, result):
        """
        Implementation of updating the results, subclasses should override.
        """
        raise NotImplementedError('This method `_update_results` should be implemented in subclasses.')

    @classmethod
    def _print_and_save_results(cls):
        """
        Implementation of printing and saving the results, subclasses should override.
        """
        raise NotImplementedError('This method `_print_and_save_results` should be implemented in subclasses.')

    def eval(self):
        """
        Default evaluation method, subclasses should override.
        """
        raise NotImplementedError('This method `eval` should be implemented in subclasses.')

    def _eval_single_episode(self, episode_data: Dict[str, Any]):
        """
        Implementation of evaluating a single episode, subclasses should override.
        """
        raise NotImplementedError('This method `_eval_single_episode` should be implemented in subclasses.')

    @classmethod
    def _get_all_episodes_setting_data(cls, episodes_config_path) -> List[Any]:
        """
        Implementation of getting all episodes setting data, subclasses should override.
        """
        raise NotImplementedError('This method `_get_all_episodes_setting_data` should be implemented in subclasses.')

    @classmethod
    def init(cls, config: EvalCfg):
        """
        Init a evaluator instance from a config.
        """
        return EvaluatorRegistry[config.eval_type].value(config)

    def _set_distributed_device_ids(self):
        import ray

        if ray.is_initialized():
            self.device_id = int(ray.get_gpu_ids()[0])
            # NOTE: let the model and env be deployed on the same GPU for now, to avoid frequent GPU data(not sure) to be transferred between different cuda devices
            # TODO(maybe): support model and env to be deployed on different GPUs
            self.config.agent.agent_settings['device_id'] = self.device_id
            self.config.env.device_id = self.device_id
        else:
            raise RuntimeError('The distributed backend is not initialized.')

    def ping(self):
        """
        Ping the evaluator to check if it is alive.
        Only used for checking the status of evaluator ray actors on distributed backend.
        """
        return True
