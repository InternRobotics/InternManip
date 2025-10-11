from enum import Enum
from transformers import AutoModel

from internmanip import model
from internmanip.configs import AgentCfg



class AgentRegistry(Enum):
    """
    Registry of agent subclasses.
    The key is the agent type.
    The value is the agent subclass.
    """
    SIMPLER = 'SimplerAgent'
    GENMANIP = 'GenmanipAgent'
    REALARX = 'RealArxAgent'
    REPLAY = 'ReplayAgent'

    @property
    def value(self):
        if self.name == 'SIMPLER':
            from internmanip.agent.simpler_agent import SimplerAgent
            return SimplerAgent
        elif self.name == 'GENMANIP':
            from internmanip.agent.genmanip_agent import GenmanipAgent
            return GenmanipAgent
        elif self.name == 'REALARX':
            from internmanip.agent.realarx_agent import RealArxAgent
            return RealArxAgent
        elif self.name == 'REPLAY':
            from internmanip.agent.replay_agent import ReplayAgent
            return ReplayAgent
        else:
            raise ValueError(f'Invalid agent type: {self.name}. Only {[agent_type.name for agent_type in AgentRegistry]} are registered.')


class BaseAgent:

    def __init__(self, config: AgentCfg):
        self.config = config

        if config.base_model_path is None:
            policy_model = AutoModel.from_config(config.model_cfg, **config.model_kwargs)
        else:
            # must ensure that if the path is a huggingface model, it should be a repo that has only one model weight
            policy_model = AutoModel.from_pretrained(config.base_model_path, **config.model_kwargs)
        self.policy_model = policy_model

    def step(self):
        raise NotImplementedError('Not implemented in base agent class')

    def reset(self):
        raise NotImplementedError('Not implemented in base agent class')

    @classmethod
    def init(cls, config: AgentCfg):
        """
        Init a agent instance from a config.
        """
        print(f'Initializing {config.agent_type} on {config.eval_type}')
        return AgentRegistry[config.eval_type].value(config)
