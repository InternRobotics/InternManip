from internmanip.configs.env.env_cfg import EnvCfg
from enum import Enum


class EnvWrapperRegistry(Enum):
    """
    Registry of env wrapper subclasses.
    The key is the env wrapper type.
    The value is the env wrapper subclass.
    """
    SIMPLER = 'SimplerEnv'
    CALVIN = 'CalvinEnv'
    GENMANIP = 'GenmanipEnv'
    DUMMY = 'DummyRealWorldEnv'

    @property
    def value(self):
        if self.name == 'SIMPLER':
            from internmanip.env.simpler_env import SimplerEnv
            return SimplerEnv
        elif self.name == 'CALVIN':
            from internmanip.env.calvin_env import CalvinEnv
            return CalvinEnv
        elif self.name == 'GENMANIP':
            from internmanip.env.genmanip_env import GenmanipEnv
            return GenmanipEnv
        elif self.name == 'DUMMY':
            from internmanip.env.dummy_realworld_env import DummyRealWorldEnv
            return DummyRealWorldEnv
        else:
            raise ValueError(f'Invalid env wrapper type: {self.name}. Only {[env_type.name for env_type in EnvWrapperRegistry]} are registered.')


class EnvWrapper:
    """
    Base class of all environments.
    """

    def __init__(self, config: EnvCfg):
        self.config = config

    def reset(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    @classmethod
    def init(cls, config: EnvCfg):
        """
        Init a env instance from a config.
        """
        return EnvWrapperRegistry[config.env_type].value(config)
