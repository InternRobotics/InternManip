from pydantic import BaseModel, field_validator, Field
from typing import Optional, Union, List
from pathlib import Path
from internmanip.configs.env.simpler_env import SimplerEnvSettings
from internmanip.configs.env.calvin_env import CalvinEnvSettings
from internmanip.configs.env.genmanip_env import GenmanipEnvSettings


class EnvCfg(BaseModel):
    env_type: str
    device_id: Optional[int] = Field(default=None, description='Specify the EGL device id(should be a physical device id) for rendering.')
    config_path: Optional[str] = Field(default=None, description='Config file path for your specified benchmark, which includes the evaluation settings(e.g. task, scene, robot, etc.)')
    env_settings: Optional[Union[SimplerEnvSettings, CalvinEnvSettings, GenmanipEnvSettings]] = Field(default=None, description='Custom settings')
    episodes_config_path: Optional[Union[str, List[str]]] = Field(default=None, description='Episodes config file path for your specified benchmark, which includes the task settings for each episode(e.g. object position, etc.)')


    @field_validator('env_type')
    def check_env_type(cls, v):
        valid_types = ['SIMPLER', 'CALVIN', 'GENMANIP', 'DUMMY']
        if v.upper() not in valid_types:
            raise ValueError(f'Invalid environment type: {v}. Only: {valid_types} are supported.')
        return v.upper()

    @field_validator('device_id')
    def check_device_id(cls, v):
        import torch
        if v is not None:
            if v < 0 or v >= torch.cuda.device_count():
                raise ValueError(f'Invalid device id: {v}')
        return v

    @field_validator('config_path')
    def check_config_path(cls, v):
        if v is not None and not Path(v).exists():
            raise ValueError(f'Invalid config path: {v}')
        return v

    @field_validator('episodes_config_path')
    def check_episodes_config_path(cls, v):
        if v is not None:
            if isinstance(v, str):
                if not Path(v).exists():
                    raise ValueError(f'Invalid episodes config path: {v}')
            elif isinstance(v, list):
                for path in v:
                    if not Path(path).exists():
                        raise ValueError(f'Invalid episodes config path: {path}')
        return v
