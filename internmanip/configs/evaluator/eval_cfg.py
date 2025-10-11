from pydantic import BaseModel, field_validator
from typing import Optional
from pathlib import Path
from internmanip.configs.agent.agent_cfg import AgentCfg
from internmanip.configs.env.env_cfg import EnvCfg
from internmanip.configs.evaluator.distributed_cfg import DistributedCfg


class EvalCfg(BaseModel):
    eval_type: str
    agent: AgentCfg
    env: EnvCfg
    logging_dir: Optional[str] = None
    distributed_cfg: Optional[DistributedCfg] = None

    @field_validator('eval_type')
    def check_eval_type(cls, v):
        valid_types = ['SIMPLER', 'CALVIN', 'GENMANIP', 'DUMMY']
        if v.upper() not in valid_types:
            raise ValueError(f'Invalid eval type: {v}. Only: {valid_types} are supported.')
        return v.upper()

    @field_validator('logging_dir')
    def check_logging_dir(cls, v):
        if v is not None and not Path(v).exists():
            try:
                print(f'Logging directory not exists, try to create it at: {Path(v).absolute()}')
                Path(v).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ValueError(f'Failed to create logging directory: {v}')
        return v
