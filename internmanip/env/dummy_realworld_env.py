from typing import List, Dict, Any, Tuple
import time
import numpy as np

from internmanip.env.base import EnvWrapper
from internmanip.configs.env.env_cfg import EnvCfg


class DummyRealWorldEnv(EnvWrapper):
    def __init__(self, config: EnvCfg):
        super().__init__(config)
        print(f'RealWorld env settings: {self.config.env_settings}')

        
    
    def get_observations(self) -> List[Dict[str, Any]]:
        positions_28 = np.zeros(28)
        velocities_28 = np.zeros(28)

        img = np.zeros((480, 640, 3), dtype=np.uint8)

        observation = {
            'robot': {
                'joints_state': {
                    'positions': positions_28,
                    'velocities': velocities_28
                },
                'sensors': {
                    'top_camera': {'rgb': img},
                    'left_camera': {'rgb': img},
                    'right_camera': {'rgb': img}
                },
                'instruction': "find the red bowl and place it in the sink",
                'step': 0,
            }
        }
        return [observation]
        
    def reset(self, env_reset_ids=None):
        print("Resetting real world environment...")
        
        self.current_step = 0
        
        obs = self.get_observations()
        
        return obs, {'episode_name': f'episode_{time.time()}'}
    
    def step(self, actions: List[Dict[str, Any]]) -> Tuple[List[Dict], List[float], List[bool], List[Dict], List[bool]]:
        self.current_step += 1
        
        obs = self.get_observations()
        
        reward = 0.0
        info = {'step': self.current_step}
        
        return obs, [reward], [None], [info], [None]


    def get_obs(self):
        return self.get_observations()
    
    @property
    def is_episode_done(self):
        return False 
    