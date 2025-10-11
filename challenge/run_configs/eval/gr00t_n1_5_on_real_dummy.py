from internmanip.configs import *
from internmanip.benchmarks.genmanip.config.env_config import FrankaCameraEnable, AlohaSplitCameraEnable


eval_cfg = EvalCfg(
    eval_type='dummy',
    agent=AgentCfg(
        agent_type='gr00t_n1_5_on_realarx',
        eval_type='realarx',
        base_model_path='/data/Checkpoints/runs/gr00t_n1_5_arx_iros_20251009/checkpoint-5000',
        agent_settings={
            'data_config': 'arx',
            'embodiment_tag': 'new_embodiment',
            'pred_action_horizon': 16,
            'action_ensemble': False,
            'adaptive_ensemble_alpha': 0,
            'eval_type': 'realarx',
        },
        model_kwargs={
            'HF_cache_dir': None,
            'torch_dtype': 'float16'
        },
        server_cfg=ServerCfg(
            server_host='localhost',
            server_port=5000
        ),
    ),
    env=EnvCfg(
        env_type='dummy',
        env_settings=GenmanipEnvSettings(
            dataset_path='./data/dataset/IROS-2025-Challenge-Manip/validation',
            res_save_path='./results',
            is_save_img=False,
            aloha_split_camera_enable=AlohaSplitCameraEnable(
                top_camera=True, left_camera=True, right_camera=True
            ),
        ),
    ),
    distributed_cfg=DistributedCfg(
        num_workers=2,
        ray_head_ip='localhost'
    )
)
