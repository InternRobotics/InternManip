from internmanip.evaluator.base import Evaluator
from internmanip.configs.evaluator.eval_cfg import EvalCfg
from internmanip.configs.env.genmanip_env import EpisodeInfo

class DummyRealWorldEvaluator(Evaluator):
    def __init__(self, config: EvalCfg):
        # 设置 episode 列表
        if config.env.env_settings.episode_list is None \
            or len(config.env.env_settings.episode_list) == 0:
            config.env.env_settings.episode_list = self._get_realworld_episodes(config)

        super().__init__(config)

    def eval(self, distributed=False):
        print("Starting real world evaluation...")
        print("Controls: 'p' to pause, 'q' to quit")

        for episode_info in self.config.env.env_settings.episode_list:
            print(f"Starting episode: {episode_info.task_name}")
            
            # 重置环境
            obs, info = self.env.reset()
            
            # 运行单个episode
            self._run_single_episode()
            
            # 询问是否继续
            if episode_info != self.config.env.env_settings.episode_list[-1]:
                cont = input("Continue to next episode? (y/n): ")
                if cont.lower() != 'y':
                    break

        print("Real world evaluation completed.")
        return 0  # 返回0，实际成功率由裁判评判

    def _run_single_episode(self):
        """运行单个episode"""
        step_count = 0
        max_steps = self.config.env.env_settings.max_step
        
        while step_count < max_steps:
            if self.env.is_episode_done:
                break
                
            # 获取观测
            obs = self.env.get_obs()
            
            # # 获取动作
            # print(obs)

            all_env_action = self.agent.step(obs)
            # print(all_env_action)
            
            # 执行动作
            obs, rewards, terminated_status, infos, truncated_status = self.env.step(all_env_action)
            
            step_count += 1

    @classmethod
    def _get_realworld_episodes(cls, config):
        """获取真机环境的 episode 配置"""
        episode_list = []
        
        if config.env.env_settings.eval_tasks:
            eval_tasks = config.env.env_settings.eval_tasks
        else:
            eval_tasks = ['real_world_demo']
        
        for task_item in eval_tasks:
            episode_info = EpisodeInfo(
                episode_path='real_world',
                task_name=task_item,
                episode_name=f'real_world_{task_item}'
            )
            episode_list.append(episode_info)
        
        print(f"Loaded {len(episode_list)} real world episodes")
        return episode_list

    def calc_task_sr(self, episode_sr_info=None):
        """计算任务成功率 - 返回0，实际由裁判评判"""
        return 0