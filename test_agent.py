import hydra
import torch
import utils

@hydra.main(config_path='/home/misha/algos/goal_sac/config/train.yaml', strict=True)
def main(cfg):
    device = torch.device(cfg.device)
    env = utils.make_env(cfg)
    obs_shape = env.observation_space['observation'].shape
    goal_shape = env.observation_space['desired_goal'].shape

    cfg.agent.params.obs_dim = obs_shape[0]
    cfg.agent.params.goal_dim = goal_shape[0]
    cfg.agent.params.action_dim = env.action_space.shape[0]
    cfg.agent.params.action_range = [
        float(env.action_space.low.min()),
        float(env.action_space.high.max())
    ]
    agent = hydra.utils.instantiate(cfg.agent)
    params = list(agent.actor.named_parameters())+list(agent.critic.named_parameters())
    for name, param in params:
        if param.requires_grad:
            print(name, param.data.shape)

    agent.save()
if __name__ == '__main__':
    main()

