from sandbox.adam.envs.atari_env import AtariEnv

env_args = dict(
    game="breakout",
    frame_skip=4,
    num_img_obs=4,
    img_downsample=2,
    two_frame_max=None,
)

env = AtariEnv(**env_args)

a = env.action_space.sample()

for _ in range(1000):
    o, r, d, info = env.step(env.action_space.sample())
    env.render(wait=25, show_full_obs=True)
    if d:
        env.reset()
