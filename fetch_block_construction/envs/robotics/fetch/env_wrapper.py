from dm_control import suite
from abc import ABC
import numpy as np
import gym
import mujoco_py
from gym.envs.registration import register


def change_fetch_model(change_model):
    import os
    import shutil
    gym_folder = os.path.dirname(gym.__file__)
    xml_folder = 'envs/robotics/assets/fetch'
    full_folder_path = os.path.join(gym_folder, xml_folder)
    xml_file_path = os.path.join(full_folder_path, 'shared.xml')
    backup_file_path = os.path.join(full_folder_path, 'shared_backup.xml')
    if change_model:
        if not os.path.exists(backup_file_path):
            shutil.copy2(xml_file_path, backup_file_path)
        shutil.copy2('fetch_yellow_obj.xml', xml_file_path)
    else:
        if os.path.exists(backup_file_path):
            shutil.copy2(backup_file_path, xml_file_path)


def make(domain_name, task_name, seed, from_pixels, height, width, cameras=range(1),
         visualize_reward=False, frame_skip=None, reward_type='dense', change_model=False):
    possibly_gym = False

    if domain_name in suite._DOMAINS:
        import dmc2gym
        env = dmc2gym.make(
            domain_name=domain_name,
            task_name=task_name,
            seed=seed,
            visualize_reward=visualize_reward,
            from_pixels=from_pixels,
            height=height,
            width=width,
            frame_skip=frame_skip
        )
        return env
    else:
        env_class = None
        possibly_gym = True

    if not possibly_gym:
        if task_name is not None:
            env = env_class()
            env.set_task(task_name)
        else:
            env = env_class()
        env = MetaWorldEnvWrapper(env, from_pixels=from_pixels, cameras=cameras, height=height, width=width)
    else:
        if 'RealArm' not in domain_name:
            change_fetch_model(change_model)
            env = gym.make(domain_name, reward_type=reward_type)
            # print('Setting target on table!')
            # env.unwrapped.target_in_the_air = False
            env = GymEnvWrapper(env, from_pixels=from_pixels, cameras=cameras, height=height, width=width)
        else:
            import gym_xarm
            env = gym.make(domain_name)
            env.env.set_reward_mode(reward_type)
            env = RealEnvWrapper(env, from_pixels=from_pixels, cameras=cameras, height=height, width=width)

    env.seed(seed)
    return env


def generate_shell_commands(domain_name, task_name, cameras, observation_type, encoder_type, work_dir,
                            pre_transform_image_size, agent, data_augs=None,
                            replay_buffer_load_dir=None,
                            seed=-1, critic_lr=1e-3, actor_lr=1e-3, frame_stack=3, num_updates=1,
                            eval_freq=5000, batch_size=128, num_train_steps=1000000, cuda=None,
                            save_tb=True, save_video=True, save_model=False,
                            reward_type='dense',
                            special_reset=None, demo_model_dir=None, demo_model_step=0,
                            demo_samples=25000,
                            demo_special_reset=None, use_demo=False, demo_weight=0,
                            change_model=False):
    if cuda is not None:
        command = 'CUDA_VISIBLE_DEVICES=' + cuda + ' python train.py'
    else:
        command = 'python train.py'
    command += ' --domain_name ' + domain_name
    if task_name is not None:
        command += ' --task_name ' + task_name
    if replay_buffer_load_dir:
        command += ' --replay_buffer_load_dir ' + replay_buffer_load_dir
    command += ' --reward_type ' + reward_type
    command += ' --cameras ' + cameras
    command += ' --frame_stack ' + str(frame_stack)
    command += ' --num_updates ' + str(num_updates)
    command += ' --observation_type ' + observation_type
    command += ' --encoder_type ' + encoder_type
    command += ' --work_dir ' + work_dir
    command += ' --pre_transform_image_size ' + str(pre_transform_image_size)
    command += ' --image_size 84'
    command += ' --agent ' + agent
    if data_augs is not None:
        command += ' --data_augs ' + data_augs
    command += ' --seed ' + str(seed)
    command += ' --critic_lr ' + str(critic_lr)
    command += ' --actor_lr ' + str(actor_lr)
    command += ' --eval_freq ' + str(eval_freq)
    command += ' --batch_size ' + str(batch_size)
    command += ' --num_train_steps ' + str(num_train_steps)
    if save_tb:
        command += ' --save_tb'
    if save_video:
        command += ' --save_video'
    if save_model:
        command += ' --save_model'
    if special_reset is not None:
        command += ' --special_reset ' + special_reset
    if use_demo:
        command += ' --use_demo'
        command += ' --demo_model_dir ' + str(demo_model_dir)
        command += ' --demo_model_step ' + str(demo_model_step)
        command += ' --demo_weight ' + str(demo_weight)
        command += ' --demo_samples ' + str(demo_samples)
        if demo_special_reset is not None:
            command += ' --demo_special_reset ' + demo_special_reset
    if change_model:
        command += ' --change_model'
    return command


class EnvWrapper(gym.Env, ABC):
    def __init__(self, env, cameras, from_pixels=True, height=100, width=100, channels_first=True):
        camera_0 = {'trackbodyid': -1, 'distance': 1.5, 'lookat': np.array((0.0, 0.6, 0)),
                    'elevation': -45.0, 'azimuth': 90}
        camera_1 = {'trackbodyid': -1, 'distance': 1.5, 'lookat': np.array((0.0, 0.6, 0)),
                    'elevation': -45.0, 'azimuth': 135}
        camera_2 = {'trackbodyid': -1, 'distance': 1.5, 'lookat': np.array((0.0, 0.6, 0)),
                    'elevation': -45.0, 'azimuth': 180}
        camera_3 = {'trackbodyid': -1, 'distance': 1.5, 'lookat': np.array((0.0, 0.6, 0)),
                    'elevation': -45.0, 'azimuth': 225}
        camera_4 = {'trackbodyid': -1, 'distance': 1.5, 'lookat': np.array((0.0, 0.6, 0)),
                    'elevation': -45.0, 'azimuth': 270}
        camera_5 = {'trackbodyid': -1, 'distance': 1.5, 'lookat': np.array((0.0, 0.6, 0)),
                    'elevation': -45.0, 'azimuth': 315}
        camera_6 = {'trackbodyid': -1, 'distance': 1.5, 'lookat': np.array((0.0, 0.6, 0)),
                    'elevation': -45.0, 'azimuth': 0}
        camera_7 = {'trackbodyid': -1, 'distance': 1.5, 'lookat': np.array((0.0, 0.6, 0)),
                    'elevation': -45.0, 'azimuth': 45}
        self.all_cameras = [camera_0, camera_1, camera_2, camera_3, camera_4, camera_5, camera_6, camera_7]

        self._env = env
        self.cameras = cameras
        self.from_pixels = from_pixels
        self.height = height
        self.width = width
        self.channels_first = channels_first

        self.special_reset = None
        self.special_reset_save = None
        self.hybrid_obs = False
        self.viewer = None

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        shape = [3 * len(cameras), height, width] if channels_first else [height, width, 3 * len(cameras)]
        self._observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8
        )

        self._state_obs = None
        self.change_camera()
        self.reset()

    def change_camera(self):
        return

    @property
    def observation_space(self):
        if self.from_pixels:
            return self._observation_space
        else:
            return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    def seed(self, seed=None):
        return self._env.seed(seed)

    def reset_model(self):
        self._env.reset()

    def viewer_setup(self, camera_id=0):
        for key, value in self.all_cameras[camera_id].items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def set_hybrid_obs(self, mode):
        self.hybrid_obs = mode

    def _get_obs(self):
        if self.from_pixels:
            imgs = []
            for c in self.cameras:
                imgs.append(self.render(mode='rgb_array', camera_id=c))
            if self.channels_first:
                pixel_obs = np.concatenate(imgs, axis=0)
            else:
                pixel_obs = np.concatenate(imgs, axis=2)
            if self.hybrid_obs:
                return [pixel_obs, self._get_hybrid_state()]
            else:
                return pixel_obs
        else:
            return self._get_state_obs()

    def _get_state_obs(self):
        return self._state_obs

    def _get_hybrid_state(self):
        return self._state_obs

    @property
    def hybrid_state_shape(self):
        if self.hybrid_obs:
            return self._get_hybrid_state().shape
        else:
            return None

    def step(self, action):

        self._state_obs, reward, done, info = self._env.step(action)
        return self._get_obs(), reward, done, info

    def reset(self, save_special_steps=False):
        self._state_obs = self._env.reset()
        return self._get_obs()

    def set_state(self, qpos, qvel):
        self._env.set_state(qpos, qvel)

    @property
    def dt(self):
        if hasattr(self._env, 'dt'):
            return self._env.dt
        else:
            return 1

    @property
    def _max_episode_steps(self):
        return self._env.max_path_length

    def do_simulation(self, ctrl, n_frames):
        self._env.do_simulatiaon(ctrl, n_frames)

    def render(self, mode='human', camera_id=0, height=None, width=None):
        if mode == 'human':
            self._env.render()

        if height is None:
            height = self.height
        if width is None:
            width = self.width

        if mode == 'rgb_array':
            if isinstance(self, GymEnvWrapper):
                self._env.unwrapped._render_callback()
            viewer = self._get_viewer(camera_id)
            # Calling render twice to fix Mujoco change of resolution bug.
            viewer.render(width, height, camera_id=-1)
            viewer.render(width, height, camera_id=-1)
            # window size used for old mujoco-py:
            data = viewer.read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            data = data[::-1, :, :]
            if self.channels_first:
                data = data.transpose((2, 0, 1))
            return data

    def close(self):
        if self.viewer is not None:
            self.viewer = None
            self._env.close()

    def _get_viewer(self, camera_id):
        if self.viewer is None:
            from mujoco_py import GlfwContext
            GlfwContext(offscreen=True)
            self.viewer = mujoco_py.MjRenderContextOffscreen(self._env.sim)
        self.viewer_setup(camera_id)
        return self.viewer

    def get_body_com(self, body_name):
        return self._env.get_body_com(body_name)

    def state_vector(self):
        return self._env.state_vector


class MetaWorldEnvWrapper(EnvWrapper):
    def change_camera(self):
        for c in self.all_cameras:
            c['lookat'] = np.array((1.3, 0.75, 0.4))
            c['distance'] = 1.2
        # Zoomed out cameras
        camera_8 = {'trackbodyid': -1, 'distance': 1.8, 'lookat': np.array((1.3, 0.75, 0.4)),
                    'elevation': -45.0, 'azimuth': 135}
        camera_9 = {'trackbodyid': -1, 'distance': 1.8, 'lookat': np.array((1.3, 0.75, 0.4)),
                    'elevation': -45.0, 'azimuth': 225}
        # Gripper head camera
        camera_10 = {'trackbodyid': -1, 'distance': 0.2, 'lookat': np.array((1.3, 0.75, 0.4)),
                     'elevation': -90, 'azimuth': 0}
        # Zoomed out head-on camera
        camera_11 = {'trackbodyid': -1, 'distance': 1.8, 'lookat': np.array((1.3, 0.75, 0.4)),
                     'elevation': -45.0, 'azimuth': 180}
        self.all_cameras.append(camera_8)
        self.all_cameras.append(camera_9)
        self.all_cameras.append(camera_10)
        self.all_cameras.append(camera_11)

    def update_tracking_cameras(self):
        import ipdb; ipdb.set_trace()

    @property
    def _max_episode_steps(self):
        return self._env.max_path_length

    def step(self, action):
        self._state_obs, reward, done, info = self._env.step(action)
        if self._env.curr_path_length >= self._env.max_path_length:
            done = True
        return self._get_obs(), reward, done, info


class GymEnvWrapper(EnvWrapper):
    def change_camera(self):
        for c in self.all_cameras:
            c['lookat'] = np.array((1.3, 0.75, 0.4))
            c['distance'] = 1.2
        # Zoomed out cameras
        camera_8 = {'trackbodyid': -1, 'distance': 1.8, 'lookat': np.array((1.3, 0.75, 0.4)),
                    'elevation': -45.0, 'azimuth': 135}
        camera_9 = {'trackbodyid': -1, 'distance': 1.8, 'lookat': np.array((1.3, 0.75, 0.4)),
                    'elevation': -45.0, 'azimuth': 225}
        # Gripper head camera
        camera_10 = {'trackbodyid': -1, 'distance': 0.2, 'lookat': np.array((1.3, 0.75, 0.4)),
                     'elevation': -90, 'azimuth': 0}
        # Zoomed out head-on camera
        camera_11 = {'trackbodyid': -1, 'distance': 1.8, 'lookat': np.array((1.3, 0.75, 0.4)),
                     'elevation': -45.0, 'azimuth': 180}
        self.all_cameras.append(camera_8)
        self.all_cameras.append(camera_9)
        self.all_cameras.append(camera_10)
        self.all_cameras.append(camera_11)

    def update_tracking_cameras(self):
        gripper_pos = self._state_obs['observation'][:3].copy()
        self.all_cameras[10]['lookat'] = gripper_pos

    def _get_obs(self):
        self.update_tracking_cameras()
        return super()._get_obs()

    @property
    def _max_episode_steps(self):
        return self._env._max_episode_steps

    def set_special_reset(self, mode):
        self.special_reset = mode

    def register_special_reset_move(self, action, reward):
        # self.update_tracking_cameras()
        # img = self.render(mode='rgb_array', camera_id=10)
        # import matplotlib.pyplot as plt
        # plt.imshow(img.transpose(1, 2, 0))
        # plt.show()
        if self.special_reset_save is not None:
            self.special_reset_save['obs'].append(self._get_obs())
            self.special_reset_save['act'].append(action)
            self.special_reset_save['reward'].append(reward)

    def go_to_pos(self, pos):
        grip_pos = self._state_obs['observation'][:3]
        action = np.zeros(4)
        for i in range(10):
            if np.linalg.norm(grip_pos - pos) < 0.02:
                break
            action[:3] = (pos - grip_pos) * 10
            self._state_obs, r, d, i = self._env.step(action)
            self.register_special_reset_move(action, r)
            grip_pos = self._state_obs['observation'][:3]

    def raise_gripper(self):
        grip_pos = self._state_obs['observation'][:3]
        raised_pos = grip_pos.copy()
        raised_pos[2] += 0.1
        self.go_to_pos(raised_pos)

    def open_gripper(self):
        action = np.array([0, 0, 0, 1])
        for i in range(2):
            self._state_obs, r, d, i = self._env.step(action)
            self.register_special_reset_move(action, r)

    def close_gripper(self):
        action = np.array([0, 0, 0, -1])
        for i in range(2):
            self._state_obs, r, d, i = self._env.step(action)
            self.register_special_reset_move(action, r)

    def reset(self, save_special_steps=False):
        self._state_obs = self._env.reset()
        if save_special_steps:
            self.special_reset_save = {'obs': [], 'act': [], 'reward': []}
            self.special_reset_save['obs'].append(self._get_obs())
        if self.special_reset == 'close' and self._env.has_object:
            obs = self._state_obs['observation']
            goal = self._state_obs['desired_goal']
            obj_pos = obs[3:6]
            goal_distance = np.linalg.norm(obj_pos - goal)
            desired_reset_pos = obj_pos + (obj_pos - goal) / goal_distance * 0.06
            desired_reset_pos_raised = desired_reset_pos.copy()
            desired_reset_pos_raised[2] += 0.1
            self.raise_gripper()
            self.go_to_pos(desired_reset_pos_raised)
            self.go_to_pos(desired_reset_pos)
        elif self.special_reset == 'grip' and self._env.has_object and not self._env.block_gripper:
            obs = self._state_obs['observation']
            obj_pos = obs[3:6]
            above_obj = obj_pos.copy()
            above_obj[2] += 0.1
            self.open_gripper()
            self.raise_gripper()
            self.go_to_pos(above_obj)
            self.go_to_pos(obj_pos)
            self.close_gripper()
        return self._get_obs()

    def _get_state_obs(self):
        obs = np.concatenate([self._state_obs['observation'],
                              self._state_obs['achieved_goal'],
                              self._state_obs['desired_goal']])
        return obs

    def _get_hybrid_state(self):
        grip_pos = self._env.sim.data.get_site_xpos('robot0:grip')
        dt = self._env.sim.nsubsteps * self._env.sim.model.opt.timestep
        grip_velp = self._env.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = gym.envs.robotics.utils.robot_get_obs(self._env.sim)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric
        robot_info = np.concatenate([grip_pos, gripper_state, grip_velp, gripper_vel])
        hybrid_obs_list = []
        if 'robot' in self.hybrid_obs:
            hybrid_obs_list.append(robot_info)
        if 'goal' in self.hybrid_obs:
            hybrid_obs_list.append(self._state_obs['desired_goal'])
        return np.concatenate(hybrid_obs_list)

    @property
    def observation_space(self):
        shape = self._get_state_obs().shape
        return gym.spaces.Box(-np.inf, np.inf, shape=shape, dtype='float32')


class RealEnvWrapper(GymEnvWrapper):
    def render(self, mode='human', camera_id=0, height=None, width=None):
        if mode == 'human':
            self._env.render()

        if height is None:
            height = self.height
        if width is None:
            width = self.width

        if mode == 'rgb_array':
            data = self._env.render(mode='rgb_array', height=height, width=width)
            if self.channels_first:
                data = data.transpose((2, 0, 1))
            if camera_id == 8:
                data = data[3:]
            return data

    def _get_obs(self):
        return self.render(mode='rgb_array', height=self.height, width=self.width)

    def _get_state_obs(self):
        return self._get_obs()

    def reset(self, save_special_steps=False):
        self._state_obs = self._env.reset(rand_pos=True)
        return self._get_obs()
