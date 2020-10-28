import os
from gym import utils as gym_utils
from fetch_block_construction.envs.robotics import utils
from fetch_block_construction.envs.robotics import fetch_env, rotations
import numpy as np
import mujoco_py
from .xml import generate_xml
import tempfile
import random


class FetchBlockHRLEnv(fetch_env.FetchEnv, gym_utils.EzPickle):
    def __init__(self, initial_qpos,
                 num_blocks=1,
                 reward_type='incremental',
                 obs_type='np',
                 render_size=42,
                 stack_only=False,
                 case="Singletower"):
        self.num_blocks = num_blocks
        self.og_num_blocks = num_blocks
        self.object_names = ['object{}'.format(i) for i in range(self.num_blocks)]
        self.stack_only = stack_only
        self.case = case

        # Ensure we get the path separator correct on windows
        # MODEL_XML_PATH = os.path.join('fetch', F'stack{self.num_blocks}.xml')
        self.two_cams = True
        with tempfile.NamedTemporaryFile(mode='wt', dir=F"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/assets/fetch/", delete=False, suffix=".xml") as fp:
            fp.write(generate_xml(self.num_blocks))
            MODEL_XML_PATH = fp.name

        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, obs_type=obs_type, render_size=render_size)

        os.remove(MODEL_XML_PATH)

        gym_utils.EzPickle.__init__(self, initial_qpos, num_blocks, reward_type, obs_type, render_size, stack_only)
        self.render_image_obs = False

    def gripper_pos_far_from_goals(self, achieved_goal, goal):
        if 'Reach' in self.case:
            return True
        gripper_pos = self.sim.data.get_site_xpos('robot0:grip').copy()

        block_goals = goal[..., :-3] # Get all the goals EXCEPT the zero'd out grip position

        distances = [
            np.linalg.norm(gripper_pos - block_goals[..., i*3:(i+1)*3], axis=-1) for i in range(self.num_blocks)
        ]
        return np.all([d > self.distance_threshold * 2 for d in distances], axis=0)

    def gripper_pos_far_from_goal(self, goal):
        if 'Reach' in self.case:
            return True
        gripper_pos = grip_pos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.case == 'PutDown':
            return np.linalg.norm(gripper_pos - goal) > self.distance_threshold*2

        block_goals = goal[..., :-3] # Get all the goals EXCEPT the zero'd out grip position

        distances = [
            np.linalg.norm(gripper_pos - block_goals[..., i*3:(i+1)*3], axis=-1) for i in range(self.num_blocks)
        ]

        return np.all([d > self.distance_threshold * 2 for d in distances], axis=0)


    def subgoal_distances(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape,(goal_a.shape,goal_b.shape)
        if self.case == 'Pickup':
            return (goal_a - goal_b)**2 * 10
        for i in range(self.num_blocks - 1):
            assert goal_a[..., i * 3:(i + 1) * 3].shape == goal_a[..., (i + 1) * 3:(i + 2) * 3].shape
        return [
            np.linalg.norm(goal_a[..., i * 3:(i + 1) * 3] - goal_b[..., i * 3:(i + 1) * 3], axis=-1) for i in
            range(self.num_blocks)
        ]
    def compute_image_reward(self, goal):
        assert self.case == 'Pickup' or self.case == 'PutDown', 'others not supported'
        if self.case == 'Pickup':
            target_block = goal.argmax()
            curr_height = self.sim.data.get_site_xpos(self.object_names[target_block])[2]
            if curr_height >= self.height_offset + 0.20:
                return 0.0
            return -1.0
        elif self.case == 'PutDown':
            block, target_block = goal[:3].argmax(), goal[3:6].argmax()
            curr_pos = self.sim.data.get_site_xpos(self.object_names[block])
            goal = self.sim.data.get_site_xpos(self.object_names[target_block]).copy()
            goal[2] += 0.05
            if np.linalg.norm(curr_pos - goal, axis=-1) <= 0.05 and self.gripper_pos_far_from_goal(goal):
                return 0.0
            return -1.0
    def compute_reward(self, achieved_goal, goal, info):
        """
        Computes reward, perhaps in an off-policy way during training. Doesn't make sense to use any of the simulator state besides that provided in achieved_goal, goal.
        :param achieved_goal:
        :param goal:
        :param info:
        :return:
        """
        subgoal_distances = self.subgoal_distances(achieved_goal, goal)
        #print("computing reward", subgoal_distances)
        if self.reward_type == "sparse":

            reward = np.min([-(d > self.distance_threshold).astype(np.float32) for d in subgoal_distances], axis=0)
            reward = np.asarray(reward)
            if "Drop" in self.case or self.case == "PutDown":
                if reward == 0.0:
                    reward = 0.0 if self.gripper_pos_far_from_goal(goal) else -1.0
            #np.putmask(reward, reward == 0, self.gripper_pos_far_from_goal(goal))
            return reward
        elif self.reward_type == "dense":
            # Dense incremental
            stacked_reward = -np.sum([(d > self.distance_threshold).astype(np.float32) for d in subgoal_distances], axis=0)
            stacked_reward = np.asarray(stacked_reward)

            reward = stacked_reward.copy()
            np.putmask(reward, reward == 0, self.gripper_pos_far_from_goals(achieved_goal, goal))

            if stacked_reward != 0:
                next_block_id = int(self.num_blocks - np.abs(stacked_reward))
                assert 0 <= next_block_id < self.num_blocks
                gripper_pos = achieved_goal[..., -3:]
                block_goals = goal[..., :-3]
                reward -= .01 * np.linalg.norm(gripper_pos - block_goals[next_block_id*3: (next_block_id+1)*3])
            return reward
        else:
            raise ("Reward not defined!")

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        obs = np.concatenate([
            grip_pos,
            gripper_state,
            grip_velp,
            gripper_vel,
        ])

        achieved_goal = []
        if self.case == 'ReachBlock':
            achieved_goal = grip_pos.copy()

        #if self.case != 'Reach':
        for i in self.block_ids:
            object_i_pos = self.sim.data.get_site_xpos(self.object_names[i])
            # rotations
            object_i_rot = rotations.mat2euler(self.sim.data.get_site_xmat(self.object_names[i]))
            # velocities
            object_i_velp = self.sim.data.get_site_xvelp(self.object_names[i]) * dt
            object_i_velr = self.sim.data.get_site_xvelr(self.object_names[i]) * dt
            # gripper state
            object_i_rel_pos = object_i_pos - grip_pos
            object_i_velp -= grip_velp

            obs = np.concatenate([
                obs,
                object_i_pos.ravel(),
                object_i_rel_pos.ravel(),
                object_i_rot.ravel(),
                object_i_velp.ravel(),
                object_i_velr.ravel()
            ])
            if self.case == 'Pickup':
                achieved_goal = np.array([1])#np.concatenate([achieved_goal, object_i_pos[2:3]])
            else:
                achieved_goal = np.concatenate([
                    achieved_goal, object_i_pos.copy()
                ])
        #achieved_goal = np.concatenate([achieved_goal, grip_pos.copy()])
        # pad goal with zeros due to gripper pos (should refactor this)
        achieved_goal = self.goal.copy() #np.concatenate([achieved_goal, np.zeros(3)])

       #else:
        if self.case == 'Reach':
            # if takes is reach then ag = [] so you're just getting gripper pos
            #achieved_goal = np.concatenate([achieved_goal, grip_pos.copy()])
            #achieved_goal = np.concatenate([achieved_goal, np.zeros(3)])
            achieved_goal[:3] = grip_pos.copy()


        # Append the grip
        achieved_goal = np.squeeze(achieved_goal)

        return_dict = {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }
        #if hasattr(self, "render_image_obs") and self.render_image_obs:
        if self.obs_type == 'dictimage':
            imobs = self.render(mode='rgb_array', size=84)
            imobs = imobs.astype(np.float32)/255
            return_dict['observation'] = imobs
        return return_dict

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()

        for i in range(self.og_num_blocks):
            site_id = self.sim.model.site_name2id('target{}'.format(i))
            self.sim.model.site_pos[site_id] = np.zeros(3) - sites_offset[i]

        for i,j in enumerate(self.block_ids):
            site_id = self.sim.model.site_name2id('target{}'.format(j))
            if self.case == "Pickup":
                self.sim.model.site_pos[site_id] = [1., 1., 1.]
            else:
                self.sim.model.site_pos[site_id] = self.goal[i * 3:(i + 1) * 3] - sites_offset[i]

        self.sim.forward()

    def reset_goal_based(self, obj_pos_list, goal, block):
        ret = self.reset()
        i = 0
        for obj_name in self.object_names:
            object_xypos = obj_pos_list[i]

            object_qpos = self.sim.data.get_joint_qpos(F"{obj_name}:joint")
            assert object_qpos.shape == (7,)
            object_qpos[:3] = object_xypos
            self.sim.data.set_joint_qpos(F"{obj_name}:joint", object_qpos)
            self.sim.forward()
            i += 1
        self._block_in_hand(block)
        self.goal = goal
        return self._get_obs()

    def get_object_list(self):
        obj_list = []
        for obj_name in self.object_names:
            object_pos = self.sim.data.get_site_xpos(obj_name).copy()
            obj_list.append(object_pos.copy())
        return obj_list

    def _reset_sim(self):
        assert self.num_blocks <= 17 # Cannot sample collision free block inits with > 17 blocks
        self.sim.set_state(self.initial_state)

        # Randomize start position of objects.

        prev_obj_xpos = []

        for obj_name in self.object_names:
            object_xypos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)

            while not ((np.linalg.norm(object_xypos - self.initial_gripper_xpos[:2]) >= 0.1) and np.all([np.linalg.norm(object_xypos - other_xpos) >= 0.06 for other_xpos in prev_obj_xpos])):
                object_xypos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)

            prev_obj_xpos.append(object_xypos)

            object_qpos = self.sim.data.get_joint_qpos(F"{obj_name}:joint")
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xypos
            object_qpos[2] = self.height_offset
            self.sim.data.set_joint_qpos(F"{obj_name}:joint", object_qpos)
            self.sim.forward()
        return True

    def _stack_a_on_b(self,a,b):
        a_pos = self.sim.data.get_site_xpos('object' + str(a))
        b_pos = self.sim.data.get_site_xpos('object' + str(b))

        a_final_pos = b_pos.copy()
        a_final_pos[2] = self.height_offset + 0.05

        # goal has original block
        # to encourage not moving original block
        # and stacked block
        self.block_ids = [a]
        self.base_block_id = b

    def _stack_on_random_block(self):
        ids = np.arange(self.og_num_blocks)
        self.num_blocks = 1
        # a is block that should be picked up
        # b is block that a should be stacked on
        # so goal is b_pos_z + height
        # find closest block

        a,b = np.random.choice(ids,2,replace=False)
        self._stack_a_on_b(a,b)

    def _move_gripper(self, location):
        dir = location - self.sim.data.get_site_xpos('robot0:grip')
        dir = np.append(dir, [0])
        i = 0
        mult_by = 10
        while np.max(np.abs(dir)[:2]) >= 0.05 and np.abs(dir[2]) >= 0.01:
            assert i < 100, "move gripper taking too long"
            dir = location - self.sim.data.get_site_xpos('robot0:grip')
            dir = np.append(dir, [0])
            np.clip(dir, -1, 1)
            self._set_action(dir * mult_by)
            try:
                self.sim.step()
            except mujoco_py.builder.MujocoException as e:
                print(e)
            i += 1
            mult_by -= 0.1
        #print(self.sim.data.get_site_xpos('robot0:grip'), location)

    def _block_in_hand(self, block):
        grip_pos = self.sim.data.get_site_xpos('robot0:grip').copy()
        #block = np.random.choice(list(range(self.num_blocks)),1,replace=False)[0]
        object_qpos = self.sim.data.get_joint_qpos(F"object{block}:joint").copy()
        object_qpos[:3] = grip_pos
        self.sim.data.set_joint_qpos(F"object{block}:joint", object_qpos)
        self._set_action(np.array([0. ,0. ,0., -1.]))
        try:
            self.sim.step()
            self.sim.step()
            self.sim.step()
        except mujoco_py.builder.MujocoException as e:
            print(e)
        


    def _sample_goal(self):
        cases = ["PickAndDrop","PickAndPlace","Reach", "Pickup", "ReachBlock", "PutDown"]
        if self.case == "All":
            case_id = np.random.randint(0, len(cases))
            case = cases[case_id]
        elif self.case in cases:
            case = self.case
        else:
            raise NotImplementedError

        goals = []
        if case == "PutDown":
            assert self.og_num_blocks == 3
            ids = np.arange(self.og_num_blocks)
            block, target_block = np.random.choice(ids, 2, replace=False)
            goal = list(np.zeros(self.og_num_blocks*2))
            goal[block] = 1
            goal[3+target_block] = 1
            goals.append(goal)
            self.block_ids = list(range(self.num_blocks))
            # ids = np.arange(self.og_num_blocks)
            # a, b = np.random.choice(ids, 2, replace=False)
            # self.chosen_block = a
            # self._block_in_hand(a)
            # # random reset other blocks
            # if random.random() > 0.5:
            #     other_block = np.random.choice(list(set(ids) - set([a, b])), 1, replace=False)[0]
            #     object_qpos = self.sim.data.get_joint_qpos(F"object{b}:joint")
            #     bpos = self.sim.data.get_site_xpos(self.object_names[other_block])
            #     object_qpos[:2] = bpos[:2]
            #     object_qpos[2] = bpos[2] + 0.05
            #     self.sim.data.set_joint_qpos(F"object{b}:joint", object_qpos)
            #     self.sim.forward()
            # goal_pos = self.sim.data.get_site_xpos('object' + str(b)).copy()
            # goal_pos[2] += 0.05
            # for i in range(self.num_blocks):
            #     if a == i:
            #         goals.append(goal_pos)
            #     else:
            #         goals.append(self.sim.data.get_site_xpos(self.object_names[i]).copy())
            # self.block_ids = list(range(self.num_blocks))
        elif case == "Pickup":
            ids = np.arange(self.og_num_blocks)
            a = np.random.choice(ids, 1, replace=False)[0]
            goal = list(np.zeros(self.og_num_blocks))
            goal[a] = 1
            goals.append(goal)
            self.block_ids = list(range(self.num_blocks))

           #  #goal_heights = np.zeros(self.num_blocks)
           #  target_height = self.height_offset + 0.20 # TODO(catc) maybe make higher
           #  # reset gripper
           #  move_to = self.sim.data.get_site_xpos('object' + str(a)).copy()
           #  move_to[2] += 0.10
           # # self.sim.data.set_site_xpos()
           #  for i in range(self.num_blocks):
           #      if a == i:
           #          goals.append([target_height])
           #      else:
           #          goals.append([self.height_offset])
           #  #goals.append(goal_heights)
           #  self.block_ids = list(range(self.num_blocks))
           # # self._move_gripper(move_to.copy())
        elif case == "PickAndDrop":
            goal_object0 = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range,
                                                                                  size=3)
            ids = np.arange(self.og_num_blocks)
            self.num_blocks = 1
            # a is block that should be picked up
            # b is block that a should be stacked on
            # so goal is b_pos_z + height
            # find closest block
            a,b = np.random.choice(ids,2,replace=False)
            a_pos = self.sim.data.get_site_xpos('object' + str(a))

            a_final_pos = a_pos.copy()
            a_final_pos[2] = self.height_offset + 0.05

            # goal has original block
            # to encourage not moving original block
            goals.append(a_final_pos)
            # and stacked block
            self.block_ids = [b]

        elif case == "PickAndPlace":

            goal_object0 = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range,
                                                                                  size=3)
            ids = np.arange(self.og_num_blocks)
            self.num_blocks = self.og_num_blocks
            # a is block that should be picked up
            # b is block that a should be stacked on
            # so goal is b_pos_z + height
            # find closest block
            a = np.random.choice(ids,1,replace=False)[0]
            a_pos = self.sim.data.get_site_xpos('object' + str(a))

            a_final_pos = goal_object0.copy() #a_pos.copy()
            #a_final_pos[2] = self.height_offset + 0.07

            # goal has original block
            # to encourage not moving original block
            for i in range(self.num_blocks):
              #  print(i, a)
                if a == i:
                    goals.append(a_final_pos)
                else:
                    #assert False, "should not ever get here"
                    goals.append(self.sim.data.get_site_xpos('object' + str(i)))
            # and stacked block
            self.block_ids = list(range(self.num_blocks))
            #print(self.block_ids)

        elif case == "Reach":
            goal_object0 = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range,
                                                                                  size=3)
            ids = np.arange(self.og_num_blocks)
            self.num_blocks = 1
            # a is block that should be picked up
            # b is block that a should be stacked on
            # so goal is b_pos_z + height
            a = np.random.choice(ids,1,replace=False)[0]
            a_pos = self.sim.data.get_site_xpos('object' + str(a))

            a_final_pos = a_pos.copy()
            a_final_pos[2] = self.height_offset + 0.07

            # goal has original block
            # to encourage not moving original block
            goals.append(a_final_pos)
            # and stacked block
            self.block_ids = [a]
        elif case == "ReachBlock":
            self.distance_threshold = 0.05
            ids = np.arange(self.og_num_blocks)
            a = np.random.choice(ids, 1, replace=False)[0]
            a_pos = self.sim.data.get_site_xpos('object' + str(a))
            a_final_pos = a_pos.copy()
            a_final_pos[2] = self.height_offset + 0.10
            goals.append(a_final_pos)
            for i in range(self.num_blocks):
                goals.append(self.sim.data.get_site_xpos('object' + str(i)).copy())
            self.block_ids = list(range(self.num_blocks))
        elif case == "Singletower":
            goal_object0 = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range,
                                                                                  size=3)
            goal_object0 += self.target_offset
            goal_object0[2] = self.height_offset

            if self.target_in_the_air and self.np_random.uniform() < 0.5 and not self.stack_only:
                # If we're only stacking, do not allow the block0 to be in the air
                goal_object0[2] += self.np_random.uniform(0, 0.45)

            # Start off goals array with the first block
            goals.append(goal_object0)

            # These below don't have goal object0 because only object0+ can be used for towers in PNP stage. In stack stage,
            previous_xys = [goal_object0[:2]]
            current_tower_heights = [goal_object0[2]]

            num_configured_blocks = self.num_blocks - 1

            for i in range(num_configured_blocks):
                if hasattr(self, "stack_only") and self.stack_only:
                    # If stack only, use the object0 position as a base
                    goal_objecti = goal_object0[:2]
                    objecti_xy = goal_objecti
                else:
                    objecti_xy = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.target_range,
                                                                                        self.target_range, size=2)
                    # Keep rolling if the other block xys are too close to the green block xy
                    # This is because the green block is sometimes lifted into the air
                    while np.linalg.norm(objecti_xy - goal_object0[0:2]) < .071:
                        objecti_xy = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.target_range, self.target_range, size=2)
                    goal_objecti = objecti_xy

                # Check if any of current block xy matches any previous xy's
                for _ in range(len(previous_xys)):
                    previous_xy = previous_xys[_]
                    if np.linalg.norm(previous_xy - objecti_xy) < .071:
                        goal_objecti = previous_xy

                        new_height_offset = current_tower_heights[_] + .05
                        current_tower_heights[_] = new_height_offset
                        goal_objecti = np.append(goal_objecti, new_height_offset)

                # If we didn't find a previous height at the xy.. just put the block at table height and update the previous xys array
                if len(goal_objecti) == 2:
                    goal_objecti = np.append(goal_objecti, self.height_offset)
                    previous_xys.append(objecti_xy)
                    current_tower_heights.append(self.height_offset)

                goals.append(goal_objecti)
        elif case == "Pyramid":
            def skew(x):
                return np.array([[0, -x[2], x[1]],
                                 [x[2], 0, -x[0]],
                                 [-x[1], x[0], 0]])

            def rot_matrix(A, B):
                """
                Rotate A onto B
                :param A:
                :param B:
                :return:
                """
                A = A/np.linalg.norm(A)
                B = B/np.linalg.norm(B)
                v = np.cross(A, B)
                s = np.linalg.norm(v)
                c = np.dot(A, B)

                R = np.identity(3) + skew(v) + np.dot(skew(v), skew(v)) * ((1 - c) / s ** 2)
                return R

            def get_xs_zs(start_point):
                start_point[2] = self.height_offset
                xs = [0, 1, .5]
                zs = [0, 0, 1]
                diagonal_start = 2
                x_bonus = 0
                z = 0
                while len(xs) <= self.num_blocks:
                    next_x = diagonal_start + x_bonus
                    xs.append(next_x)
                    zs.append(z)
                    x_bonus -= .5
                    z += 1
                    if x_bonus < -.5 * diagonal_start:
                        diagonal_start += 1
                        x_bonus = 0
                        z = 0
                return xs, zs
            x_scaling = .06
            start_point = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range,
                                                                              size=3)
            xs, zs = get_xs_zs(start_point) # Just temporary
            # xs is actually ys, because we rotate

            attempt_count = 0
            while start_point[1] + max(xs)*x_scaling > self.initial_gripper_xpos[1] + self.target_range:
                start_point = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range,
                                                                                         self.target_range,
                                                                                         size=3)
                if attempt_count > 10:
                    start_point[1] = self.initial_gripper_xpos[1] - self.target_range

                xs, zs = get_xs_zs(start_point)  # Just temporary
                attempt_count += 1

            for i in range(self.num_blocks):
                new_goal = start_point.copy()
                new_goal[0] += xs[i]*x_scaling
                new_goal[2] += zs[i]*.05

                if i > 0:
                    target_dir_vec = np.zeros(3)
                    target_dir_vec[:2] = self.initial_gripper_xpos[:2]-goals[0][:2]

                    target_dir_vec = np.array([0, 1, 0])

                    new_goal_vec = np.zeros(3)
                    new_goal_vec[:2] = new_goal[:2] - goals[0][:2]

                    new_goal = rot_matrix(new_goal_vec, target_dir_vec)@(new_goal - goals[0]) + goals[0]

                goals.append(new_goal)
        elif case == "Multitower":
            if self.num_blocks < 3:
                num_towers = 1
            elif 3 <= self.num_blocks <= 5:
                num_towers = 2
            else:
                num_towers = 3
            tower_bases = []
            tower_heights = []
            for i in range(num_towers):
                base_xy = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.target_range,
                                                                                    self.target_range, size=2)
                while not np.all(
                    [np.linalg.norm(base_xy - other_xpos) >= 0.06 for other_xpos in tower_bases]):
                    base_xy = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.target_range,
                                                                                        self.target_range, size=2)

                tower_bases.append(base_xy)
                tower_heights.append(self.height_offset)

                goal_objecti = np.zeros(3)
                goal_objecti[:2] = base_xy
                goal_objecti[2] = self.height_offset
                goals.append(goal_objecti.copy())
            # for _ in range(tower_height):
            for _ in range(self.num_blocks - num_towers):
                goal_objecti = np.zeros(3)
                goal_objecti[:2] = tower_bases[_%num_towers][:2]
                goal_objecti[2] = tower_heights[_%num_towers] + .05
                tower_heights[_%num_towers] = tower_heights[_%num_towers] + .05
                goals.append(goal_objecti.copy())
        elif case == "Togethertower":
            num_towers = 2
            tower_bases = []
            tower_heights = []
            for i in range(num_towers):
                if i == 0:
                    base_xy = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.target_range,
                                                                                    self.target_range, size=2)
                else:
                    base_xy = tower_bases[-1].copy()
                    base_xy[1] += .05

                tower_bases.append(base_xy)
                tower_heights.append(self.height_offset)

                goal_objecti = np.zeros(3)
                goal_objecti[:2] = base_xy
                goal_objecti[2] = self.height_offset
                goals.append(goal_objecti.copy())
            # for _ in range(tower_height):
            for _ in range(self.num_blocks - num_towers):
                goal_objecti = np.zeros(3)
                goal_objecti[:2] = tower_bases[_%num_towers][:2]
                goal_objecti[2] = tower_heights[_%num_towers] + .05
                tower_heights[_%num_towers] = tower_heights[_%num_towers] + .05
                goals.append(goal_objecti.copy())
        elif case == "PickAndPlace":
            goal_object0 = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range,
                                                                                  self.target_range,
                                                                                  size=3)
            goal_object0[2] = self.height_offset

            if np.random.uniform() < 0.5:
                # If we're only stacking, do not allow the block0 to be in the air
                goal_object0[2] += self.np_random.uniform(0, 0.45)

            # Start off goals array with the first block
            goals.append(goal_object0)
            for i in range(self.num_blocks - 1):
                objecti_xy = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.target_range,
                                                                                    self.target_range, size=2)
                while not np.all([np.linalg.norm(objecti_xy - goal[:2]) > .071 for goal in goals]):
                    objecti_xy = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.target_range,
                                                                                        self.target_range,
                                                                                        size=2)
                goal_objecti = np.zeros(3)
                goal_objecti[:2] = objecti_xy
                goal_objecti[2] = self.height_offset
                goals.append(goal_objecti)
        else:
            raise NotImplementedError

        goals.append([0.0, 0.0, 0.0])
        return np.concatenate(goals, axis=0).copy()

    def _is_success(self, achieved_goal, desired_goal):
        subgoal_distances = self.subgoal_distances(achieved_goal, desired_goal)
        if np.sum([-(d > self.distance_threshold).astype(np.float32) for d in subgoal_distances]) == 0:
            return True
        else:
            return False

    def _set_action(self, action):
        assert action.shape == (4,), action.shape
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        try:
            self.sim.step()
        except mujoco_py.builder.MujocoException as e:
            print(e)
            print(F"action {action}")
        self._step_callback()
        obs = self._get_obs()

        done = False

        if "image" in self.obs_type:
            assert self.case == 'Pickup' or self.case == 'PutDown'
            achieved_goal = []
            reward = self.compute_image_reward(obs['desired_goal'])
            if reward > .05:
                info = {
                    'is_success': True,
                }
            else:
                info = {
                    'is_success': False,
                }
        elif "state" in self.obs_type:
            subgoal_distances = self.subgoal_distances(obs['achieved_goal'], obs['desired_goal'])
            goal_reached_reward = np.asarray(np.min([-(d > self.distance_threshold).astype(np.float32) for d in subgoal_distances], axis=0))
            goal_reached = goal_reached_reward > -1.0
            info = {
                'is_success': self._is_success(obs['achieved_goal'], self.goal),
                'gripper_pos':self.sim.data.get_site_xpos('robot0:grip').copy(),
                'goal_reached_reward': goal_reached_reward,
                'goal_reached':goal_reached
            }
            reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        else:
            raise ("Obs_type not recognized")
        return obs, reward, done, info