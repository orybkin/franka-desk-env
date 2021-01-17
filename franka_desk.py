import numpy as np
import copy
import os
import gym
from envs.franka_desk.base_mujoco_env import BaseMujocoEnv
from metaworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv


class FrankaDesk(BaseMujocoEnv, SawyerXYZEnv):
    """ A playground environment taken from Lynch'19 (actual implementation borrowed from Tian'20)"""

    def __init__(self, env_params_dict, reset_state=None):
        hand_low = (0.05, -0.75, 0.73)
        hand_high = (0.6, -0.48, 1.0)
        obj_low=(0.15, -0.4, 0.6)
        obj_high=(0.4, -0.6, 0.6)
        
        self.action_space = gym.spaces.Box(low=np.array((-1, -1, -1)), high=np.array((1, 1, 1)))

        dirname = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        params_dict = copy.deepcopy(env_params_dict)
        _hp = self._default_hparams()
        for name, value in params_dict.items():
            print('setting param {} to value {}'.format(name, value))
            _hp[name] = value

        filename = os.path.join(dirname, "playroom.xml")

        BaseMujocoEnv.__init__(self, filename, _hp)
        SawyerXYZEnv.__init__(
                self,
                frame_skip=_hp.frame_skip,
                action_scale=_hp.action_scale,
                hand_low=hand_low,
                hand_high=hand_high,
                model_name=filename
            )
        goal_low = self.hand_low
        goal_high = self.hand_high
        self.obj_low = obj_low
        self.obj_high = obj_high
        self._adim = 4
        self._hp = _hp
        self.liftThresh = 0.04
        self.max_path_length = 100
        self.hand_init_pos = np.array([0.6, -0.48, 1.0])

    def default_ncam():
        return 1

    def _default_hparams(self):
        default_dict = {
            'verbose': False,
            'difficulty': None,
            'textured': False,
            'render_imgs': True,
            'include_objects': False,
            'frame_skip': 15,
            'action_scale': 1./5,
        }
        parent_params = super()._default_hparams()
        for k in default_dict.keys():
          parent_params[k] = default_dict[k]
        return parent_params

    #default_mocap_quat = np.array([0, -1, 0, 1])
    default_mocap_quat = np.array([0, 1, 0.5, 0])

    def set_xyz_action(self, action):
        action = np.clip(action, -1, 1)
        pos_delta = action * self.action_scale
        new_mocap_pos = self.data.mocap_pos + pos_delta[None]

        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            self.mocap_low,
            self.mocap_high,
        )
        self.data.set_mocap_pos('mocap', new_mocap_pos)
        self.data.set_mocap_quat('mocap', self.default_mocap_quat.copy())
    
    def reset_goal(self):
        # The goal is only on the sliding door
        # self._goal = np.random.uniform(0, 0.6)
        self._goal = 0.6
        
        # # Naive implementation
        # # TODO reset is recursive here
        # self.reset()
        # goal = self.sim.data.qpos[:].squeeze()
        # # qpos has 41 components: first 9 unknown, next 21 for the 3 objects, 10 unknown, 1 for the door
        #
        # self.data.qpos[40] = np.random.uniform(0, 0.3)
        #
        # # Blocks
        # for i in range(3):
        #     self.targetobj = i
        #     init_pos = np.random.uniform(
        #         self.obj_low,
        #         self.obj_high,
        #     )
        #     if not self._hp.include_objects:
        #         init_pos = [2, 2, 2]
        #     self.obj_init_pos = init_pos
        #     self.data.qpos[9 + 7 * i:12 + 7 * i] = init_pos
        #
        # # TODO make a reasonable fixed
        # # TODO add rewards w.r.t goal
        #
        # self.set_goal(goal)
    
    def _reset_hand(self, reset_hand_state=None):
        if reset_hand_state is not None:
            pos = reset_hand_state.copy()
        else:
            pos = self.hand_init_pos.copy()
            pos[0] = np.random.uniform(0.2, self.hand_high[0])
            pos[1] = np.random.uniform(-0.6, -0.5)
            # # TODO maybe change this?
            # pos[0] = np.random.uniform(0.05, self.hand_high[0])
            # pos[1] = np.random.uniform(-0.55, -0.5)
            pos[2] = np.random.uniform(self.hand_low[2], self.hand_high[2])
        for _ in range(20):
          self.data.set_mocap_pos('mocap', pos)
          self.data.set_mocap_quat('mocap', self.default_mocap_quat.copy())
          #self.do_simulation([0]*7 + [-1,1], self.frame_skip)
          self.do_simulation([-1,1], self.frame_skip)
        #rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        #self.init_fingerCOM = (rightFinger + leftFinger)/2
        self.pickCompleted = False

    def get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

    def get_body_pos(self, bodyName):
        _id = self.model.body_names.index(bodyName)
        return self.data.body_xpos[_id].copy()

    def reset(self, reset_state=None):
        # self.data.qpos[:9] = np.array([5.20311586e-01, 1.07768693e+00, 9.71443297e-01, 7.49235668e-01,
        #        -1.55658590e+00, -8.32050797e-01, 1.96473385e+00, -4.70370014e-07,
        #        1.07759349e-04])
        self._reset_hand()
        if reset_state is not None:
            if reset_state.shape[0] == 41:
                target_qpos = reset_state
                target_qvel = np.zeros_like(self.data.qvel)
            else:
                target_qpos = reset_state[:41]
                target_qvel = reset_state[41:]
            self.set_state(target_qpos, target_qvel)
        else:
            # Sliding door. 0-0.6 is the max range
            self.data.qpos[40] = np.random.uniform(0, 0.3)
            # TODO remove
            self.data.qpos[40] = 0
            # self.data.qpos[40] = 0.6
            
            # Blocks
            for i in range(3):
                self.targetobj = i
                init_pos = np.random.uniform(
                    self.obj_low,
                    self.obj_high,
                )
                if not self._hp.include_objects:
                    init_pos = [2, 2, 2]
                self.obj_init_pos = init_pos
                self.data.qpos[9+7*i:12+7*i] = init_pos
                #self._set_obj_xyz(self.obj_init_pos)
    
                # self.data.qpos[40] = 0
                for _ in range(10):
                     #self.do_simulation([0.0, 0.0] + [0]*7)
                     self.do_simulation([0.0, 0.0])
        self.update_mocap_pos()
        # self._obs_history = []
        obs = self._get_obs()
        # self._reset_eval()

        self.reset_goal()

        #Can try changing this
        return obs
        #return obs, None

    def step(self, action):
        self.set_xyz_action(action[:3])
        for i in range(10):
            self.do_simulation([action[-1], -action[-1]])
        self.update_mocap_pos()
        self.do_simulation([action[-1], -action[-1]], 1)
        obs = self._get_obs()
        # print('current', obs['state'][40])
        # print(self._goal_obj_pose[-1])
        
        return obs, self.get_reward(), np.zeros([]), {'success': self.get_success()}

    def update_mocap_pos(self):
        # print('mocap', self.data.mocap_pos)
        # print('endeff', self.get_endeff_pos())
        # print(self.data.mocap_pos-self.get_endeff_pos())
        self.data.set_mocap_pos('mocap', self.get_endeff_pos())

    def get_reward(self):
        curr_door_pos = self.sim.data.qpos[40]
        goal_door_pos = self._goal
        reward = - (curr_door_pos - goal_door_pos) ** 2
        return np.asarray(reward)

    def get_success(self):
        curr_door_pos = self.sim.data.qpos[40]
        goal_door_pos = self._goal
        distance = np.abs(curr_door_pos - goal_door_pos)
        return np.asarray(distance < 0.1)

    def has_goal(self):
        return True

    def get_endeff_pos(self):
        # x, y, z
        # x: right of the table, y towards the table, z upwards
        return self.get_body_pos('hand').copy()

    def _get_obs(self):
        obs = {}
        #joint poisitions and velocities
        obs['qpos'] = copy.deepcopy(self.sim.data.qpos[:].squeeze())
        obs['qvel'] = copy.deepcopy(self.sim.data.qvel[:].squeeze())
        obs['gripper'] = self.get_endeff_pos()
        # obs['state'] = np.concatenate([obs['gripper'], copy.deepcopy(self.sim.data.qpos[:].squeeze()),
        #                                  copy.deepcopy(self.sim.data.qvel[:].squeeze())])
        obs['state'] = np.concatenate([copy.deepcopy(self.sim.data.qpos[:].squeeze()),
                                       copy.deepcopy(self.sim.data.qvel[:].squeeze())])
        obs['object_qpos'] = copy.deepcopy(self.sim.data.qpos[9:].squeeze())

        #copy non-image data for environment's use (if needed)
        # self._last_obs = copy.deepcopy(obs)
        # self._obs_history.append(copy.deepcopy(obs))

        #get images
        obs['images'] = self.render()
        obs['env_done'] = np.asarray(0)
        return obs
  
    def valid_rollout(self):
        return True

    def current_obs(self):
        return self._get_obs()
  
    def get_goal(self):
        return self.goalim
  
    def has_goal(self):
        return True

    def reset_model(self):
        pass