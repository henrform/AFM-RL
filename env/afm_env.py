from gymnasium.core import ObsType
from typing import Literal, Any, SupportsFloat
from ppafm.io import loadXYZ
from ppafm.ocl.AFMulator import AFMulator
from ppafm.ocl.oclUtils import init_env
from ppafm.common import PpafmParameters
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema
import numpy as np
from collections import defaultdict
import gymnasium as gym
import os


class AfmEnvironment(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def _compute_imgs(self, surface_path: str, params_path: str, i_platform: int = 0,
                      angle_deg: float = 0.0, tx: float = 0.0, ty: float = 0.0) -> tuple[AFMulator, np.ndarray]:
        """
        Generates AFM images with ppafm

        Parameters
        ----------
        surface_path : str
            Path to a .xyz file containing the surface
        params_path : str
            Path to a .ini file containing the parameters for the simulation
        i_platform : int
            Index of OpenCL device

        Returns
        -------
        AFMulator
            AFMulator object used to generate images
        np.ndarray
            Generated images. The second and third axis are already reversed.
        """
        init_env(i_platform=i_platform)

        parameters = PpafmParameters.from_file(params_path)

        xyzs, Zs, qs, _ = loadXYZ(surface_path)

        theta = np.radians(angle_deg)
        c, s = np.cos(theta), np.sin(theta)

        Rz = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])

        translation = np.array([tx, ty, 0.0])

        xyzs = np.dot(xyzs, Rz.T) + translation

        gridA = np.dot(Rz, parameters.gridA).tolist()
        gridB = np.dot(Rz, parameters.gridB).tolist()
        gridC = np.dot(Rz, parameters.gridC).tolist()

        parameters.gridA = gridA
        parameters.gridB = gridB
        parameters.gridC = gridC

        new_params_path = params_path.replace(".ini", "_displaced.toml")

        parameters.to_file(new_params_path)

        afmulator = AFMulator.from_params(new_params_path)
        afm_images = afmulator(xyzs, Zs, qs)

        return afmulator, afm_images[:, ::-1, ::-1]

    def save_to_file(self, path: str):
        """
        Saves the necessary environment variables to a .npz file.

        Parameters
        ----------
        Path
            Directory where to save the environment to
        """
        np.savez(
            path,
            afm_images=self.afm_images,
            z_height_map=self.z_height_map,
            min_image=self.min_image,
            z_bounds=np.array([self.z_min, self.z_max])
        )

    # TODO: Implement render mode
    # TODO: Figure out how to handle multiple surfaces at once
    def __init__(self,
                 surface_path: str = None,
                 params_path: str = None,
                 data_file_path: str = None,
                 i_platform: int = 0,
                 sigma: int = 4,
                 num_historic_data: int = 4,
                 height_offset_reward=0.3,
                 num_actions=1,
                 render_mode: Literal[None, 'human', 'rgb'] = None,
                 norm_margin: float = 0.5,  # Margin in Angstroms for normalization masking
                 angle_deg: float = 0.0,
                 tx: float = 0.0,
                 ty: float = 0.0
                 ) -> None:
        """
        Constructor

        Parameters
        ----------
        surface_path : str
            Path to a .xyz file containing the surface
        params_path : str
            Path to a .ini file containing the parameters for the simulation
        i_platform : int
            Index of OpenCL device
        render_mode : Literal[None, 'human', 'rgb']
            Render mode to use
        """
        super().__init__()

        self.num_historic_data = num_historic_data
        self.height_offset_reward = height_offset_reward
        self.num_actions = num_actions

        if data_file_path and os.path.exists(data_file_path):
            data = np.load(data_file_path)
            self.afm_images = data['afm_images']
            self.z_height_map = data['z_height_map']
            self.min_image = data['min_image']
            self.z_min, self.z_max = data['z_bounds']
        else:
            if not surface_path or not params_path:
                raise ValueError("Must provide surface_path and params_path if data_file_path is missing.")

            # Generate images
            afmulator, self.afm_images = self._compute_imgs(surface_path, params_path, i_platform, angle_deg, tx, ty)

            # Calculate heights for each slice
            self.z_height_map = np.linspace(
                afmulator.scan_window[0][2],
                afmulator.scan_window[1][2] - afmulator.df_steps * afmulator.dz,
                afmulator.scan_dim[2] - afmulator.df_steps + 1,
            )

            # Get all minima in the z direction and only keep the highest one
            minima = np.array(argrelextrema(self.afm_images, np.less_equal, axis=2)).T
            minima_dict = defaultdict(list)
            for xx, yy, zz in minima:
                minima_dict[xx, yy].append(zz)
                minima_dict[xx, yy] = [max(minima_dict[xx, yy])]

            argmin_image = np.zeros(self.afm_images.shape[0:-1], dtype=int)
            for pixel, val in minima_dict.items():
                argmin_image[pixel] = int(val[0])

            # Convert index to actual height and smooth optimal surface
            self.min_image = self.z_height_map[argmin_image]

            self.z_min = afmulator.scan_window[0][2]
            self.z_max = afmulator.scan_window[1][2] - afmulator.df_steps * afmulator.dz

            del afmulator

        # TODO: Shift optimal height upwards?
        self.optimal_height = gaussian_filter(self.min_image, sigma=sigma) + self.height_offset_reward

        self._z_map_start = self.z_height_map[0]
        self._z_map_step = self.z_height_map[1] - self.z_height_map[0]
        self._z_map_len_minus_1 = len(self.z_height_map) - 1
        self._x_lim = self.afm_images.shape[0]
        self._y_lim = self.afm_images.shape[1]

        # --- NORMALIZATION LOGIC START ---

        # 1. Create a mask for valid simulation space (above the surface)
        # We assume the agent will terminate if it goes below min_image, so we
        # only want to normalize based on values it can actually see alive.

        # Broadcasting: (1, 1, Z) vs (X, Y, 1) -> (X, Y, Z) boolean mask
        z_grid = self.z_height_map[np.newaxis, np.newaxis, :]
        surface_grid = self.min_image[:, :, np.newaxis]

        # Include a small margin (e.g. 0.5 A) because the agent might step slightly
        # below the exact minimum before the termination condition triggers.
        valid_mask = z_grid >= (surface_grid - norm_margin)

        # Extract valid df values to find realistic min/max
        valid_df_values = self.afm_images[valid_mask]

        self.norm_bounds = {
            'x': (0.0, float(self._x_lim - 1)),
            'y': (0.0, float(self._y_lim - 1)),
            'df': (float(np.min(valid_df_values)), float(np.max(valid_df_values))),
            # dz is relative to start, max possible deviation is the full Z scan range
            'dz': (float(-(self.z_max - self.z_min)), float(self.z_max - self.z_min))
        }

        # --- NORMALIZATION LOGIC END ---

        # Define observation space
        # Everything is now float32 and normalized to [-1, 1]
        self.observation_space = gym.spaces.Dict(
            {
                "x": gym.spaces.Box(-1.0, 1.0, shape=(num_historic_data,), dtype=np.float32),
                "y": gym.spaces.Box(-1.0, 1.0, shape=(num_historic_data,), dtype=np.float32),
                "dz": gym.spaces.Box(-1.0, 1.0, shape=(num_historic_data,), dtype=np.float32),
                "df": gym.spaces.Box(-1.0, 1.0, shape=(num_historic_data,), dtype=np.float32),
            }
        )

        # Define action space
        # TODO: What is the maximum speed for the tip?
        if self.num_actions > 1:
            self.action_space = gym.spaces.Discrete(self.num_actions)
            self._action_map = np.linspace(-1.0, 1.0, self.num_actions)
        else:
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float64)

        self.reset()

    def _normalize(self, value: np.ndarray | float, key: str) -> np.ndarray | float:
        """
        Normalize a physical value to range [-1, 1].
        Formula: 2 * (x - min) / (max - min) - 1
        """
        min_v, max_v = self.norm_bounds[key]
        return 2.0 * ((value - min_v) / (max_v - min_v)) - 1.0

    def denormalize_observation(self, obs_dict: dict) -> dict:
        """
        Convert a normalized observation dictionary back to physical units.
        Useful for plotting and interpretation.
        Formula: (x + 1) / 2 * (max - min) + min
        """
        physical_obs = {}
        for key, value in obs_dict.items():
            if key in self.norm_bounds:
                min_v, max_v = self.norm_bounds[key]
                physical_obs[key] = (value + 1.0) / 2.0 * (max_v - min_v) + min_v
            else:
                physical_obs[key] = value
        return physical_obs

    def _get_closest_slice_index(self, z: float) -> np.int64:
        """
        Returns the index of the slice closest to a given z height

        Parameters
        ----------
        z : float
            Height of the desired slice

        Returns
        -------
        int : index of the closest slice
        """
        return np.abs(self.z_height_map - z).argmin()

    def _get_two_closest_z_planes(self, z):
        """
        Returns the indices of the two closest values in array to the given value.

        Parameters
        ----------
        value : float
            The target value

        Returns
        -------
        tuple : (int, int)
            Indices of the two closest values in ascending order
        """
        return np.sort(np.argsort(np.abs(self.z_height_map - z))[0:2])

    def _get_interpolated_df(self, x: int, y: int, z: float):
        """
        Interpolates a value at the given xy coordinates
        for a specific height `z` using two closest z-planes.

        Parameters
        ----------
        x : int
            The x index
        y : int
            The y index
        z : float
            The z position at which the value needs to be interpolated.

        Returns
        -------
        float
            The interpolated value at the specified z-coordinate.
        """
        z1, z2 = self._get_two_closest_z_planes(z)
        denom = (self.z_height_map[z2] - self.z_height_map[z1])
        if denom == 0:
            return self.afm_images[x, y, z1]

        k = (self.afm_images[x, y, z2] - self.afm_images[x, y, z1]) / denom
        return k * (z - self.z_height_map[z1]) + self.afm_images[x, y, z1]

    def _get_obs(self):
        """
        Convert internal state to observation format.

        Returns
        -------
        dict
            Observation dictionary with x and y position, difference to the starting height and change in frequency
        """
        return {
            "x": self._normalize(self._x, 'x').astype(np.float32),
            "y": self._normalize(self._y, 'y').astype(np.float32),
            "dz": self._normalize(self._dz, 'dz').astype(np.float32),
            "df": self._normalize(self._df, 'df').astype(np.float32)
        }

    def _get_info(self):
        """
        Compute auxiliary information for debugging.

        Returns
        -------
            dict : z height and corresponding index
        """
        return {
            "z": self.z_start,
            "z_i": self.z_start_index,
            "generated_image": self.generated_image,
        }

    # TODO: Randomize start position and rotation. Maybe also switch between different surfaces?
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        """
        Resets the environment

        Parameters
        ----------
        seed : int | None
            Seed for the random number generator
        options : dict[str, Any] | None
            Options for super class

        Returns
        -------

        """
        super().reset(seed=seed, options=options)

        if seed is not None:
            np.random.seed(seed)

        self.terminated = False

        z = self.z_max - np.random.rand()
        self.z_start_index = self._get_closest_slice_index(z)
        self.z_start = self.z_height_map[self.z_start_index]

        self._x = np.zeros(self.num_historic_data, dtype=np.int64)
        self._y = np.zeros(self.num_historic_data, dtype=np.int64)
        self._dz = np.zeros(self.num_historic_data, dtype=np.float64)
        self._df = self.afm_images[0, 0, self.z_start_index] * np.ones(self.num_historic_data, dtype=np.float64)

        self.generated_image = np.empty(self.afm_images.shape[:2])
        self.generated_image[:] = np.nan
        self.generated_image[0, 0] = self._get_interpolated_df(0, 0, self.z_start)

        return self._get_obs(), self._get_info()

    def _insert_into_array(self, array: np.ndarray, data_point):
        """
        Modifies an array by rolling it to the right and inserting a new data point at the beginning.

        Parameters
        ----------
        array : numpy.ndarray
            The input array to be modified.
        data_point : Any
            The new data point

        Returns
        -------
        numpy.ndarray
            The modified array
        """
        array = np.roll(array, 1)
        array[0] = data_point
        return array

    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one timestep within the environment.

        Args:
            action: The distance in Ångström that the tip should move.

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        if self.terminated:
            return self._get_obs(), 0.0, True, False, self._get_info()

        x_lim = self._x_lim
        y_lim = self._y_lim

        # Shift history arrays
        self._x[1:] = self._x[:-1]
        self._y[1:] = self._y[:-1]
        self._dz[1:] = self._dz[:-1]
        self._df[1:] = self._df[:-1]

        x_current, y_current = int(self._x[1]), int(self._y[1])

        # Determine new position based on raster scan pattern
        if y_current % 2 == 0:  # Even rows: moving right
            if x_current + 1 >= x_lim:
                # End of row reached, move to next row
                x_new = x_current
                y_new = y_current + 1
            else:
                # Move right
                x_new = x_current + 1
                y_new = y_current
        else:  # Odd rows: moving left
            if x_current - 1 < 0:
                # Beginning of row reached, move to next row
                x_new = x_current
                y_new = y_current + 1
            else:
                # Move left
                x_new = x_current - 1
                y_new = y_current

        self._x[0] = x_new
        self._y[0] = y_new

        # Update dz and df
        # TODO: Clip z
        if self.num_actions > 1:
            continuous_action = self._action_map[action]
            dz_new = self._dz[1] + continuous_action
        else:
            dz_new = self._dz[1] + action[0]

        self._dz[0] = dz_new
        z_new = self.z_start + dz_new


        # Calculate float index in the z_map
        idx_float = (z_new - self._z_map_start) / self._z_map_step

        # Floor to get lower index and clamp
        idx_1 = int(idx_float)
        if idx_1 < 0:
            idx_1 = 0
            idx_2 = 0
            fraction = 0.0
        elif idx_1 >= self._z_map_len_minus_1:
            idx_1 = self._z_map_len_minus_1
            idx_2 = self._z_map_len_minus_1
            fraction = 0.0
        else:
            idx_2 = idx_1 + 1
            fraction = idx_float - idx_1

        # Interpolate
        v1 = self.afm_images[x_new, y_new, idx_1]
        v2 = self.afm_images[x_new, y_new, idx_2]
        df_new = v1 + fraction * (v2 - v1)

        self._df[0] = df_new
        self.generated_image[x_new, y_new] = df_new

        # Check for crashes
        # TODO: Add tolerance?
        if z_new < self.min_image[x_new, y_new]:
            # Terminate, but return the normalized observation (which might be < -1.0)
            return self._get_obs(), -100.0, True, False, self._get_info()

        # TODO: Base reward should be set by variable
        reward = 10.0 - (z_new - self.optimal_height[x_new, y_new])

        # Termination check
        if y_new == y_lim - 1:  # Check last row
            if y_new % 2 == 0 and x_new == x_lim - 1:
                self.terminated = True
            elif y_new % 2 == 1 and x_new == 0:
                self.terminated = True

        return self._get_obs(), reward, self.terminated, False, self._get_info()
