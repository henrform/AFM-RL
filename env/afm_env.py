from gymnasium.core import ObsType
from typing import Literal, Any, SupportsFloat
from ppafm.io import loadXYZ
from ppafm.ocl.AFMulator import AFMulator
from ppafm.ocl.oclUtils import init_env
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema
import numpy as np
from collections import defaultdict
import gymnasium as gym
import os


class AfmEnvironment(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def _compute_imgs(self, surface_path: str, params_path: str, i_platform: int = 0) -> tuple[AFMulator, np.ndarray]:
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
        xyzs, Zs, qs, _ = loadXYZ(surface_path)
        afmulator = AFMulator.from_params(params_path)
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
                 render_mode: Literal[None, 'human', 'rgb'] = None
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
            afmulator, self.afm_images = self._compute_imgs(surface_path, params_path, i_platform)

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

        # Define observation space
        x_px_max, y_px_max, _ = self.afm_images.shape

        self.observation_space = gym.spaces.Dict(
            {
                "x": gym.spaces.Box(0, x_px_max - 1, shape=(num_historic_data,), dtype=np.int64),
                "y": gym.spaces.Box(0, y_px_max - 1, shape=(num_historic_data,), dtype=np.int64),
                "dz": gym.spaces.Box(-self.z_max, self.z_max, shape=(num_historic_data,), dtype=np.float64),
                "df": gym.spaces.Box(-np.inf, np.inf, shape=(num_historic_data,), dtype=np.float64),
            }
        )

        # Define action space
        # TODO: What is the maximum speed for the tip?
        if self.num_actions > 1:
            self.action_space = gym.spaces.Discrete(self.num_actions)
            self._action_map = np.linspace(-1.0, 1.0, self.num_actions)
        else:
            self.action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float64)

        self.reset()

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
        k = (self.afm_images[x, y, z2] - self.afm_images[x, y, z1]) / (self.z_height_map[z2] - self.z_height_map[z1])
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
            "x": self._x,
            "y": self._y,
            "dz": self._dz,
            "df": self._df
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
        self._dz = np.zeros(self.num_historic_data)
        self._df = self.afm_images[0, 0, self.z_start_index] * np.ones(self.num_historic_data)

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
            return self._get_obs(), 0, True, False, self._get_info()

        x_px_max, y_px_max, _ = self.afm_images.shape
        x_current, y_current = int(self._x[0]), int(self._y[0])

        # Determine new position based on raster scan pattern
        if y_current % 2 == 0:  # Even rows: moving right
            if x_current + 1 >= x_px_max:
                # End of row reached, move to next row
                self._x = self._insert_into_array(self._x, x_current)
                self._y = self._insert_into_array(self._y, y_current + 1)
            else:
                # Move right
                self._x = self._insert_into_array(self._x, x_current + 1)
                self._y = self._insert_into_array(self._y, y_current)
        else:  # Odd rows: moving left
            if x_current - 1 < 0:
                # Beginning of row reached, move to next row
                self._x = self._insert_into_array(self._x, x_current)
                self._y = self._insert_into_array(self._y, y_current + 1)
            else:
                # Move left
                self._x = self._insert_into_array(self._x, x_current - 1)
                self._y = self._insert_into_array(self._y, y_current)

        x_new, y_new = int(self._x[0]), int(self._y[0])

        # Update dz and df
        # TODO: Clip z
        if self.num_actions > 1:
            continuous_action = self._action_map[action]
            dz_new = self._dz[0] + continuous_action
        else:
            dz_new = self._dz[0] + action[0]
        self._dz = self._insert_into_array(self._dz, dz_new)
        z_new = self.z_start + dz_new
        self._df = self._insert_into_array(self._df, self._get_interpolated_df(x_new, y_new, z_new))
        self.generated_image[x_new, y_new] = self._df[0]

        # Check for crashes
        # TODO: Add tolerance?
        if z_new < self.min_image[x_new, y_new]:
            return self._get_obs(), -100, True, False, self._get_info()

        # TODO: Base reward should be set by variable
        reward = 10 - (z_new - self.optimal_height[x_new, y_new])

        if y_new % 2 == 0 and y_new == y_px_max - 1 and x_new == x_px_max - 1:
            self.terminated = True
        elif y_new % 2 == 1 and y_new == 0 and x_new == 0:
            self.terminated = True

        return self._get_obs(), reward, self.terminated, False, self._get_info()
