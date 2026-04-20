from gymnasium.core import ObsType
from typing import Literal, Any, SupportsFloat
from ppafm.io import loadXYZ
from ppafm.ocl.AFMulator import AFMulator
from ppafm.ocl.oclUtils import init_env
from ppafm.common import PpafmParameters
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema
import numpy as np
from collections import defaultdict
import gymnasium as gym
import os
import tempfile


def jitter_penalty(
    dz_window: np.ndarray,
    df_window: np.ndarray,
    x_window: np.ndarray,
    y_window: np.ndarray,
    scale: float = 1.0,
) -> float:
    """
    Penalty based on total variation of dz over the window.
    Sums the absolute differences between consecutive dz values, scaled by `scale`.
    Pass a negative scale to subtract from the reward (penalize jitter).

    Usage
    -----
    env = AfmEnvironment(
        ...,
        step_custom_reward_fns=[jitter_penalty],
        step_custom_reward_kwargs=[{'scale': -0.5}],
    )

    Parameters
    ----------
    dz_window : np.ndarray
        Array of recent dz values.
    df_window : np.ndarray
        Array of recent df values.
    x_window : np.ndarray
        Array of recent x positions.
    y_window : np.ndarray
        Array of recent y positions.
    scale : float
        Multiplicative scale factor applied to the total variation.
        Use a negative value to penalize jitter.

    Returns
    -------
    float
    """
    return scale * float(np.sum(np.abs(np.diff(dz_window))))


class AfmEnvironment(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def _load_view(self, view_idx: int) -> dict:
        """
        Ensure data for a selected view is loaded into memory.

        For directory-backed lazy views, memory-maps the AFM image stack and
        loads the small metadata arrays (height map, min image, z bounds) on
        first access, then computes the smoothed optimal-height surface.
        Already-loaded or eager views are returned unchanged.

        Parameters
        ----------
        view_idx : int
            Index into ``self.views`` of the view to load.

        Returns
        -------
        dict
            The (now populated) view dict from ``self.views``.
        """
        view = self.views[view_idx]

        if view.get('source') != 'data_dir_lazy':
            return view

        if view['z_height_map'] is not None and view['min_image'] is not None:
            return view

        afm_images = np.load(view['afm_path'], mmap_mode='r')
        with np.load(view['meta_path']) as meta:
            z_height_map = meta['z_height_map']
            min_image = meta['min_image']
            z_bounds = meta['z_bounds']

        z_min, z_max = z_bounds
        optimal_height = gaussian_filter(min_image, sigma=self.sigma) + self.height_offset_reward

        view['afm_images'] = afm_images
        view['z_height_map'] = z_height_map
        view['min_image'] = min_image
        view['z_min'] = float(z_min)
        view['z_max'] = float(z_max)
        view['optimal_height'] = optimal_height
        view['_z_map_start'] = z_height_map[0]
        view['_z_map_step'] = z_height_map[1] - z_height_map[0]
        view['_z_map_len_minus_1'] = len(z_height_map) - 1

        return view

    def _unload_view(self, view_idx: int) -> None:
        """
        Drop loaded arrays for a directory-backed lazy view.

        Resets the view's image, metadata, and derived fields back to ``None``
        so memory can be reclaimed. No-op for eagerly loaded views.

        Parameters
        ----------
        view_idx : int
            Index into ``self.views`` of the view to unload.
        """
        view = self.views[view_idx]
        if view.get('source') != 'data_dir_lazy':
            return

        view['afm_images'] = None
        view['z_height_map'] = None
        view['min_image'] = None
        view['z_min'] = None
        view['z_max'] = None
        view['optimal_height'] = None
        view['_z_map_start'] = None
        view['_z_map_step'] = None
        view['_z_map_len_minus_1'] = None

    def _compute_imgs(self, surface_path: str, params_path: str, i_platform: int = 0,
                      angle_deg: float = 0.0, tx: float = 0.0, ty: float = 0.0) -> tuple[AFMulator, np.ndarray]:
        """
        Generate AFM images with ppafm for a (rotated, translated) surface.

        Rotates the atom positions and grid vectors by ``angle_deg`` around z,
        then translates by ``(tx, ty)`` before running the AFMulator. The
        returned image stack has its second and third axes reversed so it is
        indexed as ``(x, y, z)`` in scan coordinates.

        Parameters
        ----------
        surface_path : str
            Path to a .xyz file containing the surface.
        params_path : str
            Path to a .ini file containing the parameters for the simulation.
        i_platform : int
            Index of OpenCL device.
        angle_deg : float
            In-plane rotation of the surface in degrees.
        tx : float
            Translation along x applied after rotation.
        ty : float
            Translation along y applied after rotation.

        Returns
        -------
        AFMulator
            AFMulator object used to generate images.
        np.ndarray
            Generated images (float16). The second and third axes are already
            reversed.
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

        with tempfile.NamedTemporaryFile(
            suffix=".toml", prefix="ppafm_params_", delete=False
        ) as tmp:
            new_params_path = tmp.name

        try:
            parameters.to_file(new_params_path)
            afmulator = AFMulator.from_params(new_params_path)
        finally:
            os.unlink(new_params_path)

        afm_images = afmulator(xyzs, Zs, qs)

        return afmulator, afm_images[:, ::-1, ::-1].astype(np.float16)

    def save_to_file(self, path: str, surface_idx: int | None = None):
        """
        Saves views to a folder, one subdirectory per view.

        Each view is saved as view_00000/afm_images.npy (uncompressed, float16)
        and view_00000/meta.npz (small metadata arrays). The uncompressed .npy
        format enables true memory-mapping via np.load(..., mmap_mode='r').

        Parameters
        ----------
        path : str
            Directory where to save the environment views to.
        surface_idx : int | None
            If given, only save views belonging to this surface config index.
            If None, save all views.
        """
        if surface_idx is not None:
            start, end = self.surface_view_ranges[surface_idx]
            views_to_save = self.views[start:end]
        else:
            views_to_save = self.views

        os.makedirs(path, exist_ok=True)
        for i, view in enumerate(views_to_save):
            if view.get('source') == 'data_dir_lazy':
                # i indexes the selected subset only if surface_idx is used.
                global_view_idx = self.views.index(view)
                view = self._load_view(global_view_idx)

            sp = view.get('scan_params')
            if sp is not None:
                ang = sp.get('angle_deg', 0.0)
                tx  = sp.get('tx', 0.0)
                ty  = sp.get('ty', 0.0)
                folder_name = f"view_ang{ang:g}_tx{tx:g}_ty{ty:g}"
            else:
                folder_name = f"view_{i:05d}"
            view_dir = os.path.join(path, folder_name)
            os.makedirs(view_dir, exist_ok=True)
            # Uncompressed .npy so np.load can truly memory-map it
            np.save(
                os.path.join(view_dir, "afm_images.npy"),
                view['afm_images'].astype(np.float16)
            )
            # Small metadata arrays; compression is fine here
            np.savez(
                os.path.join(view_dir, "meta.npz"),
                z_height_map=view['z_height_map'],
                min_image=view['min_image'],
                z_bounds=np.array([view['z_min'], view['z_max']])
            )

    # TODO: Implement render mode
    def __init__(self,
                 surface_configs: list[dict],
                 i_platform: int = 0,
                 sigma: int = 4,
                 num_historic_data: int = 4,
                 height_offset_reward: float = 0.3,
                 num_actions: int = 1,
                 render_mode: Literal[None, 'human', 'rgb'] = None,
                 df_scale: float = 10.0,
                 dz_scale: float = 10.0,
                 base_reward: float = 10.0,
                 crash_reward: float = -100.0,
                 termination_reward: float = 1000.0,
                 reward_ceiling_offset: float = 10.0,
                 reward_exponent: float = 1.0,
                 step_custom_reward_fns: list | None = None,
                 step_custom_reward_kwargs: list[dict] | None = None,
                 step_custom_reward_windows: list[int] | None = None,
                 include_image_in_info: bool = False,
                 unload_view_on_reset: bool = True,
                 ) -> None:
        """
        Constructor.

        Parameters
        ----------
        surface_configs : list[dict]
            List of surface configuration dicts. Each dict specifies one
            surface and must contain one of the following source options:

            - ``surface_path`` (str) + ``params_path`` (str): compute views
              from scratch using ppafm.  Optionally include ``scan_params``
              (list[dict]) with keys ``angle_deg``, ``tx``, ``ty`` for
              multiple rotated / translated views of the same surface.
            - ``data_dir_path`` (str): load pre-computed views from a
              directory containing ``view_*/`` subdirectories (or legacy
              ``view_*.npz`` files).
            - ``data_file_path`` (str): load a single legacy ``.npz`` file
              as one view.

            Example::

                surface_configs = [
                    {
                        'surface_path': 'materials/pt_111_small.xyz',
                        'params_path': 'materials/params.ini',
                        'scan_params': [
                            {'angle_deg': 0, 'tx': 0, 'ty': 0},
                            {'angle_deg': 30, 'tx': 5, 'ty': 0},
                        ],
                    },
                    {
                        'data_dir_path': 'environments/pt_111_small_rows_missing',
                    },
                ]

        i_platform : int
            Index of OpenCL device.
        sigma : int
            Standard deviation for Gaussian smoothing of the optimal height.
        num_historic_data : int
            Number of historic time-steps in the observation.
        height_offset_reward : float
            Offset above the smoothed min-image defining the optimal height.
        num_actions : int
            Number of discrete actions.  Use 1 for a continuous action space.
        render_mode : Literal[None, 'human', 'rgb']
            Render mode to use.
        df_scale : float
            Divisor applied to df observations (physical units).
        dz_scale : float
            Divisor applied to dz observations (physical units).
        base_reward : float
            Reward given when the tip is at or above the optimal height.
        crash_reward : float
            Penalty applied in the danger zone (scaled) and on a crash (full).
        termination_reward : float
            Bonus reward added on successful scan completion.
        reward_ceiling_offset : float
            Vertical margin above the optimal height where the agent is still
            considered in-bounds. If ``z_new`` exceeds
            ``optimal_height + reward_ceiling_offset``, the episode terminates
            with ``crash_reward``.
        reward_exponent : float
            Exponent used for the above-optimal-height penalty term in step().
            The reward in that region becomes
            ``base_reward - (z_new - z_opt) ** reward_exponent``.
        step_custom_reward_fns : list[callable] | None
            List of ``fn(dz_window, df_window, x_window, y_window, **kwargs) -> float``
            callables, each called every non-crash step. Their return values
            are summed and added to the reward (use negative values to
            penalize). ``None`` (default) disables custom rewards entirely.
        step_custom_reward_kwargs : list[dict] | None
            One kwargs dict per entry in *step_custom_reward_fns*.
            Defaults to empty dicts.
        step_custom_reward_windows : list[int] | None
            One window size per entry in *step_custom_reward_fns*.
            Each ``None`` entry defaults to *num_historic_data*.
        include_image_in_info : bool
            If True, the current AFM image slice is included in the info dict
            returned by step() under the key 'current_image'. This can be useful
            for debugging and visualization but may slow down training.
        unload_view_on_reset : bool
            If True (default), unload the previously active lazy-loaded view when
            switching to a different view in reset().
        """
        super().__init__()

        self.num_historic_data = num_historic_data
        self.height_offset_reward = height_offset_reward
        self.num_actions = num_actions
        self.sigma = sigma
        self.df_scale = df_scale
        self.dz_scale = dz_scale
        self.base_reward = base_reward
        self.crash_reward = crash_reward
        self.termination_reward = termination_reward
        self.reward_ceiling_offset = reward_ceiling_offset
        self.reward_exponent = reward_exponent
        self.step_custom_reward_fns = step_custom_reward_fns or []
        n = len(self.step_custom_reward_fns)
        self.step_custom_reward_kwargs = step_custom_reward_kwargs if step_custom_reward_kwargs is not None else [{}] * n
        self.step_custom_reward_windows = step_custom_reward_windows if step_custom_reward_windows is not None else [num_historic_data] * n
        self.include_image_in_info = include_image_in_info
        self.unload_view_on_reset = unload_view_on_reset

        # --- Load or compute views from all surface configs ---
        self.views = []
        self.surface_view_ranges = []  # (start_idx, end_idx) per surface config

        for config in surface_configs:
            start_idx = len(self.views)

            data_dir = config.get('data_dir_path')
            data_file = config.get('data_file_path')
            surface = config.get('surface_path')
            params = config.get('params_path')

            if data_dir and os.path.isdir(data_dir):
                # New format: view_*/ subdirectories, each with afm_images.npy + meta.npz
                view_dirs = sorted([
                    d for d in os.listdir(data_dir)
                    if d.startswith("view_") and os.path.isdir(os.path.join(data_dir, d))
                ])
                if view_dirs:
                    for vd in view_dirs:
                        vpath = os.path.join(data_dir, vd)
                        afm_path = os.path.join(vpath, "afm_images.npy")
                        meta_path = os.path.join(vpath, "meta.npz")

                        # Read shape from .npy header via memmap without loading array data.
                        afm_images = np.load(afm_path, mmap_mode='r')
                        x_lim = afm_images.shape[0]
                        y_lim = afm_images.shape[1]

                        self.views.append({
                            'source': 'data_dir_lazy',
                            'afm_path': afm_path,
                            'meta_path': meta_path,
                            'x_lim': x_lim,
                            'y_lim': y_lim,
                            'afm_images': None,
                            'z_height_map': None,
                            'min_image': None,
                            'z_min': None,
                            'z_max': None,
                            'optimal_height': None,
                            '_z_map_start': None,
                            '_z_map_step': None,
                            '_z_map_len_minus_1': None,
                        })
                else:
                    # Legacy format: flat view_*.npz files (no true memmap)
                    view_files = sorted([
                        f for f in os.listdir(data_dir)
                        if f.startswith("view_") and f.endswith(".npz")
                    ])
                    if not view_files:
                        raise ValueError(
                            f"No view_*/ directories or view_*.npz files in {data_dir}"
                        )
                    for vf in view_files:
                        data = np.load(os.path.join(data_dir, vf))
                        self.views.append(self._build_view_from_data(data, sigma))

            elif data_file and os.path.exists(data_file):
                # Legacy: single .npz file loaded as one view (no true memmap)
                data = np.load(data_file)
                self.views.append(self._build_view_from_data(data, sigma))

            else:
                if not surface or not params:
                    raise ValueError(
                        "Each surface config must provide 'data_dir_path', "
                        "'data_file_path', or both 'surface_path' and 'params_path'."
                    )

                scan_params = config.get('scan_params', [{'angle_deg': 0.0, 'tx': 0.0, 'ty': 0.0}])
                for sp in scan_params:
                    view = self._compute_view(
                        surface, params, i_platform, sigma,
                        angle_deg=sp.get('angle_deg', 0.0),
                        tx=sp.get('tx', 0.0),
                        ty=sp.get('ty', 0.0),
                    )
                    self.views.append(view)

            self.surface_view_ranges.append((start_idx, len(self.views)))

        if not self.views:
            raise ValueError("No views were loaded or computed from the provided surface_configs.")

        # Store per-view spatial dimensions
        for view in self.views:
            if 'x_lim' not in view or 'y_lim' not in view:
                view['x_lim'] = view['afm_images'].shape[0]
                view['y_lim'] = view['afm_images'].shape[1]

        # Initialize norm_bounds and spatial limits from first view
        # (updated per episode in reset to match the active view)
        ref = self.views[0]
        self._x_lim = ref['x_lim']
        self._y_lim = ref['y_lim']
        self.norm_bounds = {
            'x': (0.0, float(self._x_lim - 1)),
            'y': (0.0, float(self._y_lim - 1)),
        }

        # Define observation space
        # x, y normalized to [-1, 1]; df, dz scaled by fixed divisors
        self.observation_space = gym.spaces.Dict(
            {
                "x": gym.spaces.Box(-1.0, 1.0, shape=(num_historic_data,), dtype=np.float32),
                "y": gym.spaces.Box(-1.0, 1.0, shape=(num_historic_data,), dtype=np.float32),
                "dz": gym.spaces.Box(-np.inf, np.inf, shape=(num_historic_data,), dtype=np.float32),
                "df": gym.spaces.Box(-np.inf, np.inf, shape=(num_historic_data,), dtype=np.float32),
            }
        )

        # Define action space
        # TODO: What is the maximum speed for the tip?
        if self.num_actions > 1:
            self.action_space = gym.spaces.Discrete(self.num_actions)
            self._action_map = np.linspace(-1.0, 1.0, self.num_actions)
        else:
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float64)

        self.active_view_idx = None
        self.reset()

    def _build_view_from_data(self, data: np.lib.npyio.NpzFile | dict, sigma: int) -> dict:
        """
        Build a view dict from previously saved data.

        Extracts the AFM image stack, z-height map, min-image, and z bounds,
        and precomputes the smoothed optimal-height surface along with cached
        z-map indexing helpers.

        Parameters
        ----------
        data : np.lib.npyio.NpzFile | dict
            Mapping-like object containing ``afm_images``, ``z_height_map``,
            ``min_image``, and ``z_bounds``. When ``afm_images`` is already
            float16 (e.g. memory-mapped from ``.npy``), ``copy=False`` avoids
            an unnecessary allocation.
        sigma : int
            Standard deviation for Gaussian smoothing of ``min_image`` used to
            derive the optimal-height surface.

        Returns
        -------
        dict
            A view dict ready to be appended to ``self.views``.
        """
        afm_images = data['afm_images'].astype(np.float16, copy=False)
        z_height_map = data['z_height_map']
        min_image = data['min_image']
        z_min, z_max = data['z_bounds']

        # TODO: Shift optimal height upwards?
        optimal_height = gaussian_filter(min_image, sigma=sigma) + self.height_offset_reward

        return {
            'afm_images': afm_images,
            'z_height_map': z_height_map,
            'min_image': min_image,
            'z_min': float(z_min),
            'z_max': float(z_max),
            'optimal_height': optimal_height,
            '_z_map_start': z_height_map[0],
            '_z_map_step': z_height_map[1] - z_height_map[0],
            '_z_map_len_minus_1': len(z_height_map) - 1,
        }

    def _compute_view(self, surface_path: str, params_path: str, i_platform: int,
                      sigma: int, angle_deg: float, tx: float, ty: float) -> dict:
        """
        Compute a single view from scratch and return a view dict.

        Runs the AFMulator for the given rotation/translation, extracts the
        highest local-minimum z-slice per pixel to form ``min_image``, and
        derives the smoothed optimal-height surface.

        Parameters
        ----------
        surface_path : str
            Path to the .xyz surface file.
        params_path : str
            Path to the ppafm parameters .ini file.
        i_platform : int
            Index of OpenCL device.
        sigma : int
            Standard deviation of the Gaussian smoothing applied to
            ``min_image`` to produce the optimal-height surface.
        angle_deg : float
            In-plane rotation of the surface in degrees.
        tx : float
            Translation along x applied after rotation.
        ty : float
            Translation along y applied after rotation.

        Returns
        -------
        dict
            A view dict including the scan parameters that produced it.
        """
        afmulator, afm_images = self._compute_imgs(
            surface_path, params_path, i_platform, angle_deg, tx, ty
        )
        afm_images = afm_images.astype(np.float16)

        z_height_map = np.linspace(
            afmulator.scan_window[0][2],
            afmulator.scan_window[1][2] - afmulator.df_steps * afmulator.dz,
            afmulator.scan_dim[2] - afmulator.df_steps + 1,
        )

        # Get all minima in the z direction and only keep the highest one
        minima = np.array(argrelextrema(afm_images, np.less_equal, axis=2)).T
        minima_dict = defaultdict(list)
        for xx, yy, zz in minima:
            minima_dict[xx, yy].append(zz)
            minima_dict[xx, yy] = [max(minima_dict[xx, yy])]

        argmin_image = np.zeros(afm_images.shape[0:-1], dtype=int)
        for pixel, val in minima_dict.items():
            argmin_image[pixel] = int(val[0])

        # Convert index to actual height and smooth optimal surface
        min_image = z_height_map[argmin_image]

        z_min = afmulator.scan_window[0][2]
        z_max = afmulator.scan_window[1][2] - afmulator.df_steps * afmulator.dz

        # TODO: Shift optimal height upwards?
        optimal_height = gaussian_filter(min_image, sigma=sigma) + self.height_offset_reward

        del afmulator

        return {
            'afm_images': afm_images,
            'z_height_map': z_height_map,
            'min_image': min_image,
            'z_min': float(z_min),
            'z_max': float(z_max),
            'optimal_height': optimal_height,
            '_z_map_start': z_height_map[0],
            '_z_map_step': z_height_map[1] - z_height_map[0],
            '_z_map_len_minus_1': len(z_height_map) - 1,
            'scan_params': {'angle_deg': angle_deg, 'tx': tx, 'ty': ty},
        }

    def _normalize(self, value: np.ndarray | float, key: str) -> np.ndarray | float:
        """
        Normalize a physical value to the range [-1, 1].

        Uses ``self.norm_bounds[key]`` via ``2 * (x - min) / (max - min) - 1``.

        Parameters
        ----------
        value : np.ndarray | float
            Physical value(s) to normalize.
        key : str
            Key into ``self.norm_bounds`` (e.g. ``'x'`` or ``'y'``).

        Returns
        -------
        np.ndarray | float
            Normalized value(s) in [-1, 1].
        """
        min_v, max_v = self.norm_bounds[key]
        return 2.0 * ((value - min_v) / (max_v - min_v)) - 1.0

    def denormalize_observation(self, obs_dict: dict) -> dict:
        """
        Convert a normalized observation dictionary back to physical units.

        Useful for plotting and interpretation. Applies the inverse of
        ``_normalize`` (using the active view's ``norm_bounds``) to ``x``/``y``,
        and multiplies ``df`` / ``dz`` by ``df_scale`` / ``dz_scale``
        respectively. Unknown keys are passed through unchanged.

        Parameters
        ----------
        obs_dict : dict
            Normalized observation dict as produced by ``_get_obs``.

        Returns
        -------
        dict
            Dictionary with the same keys in physical units.
        """
        physical_obs = {}
        for key, value in obs_dict.items():
            if key in ('x', 'y'):
                min_v, max_v = self.norm_bounds[key]
                physical_obs[key] = (value + 1.0) / 2.0 * (max_v - min_v) + min_v
            elif key == 'df':
                physical_obs[key] = value * self.df_scale
            elif key == 'dz':
                physical_obs[key] = value * self.dz_scale
            else:
                physical_obs[key] = value
        return physical_obs

    def _get_closest_slice_index(self, z: float) -> np.int64:
        """
        Return the index of the z-slice closest to a given physical height.

        Parameters
        ----------
        z : float
            Physical height whose nearest slice index is requested.

        Returns
        -------
        np.int64
            Index into ``self.z_height_map`` of the closest slice.
        """
        return np.abs(self.z_height_map - z).argmin()

    def _get_two_closest_z_planes(self, z: float):
        """
        Return the indices of the two z-planes closest to a given height.

        Parameters
        ----------
        z : float
            Target physical height.

        Returns
        -------
        np.ndarray
            Length-2 array of indices into ``self.z_height_map`` in ascending
            order.
        """
        return np.sort(np.argsort(np.abs(self.z_height_map - z))[0:2])

    def _get_interpolated_df(self, x: int, y: int, z: float):
        """
        Linearly interpolate df at ``(x, y, z)`` between the two closest z-planes.

        Falls back to the value at the single closest plane when the two
        bracketing planes coincide.

        Parameters
        ----------
        x : int
            Pixel x index.
        y : int
            Pixel y index.
        z : float
            Physical z position at which df is interpolated.

        Returns
        -------
        float
            Interpolated df value at ``(x, y, z)``.
        """
        z1, z2 = self._get_two_closest_z_planes(z)
        denom = (self.z_height_map[z2] - self.z_height_map[z1])
        if denom == 0:
            return self.afm_images[x, y, z1]

        k = (self.afm_images[x, y, z2] - self.afm_images[x, y, z1]) / denom
        return k * (z - self.z_height_map[z1]) + self.afm_images[x, y, z1]

    def _get_obs(self):
        """
        Convert internal state to the gym observation format.

        Normalizes x/y history to [-1, 1] using the active view's bounds and
        scales dz/df history by ``dz_scale`` / ``df_scale``.

        Returns
        -------
        dict
            Observation dict with keys ``x``, ``y`` (normalized positions),
            ``dz`` (height change relative to start) and ``df`` (frequency
            shift), each a length-``num_historic_data`` float32 array.
        """
        return {
            "x": self._normalize(self._x, 'x').astype(np.float32),
            "y": self._normalize(self._y, 'y').astype(np.float32),
            "dz": (self._dz / self.dz_scale).astype(np.float32),
            "df": (self._df / self.df_scale).astype(np.float32)
        }

    def _get_info(self, include_image: bool = False) -> dict:
        """
        Build the auxiliary info dict returned alongside observations.

        Parameters
        ----------
        include_image : bool
            If True, include the currently generated df image under
            ``'generated_image'``. Can be useful for debugging / visualization
            but adds overhead.

        Returns
        -------
        dict
            Contains the starting z height (``'z'``) and its slice index
            (``'z_i'``), plus the generated image when requested.
        """
        info = {
            "z": self.z_start,
            "z_i": self.z_start_index,
        }
        if include_image:
            info["generated_image"] = self.generated_image

        return info

    # TODO: Randomize start position and rotation. Maybe also switch between different surfaces?
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        """
        Reset the environment at the start of a new episode.

        Randomly selects one of the loaded views, optionally unloading the
        previously active lazy view, and initializes tip position, history
        buffers, and the generated image. A random starting z height within
        the scan window is drawn.

        Parameters
        ----------
        seed : int | None
            Seed for the random number generator.
        options : dict[str, Any] | None
            Options forwarded to the ``gym.Env`` super class.

        Returns
        -------
        tuple
            ``(observation, info)`` as produced by ``_get_obs`` and
            ``_get_info``.
        """
        super().reset(seed=seed, options=options)

        if seed is not None:
            np.random.seed(seed)

        self.terminated = False

        # Randomly sample a view for this episode
        view_idx = np.random.randint(len(self.views))

        if self.unload_view_on_reset and self.active_view_idx is not None and self.active_view_idx != view_idx:
            self._unload_view(self.active_view_idx)

        view = self._load_view(view_idx)
        self.active_view_idx = view_idx

        # Set active view data as instance attributes for use in step/obs
        self.afm_images = view['afm_images']
        self.z_height_map = view['z_height_map']
        self.min_image = view['min_image']
        self.z_min = view['z_min']
        self.z_max = view['z_max']
        self.optimal_height = view['optimal_height']
        self._z_map_start = view['_z_map_start']
        self._z_map_step = view['_z_map_step']
        self._z_map_len_minus_1 = view['_z_map_len_minus_1']

        # Update active spatial dimensions and normalization bounds
        self._x_lim = view['x_lim']
        self._y_lim = view['y_lim']
        self.norm_bounds = {
            'x': (0.0, float(self._x_lim - 1)),
            'y': (0.0, float(self._y_lim - 1)),
        }

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

        # Track whether the agent has entered a valid zone (for ceiling offset logic)
        self._has_entered_valid_zone = False

        return self._get_obs(), self._get_info(include_image=self.include_image_in_info)

    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Execute one timestep within the environment.

        Advances the tip along a raster-scan pattern, applies the vertical
        action (directly or via the discrete action map), interpolates the
        new df value, computes the reward based on the agent's position
        relative to the optimal-height surface, and evaluates crash and
        termination conditions.

        Parameters
        ----------
        action : np.ndarray | int
            Vertical tip displacement in Ångström. For continuous action
            spaces a length-1 array-like; for discrete spaces an integer
            index into ``self._action_map``.

        Returns
        -------
        tuple
            ``(observation, reward, terminated, truncated, info)`` following
            the Gymnasium step API.
        """
        if self.terminated:
            return self._get_obs(), 0.0, True, False, self._get_info(include_image=self.include_image_in_info)

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

        # Check for crashes and calculate reward
        # TODO: Add tolerance?

        z_opt = self.optimal_height[x_new, y_new]
        z_min = self.min_image[x_new, y_new]

        z_ceiling = z_opt + self.reward_ceiling_offset

        # Check if agent is in a valid zone (at or above z_min and at or below z_ceiling)
        if z_new <= z_ceiling and z_new >= z_min:
            self._has_entered_valid_zone = True

        # Only apply ceiling crash if agent has been in a valid zone
        if z_new > z_ceiling and self._has_entered_valid_zone:
            self.terminated = True
            return self._get_obs(), self.crash_reward, True, False, self._get_info(include_image=self.include_image_in_info)
        elif z_new >= z_opt:
            reward = self.base_reward - ((z_new - z_opt) ** self.reward_exponent)

        elif z_new > z_min:
            danger_zone_range = z_opt - z_min
            danger_zone_range = max(danger_zone_range, 1e-6)

            fraction = (z_new - z_min) / danger_zone_range
            fraction = min(fraction, 1.0)

            reward = self.crash_reward * (1.0 - fraction)

        # 3. At or below min_image (Crash Zone)
        else:
            self.terminated = True
            return self._get_obs(), self.crash_reward, True, False, self._get_info(include_image=self.include_image_in_info)

        # Apply custom per-step rewards
        for fn, kwargs, w in zip(self.step_custom_reward_fns, self.step_custom_reward_kwargs, self.step_custom_reward_windows):
            reward += fn(self._dz[:w], self._df[:w], self._x[:w], self._y[:w], **kwargs)

        # Termination check
        if y_new == y_lim - 1:  # Check last row
            if y_new % 2 == 0 and x_new == x_lim - 1:
                self.terminated = True
                reward += self.termination_reward
            elif y_new % 2 == 1 and x_new == 0:
                self.terminated = True
                reward += self.termination_reward

        return self._get_obs(), reward, self.terminated, False, self._get_info(include_image=self.include_image_in_info)
