"""RGB-D camera sensor backed by MuJoCo offscreen rendering."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.sensor.sensor import Sensor, SensorCfg


@dataclass
class CameraSensorData:
  """Output buffers for camera sensor modalities."""

  rgb: torch.Tensor | None
  depth: torch.Tensor | None
  pos_w: torch.Tensor
  quat_w: torch.Tensor


@dataclass
class CameraSensorCfg(SensorCfg):
  """Configuration for an RGB-D camera sensor."""

  camera_name: str | None = None
  parent_body: str | None = None
  pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
  quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
  fovy: float | None = None
  width: int = 160
  height: int = 120
  data_types: tuple[str, ...] = ("depth",)
  clone_data: bool = False

  def build(self) -> CameraSensor:
    return CameraSensor(self)


class CameraSensor(Sensor[CameraSensorData]):
  """Camera sensor that renders per-environment RGB/depth frames."""

  def __init__(self, cfg: CameraSensorCfg):
    self.cfg = cfg
    self._camera_name = cfg.camera_name
    self._mj_model: mujoco.MjModel | None = None
    self._sim_model: mjwarp.Model | None = None
    self._sim_data: mjwarp.Data | None = None
    self._mj_data: mujoco.MjData | None = None
    self._renderer: mujoco.Renderer | None = None
    self._cam_id: int | None = None
    self._nworld = 0
    self._device = "cpu"
    self._data = CameraSensorData(
      rgb=None,
      depth=None,
      pos_w=torch.empty((0, 3)),
      quat_w=torch.empty((0, 4)),
    )

  def edit_spec(self, scene_spec: mujoco.MjSpec, entities: dict[str, Entity]) -> None:
    del entities  # Not needed unless parent_body references attached body name.
    if self.cfg.camera_name is not None:
      return

    if self.cfg.parent_body is not None:
      body = scene_spec.body(self.cfg.parent_body)
    else:
      body = scene_spec.worldbody
    cam = body.add_camera(
      name=self.cfg.name,
      pos=self.cfg.pos,
      quat=self.cfg.quat,
    )
    if self.cfg.fovy is not None:
      cam.fovy = float(self.cfg.fovy)
    self._camera_name = self.cfg.name

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    if self._camera_name is None:
      raise ValueError("Camera name unresolved. Set camera_name or create camera in edit_spec.")

    self._mj_model = mj_model
    self._sim_model = model
    self._sim_data = data
    self._device = device
    self._nworld = int(data.nworld)
    self._mj_data = mujoco.MjData(mj_model)
    self._renderer = mujoco.Renderer(
      model=mj_model,
      height=int(self.cfg.height),
      width=int(self.cfg.width),
    )
    self._cam_id = mj_model.camera(self._camera_name).id

    rgb = None
    if "rgb" in self.cfg.data_types:
      rgb = torch.zeros(
        (self._nworld, self.cfg.height, self.cfg.width, 3),
        dtype=torch.uint8,
        device=device,
      )
    depth = None
    if "depth" in self.cfg.data_types:
      depth = torch.zeros(
        (self._nworld, self.cfg.height, self.cfg.width, 1),
        dtype=torch.float32,
        device=device,
      )
    self._data = CameraSensorData(
      rgb=rgb,
      depth=depth,
      pos_w=torch.zeros((self._nworld, 3), dtype=torch.float32, device=device),
      quat_w=torch.zeros((self._nworld, 4), dtype=torch.float32, device=device),
    )

  @property
  def data(self) -> CameraSensorData:
    return self._data

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    del env_ids

  def update(self, dt: float) -> None:
    del dt
    self.capture()

  def capture(self) -> CameraSensorData:
    if self._mj_model is None or self._sim_data is None or self._mj_data is None:
      raise RuntimeError("CameraSensor is not initialized.")
    if self._renderer is None or self._cam_id is None:
      raise RuntimeError("Camera renderer is not initialized.")

    for env_id in range(self._nworld):
      if self._mj_model.nq > 0:
        self._mj_data.qpos[:] = self._sim_data.qpos[env_id].cpu().numpy()
        self._mj_data.qvel[:] = self._sim_data.qvel[env_id].cpu().numpy()
      if self._mj_model.nmocap > 0:
        self._mj_data.mocap_pos[:] = self._sim_data.mocap_pos[env_id].cpu().numpy()
        self._mj_data.mocap_quat[:] = self._sim_data.mocap_quat[env_id].cpu().numpy()

      mujoco.mj_forward(self._mj_model, self._mj_data)
      self._renderer.update_scene(self._mj_data, camera=self._cam_id)

      cam_pos = torch.from_numpy(self._mj_data.cam_xpos[self._cam_id].copy()).to(
        device=self._device, dtype=torch.float32
      )
      cam_rot = self._mj_data.cam_xmat[self._cam_id].reshape(3, 3)
      cam_quat = np.zeros(4, dtype=np.float64)
      mujoco.mju_mat2Quat(cam_quat, cam_rot.ravel())
      cam_quat_t = torch.from_numpy(cam_quat).to(device=self._device, dtype=torch.float32)
      self._data.pos_w[env_id] = cam_pos
      self._data.quat_w[env_id] = cam_quat_t

      if self._data.depth is not None:
        self._renderer.enable_depth_rendering()
        depth_np = self._renderer.render()
        depth_t = torch.from_numpy(np.array(depth_np, copy=True)).to(
          device=self._device, dtype=torch.float32
        )
        self._data.depth[env_id, :, :, 0] = depth_t
        self._renderer.disable_depth_rendering()

      if self._data.rgb is not None:
        rgb_np = self._renderer.render()
        rgb_t = torch.from_numpy(np.array(rgb_np, copy=True)).to(
          device=self._device, dtype=torch.uint8
        )
        self._data.rgb[env_id] = rgb_t

    if self.cfg.clone_data:
      return CameraSensorData(
        rgb=self._data.rgb.clone() if self._data.rgb is not None else None,
        depth=self._data.depth.clone() if self._data.depth is not None else None,
        pos_w=self._data.pos_w.clone(),
        quat_w=self._data.quat_w.clone(),
      )
    return self._data

