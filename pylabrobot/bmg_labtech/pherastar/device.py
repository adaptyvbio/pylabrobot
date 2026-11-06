"""BMG Labtech PHERAstar plate reader."""

from typing import Optional

from pylabrobot.bmg_labtech.clariostar.device import _CLARIOstarPlateHolder
from pylabrobot.io.ftdi import FTDI
from pylabrobot.resources import Coordinate, Resource

from .driver.master import PHERAstarDriver

# BMG's nominal size of the PHERAstar FX, in mm; not measured.
SIZE_X = 520.0
SIZE_Y = 470.0
SIZE_Z = 350.0


class PHERAstarDevice(Resource):
  """The PHERAstar as a resource: its plate holder, its drawer, and the driver that runs it."""

  def __init__(
    self,
    name: str = "PHERAstar",
    *,
    device_id: Optional[str] = None,
    io: Optional[FTDI] = None,
    size_x: float = SIZE_X,
    size_y: float = SIZE_Y,
    size_z: float = SIZE_Z,
    model: str = "PHERAstarFX",
  ):
    """
    Args:
      name: the resource name.
      device_id: the FTDI serial number; None uses the first BMG reader found.
      io: an already-built ``FTDI``, used instead of building one.
      size_x: width, in mm.
      size_y: depth, in mm.
      size_z: height, in mm.
      model: which reader this is.
    """
    super().__init__(
      name=name, size_x=size_x, size_y=size_y, size_z=size_z, category="device", model=model
    )
    self.driver = PHERAstarDriver(device_id=device_id, io=io, name=name)
    self.plate_holder = _CLARIOstarPlateHolder(f"{name}_plate_holder", self.driver.drawer)
    self.assign_child_resource(self.plate_holder, location=Coordinate.zero())
    self.driver.drawer.plate_holder = self.plate_holder

  @property
  def drawer(self):
    """The drawer (the plate carrier)."""
    return self.driver.drawer

  @property
  def temperature_controller(self):
    """The temperature controller, or None if the reader has none; known after `setup`."""
    return self.driver.temperature_controller

  async def setup(self) -> None:
    """Connect, initialize the reader and read what it is fitted with."""
    await self.driver.setup()

  async def stop(self, accept_plate_left_in_device: bool = False) -> None:
    """Switch heating off, close the drawer, and disconnect.

    Raises:
      RuntimeError: if a plate is still in the reader, unless `accept_plate_left_in_device`.
    """
    await self.driver.stop(accept_plate_left_in_device=accept_plate_left_in_device)


def PHERAstarFX(
  name: str = "PHERAstarFX", *, device_id: Optional[str] = None, io: Optional[FTDI] = None
) -> PHERAstarDevice:
  """A PHERAstar FX."""
  return PHERAstarDevice(name=name, device_id=device_id, io=io, model="PHERAstarFX")
