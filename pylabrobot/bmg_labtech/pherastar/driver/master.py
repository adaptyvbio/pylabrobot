"""PHERAstar driver: BMG's command layer, shared with the CLARIOstar, over the reader's FTDI bridge.

What a PHERAstar FX (firmware 1.32) was seen to answer while BMG's own software drove it
(2026-11-06): STATUS, DRAWER and TEMPERATURE_CONTROLLER as the CLARIOstar's, INIT with its own
parameters (`00 10 01`), and a stored-parameters read (`0e 09 00 01 5a`) in place of the
CLARIOstar's REQUEST `07`. Measurements are not driven yet: their RUN layout differs from the
CLARIOstar's after the well mask, so no measurement feature is built.
"""

import logging
from typing import Optional

from pylabrobot.bmg_labtech.clariostar.driver.master import CLARIOstarDriver
from pylabrobot.io.ftdi import FTDI

from .configuration import PHERAstarConfiguration
from .framing import FTDI_HEADER

logger = logging.getLogger(__name__)

# PHERAstar firmware versions checked on hardware.
CONFIRMED_FIRMWARE_VERSIONS = frozenset({"1.32"})


class PHERAstarDriver(CLARIOstarDriver):
  """Owns the PHERAstar connection, BMG's command layer, and the features the reader shares with
  the CLARIOstar (`drawer`, `temperature_controller`)."""

  _FRAME_HEADER = FTDI_HEADER
  _INIT_PARAMETERS = b"\x00\x10\x01"
  _DEVICE_NAME = "BMG PHERAstar"
  # The read BMG's software sends after INIT. Family 0x0E also writes stored parameters, which
  # `send_command` refuses; this exact payload is sent on its own.
  _STORED_PARAMETERS_REQUEST = bytes.fromhex("0e0900015a")
  # The reply type of the stored-parameters read.
  _STORED_PARAMETERS_REPLY = 0x07

  def __init__(
    self,
    device_id: Optional[str] = None,
    io: Optional[FTDI] = None,
    name: str = "PHERAstar",
  ):
    """
    Args:
      device_id: the FTDI serial number, the reader's own ("471-0376"); None opens the first
        reader found with VID 0x0403 / PID 0xBB68, which a CLARIOstar shares.
      io: an already-built ``FTDI``, used instead of building one.
      name: the device name, heading its setup summary.
    """
    super().__init__(use_io="ftdi", device_id=device_id, io=io, name=name)
    self.configuration: PHERAstarConfiguration = PHERAstarConfiguration()
    # The stored-parameters reply, as received; its layout is not decoded yet.
    self.stored_parameters: Optional[bytes] = None

  async def request_stored_parameters(self) -> bytes:
    """Read the reader's stored parameters (`0e 09 00 01 5a`, reply type 07) and return the
    reply payload.

    Raises:
      ValueError: if the reply is not a stored-parameters reply.
      FrameError: if every attempt gets a missing or malformed reply.
      CommandRefusedError: if the reader refused the read.
    """
    payload = self._STORED_PARAMETERS_REQUEST
    reply = await self._exchange_with_retries(payload, retries=3)
    self._check_reply(payload, reply)
    if not reply or reply[0] != self._STORED_PARAMETERS_REPLY:
      raise ValueError(f"not a stored parameters reply: {reply[:1].hex()}, {len(reply)} bytes")
    return reply

  def _check_firmware_compatibility(self) -> None:
    """Log a warning if the firmware version has not been checked on hardware."""
    version = self.configuration.firmware_version
    if version and version not in CONFIRMED_FIRMWARE_VERSIONS:
      logger.warning(
        "PHERAstar firmware %s has not been tested with this driver (tested: %s)",
        version,
        ", ".join(sorted(CONFIRMED_FIRMWARE_VERSIONS)),
      )

  async def discover(self) -> None:
    """Read the stored parameters, take the serial number from the link, and build the features
    INIT reported (the temperature controller)."""
    try:
      self.stored_parameters = await self.request_stored_parameters()
    except ValueError as error:
      logger.warning("stored parameters not read (%s)", error)
    self.configuration.serial_number = self._get_link_serial_number()
    self._check_firmware_compatibility()
    self._discover_features()

  def _discover_features(self) -> None:
    """Build the temperature controller if INIT reported one; no measurement feature yet."""
    if self.configuration.temperature_controller and self.temperature_controller is None:
      from pylabrobot.bmg_labtech.clariostar.driver.features.temperature_controller import (
        TemperatureController,
      )

      self.temperature_controller = TemperatureController(self)

  async def _shut_down(self, accept_plate_left_in_device: bool) -> None:
    """As the CLARIOstar's, but a PHERAstar rests in monitoring (`06 0001`, BMG's `Temp 00.1`):
    heating stops and the temperature sensors keep reporting."""
    controller = self.temperature_controller
    if controller is not None and controller.target_temperature is not None:
      await controller.stop_temperature_control()
    # The CLARIOstar's shutdown switches the sensors off; it skips a reader without a controller.
    self.temperature_controller = None
    try:
      await super()._shut_down(accept_plate_left_in_device)
    finally:
      self.temperature_controller = controller

  async def setup(self) -> None:
    """Connect, initialize the reader (about 15 s), read what it is fitted with, and start the
    temperature sensors, as BMG's software does."""
    logger.debug("setting up the PHERAstar on %s ...", self._describe_link())
    await self._ftdi.setup()
    await self._purge_rx()
    await self.initialize()
    await self.discover()
    await self._start_temperature_monitoring()
    logger.info("%s", self.format_setup_summary())
