"""What a PHERAstar reports about itself at INIT.

The INIT reply carries the CLARIOstar's blocks (`pylabrobot.bmg_labtech.clariostar.driver.
configuration`), each with a one-byte length instead of two. Only the blocks whose meaning was
checked against the reader are decoded; the others are kept raw in `init_blocks`.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterator, Tuple

from pylabrobot.bmg_labtech.clariostar.driver.configuration import (
  DECODED_FIRMWARE,
  INIT_REPLY_TYPE,
  DeviceConfiguration,
  _block,
)

logger = logging.getLogger(__name__)

MODEL = "PHERAstarFX"


def _short_blocks(payload: bytes) -> Iterator[Tuple[int, bytes]]:
  """Yield the INIT reply's blocks, `[id][length u8][data]`, from byte 6.

  Raises:
    ValueError: if the payload is not an INIT reply or its blocks do not end with it.
  """
  if not payload or payload[0] != INIT_REPLY_TYPE:
    raise ValueError(f"not an INIT reply: {payload[:1].hex()}")
  offset = 6
  while offset < len(payload):
    if offset + 2 > len(payload):
      raise ValueError(f"INIT reply cut short in the block at byte {offset}")
    length = payload[offset + 1]
    data = payload[offset + 2 : offset + 2 + length]
    if len(data) != length:
      raise ValueError(f"INIT reply cut short in the block at byte {offset}")
    yield payload[offset], data
    offset += 2 + length


@dataclass
class PHERAstarConfiguration(DeviceConfiguration):
  """What a PHERAstar reported about itself.

  Decoded from the INIT reply, checked against a PHERAstar FX (firmware 1.32): the firmware, the
  temperature controller and its two sensors, the two injector pumps and the absorbance
  spectrometer. The other fields stay None.
  """

  # Every INIT block as received, hex by id ("0x09"): the ones not decoded yet included.
  init_blocks: Dict[str, str] = field(default_factory=dict)

  def update_from_init_reply(self, payload: bytes) -> None:
    """Fill the fields the INIT reply carries; a block missing or too short leaves its fields None.

    Raises:
      ValueError: if the payload is not a whole INIT reply, or comes from firmware outside
        `DECODED_FIRMWARE`; the main board fields are filled either way.
    """
    blocks = dict(_short_blocks(payload))
    self.init_blocks = {f"0x{block_id:02x}": data.hex() for block_id, data in blocks.items()}
    main_board = _block(blocks, 0x01, 5)
    firmware = int.from_bytes(main_board[0:2], "big")
    self.firmware_version = f"{firmware / 1000:.2f}"
    self.firmware_build = main_board[2]
    self.mainboard_revision = main_board[4]
    self.model = MODEL
    if not DECODED_FIRMWARE[0] <= firmware <= DECODED_FIRMWARE[1]:
      raise ValueError(f"INIT reply of firmware {self.firmware_version} is not decoded here")
    for update in (
      self._update_from_temperature_controller,
      self._update_from_pumps,
      self._update_from_spectrometer,
      self._update_from_stacker,
    ):
      try:
        update(blocks)
      except ValueError as error:
        logger.warning("%s; its fields stay unknown", error)
