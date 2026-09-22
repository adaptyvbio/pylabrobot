"""Golden-frame tests: byte-for-byte wire output against a checked-in fixture.

``testdata/agile_golden_frames.json`` holds ``(command_id, payload_hex)``
sequences: the expected byte-level output for each scenario, captured from a
reference implementation of ``Agile7612Controller`` and ``AgileSrtController``
driven through a recording fake comm layer. Every test here drives the same
call against these controllers through an equivalent recording fake, with an
explicit ``axis_config`` chosen to configure both trees identically (real
per-axis speeds and home-sensor bitmasks, rather than each tree's own
no-profile fallback, so a mismatch here is a genuine packet-building bug and
not one of the three documented default-configuration differences), and
asserts the captured sequence matches the fixture exactly. The fixture is
checked in so a change in packet content, field order, or phase sequencing
fails immediately.

This is what actually exercises the per-axis homing routines, jog,
tip_force_jog, grip, and the underlying packet builders end to end -- unit
tests on individual helper methods do not catch a wrong byte inside a
19-command homing sequence the way a full recorded comparison does.
"""

from __future__ import annotations

import json
import struct
import time
import unittest
from pathlib import Path

from pylabrobot.agilent.bravo.axis_config import default_axis_config
from pylabrobot.agilent.bravo.controllers.agile_7612 import Agile7612Controller
from pylabrobot.agilent.bravo.controllers.agile_srt import AgileSrtController
from pylabrobot.agilent.bravo.controllers.base import AxisMoveInfo, JogParams
from pylabrobot.agilent.bravo.errors import BravoError
from pylabrobot.agilent.bravo.protocol.v11_comm_tests import BufferedTransport
from pylabrobot.agilent.bravo.types import ALL_AXES

_GOLDEN_PATH = Path(__file__).parent / "testdata" / "agile_golden_frames.json"
with open(_GOLDEN_PATH) as _f:
  GOLDEN: dict = json.load(_f)

# The per-axis fallback bitmask this port's controllers fall back to when an
# axis's home_flag_bitmask is left at its 0 default. Setting it explicitly
# here (rather than leaving it at 0) makes the source's own *direct* profile
# read produce the same on/off-sensor branching as this port's fallback, so
# the fixture's on-sensor and off-sensor scenarios are actually reachable
# and comparable in both trees.
_HOME_FLAG_BITMASK: dict = {"x": 1, "y": 2, "z": 4, "w": 8, "g": 1, "zg": 2}


def _matching_axis_config() -> dict:
  """Build an axis_config mapping that configures both trees identically.

  Every field but home_flag_bitmask already matches the fixture's fake
  profile through default_axis_config's own values (real per-axis speeds
  and ranges, the shared W ticks-per-uL constant); only the bitmask needs
  overriding away from its 0 default.
  """
  config = {}
  for axis in ALL_AXES:
    cfg = default_axis_config(axis)
    cfg.home_flag_bitmask = _HOME_FLAG_BITMASK[axis]
    config[axis] = cfg
  return config


class RecordingComm:
  """Fake comm layer: records every ``(command_id, payload_hex)`` sent.

  Response content is inert except for register 0x10 (home-sensor state)
  reads, whose on/off-sensor byte a test controls directly, and status
  reads, which always report settled so ``_agile_7612_wait_for_settled``
  returns on its first poll instead of looping.
  """

  def __init__(self) -> None:
    self.calls: list[tuple[int, str]] = []
    self.is_connected = True
    self.sensor_byte = 0xFF
    self.command_counts: dict = {}
    self.error_log: list = []

  @property
  def transport(self) -> "RecordingComm":
    return self

  def drain(self) -> int:
    return 0

  def send_command(self, command_id, data: bytes = b"", timeout: float = 2.0) -> bytes:
    self.calls.append((int(command_id), data.hex()))
    if len(data) > 1 and data[1] == 0x10:
      return bytes([0x00, 0x00, self.sensor_byte, 0x00, 0x00, 0x00, 0x00, 0x00])
    if len(data) > 7 and data[0] == 0x00 and data[7] == 0x90:
      return bytes([0x00, 0x00, 0xB0, 0xB0, 0xB0, 0xB0, 0x00, 0x00, 0x00, 0x00])
    if len(data) > 1 and data[0] == 0x09 and data[1] == 0x90:
      return bytes([0x00, 0x00, 0x55, 0x2A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
    return bytes(10)


def _new_controller(cls, sensor_byte: int = 0xFF) -> tuple[Agile7612Controller, RecordingComm]:
  controller = cls(BufferedTransport(), axis_config=_matching_axis_config())
  comm = RecordingComm()
  comm.sensor_byte = sensor_byte
  controller._comm = comm
  for axis in ALL_AXES:
    controller._homed[axis] = True
  return controller, comm


def _run(cls, sensor_byte: int, action) -> list[tuple[int, str]]:
  controller, comm = _new_controller(cls, sensor_byte)
  try:
    action(controller)
  except (BravoError, NotImplementedError):
    pass
  return comm.calls


class GoldenFrameTestCase(unittest.TestCase):
  """Base class: silences real sleeps and time so polling loops run fast.

  ``time.sleep`` is a no-op here, and ``time.monotonic`` returns a fake
  clock that advances a fixed step on every call, so a wait that only
  resolves after real elapsed time (e.g. G-homing's settle/stall wait,
  which requires either observed motion or a minimum elapsed time before
  trusting a settled status) reaches that deadline in a small, bounded
  number of iterations instead of spinning for real wall-clock seconds
  with sleep silenced.
  """

  _FAKE_TIME_STEP_S = 0.01

  def setUp(self) -> None:
    self._real_sleep = time.sleep
    self._real_monotonic = time.monotonic
    self._fake_time = 0.0
    time.sleep = lambda *_a, **_k: None
    time.monotonic = self._advance_fake_time

  def tearDown(self) -> None:
    time.sleep = self._real_sleep
    time.monotonic = self._real_monotonic

  def _advance_fake_time(self) -> float:
    self._fake_time += self._FAKE_TIME_STEP_S
    return self._fake_time

  def assert_matches_golden(self, scenario: str, calls: list) -> None:
    expected = [tuple(pair) for pair in GOLDEN[scenario]]
    self.assertEqual(calls, expected, f"{scenario}: captured frames diverge from golden")


class Agile7612HomingGoldenTests(GoldenFrameTestCase):
  def test_home_x_on_sensor(self):
    calls = _run(Agile7612Controller, 0xFF, lambda c: c._home_x())
    self.assert_matches_golden("agile7612_home_x_on_sensor", calls)

  def test_home_x_off_sensor(self):
    calls = _run(Agile7612Controller, 0x00, lambda c: c._home_x())
    self.assert_matches_golden("agile7612_home_x_off_sensor", calls)

  def test_home_y_on_sensor(self):
    calls = _run(Agile7612Controller, 0xFF, lambda c: c._home_y())
    self.assert_matches_golden("agile7612_home_y_on_sensor", calls)

  def test_home_y_off_sensor(self):
    calls = _run(Agile7612Controller, 0x00, lambda c: c._home_y())
    self.assert_matches_golden("agile7612_home_y_off_sensor", calls)

  def test_home_z_on_sensor(self):
    calls = _run(Agile7612Controller, 0xFF, lambda c: c._home_z())
    self.assert_matches_golden("agile7612_home_z_on_sensor", calls)

  def test_home_z_off_sensor(self):
    calls = _run(Agile7612Controller, 0x00, lambda c: c._home_z())
    self.assert_matches_golden("agile7612_home_z_off_sensor", calls)

  def test_home_w_on_sensor(self):
    calls = _run(Agile7612Controller, 0xFF, lambda c: c._home_w())
    self.assert_matches_golden("agile7612_home_w_on_sensor", calls)

  def test_home_w_off_sensor(self):
    calls = _run(Agile7612Controller, 0x00, lambda c: c._home_w())
    self.assert_matches_golden("agile7612_home_w_off_sensor", calls)

  def test_home_g(self):
    calls = _run(Agile7612Controller, 0xFF, lambda c: c._home_g())
    self.assert_matches_golden("agile7612_home_g", calls)

  def test_home_zg(self):
    calls = _run(Agile7612Controller, 0xFF, lambda c: c._home_zg())
    self.assert_matches_golden("agile7612_home_zg", calls)

  def test_home_axes_order(self):
    calls = _run(Agile7612Controller, 0xFF, lambda c: c.home_axes(list(ALL_AXES)))
    self.assert_matches_golden("agile7612_home_axes_order", calls)


class Agile7612MotionGoldenTests(GoldenFrameTestCase):
  def test_move(self):
    def action(c):
      c.move(
        [
          AxisMoveInfo(axis="x", position=100.0, velocity=50.0, acceleration=100.0, absolute=True),
          AxisMoveInfo(axis="g", position=2.0, velocity=10.0, acceleration=50.0, absolute=True),
        ],
        wait=True,
      )

    calls = _run(Agile7612Controller, 0xFF, action)
    self.assert_matches_golden("agile7612_move", calls)

  def test_jog(self):
    def action(c):
      c.jog(
        JogParams(
          axis="z",
          velocity=5.0,
          acceleration=20.0,
          max_position=50.0,
          tolerance=1.0,
          peak_current=0.2,
        )
      )

    calls = _run(Agile7612Controller, 0xFF, action)
    self.assert_matches_golden("agile7612_jog", calls)

  def test_tip_force_jog(self):
    calls = _run(Agile7612Controller, 0xFF, lambda c: c.tip_force_jog("z", 0.15, 30.0))
    self.assert_matches_golden("agile7612_tip_force_jog", calls)

  def test_grip(self):
    calls = _run(Agile7612Controller, 0xFF, lambda c: c.grip("slow", 3.0))
    self.assert_matches_golden("agile7612_grip", calls)


class MoveOriginOffsetTests(GoldenFrameTestCase):
  """Zg parks at firmware -20mm (hardcoded, independent of any axis_config), so an
  absolute move to Zg is the one case in this default configuration where
  _move_origin is nonzero -- exercising it directly, since none of the golden
  move scenarios happen to touch Zg.
  """

  def test_absolute_move_to_zg_subtracts_the_firmware_park_offset(self):
    controller, comm = _new_controller(Agile7612Controller)
    controller.move(
      [AxisMoveInfo(axis="zg", position=10.0, velocity=25.0, acceleration=250.0, absolute=True)],
      wait=True,
    )

    prepare_move_calls = [hexdata for cid, hexdata in comm.calls if cid == 0xA2]
    self.assertEqual(len(prepare_move_calls), 1)
    payload = bytes.fromhex(prepare_move_calls[0])
    position_ticks = struct.unpack_from("<f", payload, 1)[0]

    origin = controller._move_origin("zg")
    self.assertEqual(origin, 20.0)  # 0.0 homing_offset - (-20.0) firmware park
    expected_ticks = controller._to_ticks("zg", 10.0 - origin)
    self.assertAlmostEqual(position_ticks, expected_ticks, places=3)


class SrtHomingGoldenTests(GoldenFrameTestCase):
  def test_home_x_on_sensor(self):
    calls = _run(AgileSrtController, 0xFF, lambda c: c._srt_home_axis("x"))
    self.assert_matches_golden("srt_home_x_on_sensor", calls)

  def test_home_x_off_sensor(self):
    calls = _run(AgileSrtController, 0x00, lambda c: c._srt_home_axis("x"))
    self.assert_matches_golden("srt_home_x_off_sensor", calls)

  def test_home_y_on_sensor(self):
    calls = _run(AgileSrtController, 0xFF, lambda c: c._srt_home_axis("y"))
    self.assert_matches_golden("srt_home_y_on_sensor", calls)

  def test_home_y_off_sensor(self):
    calls = _run(AgileSrtController, 0x00, lambda c: c._srt_home_axis("y"))
    self.assert_matches_golden("srt_home_y_off_sensor", calls)

  def test_home_z_on_sensor(self):
    calls = _run(AgileSrtController, 0xFF, lambda c: c._srt_home_axis("z"))
    self.assert_matches_golden("srt_home_z_on_sensor", calls)

  def test_home_z_off_sensor(self):
    calls = _run(AgileSrtController, 0x00, lambda c: c._srt_home_axis("z"))
    self.assert_matches_golden("srt_home_z_off_sensor", calls)

  def test_home_w_on_sensor(self):
    calls = _run(AgileSrtController, 0xFF, lambda c: c._srt_home_axis("w"))
    self.assert_matches_golden("srt_home_w_on_sensor", calls)

  def test_home_w_off_sensor(self):
    calls = _run(AgileSrtController, 0x00, lambda c: c._srt_home_axis("w"))
    self.assert_matches_golden("srt_home_w_off_sensor", calls)

  def test_home_axes_order(self):
    calls = _run(AgileSrtController, 0xFF, lambda c: c.home_axes(["x", "y", "z", "w"]))
    self.assert_matches_golden("srt_home_axes_order", calls)


class PositionDecodeTests(unittest.TestCase):
  """Directly exercises _read_raw_position's byte decode, which the golden-frame
  scenarios cannot: RecordingComm's generic response is an all-zero, symmetric
  10 bytes, so a big-endian-vs-little-endian decode bug is invisible there.

  The register is a mantissa/exponent pair, not a plain integer: bytes [2:4]
  (big-endian) are a 16-bit mantissa and byte 6 is an exponent, encoding
  ``ticks = mantissa * 2**(exponent - 15)``. The X fixtures below are the
  exact response bytes captured on real hardware (Agile 7612, firmware
  5.4.7) at four independently-verified X positions -- not synthesized --
  so this pins the real wire format, not just an internally-consistent
  formula.
  """

  def _response_for(self, mantissa: int, exponent: int) -> bytes:
    return bytes(
      [0x00, 0x00, (mantissa >> 8) & 0xFF, mantissa & 0xFF, 0x00, 0x00, exponent, 0x00, 0x00, 0x00]
    )

  def _patch_position_register(self, comm: RecordingComm, response: bytes) -> None:
    real_send_command = comm.send_command

    def send_command(command_id, data: bytes = b"", timeout: float = 2.0) -> bytes:
      if len(data) > 1 and data[1] == 0x07:
        comm.calls.append((int(command_id), data.hex()))
        return response
      return real_send_command(command_id, data, timeout)

    comm.send_command = send_command  # type: ignore[method-assign]

  def test_controller_1_axis_decodes_hardware_captured_x_positions(self):
    # (mantissa, exponent, expected mm), all captured on real hardware.
    hardware_samples = [
      (0x7B08, 0x0D, 25.0),
      (0x7B08, 0x0E, 50.0),
      (0x5C46, 0x0F, 75.0),
      (0x7B08, 0x0F, 100.0),
    ]
    for mantissa, exponent, expected_mm in hardware_samples:
      with self.subTest(mantissa=hex(mantissa), exponent=hex(exponent)):
        controller, comm = _new_controller(Agile7612Controller)
        self._patch_position_register(comm, self._response_for(mantissa, exponent))
        position = controller.get_position("x")
        self.assertAlmostEqual(position, expected_mm, places=2)

  def test_controller_1_axis_decodes_the_mantissa_big_endian(self):
    controller, comm = _new_controller(Agile7612Controller)
    # A deliberately asymmetric mantissa; exponent 15 (bias) keeps the
    # scale factor at 1 so the mantissa alone determines the tick count.
    self._patch_position_register(comm, self._response_for(0x1234, 0x0F))

    position = controller.get_position("x")

    ticks_per_eng_unit = controller._ticks_per_unit["x"]
    expected = float(0x1234) / ticks_per_eng_unit
    self.assertAlmostEqual(position, expected)
    # A little-endian misreading of the same two bytes (0x3412) would give a
    # visibly different result, so this also fails if the byte order flips.
    wrong_le = float(0x3412) / ticks_per_eng_unit
    self.assertNotAlmostEqual(position, wrong_le)

  def test_controller_2_axis_decodes_a_hardware_captured_negative_zg_position(self):
    # Captured on real hardware right after homing: response
    # 910784f800000e020589 -> mantissa 0x84F8 as signed int16 = -31496,
    # exponent 0x0E -> -15748 ticks -> -20.000mm at 787.4 ticks/mm, exactly
    # _FIRMWARE_PARK_MM["zg"]. Decoding the mantissa as a full
    # two's-complement int16 is required for this: sign-plus-15-bit-
    # magnitude gives a meaningless -0.808 for the same bytes.
    controller, comm = _new_controller(Agile7612Controller)
    self._patch_position_register(comm, bytes.fromhex("910784f800000e020589"))

    position = controller.get_position("zg")

    self.assertAlmostEqual(position, -20.0, places=2)


if __name__ == "__main__":
  unittest.main()
