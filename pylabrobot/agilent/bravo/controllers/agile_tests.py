import inspect
import struct
import time
import unittest

from pylabrobot.agilent.bravo.axis_config import AxisConfig, default_axis_config
from pylabrobot.agilent.bravo.controllers.agile import AgileController
from pylabrobot.agilent.bravo.controllers.agile_7612 import Agile7612Controller
from pylabrobot.agilent.bravo.controllers.agile_srt import AgileSrtController
from pylabrobot.agilent.bravo.controllers.base import AxisMoveInfo, BravoController, JogParams
from pylabrobot.agilent.bravo.errors import BravoError, ErrorType, RabbitErrorCode
from pylabrobot.agilent.bravo.protocol.agile_packet import (
  AGILE_PACKET_SIZE,
  UNIQUE_VALUE_EXPECTED,
  crc8,
)
from pylabrobot.agilent.bravo.protocol.commands import CommandID
from pylabrobot.agilent.bravo.protocol.v11_comm_tests import BufferedTransport
from pylabrobot.agilent.bravo.types import ALL_AXES, DEFAULT_W_TICKS_PER_UL


def _v11_frame(error_code: int, data: bytes = b"") -> bytes:
  """Build a legacy-Agile V11 response frame: ``[length_u16][error][data]``."""
  payload = bytes([error_code]) + data
  return struct.pack("<H", len(payload)) + payload


def _v11_7612_frame(command_id: int, error_code: int, data: bytes = b"") -> bytes:
  """Build an Agile 7612 V11 response frame: ``[cmd][length_u16][error][data]``."""
  payload = bytes([error_code]) + data
  return struct.pack("<B", command_id) + struct.pack("<H", len(payload)) + payload


def _agile_reply_packet(register_value: int) -> bytes:
  """Build a 10-byte legacy-Agile reply packet with a valid CRC-8/SMBUS checksum.

  The register value lands where ``AgileReply.get_register_value`` reads it:
  4 bytes at absolute packet offset 4.
  """
  pkt = bytearray(AGILE_PACKET_SIZE)
  pkt[0] = 0x01
  pkt[1] = 0x00
  struct.pack_into("<I", pkt, 4, register_value)
  pkt[9] = crc8(pkt, 9)
  return bytes(pkt)


def _agile_7612_verify_packet(register_value: int) -> bytes:
  """Build the 10-byte payload ``Agile7612Controller._verify_controller`` expects.

  Unlike the legacy Agile reply, this is read directly with
  ``struct.unpack_from("<H", response, 2)``, with no CRC check.
  """
  pkt = bytearray(AGILE_PACKET_SIZE)
  struct.pack_into("<H", pkt, 2, register_value)
  return bytes(pkt)


class InheritanceChainTests(unittest.TestCase):
  def test_agile_7612_extends_agile_controller(self):
    self.assertTrue(issubclass(Agile7612Controller, AgileController))

  def test_agile_srt_extends_agile_7612_controller(self):
    self.assertTrue(issubclass(AgileSrtController, Agile7612Controller))

  def test_every_controller_extends_bravo_controller(self):
    for cls in (AgileController, Agile7612Controller, AgileSrtController):
      self.assertTrue(issubclass(cls, BravoController))


class ModelIdentityTests(unittest.TestCase):
  def test_agile_controller_has_gripper(self):
    self.assertTrue(AgileController.has_gripper)

  def test_agile_7612_controller_has_gripper(self):
    self.assertTrue(Agile7612Controller.has_gripper)

  def test_srt_has_no_gripper(self):
    self.assertFalse(AgileSrtController.has_gripper)

  def test_model_names_identify_each_hardware_generation(self):
    self.assertEqual(AgileController.model_name, "Bravo")
    self.assertEqual(Agile7612Controller.model_name, "Bravo 7612")
    self.assertEqual(AgileSrtController.model_name, "Bravo SRT")


class HeadTypeTrackingTests(unittest.TestCase):
  def test_plain_agile_controller_falls_back_to_base_default(self):
    # AgileController does not track a head type; controllers that do not
    # track one report the 96-channel default.
    controller = AgileController(BufferedTransport())
    self.assertEqual(controller.get_head_type(), "96_d_70")

  def test_agile_7612_get_head_type_reflects_the_most_recent_set_head_type(self):
    controller = Agile7612Controller(BufferedTransport())
    controller.set_head_type("384_d_70")
    self.assertEqual(controller.get_head_type(), "384_d_70")


class SendsBytesThroughTransportTests(unittest.TestCase):
  def test_agile_controller_send_command_writes_to_transport(self):
    frame = _v11_frame(RabbitErrorCode.NONE)
    transport = BufferedTransport(frame)
    controller = AgileController(transport)

    controller.send_command(CommandID.PING_DEVICE)

    self.assertEqual(len(transport.sent), 1)
    self.assertGreater(len(transport.sent[0]), 0)

  def test_agile_7612_controller_send_command_writes_to_transport(self):
    frame = _v11_7612_frame(CommandID.PING_DEVICE, RabbitErrorCode.NONE)
    transport = BufferedTransport(frame)
    controller = Agile7612Controller(transport)

    controller.send_command(CommandID.PING_DEVICE)

    self.assertEqual(len(transport.sent), 1)
    self.assertGreater(len(transport.sent[0]), 0)

  def test_srt_controller_send_command_writes_to_transport(self):
    frame = _v11_7612_frame(CommandID.PING_DEVICE, RabbitErrorCode.NONE)
    transport = BufferedTransport(frame)
    controller = AgileSrtController(transport)

    controller.send_command(CommandID.PING_DEVICE)

    self.assertEqual(len(transport.sent), 1)
    self.assertGreater(len(transport.sent[0]), 0)


class InitializeHandshakeTests(unittest.TestCase):
  def test_agile_controller_initialize_queries_firmware_and_verifies_controller(self):
    fw_frame = _v11_frame(RabbitErrorCode.NONE, b"1.2.3\x00")
    verify_frame = _v11_frame(RabbitErrorCode.NONE, _agile_reply_packet(UNIQUE_VALUE_EXPECTED))
    transport = BufferedTransport(fw_frame + verify_frame)
    controller = AgileController(transport)

    controller.initialize()

    self.assertEqual(controller.firmware_version.master, "1.2.3")
    self.assertEqual(len(transport.sent), 2)

  def test_agile_controller_initialize_raises_when_controller_unverified(self):
    fw_frame = _v11_frame(RabbitErrorCode.NONE, b"1.2.3\x00")
    verify_frame = _v11_frame(RabbitErrorCode.NONE, _agile_reply_packet(0x1234))
    transport = BufferedTransport(fw_frame + verify_frame)
    controller = AgileController(transport)

    with self.assertRaises(BravoError) as ctx:
      controller.initialize()
    self.assertEqual(ctx.exception.error_type, ErrorType.CONTROLLER_UNIDENTIFIED)

  def test_agile_7612_controller_initialize_queries_firmware_and_verifies_controller(self):
    fw_frame = _v11_7612_frame(CommandID.QUERY_VERSION, RabbitErrorCode.NONE, b"5.4.6\x00")
    verify_frame = _v11_7612_frame(
      CommandID.DIRECT_AGILE_COMMAND, RabbitErrorCode.NONE, _agile_7612_verify_packet(0x2A55)
    )
    transport = BufferedTransport(fw_frame + verify_frame)
    controller = Agile7612Controller(transport)

    controller.initialize()

    self.assertEqual(controller.firmware_version.master, "5.4.6")

  def test_agile_7612_controller_initialize_raises_when_controller_unverified(self):
    fw_frame = _v11_7612_frame(CommandID.QUERY_VERSION, RabbitErrorCode.NONE, b"5.4.6\x00")
    verify_frame = _v11_7612_frame(
      CommandID.DIRECT_AGILE_COMMAND, RabbitErrorCode.NONE, _agile_7612_verify_packet(0x1234)
    )
    transport = BufferedTransport(fw_frame + verify_frame)
    controller = Agile7612Controller(transport)

    with self.assertRaises(BravoError) as ctx:
      controller.initialize()
    self.assertEqual(ctx.exception.error_type, ErrorType.CONTROLLER_UNIDENTIFIED)

  def test_srt_controller_initialize_queries_firmware_and_verifies_controller(self):
    fw_frame = _v11_7612_frame(CommandID.QUERY_VERSION, RabbitErrorCode.NONE, b"5.4.3\x00")
    verify_frame = _v11_7612_frame(
      CommandID.DIRECT_AGILE_COMMAND, RabbitErrorCode.NONE, _agile_7612_verify_packet(0x2A55)
    )
    transport = BufferedTransport(fw_frame + verify_frame)
    controller = AgileSrtController(transport)

    controller.initialize()

    self.assertEqual(controller.firmware_version.master, "5.4.3")

  def _handshake_transport(self, version: bytes = b"5.4.6\x00") -> BufferedTransport:
    fw_frame = _v11_7612_frame(CommandID.QUERY_VERSION, RabbitErrorCode.NONE, version)
    verify_frame = _v11_7612_frame(
      CommandID.DIRECT_AGILE_COMMAND, RabbitErrorCode.NONE, _agile_7612_verify_packet(0x2A55)
    )
    return BufferedTransport(fw_frame + verify_frame)

  def test_agile_controller_initialize_clears_stale_homed_state(self):
    fw_frame = _v11_frame(RabbitErrorCode.NONE, b"1.2.3\x00")
    verify_frame = _v11_frame(RabbitErrorCode.NONE, _agile_reply_packet(UNIQUE_VALUE_EXPECTED))
    controller = AgileController(BufferedTransport(fw_frame + verify_frame))
    controller._homed["x"] = True

    controller.initialize()

    self.assertFalse(controller._homed["x"])

  def test_agile_7612_controller_initialize_clears_stale_homed_and_tracked_state(self):
    controller = Agile7612Controller(self._handshake_transport())
    controller._homed["x"] = True
    controller._home_raw["x"] = 12345.0
    controller._tracked_position["x"] = 42.0

    controller.initialize()

    self.assertFalse(controller._homed["x"])
    self.assertEqual(controller._home_raw, {})
    self.assertEqual(controller._tracked_position, {})


class AxisConfigDefaultsTests(unittest.TestCase):
  def test_no_axis_config_uses_defaults_for_every_axis(self):
    controller = Agile7612Controller(BufferedTransport())
    for axis in ALL_AXES:
      self.assertEqual(controller._axis_config[axis], default_axis_config(axis))

  def test_srt_also_defaults_every_axis_with_no_axis_config(self):
    controller = AgileSrtController(BufferedTransport())
    for axis in ALL_AXES:
      self.assertEqual(controller._axis_config[axis], default_axis_config(axis))

  def test_explicit_axis_config_overrides_the_given_axis(self):
    override = AxisConfig(
      axis="zg",
      ticks_per_eng_unit=999.0,
      range=default_axis_config("zg").range,
      homing_offset=-20.0,
      home_complete_register=0x5F,
    )
    controller = Agile7612Controller(BufferedTransport(), axis_config={"zg": override})

    self.assertEqual(controller._axis_config["zg"].home_complete_register, 0x5F)
    self.assertEqual(controller._axis_config["zg"].homing_offset, -20.0)
    self.assertEqual(controller._ticks_per_unit["zg"], 999.0)

  def test_axis_config_override_does_not_touch_other_axes(self):
    override = AxisConfig(
      axis="zg", ticks_per_eng_unit=999.0, range=default_axis_config("zg").range
    )
    controller = Agile7612Controller(BufferedTransport(), axis_config={"zg": override})

    for axis in ALL_AXES:
      if axis != "zg":
        self.assertEqual(controller._axis_config[axis], default_axis_config(axis))
    self.assertEqual(controller._ticks_per_unit["w"], DEFAULT_W_TICKS_PER_UL)

  def test_w_ticks_per_unit_is_48_when_every_axis_is_given_its_own_default(self):
    # Building the axis_config mapping explicitly (rather than omitting it)
    # must not silently change the W scale: both paths through AxisConfig
    # resolve to the same DEFAULT_W_TICKS_PER_UL constant, so there is no
    # distinction between "axis missing from the mapping" and "axis present
    # with its own default".
    explicit = {axis: default_axis_config(axis) for axis in ALL_AXES}
    controller = Agile7612Controller(BufferedTransport(), axis_config=explicit)
    self.assertEqual(controller._ticks_per_unit["w"], 48.0)


class GHomingSearchPhaseWaitTests(unittest.TestCase):
  """_wait_for_g_homing_search_phase: G homes against a mechanical stop, not
  a sensor (confirmed on hardware), so its status never reaches the generic
  0xB0-0xBF settled band _agile_7612_wait_for_settled requires -- it goes
  0x86 and stays there, with position frozen, once the stall happens. This
  method is the G-only replacement for the two homing search-phase waits in
  _home_g; every other axis (and _home_g's own pre/post move-to-0 steps)
  still use the unmodified _agile_7612_wait_for_settled.
  """

  def setUp(self):
    self.controller = Agile7612Controller(BufferedTransport())
    self.fault_reset_calls = 0
    self.controller._agile_7612_fault_reset_ctrl2 = self._record_fault_reset  # type: ignore[method-assign]

  def _record_fault_reset(self) -> None:
    self.fault_reset_calls += 1

  def _script_status(self, byte: int):
    self.controller._agile_7612_status_read = (  # type: ignore[method-assign]
      lambda *args, **kwargs: bytes([0, 0, byte, 0, 0, 0])
    )

  def test_default_timeout_covers_a_real_profiles_slower_g_homing_speed(self):
    # Confirmed on hardware: with a real profile's slower G homing speed,
    # the negative/slow approach phase is still moving, not stalled, at 15s
    # and needs most of 60s to finish. A stall still returns early via the
    # 0x86 + stable-position check well before any timeout fires, so a 60s
    # ceiling only affects genuinely slow phases, not stall detection
    # latency.
    self.assertEqual(Agile7612Controller._G_HOMING_SEARCH_TIMEOUT_S, 60.0)

  def test_completes_on_stall_when_status_is_0x86_and_position_is_stable(self):
    self._script_status(0x86)
    self.controller._read_raw_position = lambda axis: 42.0  # type: ignore[method-assign]

    self.controller._wait_for_g_homing_search_phase(
      timeout=1.0, stable_duration=0.05, position_tolerance=0.01
    )

    self.assertEqual(self.fault_reset_calls, 1)

  def test_keeps_waiting_while_position_is_still_changing(self):
    # A 0x86 status alone is not enough to call a phase complete -- G could
    # read 0x86 for an instant while genuinely still moving. Only a
    # position that has stopped changing for stable_duration is a real
    # stall, not just a momentary status reading.
    positions = [0.0, 5.0, 10.0, 15.0] + [15.0] * 200  # pads well past any poll count
    self._script_status(0x86)
    calls_made: list = []

    def fake_read_position(axis: str) -> float:
      calls_made.append(None)
      return positions[len(calls_made) - 1]

    self.controller._read_raw_position = fake_read_position  # type: ignore[method-assign]

    self.controller._wait_for_g_homing_search_phase(
      timeout=2.0, stable_duration=0.05, position_tolerance=0.01
    )

    # It must have actually observed the changing readings (0.0/5.0/10.0)
    # before settling on the stable 15.0 -- completing on the very first
    # poll (still reading 0.0) would mean it never checked for motion at all.
    self.assertGreater(len(calls_made), 4)
    self.assertEqual(self.fault_reset_calls, 1)

  def test_times_out_when_neither_settle_nor_stall_is_ever_seen(self):
    # Status never reaches 0x86 or a settled 0xB_ band, and position never
    # stops changing -- this phase genuinely never finishes.
    self._script_status(0x80)
    counter = {"n": 0}

    def ever_changing_position(axis: str) -> float:
      counter["n"] += 1
      return float(counter["n"])

    self.controller._read_raw_position = ever_changing_position  # type: ignore[method-assign]

    with self.assertRaises(BravoError) as ctx:
      self.controller._wait_for_g_homing_search_phase(
        timeout=0.2, stable_duration=0.05, position_tolerance=0.01
      )

    self.assertEqual(ctx.exception.error_type, ErrorType.MOVE_TIMEOUT)
    self.assertEqual(self.fault_reset_calls, 0)

  def test_still_accepts_a_genuine_settle_without_requiring_a_stall(self):
    # Some future firmware/hardware state might report a real 0xB_ settle
    # for G -- this must not require a stall to complete.
    self._script_status(0xB4)
    self.controller._read_raw_position = lambda axis: 7.0  # type: ignore[method-assign]

    self.controller._wait_for_g_homing_search_phase(
      timeout=1.0, stable_duration=0.05, position_tolerance=0.01, min_settle_time=0.05
    )

    # A genuine settle is not a fault: it must not reset controller 2.
    self.assertEqual(self.fault_reset_calls, 0)

  def test_does_not_accept_a_stale_0x00_before_motion_starts(self):
    # A status read of 0x00 right after move_go can still reflect the
    # axis's state from before the move was triggered -- 0x00 means both
    # "idle, never moved" and "idle, already done". Without either
    # observed motion or a minimum elapsed time, this must not be trusted.
    self._script_status(0x00)
    self.controller._read_raw_position = lambda axis: 3.0  # type: ignore[method-assign]  # never moves

    start = time.monotonic()
    self.controller._wait_for_g_homing_search_phase(
      timeout=1.0, stable_duration=0.05, position_tolerance=0.01, min_settle_time=0.15
    )
    elapsed = time.monotonic() - start

    self.assertGreaterEqual(elapsed, 0.15)
    self.assertEqual(self.fault_reset_calls, 0)


class SrtGripperlessTests(unittest.TestCase):
  def setUp(self):
    self.controller = AgileSrtController(BufferedTransport())

  def test_detect_gripper_raises_not_implemented_naming_the_model(self):
    with self.assertRaises(NotImplementedError) as ctx:
      self.controller.detect_gripper()
    self.assertIn("Bravo SRT", str(ctx.exception))

  def test_grip_raises_not_implemented_naming_the_model(self):
    with self.assertRaises(NotImplementedError) as ctx:
      self.controller.grip("slow", 5.0)
    self.assertIn("Bravo SRT", str(ctx.exception))

  def test_open_gripper_raises_not_implemented_naming_the_model(self):
    with self.assertRaises(NotImplementedError) as ctx:
      self.controller.open_gripper()
    self.assertIn("Bravo SRT", str(ctx.exception))

  def test_is_plate_in_gripper_raises_not_implemented_naming_the_model(self):
    with self.assertRaises(NotImplementedError) as ctx:
      self.controller.is_plate_in_gripper()
    self.assertIn("Bravo SRT", str(ctx.exception))

  def test_scan_stack_with_gripper_raises_not_implemented_naming_the_model(self):
    with self.assertRaises(NotImplementedError) as ctx:
      self.controller.scan_stack_with_gripper(start_zg=0.0, end_zg=10.0, speed="slow")
    self.assertIn("Bravo SRT", str(ctx.exception))

  def test_home_axes_rejects_gripper_axes(self):
    with self.assertRaises(BravoError):
      self.controller.home_axes(["g"])

  def test_home_axes_rejects_gripper_axes_naming_the_model(self):
    with self.assertRaises(BravoError) as ctx:
      self.controller.home_axes(["zg"])
    self.assertIn("Bravo SRT", str(ctx.exception))

  def test_jog_not_yet_implemented(self):
    params = JogParams(
      axis="x", velocity=1.0, acceleration=1.0, max_position=10.0, tolerance=0.1, peak_current=0.1
    )
    with self.assertRaises(BravoError):
      self.controller.jog(params)


class MoveTests(unittest.TestCase):
  def test_agile_controller_move_sends_prepare_move_and_move_go(self):
    frame = _v11_frame(RabbitErrorCode.NONE) * 2
    transport = BufferedTransport(frame)
    controller = AgileController(transport)

    controller.move(
      [AxisMoveInfo(axis="x", position=10.0, velocity=50.0, acceleration=100.0)], wait=False
    )

    # One PREPARE_MOVE, plus one MoveGo for controller 1.
    self.assertEqual(len(transport.sent), 2)

  def test_agile_7612_controller_move_requires_axis_homed(self):
    controller = Agile7612Controller(BufferedTransport())

    with self.assertRaises(BravoError) as ctx:
      controller.move([AxisMoveInfo(axis="x", position=10.0)])
    self.assertEqual(ctx.exception.error_type, ErrorType.COULD_NOT_MOVE_TO_POSITION)

  def test_agile_7612_controller_move_sends_bytes_once_homed(self):
    frame = _v11_7612_frame(CommandID.PREPARE_MOVE, RabbitErrorCode.NONE) + _v11_7612_frame(
      CommandID.DIRECT_AGILE_COMMAND, RabbitErrorCode.NONE
    )
    transport = BufferedTransport(frame)
    controller = Agile7612Controller(transport)
    controller._homed["x"] = True

    controller.move(
      [AxisMoveInfo(axis="x", position=10.0, velocity=50.0, acceleration=100.0)], wait=False
    )

    # PREPARE_MOVE and the per-axis trigger both went out; a trailing
    # fault-reset attempt (which the empty rest of the buffer times out) may
    # add more, so this only checks the floor.
    self.assertGreaterEqual(len(transport.sent), 2)

  def test_agile_7612_controller_move_rejects_out_of_range_target(self):
    controller = Agile7612Controller(BufferedTransport())
    controller._homed["x"] = True

    with self.assertRaises(BravoError) as ctx:
      controller.move([AxisMoveInfo(axis="x", position=99_999.0)])
    self.assertEqual(ctx.exception.error_type, ErrorType.COULD_NOT_MOVE_TO_POSITION)


class ProfileReplacementLogicTests(unittest.TestCase):
  """Exercises the typed AxisConfig reads that replaced the profile reflection."""

  def test_home_reg_for_axis_reads_the_configured_value(self):
    cfg = default_axis_config("z")
    cfg.home_complete_register = 0x0160
    controller = Agile7612Controller(BufferedTransport(), axis_config={"z": cfg})
    self.assertEqual(controller._home_reg_for_axis("z"), 0x0160)

  def test_home_reg_for_axis_default_is_zero(self):
    controller = Agile7612Controller(BufferedTransport())
    self.assertEqual(controller._home_reg_for_axis("x"), 0)

  def test_home_sensor_bitmask_falls_back_to_the_per_axis_default_when_unset(self):
    controller = Agile7612Controller(BufferedTransport())
    self.assertEqual(controller._home_sensor_bitmask("x"), 1)
    self.assertEqual(controller._home_sensor_bitmask("y"), 2)
    self.assertEqual(controller._home_sensor_bitmask("z"), 4)
    self.assertEqual(controller._home_sensor_bitmask("w"), 8)
    self.assertEqual(controller._home_sensor_bitmask("g"), 1)
    self.assertEqual(controller._home_sensor_bitmask("zg"), 2)

  def test_home_sensor_bitmask_uses_the_configured_value_when_set(self):
    cfg = default_axis_config("x")
    cfg.home_flag_bitmask = 0x40
    controller = Agile7612Controller(BufferedTransport(), axis_config={"x": cfg})
    self.assertEqual(controller._home_sensor_bitmask("x"), 0x40)

  def test_homing_depart_direction_uses_the_configured_flag(self):
    cfg = default_axis_config("x")
    cfg.home_in_positive_direction = True
    controller = Agile7612Controller(BufferedTransport(), axis_config={"x": cfg})
    self.assertEqual(controller._homing_depart_direction("x"), -1)
    self.assertEqual(controller._homing_depart_direction("y"), 1)

  def test_speed_for_level_uses_the_configured_profile(self):
    controller = Agile7612Controller(BufferedTransport())
    self.assertEqual(controller._speed_for_level("x", "fast"), (400.0, 2000.0))

  def test_speed_for_level_falls_back_when_the_level_is_missing(self):
    cfg = default_axis_config("x")
    cfg.speeds = {}
    controller = Agile7612Controller(BufferedTransport(), axis_config={"x": cfg})
    self.assertEqual(controller._speed_for_level("x", "fast"), (50.0, 100.0))

  def test_get_park_position_reads_the_configured_homing_offset(self):
    cfg = default_axis_config("zg")
    cfg.homing_offset = -20.0
    controller = Agile7612Controller(BufferedTransport(), axis_config={"zg": cfg})
    self.assertEqual(controller.get_park_position("zg"), -20.0)

  def test_srt_home_reg_for_axis_sets_the_0x0100_bit(self):
    cfg = default_axis_config("z")
    cfg.home_complete_register = 0x60
    controller = AgileSrtController(BufferedTransport(), axis_config={"z": cfg})
    self.assertEqual(controller._home_reg_for_axis("z"), 0x0160)


class DiagnosticsTests(unittest.TestCase):
  def test_get_diagnostics_reports_disconnected(self):
    controller = Agile7612Controller(BufferedTransport(connected=False))
    self.assertEqual(controller.get_diagnostics(), {"connected": False})

  def test_get_diagnostics_reports_command_counts_and_errors(self):
    frame = _v11_7612_frame(CommandID.PING_DEVICE, RabbitErrorCode.NONE)
    controller = Agile7612Controller(BufferedTransport(frame))

    controller.send_command(CommandID.PING_DEVICE)
    diag = controller.get_diagnostics()

    self.assertTrue(diag["connected"])
    command_counts = diag["command_counts"]
    assert isinstance(command_counts, dict)
    self.assertEqual(command_counts["PING_DEVICE"], 1)
    self.assertEqual(diag["error_count"], 0)


class DrainDelegationTests(unittest.TestCase):
  def test_drain_tcp_buffer_delegates_to_transport_drain(self):
    controller = Agile7612Controller(BufferedTransport())
    # Must not raise: Transport.drain() is unconditional on the ABC.
    controller._drain_tcp_buffer()


class TimeoutUnitsAreSecondsTests(unittest.TestCase):
  def test_agile_controller_move_default_timeout_is_30_seconds(self):
    default = inspect.signature(AgileController.move).parameters["timeout"].default
    self.assertEqual(default, 30.0)
    self.assertIsInstance(default, float)

  def test_agile_7612_controller_move_default_timeout_is_30_seconds(self):
    default = inspect.signature(Agile7612Controller.move).parameters["timeout"].default
    self.assertEqual(default, 30.0)
    self.assertIsInstance(default, float)

  def test_send_agile_default_timeout_is_2_seconds(self):
    default = inspect.signature(AgileController._send_agile).parameters["timeout"].default
    self.assertEqual(default, 2.0)
    self.assertIsInstance(default, float)

  def test_ping_uses_a_one_second_timeout(self):
    # A regression that reintroduces a millisecond-scale value (e.g. 1000
    # instead of 1.0) here would be a thousand-fold unit error.
    frame = _v11_frame(RabbitErrorCode.NONE)
    transport = BufferedTransport(frame, connected=True)
    controller = AgileController(transport)
    self.assertTrue(controller.ping())


class IsPlateInGripperTests(unittest.TestCase):
  def setUp(self):
    self.controller = AgileController(BufferedTransport())

  def test_returns_false_and_logs_when_g_is_unhomed(self):
    # An unhomed G's position has no calibrated reference frame --
    # get_position() can return an arbitrary raw register reading -- so it
    # must not be compared against the open position, which would produce
    # a false "plate detected".
    self.assertFalse(self.controller.is_axis_homed("g"))
    with self.assertLogs("pylabrobot.agilent.bravo.controllers.agile", "WARNING") as logs:
      self.assertFalse(self.controller.is_plate_in_gripper())
    self.assertIn("not homed", "\n".join(logs.output))

  def test_evaluates_position_normally_once_g_is_homed(self):
    # The unhomed guard must not block the ordinary, already-working path.
    self.controller._homed["g"] = True
    self.controller.get_position = lambda axis: 5.0  # type: ignore[method-assign]
    self.assertTrue(self.controller.is_plate_in_gripper())


if __name__ == "__main__":
  unittest.main()
