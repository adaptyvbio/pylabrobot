"""PHERAstar driver tests over a fake io. Every frame was recorded from a PHERAstar FX (FTDI,
firmware 1.32, serial 471-0376) while BMG's own software drove it, on 2026-11-06."""

import unittest

from pylabrobot.bmg_labtech.clariostar.driver.master_tests import FakeFTDI
from pylabrobot.bmg_labtech.pherastar.driver.configuration import PHERAstarConfiguration
from pylabrobot.bmg_labtech.pherastar.driver.master import PHERAstarDriver
from pylabrobot.bmg_labtech.protocol.framing import ftdi_extract_payload, ftdi_validate_frame

INIT_FRAME = bytes.fromhex("02000d09010000100100002a0d")
INIT_REPLY = bytes.fromhex(
  "02005e091325002100010106052801040102090c03046a107d0000f623044d280d0220000e032700000f1b0305"
  "50313a37333732333020440d0a50323a373337323330204400110300000012020faa1f090100ec181704370021"
  "000b800d"
)
PARAMETERS_FRAME = bytes.fromhex("02000d090e0900015a00008a0d")
STATUS_QUERY_FRAME = bytes.fromhex("02000909800000940d")
DRAWER_OPEN_FRAME = bytes.fromhex("02000e0903010000000000001d0d")
DRAWER_CLOSE_FRAME = bytes.fromhex("02000e0903000000000000001c0d")
TEMPERATURE_25_FRAME = bytes.fromhex("02000b090600fa0001160d")
TEMPERATURE_MONITOR_FRAME = bytes.fromhex("02000b0906000100001d0d")


def frame(payload_hex: str) -> bytes:
  """A reply frame with the PHERAstar's header around ``payload_hex``."""
  payload = bytes.fromhex(payload_hex)
  body = bytes([0x02]) + (len(payload) + 8).to_bytes(2, "big") + bytes([0x09]) + payload
  return body + (sum(body) & 0xFFFFFF).to_bytes(3, "big") + b"\x0d"


# Status replies seen: idle with a plate in, busy moving the drawer, idle with the drawer out.
STATUS_IDLE_PLATE_IN = frame("0105060601010000000000012f013300")
STATUS_BUSY_DRAWER = frame("0125062601010300000000012d013400")
STATUS_DRAWER_OUT = frame("0105062101010000000000012c013400")
STATUS_HEATING = frame("010516060101000000000001300134c0")


def make_driver(io: FakeFTDI) -> PHERAstarDriver:
  driver = PHERAstarDriver(io=io)
  driver._ftdi._packet_read_timeout = 0.01
  driver._ftdi._frame_stall_timeout = 0.05
  driver._RETRY_SETTLE = 0
  driver.default_poll_interval = 0
  driver.drawer.default_poll_interval = 0
  return driver


class FramingTests(unittest.TestCase):
  def test_recorded_frames_carry_the_pherastar_header(self):
    for recorded in (
      INIT_FRAME,
      INIT_REPLY,
      PARAMETERS_FRAME,
      STATUS_QUERY_FRAME,
      DRAWER_OPEN_FRAME,
    ):
      ftdi_validate_frame(recorded, header=0x09)

  def test_helper_frames_match_recorded_ones(self):
    self.assertEqual(frame("80"), STATUS_QUERY_FRAME)
    self.assertEqual(frame("030100000000"), DRAWER_OPEN_FRAME)


class StatusTests(unittest.TestCase):
  def test_flags_and_temperatures_from_recorded_replies(self):
    idle = PHERAstarDriver._parse_status(ftdi_extract_payload(STATUS_IDLE_PLATE_IN))
    self.assertFalse(idle.busy)
    self.assertFalse(idle.drawer_open)
    self.assertTrue(idle.plate_detected)
    self.assertEqual((idle.temperature_bottom, idle.temperature_top), (30.3, 30.7))
    self.assertTrue(PHERAstarDriver._parse_status(ftdi_extract_payload(STATUS_BUSY_DRAWER)).busy)
    self.assertTrue(
      PHERAstarDriver._parse_status(ftdi_extract_payload(STATUS_DRAWER_OUT)).drawer_open
    )


class ConfigurationTests(unittest.TestCase):
  def test_init_reply(self):
    configuration = PHERAstarConfiguration()
    configuration.update_from_init_reply(ftdi_extract_payload(INIT_REPLY))
    self.assertEqual(configuration.model, "PHERAstarFX")
    self.assertEqual(configuration.firmware_version, "1.32")
    self.assertTrue(configuration.temperature_controller)
    self.assertEqual(configuration.temperature_controller_channels, 2)
    self.assertEqual(configuration.pumps, ("XCalibur", "XCalibur", None))
    self.assertTrue(configuration.spectrometer)
    self.assertFalse(configuration.stacker_connected)
    self.assertEqual(
      sorted(configuration.init_blocks),
      ["0x01", "0x09", "0x0d", "0x0e", "0x0f", "0x11", "0x12", "0x1f"],
    )
    # Not decoded for the PHERAstar yet.
    self.assertIsNone(configuration.dual_photomultiplier_tubes)
    self.assertIsNone(configuration.barcode_readers)

  def test_a_cut_reply_is_refused(self):
    with self.assertRaises(ValueError):
      PHERAstarConfiguration().update_from_init_reply(ftdi_extract_payload(INIT_REPLY)[:-3])


class CommandTests(unittest.IsolatedAsyncioTestCase):
  async def test_initialize_sends_bmgs_init(self):
    io = FakeFTDI()
    driver = make_driver(io)
    io.replies = [INIT_REPLY, STATUS_BUSY_DRAWER, STATUS_IDLE_PLATE_IN]
    await driver.initialize()
    self.assertEqual(io.written[0], INIT_FRAME)
    self.assertEqual(driver.configuration.firmware_version, "1.32")

  async def test_stored_parameters_read(self):
    io = FakeFTDI()
    driver = make_driver(io)
    io.replies = [frame("0705002001010000")]
    reply = await driver.request_stored_parameters()
    self.assertEqual(io.written, [PARAMETERS_FRAME])
    self.assertEqual(reply[0], 0x07)

  async def test_drawer_frames(self):
    io = FakeFTDI()
    driver = make_driver(io)
    io.replies = [STATUS_IDLE_PLATE_IN, STATUS_BUSY_DRAWER, STATUS_DRAWER_OUT, STATUS_DRAWER_OUT]
    await driver.drawer.open()
    self.assertIn(DRAWER_OPEN_FRAME, io.written)
    self.assertTrue(driver.drawer.drawer_open)

  async def test_temperature_frames(self):
    io = FakeFTDI()
    driver = make_driver(io)
    driver.configuration.temperature_controller = True
    driver._discover_features()
    controller = driver.temperature_controller
    assert controller is not None
    io.replies = [STATUS_HEATING, STATUS_HEATING, STATUS_HEATING, STATUS_HEATING]
    await controller.start_temperature_control(25.0)
    self.assertIn(TEMPERATURE_25_FRAME, io.written)
    io.written.clear()
    io.replies = [STATUS_HEATING, STATUS_HEATING]
    await controller.stop_temperature_control()
    self.assertIn(TEMPERATURE_MONITOR_FRAME, io.written)

  async def test_shutdown_leaves_the_sensors_monitoring(self):
    io = FakeFTDI()
    driver = make_driver(io)
    driver.configuration.temperature_controller = True
    driver._discover_features()
    controller = driver.temperature_controller
    assert controller is not None
    controller.target_temperature = 25.0
    io.replies = [STATUS_HEATING] * 8 + [STATUS_IDLE_PLATE_IN] * 4
    await driver._shut_down(accept_plate_left_in_device=True)
    self.assertIn(TEMPERATURE_MONITOR_FRAME, io.written)
    self.assertNotIn(frame("060000"), io.written)
    self.assertIs(driver.temperature_controller, controller)


if __name__ == "__main__":
  unittest.main()
