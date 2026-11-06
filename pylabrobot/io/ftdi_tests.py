import asyncio
import logging
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Union
from unittest import mock

from pylabrobot.events import EventBus, use_event_bus
from pylabrobot.io import capture as capture_module
from pylabrobot.io import ftd2xx
from pylabrobot.io import ftdi as ftdi_module
from pylabrobot.io.capture import CaptureReader, capturer
from pylabrobot.io.ftd2xx import D2XXError, FTStatus
from pylabrobot.io.ftd2xx_tests import CLARIOSTAR, FT232R, OPEN_ELSEWHERE, PHERASTAR, FakeFTD2XX
from pylabrobot.io.ftdi import (
  FTDI,
  HAS_PYLIBFTDI,
  HAS_PYUSB,
  FTDICommand,
  FTDIValidator,
  is_ftdi_transport_error,
)
from pylabrobot.io.validation_utils import LOG_LEVEL_IO


class _RecordingHandler(logging.Handler):
  """Captures every record emitted to a logger, whatever method produced it - so the
  test cannot be fooled by swapping logger.log for logger.debug/info."""

  def __init__(self) -> None:
    super().__init__(level=LOG_LEVEL_IO)
    self.records: list[logging.LogRecord] = []

  def emit(self, record: logging.LogRecord) -> None:
    self.records.append(record)


@unittest.skipUnless(HAS_PYLIBFTDI and HAS_PYUSB, "pylibftdi/pyusb not installed")
class FTDIEmptyReadTests(unittest.IsolatedAsyncioTestCase):
  """Empty reads must be neither logged nor captured (mirrors io.serial). Polling a quiet
  device reads b'' repeatedly; logging/capturing each one floods the log and capture.
  Locks the `if len(data) != 0:` guard on read()/readline() - asserting on real log
  records (not a mocked method) so re-routing the log call can't fake a pass."""

  def setUp(self) -> None:
    self._handler = _RecordingHandler()
    self._logger = logging.getLogger("pylabrobot.io.ftdi")
    self._prev_level = self._logger.level
    self._logger.setLevel(LOG_LEVEL_IO)
    self._logger.addHandler(self._handler)
    self.addCleanup(self._logger.setLevel, self._prev_level)
    self.addCleanup(self._logger.removeHandler, self._handler)

  def _ftdi(self, return_value: bytes) -> FTDI:
    # Bypass setup() (no hardware) by driving the underlying device read directly.
    dev = FTDI(human_readable_device_name="test", device_id="test")
    dev._dev = mock.Mock()
    dev._dev.read.return_value = return_value
    dev._dev.readline.return_value = return_value
    return dev

  async def test_empty_read_is_not_logged_or_captured(self) -> None:
    dev = self._ftdi(b"")
    with mock.patch.object(ftdi_module.capturer, "record") as mock_record:
      self.assertEqual(await dev.read(4), b"")
    self.assertEqual(self._handler.records, [])
    mock_record.assert_not_called()

  async def test_empty_readline_is_not_logged_or_captured(self) -> None:
    dev = self._ftdi(b"")
    with mock.patch.object(ftdi_module.capturer, "record") as mock_record:
      with self.assertRaises(TimeoutError):
        await dev.readline(timeout=0.05)
    self.assertEqual(self._handler.records, [])
    mock_record.assert_not_called()

  async def test_nonempty_read_is_logged_and_captured(self) -> None:
    dev = self._ftdi(b"\x01\x02")
    with mock.patch.object(ftdi_module.capturer, "record") as mock_record:
      self.assertEqual(await dev.read(2), b"\x01\x02")
    self.assertEqual(len(self._handler.records), 1)
    self.assertIn(str(b"\x01\x02"), self._handler.records[0].getMessage())
    mock_record.assert_called_once()


class _BlockingDevice:
  """A pylibftdi device stand-in whose calls block, standing in for a bulk transfer.

  Args:
    to_read: bytes the device hands out, oldest first.
    block_s: how long every call sleeps before returning.
  """

  def __init__(self, to_read: bytes = b"", block_s: float = 0.0, text_mode: bool = False) -> None:
    self._to_read = bytearray(to_read)
    self._block_s = block_s
    self._text_mode = text_mode
    self.written = bytearray()
    self.reads = 0
    self.ftdi_fn = mock.Mock()

  def read(self, num_bytes: int) -> Union[bytes, str]:
    data = bytes(self._to_read[:num_bytes])
    del self._to_read[:num_bytes]
    self.reads += 1
    time.sleep(self._block_s)
    # pylibftdi decodes with latin-1 when the device was opened in text mode.
    return data.decode("latin-1") if self._text_mode else data

  def write(self, data: bytes) -> int:
    time.sleep(self._block_s)
    self.written.extend(data)
    return len(data)


@unittest.skipUnless(HAS_PYLIBFTDI and HAS_PYUSB, "pylibftdi/pyusb not installed")
class FTDICancellationTests(unittest.IsolatedAsyncioTestCase):
  """Running read/write on the executor makes them cancellable for the first time, and a call
  already handed to the worker cannot be recalled. Locks the guarantee that neither end loses
  data when its caller goes away: a cancelled read's bytes are already out of the chip's buffer,
  so dropping them would leave the next read part-way through a response."""

  def _ftdi(self, device: _BlockingDevice) -> FTDI:
    # Bypass setup() (no hardware) by driving the underlying device directly.
    io = FTDI(human_readable_device_name="test", device_id="test")
    io._dev = device
    io._executor = ThreadPoolExecutor(max_workers=1)
    self.addCleanup(io._executor.shutdown)
    return io

  async def test_cancelled_read_keeps_its_bytes_for_the_next_read(self) -> None:
    device = _BlockingDevice(to_read=b"ABCD", block_s=0.2)
    io = self._ftdi(device)

    with self.assertRaises(asyncio.TimeoutError):
      await asyncio.wait_for(io.read(4), timeout=0.05)

    self.assertEqual(await io.read(4), b"ABCD")
    # The second read waited for the one in flight rather than issuing another.
    self.assertEqual(device.reads, 1)

  async def test_cancelled_read_is_captured_once(self) -> None:
    device = _BlockingDevice(to_read=b"ABCD", block_s=0.2)
    io = self._ftdi(device)
    records: List[Any] = []

    with mock.patch.object(ftdi_module.capturer, "record", records.append):
      with self.assertRaises(asyncio.TimeoutError):
        await asyncio.wait_for(io.read(4), timeout=0.05)
      await io.read(4)

    self.assertEqual([(r.action, r.data) for r in records], [("read", "41424344")])

  async def test_cancelled_write_still_reaches_the_device(self) -> None:
    device = _BlockingDevice(block_s=0.2)
    io = self._ftdi(device)
    records: List[Any] = []

    with mock.patch.object(ftdi_module.capturer, "record", records.append):
      with self.assertRaises(asyncio.TimeoutError):
        await asyncio.wait_for(io.write(b"\x01\x02"), timeout=0.05)
      await asyncio.sleep(0.3)  # let the worker finish

    self.assertEqual(bytes(device.written), b"\x01\x02")
    self.assertEqual([(r.action, r.data) for r in records], [("write", "0102")])

  async def test_purge_discards_bytes_held_for_the_next_read(self) -> None:
    device = _BlockingDevice(to_read=b"ABCD", block_s=0.2)
    io = self._ftdi(device)

    with self.assertRaises(asyncio.TimeoutError):
      await asyncio.wait_for(io.read(4), timeout=0.05)
    await io.usb_purge_rx_buffer()

    # A purge means the caller has decided everything in flight is stale.
    self.assertEqual(await io.read(4), b"")

  async def test_a_text_mode_device_still_reads(self) -> None:
    """pylibftdi returns str, decoded with latin-1, if the device was opened in text mode. setup()
    opens it in byte mode, so this only guards against losing the tolerance upstream had."""
    io = self._ftdi(_BlockingDevice(to_read=b"\xff\x01", text_mode=True))

    self.assertEqual(await io.read(2), b"\xff\x01")

  async def test_second_concurrent_reader_gets_an_empty_read(self) -> None:
    """Two callers on one io stay unsupported, but the second one now backs off rather than
    taking bytes out of the middle of the first one's response."""
    io = self._ftdi(_BlockingDevice(to_read=b"ABCD", block_s=0.01))
    received: dict = {"first": [], "second": []}

    async def reader(name: str) -> None:
      for _ in range(4):
        received[name].append(await io.read(1))

    await asyncio.gather(reader("first"), reader("second"))

    self.assertEqual(received["first"], [b"A", b"B", b"C", b"D"])
    self.assertEqual(received["second"], [b"", b"", b"", b""])


class FTDITopologyLifecycleTests(unittest.IsolatedAsyncioTestCase):
  """Topology selection remains portable and always releases its host resources."""

  @staticmethod
  def _ftdi_without_optional_dependency_checks() -> FTDI:
    io = FTDI.__new__(FTDI)
    io._interface_select = None
    io._detached_kernel_driver = None
    io._dev = None
    io._executor = None
    io._unread = bytearray()
    io._pending_read = None
    return io

  async def test_detach_tolerates_a_backend_without_kernel_driver_support(self) -> None:
    io = self._ftdi_without_optional_dependency_checks()
    device = mock.Mock()
    device.is_kernel_driver_active.side_effect = NotImplementedError
    dispose_resources = mock.Mock()
    usb_module = SimpleNamespace(util=SimpleNamespace(dispose_resources=dispose_resources))

    with (
      mock.patch.object(ftdi_module, "usb", usb_module, create=True),
      mock.patch.object(io, "_usb_device_at", return_value=device),
    ):
      io._detach_kernel_driver(3, 17)

    self.assertIsNone(io._detached_kernel_driver)
    device.detach_kernel_driver.assert_not_called()
    dispose_resources.assert_called_once_with(device)

  async def test_detach_tolerates_an_unsupported_detach_operation(self) -> None:
    io = self._ftdi_without_optional_dependency_checks()
    device = mock.Mock()
    device.is_kernel_driver_active.return_value = True
    device.detach_kernel_driver.side_effect = NotImplementedError
    dispose_resources = mock.Mock()
    usb_module = SimpleNamespace(util=SimpleNamespace(dispose_resources=dispose_resources))

    with (
      mock.patch.object(ftdi_module, "usb", usb_module, create=True),
      mock.patch.object(io, "_usb_device_at", return_value=device),
    ):
      io._detach_kernel_driver(3, 17)

    self.assertIsNone(io._detached_kernel_driver)
    device.detach_kernel_driver.assert_called_once_with(0)
    dispose_resources.assert_called_once_with(device)

  async def test_stop_reattaches_and_shuts_down_after_close_failure(self) -> None:
    io = self._ftdi_without_optional_dependency_checks()
    device = mock.Mock()
    device.close.side_effect = RuntimeError("simulated close failure")
    io._dev = device
    io._executor = ThreadPoolExecutor(max_workers=1)
    io._detached_kernel_driver = (3, 17, 0)

    with (
      mock.patch.object(io, "_reattach_kernel_driver") as reattach,
      self.assertRaisesRegex(RuntimeError, "close failure"),
    ):
      await io.stop()

    reattach.assert_called_once_with()
    self.assertIsNone(io._dev)
    self.assertIsNone(io._executor)

  async def test_cancelled_stop_waits_for_close_and_reattaches(self) -> None:
    io = self._ftdi_without_optional_dependency_checks()
    close_started = threading.Event()

    def close() -> None:
      close_started.set()
      time.sleep(0.05)

    io._dev = SimpleNamespace(close=close)
    io._executor = ThreadPoolExecutor(max_workers=1)
    io._detached_kernel_driver = (3, 17, 0)

    with mock.patch.object(io, "_reattach_kernel_driver") as reattach:
      stop_task = asyncio.create_task(io.stop())
      for _ in range(100):
        if close_started.is_set():
          break
        await asyncio.sleep(0.001)
      self.assertTrue(close_started.is_set())
      stop_task.cancel()
      for _ in range(200):
        if stop_task.done():
          break
        await asyncio.sleep(0.001)
      self.assertTrue(stop_task.done())
      with self.assertRaises(asyncio.CancelledError):
        await stop_task

    reattach.assert_called_once_with()
    self.assertIsNone(io._dev)
    self.assertIsNone(io._executor)


@unittest.skipUnless(HAS_PYLIBFTDI and HAS_PYUSB, "pylibftdi/pyusb not installed")
class FTDIEventTests(unittest.IsolatedAsyncioTestCase):
  """`read` now returns from a buffer and `write` is shielded, so both `emit_event` sites moved."""

  async def test_read_and_write_still_emit_io_events(self) -> None:
    io = FTDI(human_readable_device_name="test", device_id="test")
    io._dev = _BlockingDevice(to_read=b"\x03\x04")
    events: List[Any] = []
    bus = EventBus()
    bus.subscribe(events.append)

    with use_event_bus(bus):
      await io.write(b"\x01\x02")
      await io.read(2)

    self.assertEqual(
      [(event.name, event.data["data"]) for event in events],
      [("io.write", "0102"), ("io.read", "0304")],
    )


@unittest.skipUnless(HAS_PYLIBFTDI and HAS_PYUSB, "pylibftdi/pyusb not installed")
class FTDICaptureTests(unittest.IsolatedAsyncioTestCase):
  """A capture keeps its payload, so FTDIValidator can replay it. FTDICommand declared `data`
  without assigning it, so every recorded FTDI command was written out without its bytes."""

  def setUp(self) -> None:
    # CaptureReader.done() leaves the capture/validation flag set, which blocks the next FTDI().
    self.addCleanup(setattr, capture_module, "_capture_or_validation_active", False)

  def test_command_carries_its_data(self) -> None:
    # The capture file is written from __dict__, so an unassigned field is a lost payload.
    command = FTDICommand(device_id="test", action="write", data="dead")
    self.assertEqual(command.__dict__["data"], "dead")

  async def test_capture_replays_through_the_validator(self) -> None:
    io = FTDI(human_readable_device_name="test", device_id="test")
    io._dev = _BlockingDevice(to_read=b"\x03\x04")

    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "capture.json"
      capturer.start(path)
      self.addCleanup(lambda: capturer.capture_active and capturer.stop())
      await io.set_baudrate(115200)
      await io.write(b"\x01\x02")
      await io.read(2)
      capturer.stop()

      reader = CaptureReader(str(path))
      validator = FTDIValidator(reader, "test", "test")
      await validator.set_baudrate(115200)
      await validator.write(b"\x01\x02")
      self.assertEqual(await validator.read(2), b"\x03\x04")
      reader.done()


@unittest.skipUnless(HAS_PYLIBFTDI and HAS_PYUSB, "pylibftdi/pyusb not installed")
class FTDIReadlineTests(unittest.IsolatedAsyncioTestCase):
  """pylibftdi's readline raises TypeError unless the device was opened in text mode, and setup()
  opens it in byte mode, so the line is assembled here instead."""

  def _ftdi(self, to_read: bytes) -> FTDI:
    io = FTDI(human_readable_device_name="test", device_id="test")
    io._dev = _BlockingDevice(to_read=to_read)
    return io

  async def test_line_is_returned_with_its_terminator(self) -> None:
    io = self._ftdi(b"ok\ntrailing")
    self.assertEqual(await io.readline(), b"ok\n")
    self.assertEqual(await io.readline(terminator=b"g"), b"trailing")

  async def test_incomplete_line_times_out(self) -> None:
    io = self._ftdi(b"partial")
    with self.assertRaises(TimeoutError):
      await io.readline(timeout=0.05)

  async def test_empty_terminator_is_rejected(self) -> None:
    io = self._ftdi(b"")
    with self.assertRaises(ValueError):
      await io.readline(terminator=b"")


class FTDID2XXBackendTests(unittest.IsolatedAsyncioTestCase):
  """`FTDI(backend="d2xx")` over a fake FTD2XX library, with neither pylibftdi nor pyusb."""

  def setUp(self) -> None:
    self.library = FakeFTD2XX([PHERASTAR, FT232R])
    for patcher in (
      mock.patch.object(ftd2xx, "load_library", return_value=self.library),
      mock.patch.object(ftdi_module, "HAS_PYLIBFTDI", False),
      mock.patch.object(ftdi_module, "HAS_PYUSB", False),
    ):
      patcher.start()
      self.addCleanup(patcher.stop)

  async def test_an_io_runs_over_d2xx_without_pylibftdi(self) -> None:
    io = FTDI(human_readable_device_name="reader", vid=0x0403, pid=0xBB68, backend="d2xx")
    self.addAsyncCleanup(io.stop)
    self.assertEqual(io.backend, "d2xx")

    await io.setup()
    self.assertEqual(io.device_id, "471-0376")
    await io.set_baudrate(125000)
    await io.set_line_property(8, 0, 0)
    await io.set_latency_timer(2)
    await io.usb_purge_rx_buffer()
    self.assertEqual(await io.write(b"\x02ping\x0d"), 6)
    self.library.received += b"\x02pong\x0d"
    self.assertEqual(await io.read(25), b"\x02pong\x0d")
    self.library.modem_status = 0x6030
    self.assertEqual(await io.poll_modem_status(), 0x6030)
    await io.stop()

    self.assertEqual(
      self.library.calls,
      [
        ("FT_SetVIDPID", 0x0403, 0xBB68),
        ("FT_CreateDeviceInfoList",),
        ("FT_GetDeviceInfoDetail", 0),
        ("FT_GetDeviceInfoDetail", 1),
        ("FT_OpenEx", b"471-0376", 1),
        ("FT_SetTimeouts", 5000, 5000),
        ("FT_SetFlowControl", 0, 0, 0),
        ("FT_SetBaudRate", 9600),
        ("FT_SetLatencyTimer", 16),
        ("FT_SetBaudRate", 125000),
        ("FT_SetDataCharacteristics", 8, 0, 0),
        ("FT_SetLatencyTimer", 2),
        ("FT_Purge", 1),
        ("FT_Write", b"\x02ping\x0d"),
        ("FT_GetQueueStatus",),
        ("FT_Read", 6),
        ("FT_GetModemStatus",),
        ("FT_Close",),
      ],
    )
    self.assertEqual(
      io.serialize(),
      {
        "human_readable_device_name": "reader",
        "device_id": "471-0376",
        "vid": 0x0403,
        "pid": 0xBB68,
        "usb_address": None,
        "backend": "d2xx",
      },
    )

  async def test_a_device_id_is_opened_by_serial_number_without_listing(self) -> None:
    io = FTDI(human_readable_device_name="reader", device_id="471-0376", backend="d2xx")
    self.addAsyncCleanup(io.stop)
    await io.setup()
    self.assertEqual(self.library.calls[0], ("FT_OpenEx", b"471-0376", 1))
    self.assertNotIn("FT_CreateDeviceInfoList", [call[0] for call in self.library.calls])

  async def test_the_vid_pid_must_match_exactly_one_device_not_open_elsewhere(self) -> None:
    cases = [
      ([FT232R], "No FTDI D2XX devices found.*A50285BI \\(VID:PID 0403:6001"),
      ([OPEN_ELSEWHERE], "No FTDI D2XX devices found.*a device open in another process"),
      ([], "No FTDI D2XX devices found.*D2XX devices: none"),
      ([PHERASTAR, CLARIOSTAR], "Multiple FTDI D2XX devices found.*471-0376.*430-0001"),
    ]
    for devices, message in cases:
      with self.subTest(message=message):
        self.library.devices = devices
        io = FTDI(human_readable_device_name="reader", vid=0x0403, pid=0xBB68, backend="d2xx")
        with self.assertRaisesRegex(RuntimeError, message):
          await io.setup()
        self.assertIsNone(io._dev)
        self.assertIsNone(io._executor)
        self.assertIsNone(self.library.opened)

  async def test_a_device_that_does_not_open_is_reported_by_name(self) -> None:
    self.library.failures["FT_OpenEx"] = FTStatus.FT_DEVICE_NOT_OPENED  # open in BMG's software
    io = FTDI(human_readable_device_name="reader", device_id="471-0376", backend="d2xx")
    with self.assertRaisesRegex(RuntimeError, "Failed to open FTDI device for 'reader'") as raised:
      await io.setup()
    self.assertIsInstance(raised.exception.__cause__, D2XXError)
    self.assertIsNone(io._dev)
    self.assertIsNone(io._executor)

  def test_libftdi_only_arguments_and_unknown_backends_are_refused(self) -> None:
    with self.assertRaisesRegex(ValueError, "usb_address is libftdi-only"):
      FTDI(human_readable_device_name="reader", usb_address="1-2", backend="d2xx")
    with self.assertRaisesRegex(ValueError, "interface_select is libftdi-only"):
      FTDI(human_readable_device_name="reader", interface_select=1, backend="d2xx")
    with self.assertRaisesRegex(ValueError, "backend must be"):
      FTDI(human_readable_device_name="reader", backend="pyftdi")  # type: ignore[arg-type]

  def test_d2xx_errors_are_ftdi_transport_errors(self) -> None:
    self.assertTrue(is_ftdi_transport_error(D2XXError("FT_Read", FTStatus.FT_IO_ERROR)))
    self.assertFalse(is_ftdi_transport_error(RuntimeError("not a transport error")))


class FTDIDefaultBackendTests(unittest.TestCase):
  """With no backend given, `FTDI` is what it was: pylibftdi over libusb."""

  def test_the_default_backend_needs_pylibftdi(self) -> None:
    with (
      mock.patch.object(ftdi_module, "HAS_PYLIBFTDI", False),
      mock.patch.object(ftdi_module, "_FTDI_ERROR", ImportError("absent"), create=True),
    ):
      with self.assertRaisesRegex(RuntimeError, "pylibftdi is not installed"):
        FTDI(human_readable_device_name="reader")
      with self.assertRaisesRegex(RuntimeError, "pylibftdi is not installed"):
        FTDI(human_readable_device_name="reader", backend=None)

  def test_the_default_backend_opens_a_pylibftdi_device_and_never_d2xx(self) -> None:
    device_class = mock.Mock()
    with (
      mock.patch.object(ftdi_module, "HAS_PYLIBFTDI", True),
      mock.patch.object(ftdi_module, "HAS_PYUSB", True),
      mock.patch.object(ftdi_module, "Device", device_class, create=True),
      mock.patch.object(ftd2xx, "load_library", side_effect=AssertionError("D2XX loaded")),
    ):
      io = FTDI(human_readable_device_name="reader", device_id="S1")
      self.assertEqual(io.backend, "libftdi")
      self.assertEqual(io.serialize()["backend"], "libftdi")
      with mock.patch.object(io, "_resolve_device_serial", return_value="S1"):
        io._setup_sync()

    device_class.assert_called_once_with(
      lazy_open=True, device_id="S1", pid=None, vid=None, interface_select=None
    )
    device_class.return_value.open.assert_called_once_with()
    self.assertIs(io._dev, device_class.return_value)


if __name__ == "__main__":
  unittest.main()
