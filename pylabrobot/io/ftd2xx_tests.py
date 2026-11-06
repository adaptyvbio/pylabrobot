"""The D2XX binding over a fake FTD2XX library: Python callables standing in for its functions.

Expected D2XX values are written out as numbers, as ftd2xx.h defines them, so a wrong constant in
the binding fails here.
"""

import ctypes
import unittest
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

from pylabrobot.io import ftd2xx
from pylabrobot.io.ftd2xx import D2XXDevice, D2XXDeviceInfo, D2XXError, FTStatus

# Device list entries, as FT_GetDeviceInfoDetail fills them. The PHERAstar FX's serial number is
# the one BMG's software opens; the descriptions, location IDs and the other serial numbers are
# made up.
PHERASTAR = {
  "flags": 0x2,  # FT_FLAGS_HISPEED
  "type": 8,  # FT_DEVICE_232H
  "vid": 0x0403,
  "pid": 0xBB68,
  "location": 0x21,
  "serial": "471-0376",
  "description": "BMG reader",
}
CLARIOSTAR = dict(PHERASTAR, location=0x22, serial="430-0001")
FT232R = {
  "flags": 0x0,
  "type": 5,  # FT_DEVICE_232R
  "vid": 0x0403,
  "pid": 0x6001,
  "location": 0x31,
  "serial": "A50285BI",
  "description": "FT232R USB UART",
}
# A device open in another process: D2XX fills in its flags only.
OPEN_ELSEWHERE = {
  "flags": 0x3,  # FT_FLAGS_OPENED | FT_FLAGS_HISPEED
  "type": 0,
  "vid": 0,
  "pid": 0,
  "location": 0,
  "serial": "",
  "description": "",
}


def _out(reference: Any, ctype: Any) -> Any:
  """What an output argument, a `ctypes.byref`, refers to."""
  return ctypes.cast(reference, ctypes.POINTER(ctype)).contents


class FakeFTD2XX:
  """FTD2XX's functions as Python callables, over a model of D2XX's device list and of the
  driver's buffers for the one device open.

  Every call is recorded in `calls` as (function, arguments), leaving out the handle and the
  references a function writes its results through.
  """

  HANDLE = 0x5EED
  # Functions that only take their arguments and return a status.
  _SETTINGS = {
    "FT_SetTimeouts",
    "FT_SetBaudRate",
    "FT_SetDataCharacteristics",
    "FT_SetFlowControl",
    "FT_SetLatencyTimer",
    "FT_SetRts",
    "FT_ClrRts",
    "FT_SetDtr",
    "FT_ClrDtr",
    "FT_ResetDevice",
  }

  def __init__(
    self, devices: Optional[List[Dict[str, Any]]] = None, has_set_vid_pid: bool = True
  ) -> None:
    """
    Args:
      devices: D2XX's device list.
      has_set_vid_pid: whether the library has FT_SetVIDPID, as libftd2xx has and ftd2xx.dll not.
    """
    self.devices = devices if devices is not None else [PHERASTAR]
    self.has_set_vid_pid = has_set_vid_pid
    self.calls: List[Tuple[Any, ...]] = []
    self.failures: Dict[str, int] = {}  # function -> the status it returns instead of FT_OK
    self.received = bytearray()  # what the driver holds for FT_Read
    self.arriving: List[bytes] = []  # bytes arriving before each FT_GetQueueStatus, in turn
    self.written = bytearray()
    self.modem_status = 0
    self.opened: Optional[bytes] = None  # the serial number of the device open

  def _status(self, function: str, *args: Any) -> int:
    self.calls.append((function, *args))
    return self.failures.get(function, FTStatus.FT_OK)

  def _status_on_handle(self, function: str, handle: Any, *args: Any) -> int:
    if self.opened is None or handle.value != self.HANDLE:
      self.calls.append((function, *args))
      return FTStatus.FT_INVALID_HANDLE
    return self._status(function, *args)

  def __getattr__(self, name: str) -> Any:
    if name in self._SETTINGS:
      return lambda handle, *args: self._status_on_handle(name, handle, *args)
    if name == "FT_SetVIDPID" and self.has_set_vid_pid:
      return lambda vid, pid: self._status(name, vid, pid)
    raise AttributeError(name)

  def FT_CreateDeviceInfoList(self, count: Any) -> int:
    _out(count, ctypes.c_uint32).value = len(self.devices)
    return self._status("FT_CreateDeviceInfoList")

  def FT_GetDeviceInfoDetail(
    self,
    index: int,
    flags: Any,
    device_type: Any,
    device_id: Any,
    location_id: Any,
    serial_number: Any,
    description: Any,
    handle: Any,
  ) -> int:
    device = self.devices[index]
    _out(flags, ctypes.c_uint32).value = device["flags"]
    _out(device_type, ctypes.c_uint32).value = device["type"]
    _out(device_id, ctypes.c_uint32).value = device["vid"] << 16 | device["pid"]
    _out(location_id, ctypes.c_uint32).value = device["location"]
    serial_number.value = device["serial"].encode()
    description.value = device["description"].encode()
    return self._status("FT_GetDeviceInfoDetail", index)

  def FT_OpenEx(self, serial_number: bytes, flags: int, handle: Any) -> int:
    status = self._status("FT_OpenEx", serial_number, flags)
    if status != FTStatus.FT_OK:
      return status
    if not any(device["serial"].encode() == serial_number for device in self.devices):
      return FTStatus.FT_DEVICE_NOT_FOUND
    _out(handle, ctypes.c_void_p).value = self.HANDLE
    self.opened = serial_number
    return status

  def FT_Close(self, handle: Any) -> int:
    status = self._status_on_handle("FT_Close", handle)
    self.opened = None
    return status

  def FT_GetQueueStatus(self, handle: Any, count: Any) -> int:
    status = self._status_on_handle("FT_GetQueueStatus", handle)
    if status == FTStatus.FT_OK:
      if self.arriving:
        self.received += self.arriving.pop(0)
      _out(count, ctypes.c_uint32).value = len(self.received)
    return status

  def FT_Read(self, handle: Any, buffer: Any, count: int, received: Any) -> int:
    status = self._status_on_handle("FT_Read", handle, count)
    if status == FTStatus.FT_OK:
      data = bytes(self.received[:count])
      del self.received[:count]
      ctypes.memmove(buffer, data, len(data))
      _out(received, ctypes.c_uint32).value = len(data)
    return status

  def FT_Write(self, handle: Any, data: bytes, count: int, written: Any) -> int:
    status = self._status_on_handle("FT_Write", handle, data[:count])
    if status == FTStatus.FT_OK:
      self.written += data[:count]
      _out(written, ctypes.c_uint32).value = count
    return status

  def FT_Purge(self, handle: Any, mask: int) -> int:
    status = self._status_on_handle("FT_Purge", handle, mask)
    if status == FTStatus.FT_OK and mask & 1:  # FT_PURGE_RX
      self.received.clear()
    return status

  def FT_GetModemStatus(self, handle: Any, modem_status: Any) -> int:
    status = self._status_on_handle("FT_GetModemStatus", handle)
    if status == FTStatus.FT_OK:
      _out(modem_status, ctypes.c_uint32).value = self.modem_status
    return status


class D2XXListingTests(unittest.TestCase):
  def test_each_device_is_listed_with_the_vid_and_pid_its_id_holds(self):
    devices = ftd2xx.list_devices(FakeFTD2XX([PHERASTAR, FT232R]))
    self.assertEqual(
      devices,
      [
        D2XXDeviceInfo(
          index=0,
          flags=0x2,
          type=8,
          vid=0x0403,
          pid=0xBB68,
          location_id=0x21,
          serial_number="471-0376",
          description="BMG reader",
        ),
        D2XXDeviceInfo(
          index=1,
          flags=0x0,
          type=5,
          vid=0x0403,
          pid=0x6001,
          location_id=0x31,
          serial_number="A50285BI",
          description="FT232R USB UART",
        ),
      ],
    )
    self.assertFalse(devices[0].opened)

  def test_a_device_open_in_another_process_is_listed_as_open_without_details(self):
    (device,) = ftd2xx.list_devices(FakeFTD2XX([OPEN_ELSEWHERE]))
    self.assertTrue(device.opened)
    self.assertEqual((device.serial_number, device.vid, device.pid), ("", 0, 0))

  def test_a_failed_listing_raises(self):
    library = FakeFTD2XX()
    library.failures["FT_CreateDeviceInfoList"] = FTStatus.FT_IO_ERROR
    with self.assertRaisesRegex(D2XXError, r"FT_CreateDeviceInfoList returned FT_IO_ERROR \(4\)"):
      ftd2xx.list_devices(library)

  def test_a_custom_vid_pid_is_registered_where_the_library_needs_it(self):
    libftd2xx = FakeFTD2XX()
    ftd2xx.register_vid_pid(0x0403, 0xBB68, libftd2xx)
    self.assertEqual(libftd2xx.calls, [("FT_SetVIDPID", 0x0403, 0xBB68)])

    windows_dll = FakeFTD2XX(has_set_vid_pid=False)
    ftd2xx.register_vid_pid(0x0403, 0xBB68, windows_dll)
    self.assertEqual(windows_dll.calls, [])

  def test_a_library_that_does_not_load_names_the_file(self):
    with self.assertRaisesRegex(RuntimeError, "D2XX library could not be loaded.*no-libftd2xx"):
      ftd2xx.load_library("/no-libftd2xx/libftd2xx.so")


class D2XXDeviceTests(unittest.TestCase):
  def _open(self, library: Optional[FakeFTD2XX] = None) -> Tuple[D2XXDevice, FakeFTD2XX]:
    library = library if library is not None else FakeFTD2XX()
    device = D2XXDevice("471-0376", library=library)
    device.open()
    library.calls.clear()
    return device, library

  def test_open_by_serial_number_with_pylibftdi_settings_and_libftdi_timeouts(self):
    library = FakeFTD2XX()
    device = D2XXDevice("471-0376", library=library)
    self.assertTrue(device.closed)
    device.open()
    self.assertFalse(device.closed)
    self.assertEqual(
      library.calls,
      [
        ("FT_OpenEx", b"471-0376", 1),  # FT_OPEN_BY_SERIAL_NUMBER
        ("FT_SetTimeouts", 5000, 5000),
        ("FT_SetFlowControl", 0x0000, 0, 0),  # FT_FLOW_NONE
        ("FT_SetBaudRate", 9600),
        ("FT_SetLatencyTimer", 16),
      ],
    )
    self.assertEqual(device.baudrate, 9600)

  def test_a_device_that_does_not_open_stays_closed(self):
    library = FakeFTD2XX()
    library.failures["FT_OpenEx"] = FTStatus.FT_DEVICE_NOT_OPENED  # open in BMG's software
    device = D2XXDevice("471-0376", library=library)
    with self.assertRaises(D2XXError) as raised:
      device.open()
    self.assertEqual((raised.exception.function, raised.exception.status), ("FT_OpenEx", 3))
    self.assertEqual(str(raised.exception), "FT_OpenEx returned FT_DEVICE_NOT_OPENED (3)")
    self.assertTrue(device.closed)

  def test_a_setting_that_fails_on_open_closes_the_device_again(self):
    library = FakeFTD2XX()
    library.failures["FT_SetTimeouts"] = FTStatus.FT_IO_ERROR
    device = D2XXDevice("471-0376", library=library)
    with self.assertRaisesRegex(D2XXError, "FT_SetTimeouts"):
      device.open()
    self.assertTrue(device.closed)
    self.assertEqual(library.calls[-1], ("FT_Close",))
    self.assertIsNone(library.opened)

  def test_read_takes_what_the_driver_holds_up_to_the_length(self):
    device, library = self._open()
    library.received += b"ABCDE"
    with mock.patch.object(ftd2xx.time, "sleep") as sleep:
      self.assertEqual(device.read(3), b"ABC")
      self.assertEqual(device.read(25), b"DE")
    sleep.assert_not_called()
    self.assertEqual(
      [call for call in library.calls if call[0] == "FT_Read"], [("FT_Read", 3), ("FT_Read", 2)]
    )

  def test_an_empty_read_waits_one_latency_period_and_returns_nothing(self):
    device, library = self._open()
    device.ftdi_fn.ftdi_set_latency_timer(2)
    with mock.patch.object(ftd2xx.time, "sleep") as sleep:
      self.assertEqual(device.read(25), b"")
    sleep.assert_called_once_with(0.002)
    self.assertNotIn("FT_Read", [call[0] for call in library.calls])

  def test_bytes_arriving_within_the_latency_period_are_read(self):
    device, library = self._open()
    library.arriving = [b"", b"\x02\x00"]
    with mock.patch.object(ftd2xx.time, "sleep") as sleep:
      self.assertEqual(device.read(25), b"\x02\x00")
    sleep.assert_called_once_with(0.016)  # the latency timer open sets

  def test_write(self):
    device, library = self._open()
    self.assertEqual(device.write(b"\x02\x00\x09"), 3)
    self.assertEqual(device.write(b""), 0)
    self.assertEqual(bytes(library.written), b"\x02\x00\x09")
    self.assertEqual(library.calls, [("FT_Write", b"\x02\x00\x09")])

  def test_each_libftdi_function_runs_its_d2xx_counterpart(self):
    device, library = self._open()
    fn = device.ftdi_fn
    cases = [
      (lambda: fn.ftdi_setrts(1), ("FT_SetRts",)),
      (lambda: fn.ftdi_setrts(0), ("FT_ClrRts",)),
      (lambda: fn.ftdi_setdtr(True), ("FT_SetDtr",)),
      (lambda: fn.ftdi_setdtr(False), ("FT_ClrDtr",)),
      (lambda: fn.ftdi_usb_reset(), ("FT_ResetDevice",)),
      (lambda: fn.ftdi_set_latency_timer(2), ("FT_SetLatencyTimer", 2)),
      # libftdi BITS_8, STOP_BIT_1, NONE -> FT_BITS_8, FT_STOP_BITS_1, FT_PARITY_NONE
      (lambda: fn.ftdi_set_line_property(8, 0, 0), ("FT_SetDataCharacteristics", 8, 0, 0)),
      # BITS_7, STOP_BIT_2, EVEN -> FT_BITS_7, FT_STOP_BITS_2, FT_PARITY_EVEN
      (lambda: fn.ftdi_set_line_property(7, 2, 2), ("FT_SetDataCharacteristics", 7, 2, 2)),
      # ODD, MARK, SPACE -> FT_PARITY_ODD, FT_PARITY_MARK, FT_PARITY_SPACE
      (lambda: fn.ftdi_set_line_property(8, 0, 1), ("FT_SetDataCharacteristics", 8, 0, 1)),
      (lambda: fn.ftdi_set_line_property(8, 0, 3), ("FT_SetDataCharacteristics", 8, 0, 3)),
      (lambda: fn.ftdi_set_line_property(8, 0, 4), ("FT_SetDataCharacteristics", 8, 0, 4)),
      # SIO_* -> FT_FLOW_*, with libftdi's XON/XOFF characters, 0
      (lambda: fn.ftdi_setflowctrl(0x0), ("FT_SetFlowControl", 0x0000, 0, 0)),
      (lambda: fn.ftdi_setflowctrl(0x100), ("FT_SetFlowControl", 0x0100, 0, 0)),
      (lambda: fn.ftdi_setflowctrl(0x200), ("FT_SetFlowControl", 0x0200, 0, 0)),
      (lambda: fn.ftdi_setflowctrl(0x400), ("FT_SetFlowControl", 0x0400, 0, 0)),
      (lambda: fn.ftdi_usb_purge_rx_buffer(), ("FT_Purge", 1)),  # FT_PURGE_RX
      (lambda: fn.ftdi_usb_purge_tx_buffer(), ("FT_Purge", 2)),  # FT_PURGE_TX
    ]
    for call, expected in cases:
      with self.subTest(expected=expected):
        library.calls.clear()
        self.assertEqual(call(), 0)
        self.assertEqual(library.calls, [expected])

    library.calls.clear()
    device.baudrate = 125000
    self.assertEqual(library.calls, [("FT_SetBaudRate", 125000)])
    self.assertEqual(device.baudrate, 125000)

  def test_a_purge_drops_what_the_driver_holds(self):
    device, library = self._open()
    library.received += b"stale"
    device.ftdi_fn.ftdi_usb_purge_rx_buffer()
    with mock.patch.object(ftd2xx.time, "sleep"):
      self.assertEqual(device.read(25), b"")

  def test_values_d2xx_has_no_counterpart_for_are_refused_before_any_call(self):
    device, library = self._open()
    with self.assertRaisesRegex(ValueError, "STOP_BIT_15"):
      device.ftdi_fn.ftdi_set_line_property(8, 1, 0)  # 1.5 stop bits
    with self.assertRaises(ValueError):
      device.ftdi_fn.ftdi_set_line_property(6, 0, 0)
    with self.assertRaises(ValueError):
      device.ftdi_fn.ftdi_set_line_property(8, 0, 5)
    with self.assertRaises(ValueError):
      device.ftdi_fn.ftdi_setflowctrl(0x300)
    self.assertEqual(library.calls, [])

  def test_modem_status_fills_libftdi_unsigned_short(self):
    device, library = self._open()
    # Line status (high byte): transmitter empty (0x60) and overrun (0x02); modem status (low
    # byte): CTS (0x10) and DSR (0x20).
    library.modem_status = 0x6230
    through_byref = ctypes.c_ushort(0)
    self.assertEqual(device.ftdi_fn.ftdi_poll_modem_status(ctypes.byref(through_byref)), 0)
    self.assertEqual(through_byref.value, 0x6230)
    through_pointer = ctypes.c_ushort(0)
    device.ftdi_fn.ftdi_poll_modem_status(ctypes.pointer(through_pointer))
    self.assertEqual(through_pointer.value, 0x6230)

  def test_a_status_other_than_ft_ok_raises_naming_the_function_and_the_status(self):
    device, library = self._open()
    library.failures["FT_SetBaudRate"] = FTStatus.FT_INVALID_BAUD_RATE
    with self.assertRaises(D2XXError) as raised:
      device.baudrate = 3
    self.assertEqual(str(raised.exception), "FT_SetBaudRate returned FT_INVALID_BAUD_RATE (7)")
    self.assertEqual(device.baudrate, 9600)

    library.failures["FT_GetQueueStatus"] = FTStatus.FT_IO_ERROR  # e.g. the cable was pulled
    with self.assertRaisesRegex(D2XXError, r"FT_GetQueueStatus returned FT_IO_ERROR \(4\)"):
      device.read(25)
    self.assertEqual(str(D2XXError("FT_Read", 42)), "FT_Read returned unknown status (42)")

  def test_close_gives_up_the_handle_once(self):
    device, library = self._open()
    device.close()
    device.close()
    self.assertTrue(device.closed)
    self.assertEqual(library.calls, [("FT_Close",)])
    with self.assertRaises(D2XXError) as raised:
      device.read(1)
    self.assertEqual(raised.exception.status, FTStatus.FT_INVALID_HANDLE)
    self.assertIn("not open", str(raised.exception))
    with self.assertRaises(D2XXError):
      device.write(b"")
    self.assertEqual(library.calls, [("FT_Close",)])


if __name__ == "__main__":
  unittest.main()
