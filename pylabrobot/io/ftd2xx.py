"""FTDI's own D2XX driver through ctypes, for `FTDI(backend="d2xx")`.

On Windows, an FTDI chip on FTDI's driver (the FTDIBUS service) is reached through that driver's
`ftd2xx.dll`, so the device's driver does not have to be replaced with WinUSB or libusbK (Zadig),
and vendor software on the same driver keeps working. One process at a time has a device open:
close the vendor software before opening it here. On Linux and macOS, FTDI's `libftd2xx` works the
same way, once the kernel's `ftdi_sio` (Linux) or Apple's FTDI driver (macOS) has let go of the
chip.

`D2XXDevice` offers the part of pylibftdi's `Device` that `FTDI` uses; its `ftdi_fn` runs the
libftdi functions `FTDI` calls with their D2XX counterparts. `list_devices` lists what D2XX sees.
Every D2XX status other than FT_OK raises `D2XXError`.
"""

import ctypes
import enum
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class FTStatus(enum.IntEnum):
  """FT_STATUS, what every D2XX function returns (ftd2xx.h)."""

  FT_OK = 0
  FT_INVALID_HANDLE = 1
  FT_DEVICE_NOT_FOUND = 2
  FT_DEVICE_NOT_OPENED = 3
  FT_IO_ERROR = 4
  FT_INSUFFICIENT_RESOURCES = 5
  FT_INVALID_PARAMETER = 6
  FT_INVALID_BAUD_RATE = 7
  FT_DEVICE_NOT_OPENED_FOR_ERASE = 8
  FT_DEVICE_NOT_OPENED_FOR_WRITE = 9
  FT_FAILED_TO_WRITE_DEVICE = 10
  FT_EEPROM_READ_FAILED = 11
  FT_EEPROM_WRITE_FAILED = 12
  FT_EEPROM_ERASE_FAILED = 13
  FT_EEPROM_NOT_PRESENT = 14
  FT_EEPROM_NOT_PROGRAMMED = 15
  FT_INVALID_ARGS = 16
  FT_NOT_SUPPORTED = 17
  FT_OTHER_ERROR = 18
  FT_DEVICE_LIST_NOT_READY = 19


FT_OPEN_BY_SERIAL_NUMBER = 1
FT_FLAGS_OPENED = 1  # bit 0 of a device list entry's flags: the device is open
FT_PURGE_RX = 1
FT_PURGE_TX = 2

# D2XX's line properties and flow control (ftd2xx.h).
FT_BITS_7, FT_BITS_8 = 7, 8
FT_STOP_BITS_1, FT_STOP_BITS_2 = 0, 2
FT_PARITY_NONE, FT_PARITY_ODD, FT_PARITY_EVEN, FT_PARITY_MARK, FT_PARITY_SPACE = 0, 1, 2, 3, 4
FT_FLOW_NONE, FT_FLOW_RTS_CTS, FT_FLOW_DTR_DSR, FT_FLOW_XON_XOFF = 0x0000, 0x0100, 0x0200, 0x0400

# libftdi's values, as `FTDI` passes them (ftdi.h's enums and SIO_* flow control), to D2XX's. They
# agree wherever both have one; D2XX has no 1.5 stop bits (libftdi's STOP_BIT_15, 1).
_WORD_LENGTHS = {7: FT_BITS_7, 8: FT_BITS_8}  # BITS_7, BITS_8
_STOP_BITS = {0: FT_STOP_BITS_1, 2: FT_STOP_BITS_2}  # STOP_BIT_1, STOP_BIT_2
_PARITIES = {  # NONE, ODD, EVEN, MARK, SPACE
  0: FT_PARITY_NONE,
  1: FT_PARITY_ODD,
  2: FT_PARITY_EVEN,
  3: FT_PARITY_MARK,
  4: FT_PARITY_SPACE,
}
_FLOW_CONTROLS = {  # SIO_DISABLE_FLOW_CTRL, SIO_RTS_CTS_HS, SIO_DTR_DSR_HS, SIO_XON_XOFF_HS
  0x0000: FT_FLOW_NONE,
  0x0100: FT_FLOW_RTS_CTS,
  0x0200: FT_FLOW_DTR_DSR,
  0x0400: FT_FLOW_XON_XOFF,
}

_TIMEOUT_MS = 5000  # FT_Read and FT_Write timeouts: libftdi's usb_read_timeout and write timeout
_OPEN_BAUDRATE = 9600  # what pylibftdi's open sets
_OPEN_LATENCY_MS = 16  # likewise: the chip's default

_DWORD = ctypes.c_uint32  # DWORD and ULONG: 32 bits on Windows and in libftd2xx's WinTypes.h
_P_DWORD = ctypes.POINTER(_DWORD)
_HANDLE = ctypes.c_void_p
_SIGNATURES: Dict[str, List[Any]] = {
  "FT_CreateDeviceInfoList": [_P_DWORD],
  "FT_GetDeviceInfoDetail": [
    _DWORD,
    _P_DWORD,
    _P_DWORD,
    _P_DWORD,
    _P_DWORD,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.POINTER(_HANDLE),
  ],
  "FT_OpenEx": [ctypes.c_void_p, _DWORD, ctypes.POINTER(_HANDLE)],
  "FT_Close": [_HANDLE],
  "FT_Read": [_HANDLE, ctypes.c_void_p, _DWORD, _P_DWORD],
  "FT_Write": [_HANDLE, ctypes.c_void_p, _DWORD, _P_DWORD],
  "FT_GetQueueStatus": [_HANDLE, _P_DWORD],
  "FT_SetTimeouts": [_HANDLE, _DWORD, _DWORD],
  "FT_SetBaudRate": [_HANDLE, _DWORD],
  "FT_SetDataCharacteristics": [_HANDLE, ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_ubyte],
  "FT_SetFlowControl": [_HANDLE, ctypes.c_ushort, ctypes.c_ubyte, ctypes.c_ubyte],
  "FT_SetLatencyTimer": [_HANDLE, ctypes.c_ubyte],
  "FT_SetRts": [_HANDLE],
  "FT_ClrRts": [_HANDLE],
  "FT_SetDtr": [_HANDLE],
  "FT_ClrDtr": [_HANDLE],
  "FT_ResetDevice": [_HANDLE],
  "FT_Purge": [_HANDLE, _DWORD],
  "FT_GetModemStatus": [_HANDLE, _P_DWORD],
}
# Only in libftd2xx (Linux, macOS): Windows learns custom VID/PIDs from the driver's INF file.
_OPTIONAL_SIGNATURES: Dict[str, List[Any]] = {"FT_SetVIDPID": [_DWORD, _DWORD]}

if sys.platform == "win32":
  _LIBRARY_NAMES = ("ftd2xx.dll", "ftd2xx64.dll")
elif sys.platform == "darwin":
  _LIBRARY_NAMES = ("libftd2xx.dylib",)
else:
  _LIBRARY_NAMES = ("libftd2xx.so",)

_library: Optional[Any] = None  # the library `load_library` loaded by its default names


class D2XXError(Exception):
  """A D2XX function returned a status other than FT_OK, or was called on a closed device."""

  def __init__(self, function: str, status: int, detail: str = ""):
    super().__init__(function, status, detail)
    self.function = function
    self.status = status
    self.detail = detail

  def __str__(self) -> str:
    try:
      name = FTStatus(self.status).name
    except ValueError:
      name = "unknown status"
    message = f"{self.function} returned {name} ({self.status})"
    return f"{message}: {self.detail}" if self.detail else message


def _check(function: str, status: int) -> None:
  """Raise `D2XXError` unless `status` is FT_OK."""
  if status != FTStatus.FT_OK:
    raise D2XXError(function, status)


def _declare_functions(library: Any) -> None:
  """Give the D2XX functions their C signatures; the optional ones only where the library has
  them."""
  for name, argtypes in _SIGNATURES.items():
    function = getattr(library, name)
    function.argtypes, function.restype = argtypes, _DWORD
  for name, argtypes in _OPTIONAL_SIGNATURES.items():
    function = getattr(library, name, None)
    if function is not None:
      function.argtypes, function.restype = argtypes, _DWORD


def _open_library(name: str) -> Any:
  if sys.platform == "win32":
    return ctypes.WinDLL(name)  # the D2XX functions are WINAPI (stdcall) on 32-bit Windows
  return ctypes.CDLL(name)


def load_library(path: Optional[str] = None) -> Any:
  """FTDI's D2XX library, with its functions' signatures declared.

  Args:
    path: the library file. None loads `ftd2xx.dll` on Windows, `libftd2xx.dylib` on macOS and
      `libftd2xx.so` elsewhere, once, by the system's search rules.

  Raises:
    RuntimeError: if the library cannot be loaded.
  """
  global _library
  if path is None and _library is not None:
    return _library
  errors = []
  for name in (path,) if path is not None else _LIBRARY_NAMES:
    try:
      library = _open_library(name)
    except OSError as e:
      errors.append(f"{name}: {e}")
      continue
    _declare_functions(library)
    if path is None:
      _library = library
    return library
  raise RuntimeError(
    "FTDI's D2XX library could not be loaded. Install FTDI's D2XX driver "
    "(https://ftdichip.com/drivers/d2xx-drivers/); vendor software that talks to the device "
    "usually has. " + "; ".join(errors)
  )


@dataclass(frozen=True)
class D2XXDeviceInfo:
  """One entry of D2XX's device list.

  D2XX lists only devices on FTDI's driver. A device open in another process is listed as `opened`
  with every other field empty.
  """

  index: int
  flags: int
  type: int  # FT_DEVICE: the chip, e.g. 8 for an FT232H
  vid: int
  pid: int
  location_id: int
  serial_number: str
  description: str

  @property
  def opened(self) -> bool:
    return bool(self.flags & FT_FLAGS_OPENED)


def list_devices(library: Optional[Any] = None) -> List[D2XXDeviceInfo]:
  """The devices D2XX sees, from FT_CreateDeviceInfoList and FT_GetDeviceInfoDetail.

  Args:
    library: the D2XX library; None uses `load_library()`.
  """
  library = library if library is not None else load_library()
  count = _DWORD(0)
  _check("FT_CreateDeviceInfoList", library.FT_CreateDeviceInfoList(ctypes.byref(count)))
  devices = []
  for index in range(count.value):
    flags, device_type, device_id, location_id = _DWORD(0), _DWORD(0), _DWORD(0), _DWORD(0)
    serial_number = ctypes.create_string_buffer(16)
    description = ctypes.create_string_buffer(64)
    handle = _HANDLE()
    _check(
      "FT_GetDeviceInfoDetail",
      library.FT_GetDeviceInfoDetail(
        index,
        ctypes.byref(flags),
        ctypes.byref(device_type),
        ctypes.byref(device_id),
        ctypes.byref(location_id),
        serial_number,
        description,
        ctypes.byref(handle),
      ),
    )
    devices.append(
      D2XXDeviceInfo(
        index=index,
        flags=flags.value,
        type=device_type.value,
        vid=(device_id.value >> 16) & 0xFFFF,  # the ID holds the VID in its high word
        pid=device_id.value & 0xFFFF,
        location_id=location_id.value,
        serial_number=serial_number.value.decode("latin-1"),
        description=description.value.decode("latin-1"),
      )
    )
  return devices


def register_vid_pid(vid: int, pid: int, library: Optional[Any] = None) -> None:
  """Let D2XX list and open devices with this VID/PID (FT_SetVIDPID).

  libftd2xx (Linux, macOS) only knows FTDI's own PIDs until told another, such as a vendor's
  custom PID; on Windows the driver's INF file names them and ftd2xx.dll has no FT_SetVIDPID, so
  this does nothing there.

  Args:
    library: the D2XX library; None uses `load_library()`.
  """
  library = library if library is not None else load_library()
  set_vid_pid = getattr(library, "FT_SetVIDPID", None)
  if set_vid_pid is not None:
    _check("FT_SetVIDPID", set_vid_pid(vid, pid))


class D2XXFunctions:
  """The libftdi functions `FTDI` calls through pylibftdi's `Device.ftdi_fn`, run with their D2XX
  counterparts on a `D2XXDevice`.

  Each returns 0, libftdi's success, and raises `D2XXError` where D2XX fails, and `ValueError` for
  a value D2XX has no counterpart for.
  """

  def __init__(self, device: "D2XXDevice"):
    self._device = device

  def ftdi_setrts(self, level: Any) -> int:
    self._device.call("FT_SetRts" if level else "FT_ClrRts")
    return 0

  def ftdi_setdtr(self, level: Any) -> int:
    self._device.call("FT_SetDtr" if level else "FT_ClrDtr")
    return 0

  def ftdi_usb_reset(self) -> int:
    """Reset the chip's serial engine (FT_ResetDevice), as libftdi's SIO reset does."""
    self._device.call("FT_ResetDevice")
    return 0

  def ftdi_set_latency_timer(self, latency: int) -> int:
    self._device.call("FT_SetLatencyTimer", latency)
    self._device.latency_ms = latency
    return 0

  def ftdi_set_line_property(self, bits: int, stopbits: int, parity: int) -> int:
    """FT_SetDataCharacteristics, with libftdi's `ftdi_bits_type`, `ftdi_stopbits_type` and
    `ftdi_parity_type` values.

    Raises:
      ValueError: for a value D2XX has none for, such as libftdi's STOP_BIT_15 (1.5 stop bits).
    """
    if bits not in _WORD_LENGTHS:
      raise ValueError(f"D2XX takes 7 or 8 data bits, got {bits}")
    if stopbits not in _STOP_BITS:
      raise ValueError(
        f"D2XX takes libftdi's STOP_BIT_1 (0) or STOP_BIT_2 (2), got {stopbits}"
        + (" (STOP_BIT_15, 1.5 stop bits)" if stopbits == 1 else "")
      )
    if parity not in _PARITIES:
      raise ValueError(f"D2XX takes libftdi's parity 0 to 4 (NONE to SPACE), got {parity}")
    self._device.call(
      "FT_SetDataCharacteristics", _WORD_LENGTHS[bits], _STOP_BITS[stopbits], _PARITIES[parity]
    )
    return 0

  def ftdi_setflowctrl(self, flowctrl: int) -> int:
    """FT_SetFlowControl, with libftdi's SIO_* values. XON/XOFF characters are 0, as libftdi's
    `ftdi_setflowctrl` sends them.

    Raises:
      ValueError: for a value that is not one of libftdi's flow controls.
    """
    if flowctrl not in _FLOW_CONTROLS:
      raise ValueError(
        "D2XX takes libftdi's SIO_DISABLE_FLOW_CTRL (0x0), SIO_RTS_CTS_HS (0x100), "
        f"SIO_DTR_DSR_HS (0x200) or SIO_XON_XOFF_HS (0x400), got {flowctrl:#x}"
      )
    self._device.call("FT_SetFlowControl", _FLOW_CONTROLS[flowctrl], 0, 0)
    return 0

  def ftdi_usb_purge_rx_buffer(self) -> int:
    self._device.call("FT_Purge", FT_PURGE_RX)
    return 0

  def ftdi_usb_purge_tx_buffer(self) -> int:
    self._device.call("FT_Purge", FT_PURGE_TX)
    return 0

  def ftdi_poll_modem_status(self, status: Any) -> int:
    """FT_GetModemStatus, into libftdi's `unsigned short *status`.

    Args:
      status: a reference to a `ctypes.c_ushort` (`ctypes.byref` or `ctypes.pointer`). It receives
        the modem status in its low byte and the line status in its high byte, as libftdi lays them
        out; libftd2xx (Linux, macOS) reports no line status.
    """
    value = _DWORD(0)
    self._device.call("FT_GetModemStatus", ctypes.byref(value))
    ctypes.cast(status, ctypes.POINTER(ctypes.c_ushort)).contents.value = value.value & 0xFFFF
    return 0


class D2XXDevice:
  """An FTDI device opened through D2XX by its serial number, with the part of pylibftdi's `Device`
  that `FTDI` uses: `open`, `close`, `closed`, `read`, `write`, `baudrate` and `ftdi_fn`.

  Not thread-safe: `FTDI` makes every call from one worker thread.
  """

  def __init__(self, serial_number: str, library: Optional[Any] = None):
    """
    Args:
      serial_number: the device's serial number, as D2XX lists it.
      library: the D2XX library; None uses `load_library()` when the device opens.
    """
    self.serial_number = serial_number
    self._library = library
    self._handle: Optional[ctypes.c_void_p] = None
    self._baudrate = _OPEN_BAUDRATE
    # How long an empty read waits for bytes: the chip sends what it has at least this often.
    self.latency_ms = _OPEN_LATENCY_MS
    self.ftdi_fn = D2XXFunctions(self)

  @property
  def library(self) -> Any:
    if self._library is None:
      self._library = load_library()
    return self._library

  def _require_handle(self, function: str) -> ctypes.c_void_p:
    """The open device's handle, for calling `function`.

    Raises:
      D2XXError: if the device is not open.
    """
    if self._handle is None:
      raise D2XXError(function, FTStatus.FT_INVALID_HANDLE, "the device is not open")
    return self._handle

  def call(self, function: str, *args: Any) -> None:
    """Call a D2XX function on this device's handle, with `args` after it.

    Raises:
      D2XXError: if the device is not open, or the function does not return FT_OK.
    """
    handle = self._require_handle(function)
    _check(function, getattr(self.library, function)(handle, *args))

  def open(self) -> None:
    """Open the device by its serial number (FT_OpenEx), then set libftdi's timeouts and what
    pylibftdi's open sets: no flow control, 9600 baud, latency timer 16 ms.

    Raises:
      D2XXError: if it does not open (not connected, or open in another process), or a setting
        fails; the device is closed again.
    """
    if self._handle is not None:
      return
    handle = _HANDLE()
    _check(
      "FT_OpenEx",
      self.library.FT_OpenEx(
        self.serial_number.encode("latin-1"), FT_OPEN_BY_SERIAL_NUMBER, ctypes.byref(handle)
      ),
    )
    self._handle = handle
    try:
      self.call("FT_SetTimeouts", _TIMEOUT_MS, _TIMEOUT_MS)
      self.ftdi_fn.ftdi_setflowctrl(0)
      self.baudrate = _OPEN_BAUDRATE
      self.ftdi_fn.ftdi_set_latency_timer(_OPEN_LATENCY_MS)
    except BaseException:
      self.close()
      raise

  @property
  def closed(self) -> bool:
    return self._handle is None

  def close(self) -> None:
    """Close the device (FT_Close); nothing if it is not open. The handle is given up even if
    FT_Close fails."""
    handle, self._handle = self._handle, None
    if handle is not None:
      _check("FT_Close", self.library.FT_Close(handle))

  @property
  def baudrate(self) -> int:
    return self._baudrate

  @baudrate.setter
  def baudrate(self, value: int) -> None:
    self.call("FT_SetBaudRate", value)
    self._baudrate = value

  def _queue_status(self) -> int:
    waiting = _DWORD(0)
    self.call("FT_GetQueueStatus", ctypes.byref(waiting))
    return waiting.value

  def read(self, length: int) -> bytes:
    """Up to `length` bytes the driver has received, without waiting for `length` of them.

    If none has arrived, it waits one latency period, when the chip sends what it has, and returns
    what came then, possibly b"": as libftdi's read returns on the chip's first packet without data.
    """
    waiting = self._queue_status()
    if waiting == 0:
      time.sleep(self.latency_ms / 1000)
      waiting = self._queue_status()
    count = min(waiting, length)
    if count <= 0:
      return b""
    buffer = ctypes.create_string_buffer(count)
    received = _DWORD(0)
    self.call("FT_Read", buffer, count, ctypes.byref(received))  # returns at once: count is queued
    return buffer.raw[: received.value]

  def write(self, data: bytes) -> int:
    """Write `data` (FT_Write); return how many bytes were written, fewer on a write timeout."""
    data = bytes(data)
    if not data:
      self._require_handle("FT_Write")
      return 0
    written = _DWORD(0)
    self.call("FT_Write", data, len(data), ctypes.byref(written))
    return written.value
