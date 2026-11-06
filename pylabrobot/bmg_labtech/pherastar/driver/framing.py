"""PHERAstar wire framing: BMG's (`pylabrobot.bmg_labtech.protocol.framing`) with header byte 0x09.

FTDI only: VID 0x0403 / PID 0xBB68, the same as the CLARIOstar's FTDI bridge, at 125000 baud, 8N1,
latency 2 ms, as BMG's own software opens it (PHERAstar FX, firmware 1.32, 2026-11-06).
"""

# The PHERAstar's header byte, in every frame it sends and receives.
FTDI_HEADER = 0x09
