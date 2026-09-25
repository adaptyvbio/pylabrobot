"""The 384-head: the block of 384 dispensing channels that works a whole plate at once."""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

from pylabrobot.hamilton.star.driver.features.head import Head, HeadConfiguration
from pylabrobot.resources.coordinate import Coordinate
from pylabrobot.resources.hamilton.tip_creators import HamiltonTip
from pylabrobot.resources.resource import Resource
from pylabrobot.resources.tip import Tip
from pylabrobot.resources.tip_rack import TipRack

if TYPE_CHECKING:
  from pylabrobot.hamilton.star.driver.master import STARDriver

logger = logging.getLogger(__name__)


@dataclass
class Head384Configuration(HeadConfiguration):
  """Device facts for the installed 384-head.

  What this head adds to `HeadConfiguration` is what it reports about itself beyond the shared
  flags, and the head type that resolves what a dispensing or squeezer increment is worth: the
  three heads share a piston travel but not a bore, and are geared differently.

  Its drive windows do not move with firmware, so they are plain values rather than properties -
  what varies here is which head is fitted, not the generation.
  """

  module: str = "D0"
  """What the drive documents; no 384-head has been probed."""
  retract_command: str = "JV"
  initialize_command: str = "JI"
  tip_presence_command: str = "QK"
  position_command: str = "QJ"
  defined_position_command: str = "EN"
  y_parameter: str = "yk"
  z_parameter: str = "je"
  traverse_z_parameter: str = "zf"
  z_end_parameter: str = "zg"
  x_offset_parameter: str = "kd"
  head_types: Dict[int, str] = field(
    default_factory=lambda: {
      0: "Low volume head",
      1: "High volume head",
      2: "STP head",  # shifted tip pickup
    }
  )
  drive_parameters: Dict[str, int] = field(
    default_factory=lambda: {"yv": 5, "yr": 3, "zv": 5, "zr": 3}
  )
  # The generation the drive windows below were taken from. A head older than this documents
  # different ones, and nothing here resolves them per generation.
  first_documented_firmware_year: int = 2009

  supports_lld_absolute_threshold_check: Optional[bool] = None

  channel_pitch: float = 4.5
  channel_columns: int = 24
  channel_rows: int = 16

  dispensing_drive_mm_per_increment: float = 0.00063333

  # This head's Y and Z drives count acceleration in thousands of increments per second squared,
  # unlike the positions and speeds they count in single ones, so these are 1000x the position
  # resolutions.
  y_drive_acceleration_mm_per_increment: float = 15.625
  z_drive_acceleration_mm_per_increment: float = 5.0

  y_range_increments: Tuple[int, int] = (7100, 36100)  # type: ignore[assignment]
  y_speed_range_increments: Tuple[int, int] = (50, 20000)  # type: ignore[assignment]
  y_acceleration_range_increments: Tuple[int, int] = (5, 32)  # type: ignore[assignment]
  z_range_increments: Tuple[int, int] = (33200, 67200)  # type: ignore[assignment]
  z_acceleration_range_increments: Tuple[int, int] = (5, 100)

  # What this head's drives start from. Its accelerations are counted in thousands, so those two
  # are written small where the 96-head's are not.
  y_speed_default_increments: int = 20000
  y_acceleration_default_increments: int = 32
  z_acceleration_default_increments: int = 80

  predefined_y_position_origin: int = 22000
  predefined_z_position_origin: int = 35000

  y_drive_current_limit_default: int = 4
  z_drive_current_limit_default: int = 7
  current_limit_range: Tuple[int, int] = (0, 7)

  # The top of what this head's master commands accept for a height, which is lower than what its
  # Z drive reaches: the field is four digits in 0.1 mm and the commands document 3270 as its
  # maximum, where the drive itself goes to 336.0 mm.
  defined_position_minimum_height_default: float = 327.0

  tip_command_y_range: Tuple[float, float] = (110.0, 564.0)  # type: ignore[assignment]

  def _require_head_type(self) -> str:
    """The head type, for the facts only it decides.

    Returns:
      Which head is fitted.

    Raises:
      RuntimeError: If it has not been read, or is one this driver does not know.
    """
    if self.head_type is None or self.head_type == "unknown":
      raise RuntimeError(
        "the 384-head's type is not known, and it is what decides how much a dispensing or "
        "squeezer increment is worth; have you called `star.setup()`?"
      )
    return self.head_type

  @property
  def dispensing_drive_uL_per_increment(self) -> float:
    """What one increment of the dispensing drive holds, in uL.

    The three heads share a piston travel but not a bore, so this is the head type's to decide and
    is not known until the head has said which it is - guessing would mis-volume every aspirate.

    Returns:
      The volume one increment holds, in uL.

    Raises:
      RuntimeError: If the head type has not been read.
    """
    head_type = self._require_head_type()
    if head_type == "Low volume head":
      return 0.000974941
    if head_type == "High volume head":
      return 0.00143754
    return 0.00186531

  @property
  def squeezer_drive_mm_per_increment(self) -> float:
    """How far one increment of the squeezer drive travels, in mm.

    Geared differently on the low volume head, so this is the head type's to decide as the
    dispensing volume above is.

    Returns:
      The distance one increment travels, in mm.

    Raises:
      RuntimeError: If the head type has not been read.
    """
    return 0.00091813 if self._require_head_type() == "Low volume head" else 0.00035866

  # -- windows the dispensing and squeezer drives work in ----------------------------------------

  @property
  def dispensing_drive_range(self) -> Tuple[float, float]:
    """Aspirate/dispense piston volume window (uL); applies to both aspirate and dispense."""
    return (0.0, self.dispensing_drive_increments_to_uL(60950))

  @property
  def dispensing_drive_speed_range(self) -> Tuple[float, float]:
    """Dispensing-drive speed window (uL/s)."""
    # The drive counts its speed in tens of increments per second, so both ends are scaled.
    return (
      self.dispensing_drive_increments_to_uL(5 * 10),
      self.dispensing_drive_increments_to_uL(25000 * 10),
    )

  @property
  def dispensing_drive_speed_default(self) -> float:
    """Dispensing-drive default speed (uL/s)."""
    return self.dispensing_drive_increments_to_uL(50000)

  @property
  def dispensing_drive_acceleration_default(self) -> float:
    """Dispensing-drive default acceleration (uL/s2)."""
    return self.dispensing_drive_increments_to_uL(9000000)

  @property
  def squeezer_drive_speed_default(self) -> float:
    """Squeezer-drive default speed (mm/s); the low volume head runs slower."""
    increments = 16000 if self._require_head_type() == "Low volume head" else 40000
    return self.squeezer_drive_increments_to_mm(increments)

  @property
  def squeezer_drive_acceleration_default(self) -> float:
    """Squeezer-drive default acceleration (mm/s2); the low volume head runs gentler."""
    increments = 100000 if self._require_head_type() == "Low volume head" else 250000
    return self.squeezer_drive_increments_to_mm(increments)


class Head384(Head):
  """The 384-head.

  Reached as `driver.head384`, on a device that has one. It is addressed as `D0`, but the
  commands that move it go to the master, so this feature speaks to both.
  """

  configuration: Head384Configuration

  def __init__(self, driver: "STARDriver", configuration: Optional[Head384Configuration] = None):
    """
    Args:
      driver: the driver to send commands through.
      configuration: the head's device facts. Defaults to `Head384Configuration()`.
    """
    super().__init__(driver, configuration or Head384Configuration())

  # ----------------------------------------
  # Setup
  # ----------------------------------------

  # -- discovery ---------------------------------------------------------------------------------

  def _record_hardware(self, hardware: List[str]) -> None:
    """Record whether this head runs the absolute-threshold cLLD check.

    Index 1 was reserve until 2015, so a head older than that reads back 0 there whether or not it
    would do the check.

    Args:
      hardware: the tokens `request_hardware` read.
    """
    self.configuration.supports_lld_absolute_threshold_check = bool(int(hardware[1]))

  # -- initialization ----------------------------------------------------------------------------

  # TODO: confirm that `xd` on this head's `JI` is the sign of the position, which is what
  # `Head.initialize` derives it as. The one capture of the command carries `xd1` at `xs01613`,
  # which reads as -161.3 mm.
  #
  # A negative X at channel A1 is ordinary on this head rather than a sign that the reading is
  # wrong: A1 sits `configuration.x_offset` left of the carriage reference, measured at 368.5 mm on
  # the device this was read from, so the arm at 800 mm puts A1 at 431.5 mm and A1 at 31.5 mm has
  # the arm at 400 mm. Every arm position left of the offset puts A1 at a negative deck X, and
  # -161.3 mm is within the arm's travel.
  #
  # What does not fit is `configuration.min_x_clear_of_left_side_panel`, which the device report
  # gives as -100.0 mm: with a left side panel fitted, `XArm.narrow_travel_for_left_side_panel`
  # holds A1 to the right of that, and -161.3 mm is past it. Either the capture was taken with no
  # panel on, or `xd` is a direction flag of the command's own rather than the sign of a deck
  # coordinate, and deriving it here is wrong.
  #
  # Eject at a position right of the deck origin and read `C0 QJ` back to tell the two apart: a
  # sign answers `xd0` there, a flag of its own does not follow the position.

  # ----------------------------------------
  # Movement
  # ----------------------------------------

  # -- dispensing drive --------------------------------------------------------------------------

  # TODO: establish what this head's dispensing drive reports before reading it. A real 384-head
  # answers `D0 RD` with two signed five-digit values - "+06512 +06512" on the one head this was
  # read from - and nothing says what they are: two drives, one value written twice, or a position
  # and something else. The 96-head's `H0 RD` is one position, and the master's command table does
  # not document this one at all. The sample was a single idle reading with nothing to compare it
  # against; read it again at a known, different piston position, and whether the two values move
  # together settles it.

  # -- y position --------------------------------------------------------------------------------

  async def _unchecked_fw_safety_move_to_y_position(
    self,
    y: float,
    minimum_height_at_beginning_of_a_command: Optional[float] = None,
  ):
    """Move the head along Y through the master, unguarded and unrecorded. `C0 EY`.

    The sibling of `_unchecked_fw_move_to_coordinate` for one axis: the master raises the head to
    the height given before it travels, where `move_to_y_position` drives the head's own Y drive
    and leaves the height to the caller. The 96-head has no command of this shape, so this is the
    384-head's own.

    Args:
      y: where to put channel A1, in mm.
      minimum_height_at_beginning_of_a_command: the height the head is at before it travels, in
        mm. Defaults to `configuration.defined_position_minimum_height_default`.
    """
    c = self.configuration
    if minimum_height_at_beginning_of_a_command is None:
      minimum_height_at_beginning_of_a_command = c.defined_position_minimum_height_default
    parameters: Dict[str, Any] = {
      c.y_parameter: f"{round(y * 10):04}",
      c.traverse_z_parameter: f"{round(minimum_height_at_beginning_of_a_command * 10):04}",
    }
    return await self._driver.send_command(
      module="C0",
      command="EY",
      **parameters,
    )

  # ----------------------------------------
  # Tip pickup and drop
  # ----------------------------------------

  # -- pickup ------------------------------------------------------------------------------------

  async def _unchecked_fw_pick_up_tips(
    self,
    x_position: int,
    x_direction: int,
    y_position: int,
    tip_type_table_index: int,
    z_pick_up_position: int,
    minimum_traverse_height_at_beginning_of_a_command: int,
    minimum_height_at_command_end: int,
    pick_up_method: int = 0,
    centering: bool = True,
    read_timeout: int = 120,
  ):
    """Send the pick-up as it is given, in tenths of a millimetre. `C0 JB`.

    Args:
      x_position: X of tip A1, as an absolute value, 0 to 30000.
      x_direction: sign of the X position: 0 positive, 1 negative.
      y_position: Y of well A1, 1100 to 5640.
      tip_type_table_index: the tip type table entry to mount, 0 to 99.
      z_pick_up_position: the collar bearing position, 0 to 3270.
      minimum_traverse_height_at_beginning_of_a_command: 0 to 3270.
      minimum_height_at_command_end: 0 to 3270.
      pick_up_method: 0 from a rack, 1 from the CoRe 384 tip wash station, 2 with full volume
        blowout.
      centering: whether the head centres itself on the rack as it collects.
      read_timeout: how long to wait for the answer, in s.
    """
    return await self._driver.send_command(
      module="C0",
      command="JB",
      subsystem=self.configuration.module,
      read_timeout=read_timeout,
      xs=f"{x_position:05}",
      xd=x_direction,
      yk=f"{y_position:04}",
      tt=f"{tip_type_table_index:02}",
      iu=pick_up_method,
      je=f"{z_pick_up_position:04}",
      zf=f"{minimum_traverse_height_at_beginning_of_a_command:04}",
      zg=f"{minimum_height_at_command_end:04}",
      ii=int(centering),
    )

  async def pick_up_tips(
    self,
    tip_rack: TipRack,
    offset: Optional[Coordinate] = None,
    tip_pickup_method: Literal["from_rack", "from_wash_station", "full_blowout"] = "from_rack",
    centering: bool = True,
    minimum_height_command_end: Optional[float] = None,
    minimum_traverse_height_start: Optional[float] = None,
  ) -> None:
    """Pick up a rack of tips on the whole head, as legacy's `pick_up_tips_core384`. `C0 JB`.

    Head channel A1 goes to the centre of spot A1, at the spot's Z. Once the device has picked them
    up, the tip in each spot is mounted on the shaft of the channel with the spot's index.

    Args:
      tip_rack: a 384 tip rack. Spots without a tip give none.
      offset: added to spot A1's centre, in mm.
      tip_pickup_method: where the tips are collected from.
      centering: whether the head centres itself on the rack as it collects.
      minimum_height_command_end: in mm. `configuration.traversal_z_position` when None.
      minimum_traverse_height_start: in mm. `configuration.traversal_z_position` when None.

    Raises:
      ValueError: If the rack does not have 384 spots or holds no tips, or a position cannot be
        reached.
      TypeError: If its tips are not Hamilton tips.
      RuntimeError: If the driver was given no deck.
    """
    deck = self._driver.deck
    if deck is None:
      raise RuntimeError("tip commands are placed from the deck; this driver was given none")
    if tip_rack.num_items != 384:
      raise ValueError("Tip rack must have 384 tips")
    tips = [
      spot.tip_for_pickup() if not spot.tracks_tips or spot.tip is not None else None
      for spot in tip_rack.get_all_items()
    ]
    prototypical_tip = next((tip for tip in tips if tip is not None), None)
    if prototypical_tip is None:
      raise ValueError("No tips found in the tip rack.")
    if not isinstance(prototypical_tip, HamiltonTip):
      raise TypeError("Tip type must be HamiltonTip.")
    # TODO: a captured pick-up defined its tips as collar type 6, where `TipSize.CORE_384_HEAD_TIP`
    # is 4 and is what a 384 tip resource carries into the table. Establish which collar this head
    # takes before a rack is collected by this path.
    tip_type_index = await self._driver.get_or_assign_tip_type_index(prototypical_tip)

    # TODO: establish whether this head lowers its dispensing drive itself before it collects tips
    # from a rack. The 96-head does not: `Head96.pick_up_tips` first sends its drive to
    # `configuration.dispensing_drive_position_before_rack_pickup` with
    # `move_dispensing_drive_to_position`, so that tips are not mounted against a raised piston.
    # Nothing here does the same, and nothing can: this head has no method that moves that drive,
    # though `Head384Configuration` carries every value one would need - the increment size, the
    # window, the speed and the acceleration. Whether it needs one is untested either way, since
    # the command this sends is pinned to a capture rather than to a pick-up this driver drove.
    #
    # Raise the piston deliberately, collect a rack, and look at what the head comes away with. If
    # the tips seat badly, this head needs the move and the command that makes it.

    location = tip_rack.get_item("A1").get_location_wrt(deck, x="c", y="c", z="b") + (
      offset or Coordinate.zero()
    )
    traverse_z, end_z = self._resolve_tip_command_heights(
      minimum_traverse_height_start, minimum_height_command_end
    )
    self._check_tip_command(location, traverse_z, end_z, skip_z=True)

    await self._unchecked_fw_pick_up_tips(
      x_position=abs(round(location.x * 10)),
      x_direction=0 if location.x >= 0 else 1,
      y_position=round(location.y * 10),
      tip_type_table_index=tip_type_index,
      z_pick_up_position=round(location.z * 10),
      minimum_traverse_height_at_beginning_of_a_command=round(traverse_z * 10),
      minimum_height_at_command_end=round(end_z * 10),
      pick_up_method={"from_rack": 0, "from_wash_station": 1, "full_blowout": 2}[tip_pickup_method],
      centering=centering,
    )

    if self.resource is not None:
      for shaft, tip in zip(self.resource.get_all_items(), tips):
        if tip is not None:
          shaft.mount_tip(tip)
    await self._record_after_tip_command()

  # -- drop --------------------------------------------------------------------------------------

  async def _unchecked_fw_discard_tips(
    self,
    x_position: int,
    x_direction: int,
    y_position: int,
    z_deposit_position: int,
    minimum_traverse_height_at_beginning_of_a_command: int,
    minimum_height_at_command_end: int,
    discard_method: int = 0,
    read_timeout: int = 120,
  ):
    """Send the discard as it is given, in tenths of a millimetre. `C0 JC`.

    Args:
      x_position: X of well A1, as an absolute value, 0 to 30000.
      x_direction: sign of the X position: 0 positive, 1 negative.
      y_position: Y of well A1, 1100 to 5640.
      z_deposit_position: the collar bearing position, 0 to 3270.
      minimum_traverse_height_at_beginning_of_a_command: 0 to 3270.
      minimum_height_at_command_end: 0 to 3270.
      discard_method: 0 discards the tips, 1 the tip tool.
      read_timeout: how long to wait for the answer, in s.
    """
    return await self._driver.send_command(
      module="C0",
      command="JC",
      subsystem=self.configuration.module,
      read_timeout=read_timeout,
      xs=f"{x_position:05}",
      xd=x_direction,
      yk=f"{y_position:04}",
      je=f"{z_deposit_position:04}",
      zf=f"{minimum_traverse_height_at_beginning_of_a_command:04}",
      zg=f"{minimum_height_at_command_end:04}",
      jd=discard_method,
    )

  async def drop_tips(
    self,
    resource: Resource,
    offset: Optional[Coordinate] = None,
    minimum_height_command_end: Optional[float] = None,
    minimum_traverse_height_start: Optional[float] = None,
  ) -> None:
    """Drop the head's tips into a tip rack or anywhere else, as legacy's `discard_tips_core384`.
    `C0 JC`.

    Into a tip rack, head channel A1 goes to the centre of spot A1, at the spot's Z, and each
    channel's tip goes into the spot with its index. Anywhere else, the head is centred over the
    resource, and the tips belong to nothing afterwards.

    Args:
      resource: a 384 tip rack, or anything else, such as the trash.
      offset: added to where the head goes, in mm.
      minimum_height_command_end: in mm. `configuration.traversal_z_position` when None.
      minimum_traverse_height_start: in mm. `configuration.traversal_z_position` when None.

    Raises:
      ValueError: If a tip rack does not have 384 spots, or a position cannot be reached.
      RuntimeError: If the driver was given no deck.
    """
    deck = self._driver.deck
    if deck is None:
      raise RuntimeError("tip commands are placed from the deck; this driver was given none")
    if isinstance(resource, TipRack):
      if resource.num_items != 384:
        raise ValueError("Tip rack must have 384 tips")
      location = resource.get_item("A1").get_location_wrt(deck, x="c", y="c", z="b")
    else:
      location = self._position_centred_in(resource)
    location += offset or Coordinate.zero()
    traverse_z, end_z = self._resolve_tip_command_heights(
      minimum_traverse_height_start, minimum_height_command_end
    )
    self._check_tip_command(location, traverse_z, end_z, skip_z=True)

    await self._unchecked_fw_discard_tips(
      x_position=abs(round(location.x * 10)),
      x_direction=0 if location.x >= 0 else 1,
      y_position=round(location.y * 10),
      z_deposit_position=round(location.z * 10),
      minimum_traverse_height_at_beginning_of_a_command=round(traverse_z * 10),
      minimum_height_at_command_end=round(end_z * 10),
    )

    if self.resource is not None:
      for i, shaft in enumerate(self.resource.get_all_items()):
        if not shaft.has_tip():
          continue
        tip = shaft.release_tip()
        if isinstance(resource, TipRack) and isinstance(tip, Tip):
          spot = resource.get_item(i)
          if spot.tracks_tips:
            spot.assign_tip(tip)
    await self._record_after_tip_command()

  # ----------------------------------------
  # Liquid handling
  # ----------------------------------------

  # -- aspirate ----------------------------------------------------------------------------------

  async def _unchecked_fw_aspirate(
    self,
    x_position: int,
    x_direction: int,
    y_position: int,
    minimum_traverse_height_at_beginning_of_a_command: int,
    minimum_height_at_command_end: int,
    liquid_surface_no_lld: int,
    minimum_height: int,
    aspiration_volume: int,
    aspiration_type: int = 0,
    lld_search_height: int = 3270,
    pull_out_distance_transport_air: int = 50,
    second_section_height: int = 0,
    second_section_ratio: int = 0,
    immersion_depth: int = 0,
    immersion_depth_direction: int = 0,
    surface_following_distance: int = 0,
    aspiration_speed: int = 2000,
    transport_air_volume: int = 0,
    blow_out_air_volume: int = 100,
    pre_wetting_volume: int = 0,
    lld_mode: int = 1,
    gamma_lld_sensitivity: int = 1,
    swap_speed: int = 100,
    settling_time: int = 0,
    homogenization_volume: int = 0,
    homogenization_cycles: int = 0,
    homogenization_position_from_liquid_surface: int = 0,
    homogenization_speed: int = 2000,
    homogenization_surface_following_distance: int = 0,
    capacitive_lld_gain: Optional[int] = None,
    capacitive_lld_offset: Optional[int] = None,
    read_timeout: int = 120,
  ):
    """Send the aspiration as it is given, in the units the command counts in. `C0 JA`.

    Positions and distances are in tenths of a millimetre, volumes in hundredths of a microlitre,
    speeds in tenths of a unit per second.

    Args:
      x_position: X of well A1, as an absolute value, 0 to 30000.
      x_direction: sign of the X position: 0 positive, 1 negative.
      y_position: Y of well A1, 1100 to 5640.
      minimum_traverse_height_at_beginning_of_a_command: 0 to 3270.
      minimum_height_at_command_end: 0 to 3270.
      liquid_surface_no_lld: where the liquid stands when the head runs without LLD, 0 to 3270.
      minimum_height: the deepest the tips may go, 0 to 3270.
      aspiration_volume: 0 to 9400.
      aspiration_type: 0 simple, 1 sequence, 2 cup emptied.
      lld_search_height: 0 to 3270.
      pull_out_distance_transport_air: how far to withdraw to take transport air, 0 to 3270.
      second_section_height: the tube's second section, measured from `minimum_height`, 0 to 3270.
      second_section_ratio: 0 to 10000.
      immersion_depth: 0 to 250.
      immersion_depth_direction: 0 goes deeper, 1 goes up out of the liquid.
      surface_following_distance: how far the surface sinks over the aspiration, 0 to 250.
      aspiration_speed: 3 to 2400.
      transport_air_volume: 0 to 1000.
      blow_out_air_volume: 0 to 8760.
      pre_wetting_volume: 0 to 8760.
      lld_mode: 0 off, 1 gamma.
      gamma_lld_sensitivity: 1 high to 4 low.
      swap_speed: how fast the tips leave the liquid, 3 to 1000.
      settling_time: in tenths of a second, 0 to 99.
      homogenization_volume: 0 to 8760.
      homogenization_cycles: 0 to 99.
      homogenization_position_from_liquid_surface: 0 to 250.
      homogenization_speed: 3 to 2400.
      homogenization_surface_following_distance: 0 to 250.
      capacitive_lld_gain: in AD steps, 0 to 1023. Left out of the command when None, which is
        what Venus sends.
      capacitive_lld_offset: in AD steps, 0 to 1023. Left out of the command when None.
      read_timeout: how long to wait for the answer, in s.
    """
    parameters: Dict[str, Any] = {
      "ja": aspiration_type,
      "xs": f"{x_position:05}",
      "xd": x_direction,
      "yk": f"{y_position:04}",
      "zf": f"{minimum_traverse_height_at_beginning_of_a_command:04}",
      "zg": f"{minimum_height_at_command_end:04}",
      "jz": f"{lld_search_height:04}",
      "jt": f"{liquid_surface_no_lld:04}",
      "jm": f"{minimum_height:04}",
      "jw": f"{immersion_depth:03}",
      "jx": immersion_depth_direction,
      "jh": f"{surface_following_distance:03}",
      "jf": f"{aspiration_volume:05}",
      "jg": f"{aspiration_speed:04}",
      "ju": f"{transport_air_volume:04}",
      "jv": f"{blow_out_air_volume:05}",
      "jy": f"{pre_wetting_volume:05}",
      "jq": lld_mode,
      "jp": gamma_lld_sensitivity,
      "js": f"{swap_speed:04}",
      "ji": f"{settling_time:02}",
      "jj": f"{homogenization_volume:05}",
      "jk": f"{homogenization_cycles:02}",
      "jl": f"{homogenization_position_from_liquid_surface:03}",
      "jn": f"{homogenization_speed:04}",
      "zw": f"{second_section_height:04}",
      "zs": f"{second_section_ratio:05}",
      "mk": f"{homogenization_surface_following_distance:03}",
      "pq": f"{pull_out_distance_transport_air:04}",
    }
    if capacitive_lld_gain is not None:
      parameters["ig"] = f"{capacitive_lld_gain:04}"
    if capacitive_lld_offset is not None:
      parameters["ih"] = f"{capacitive_lld_offset:04}"

    return await self._driver.send_command(
      module="C0",
      command="JA",
      subsystem=self.configuration.module,
      read_timeout=read_timeout,
      **parameters,
    )

  # -- dispense ----------------------------------------------------------------------------------

  async def _unchecked_fw_dispense(
    self,
    x_position: int,
    x_direction: int,
    y_position: int,
    minimum_traverse_height_at_beginning_of_a_command: int,
    minimum_height_at_command_end: int,
    liquid_surface_no_lld: int,
    minimum_height: int,
    dispense_volume: int,
    dispensing_mode: int = 0,
    second_section_height: int = 0,
    second_section_ratio: int = 0,
    lld_search_height: int = 3270,
    pull_out_distance_transport_air: int = 50,
    immersion_depth: int = 0,
    immersion_depth_direction: int = 0,
    surface_following_distance: int = 0,
    dispense_speed: int = 2000,
    cut_off_speed: int = 1500,
    stop_back_volume: int = 0,
    transport_air_volume: int = 0,
    blow_out_air_volume: int = 0,
    lld_mode: int = 1,
    gamma_lld_sensitivity: int = 1,
    side_touch_off_distance: int = 0,
    swap_speed: int = 100,
    settling_time: int = 0,
    mix_volume: int = 0,
    mix_cycles: int = 0,
    mix_position_from_liquid_surface: int = 0,
    mix_speed: int = 2000,
    mix_surface_following_distance: int = 0,
    capacitive_lld_gain: Optional[int] = None,
    capacitive_lld_offset: Optional[int] = None,
    read_timeout: int = 120,
  ):
    """Send the dispense as it is given, in the units the command counts in. `C0 JD`.

    Units as `_unchecked_fw_aspirate` takes them.

    Args:
      x_position: X of well A1, as an absolute value, 0 to 30000.
      x_direction: sign of the X position: 0 positive, 1 negative.
      y_position: Y of well A1, 1100 to 5640.
      minimum_traverse_height_at_beginning_of_a_command: 0 to 3270.
      minimum_height_at_command_end: 0 to 3270.
      liquid_surface_no_lld: where the liquid stands when the head runs without LLD, 0 to 3270.
      minimum_height: the deepest the tips may go, 0 to 3270.
      dispense_volume: 0 to 8760.
      dispensing_mode: 0 partial volume in jet mode, 1 blow out in jet mode, 2 partial volume at
        the surface, 3 blow out at the surface, 4 empty the tip at a fixed position.
      second_section_height: the tube's second section, measured from `minimum_height`, 0 to 3270.
      second_section_ratio: 0 to 10000.
      lld_search_height: 0 to 3270.
      pull_out_distance_transport_air: how far to withdraw to take transport air, 0 to 3270.
      immersion_depth: 0 to 250.
      immersion_depth_direction: 0 goes deeper, 1 goes up out of the liquid.
      surface_following_distance: how far the surface rises over the dispense, 0 to 250.
      dispense_speed: 3 to 2400.
      cut_off_speed: 3 to 2400.
      stop_back_volume: 0 to 2000.
      transport_air_volume: 0 to 1000.
      blow_out_air_volume: 0 to 8760.
      lld_mode: 0 off, 1 gamma.
      gamma_lld_sensitivity: 1 high to 4 low.
      side_touch_off_distance: 0 to 90. Anything above 0 turns LLD off.
      swap_speed: how fast the tips leave the liquid, 3 to 1000.
      settling_time: in tenths of a second, 0 to 99.
      mix_volume: 0 to 8760.
      mix_cycles: 0 to 99.
      mix_position_from_liquid_surface: 0 to 250.
      mix_speed: 3 to 2400.
      mix_surface_following_distance: 0 to 250.
      capacitive_lld_gain: in AD steps, 0 to 1023. Left out of the command when None, which is
        what Venus sends.
      capacitive_lld_offset: in AD steps, 0 to 1023. Left out of the command when None.
      read_timeout: how long to wait for the answer, in s.
    """
    parameters: Dict[str, Any] = {
      "jo": dispensing_mode,
      "xs": f"{x_position:05}",
      "xd": x_direction,
      "yk": f"{y_position:04}",
      "jm": f"{minimum_height:04}",
      "jz": f"{lld_search_height:04}",
      "jt": f"{liquid_surface_no_lld:04}",
      "jw": f"{immersion_depth:03}",
      "jx": immersion_depth_direction,
      "jh": f"{surface_following_distance:03}",
      "zf": f"{minimum_traverse_height_at_beginning_of_a_command:04}",
      "zg": f"{minimum_height_at_command_end:04}",
      "jb": f"{dispense_volume:05}",
      "jc": f"{dispense_speed:04}",
      "jr": f"{cut_off_speed:04}",
      "im": f"{stop_back_volume:04}",
      "ju": f"{transport_air_volume:04}",
      "jv": f"{blow_out_air_volume:05}",
      "jq": lld_mode,
      "jp": gamma_lld_sensitivity,
      "js": f"{swap_speed:04}",
      "ji": f"{settling_time:02}",
      "jj": f"{mix_volume:05}",
      "jk": f"{mix_cycles:02}",
      "jl": f"{mix_position_from_liquid_surface:03}",
      "jn": f"{mix_speed:04}",
      "zw": f"{second_section_height:04}",
      "ij": f"{side_touch_off_distance:02}",
      "zs": f"{second_section_ratio:05}",
      "mk": f"{mix_surface_following_distance:03}",
      "pq": f"{pull_out_distance_transport_air:04}",
    }
    if capacitive_lld_gain is not None:
      parameters["ig"] = f"{capacitive_lld_gain:04}"
    if capacitive_lld_offset is not None:
      parameters["ih"] = f"{capacitive_lld_offset:04}"

    return await self._driver.send_command(
      module="C0",
      command="JD",
      subsystem=self.configuration.module,
      read_timeout=read_timeout,
      **parameters,
    )

  # ----------------------------------------
  # Wash
  # ----------------------------------------

  # The two commands below drive the CoRe 384 tip wash station. Their field widths are the
  # specification's; no capture pins them, because the device this head was read on has no wash
  # station fitted.

  async def _unchecked_fw_wash_tips(
    self,
    x_position: int,
    x_direction: int,
    y_position: int,
    wash_z_position: int,
    minimum_height: int,
    minimum_traverse_height_at_beginning_of_a_command: int,
    wash_volume: int,
    wash_cycles: int,
    surface_following_distance: int = 0,
    wash_speed: int = 2000,
    read_timeout: int = 120,
  ):
    """Send the wash as it is given, in the units the command counts in. `C0 JG`.

    Args:
      x_position: wash X of well A1, as an absolute value, 0 to 30000.
      x_direction: sign of the X position: 0 positive, 1 negative.
      y_position: wash Y of well A1, 1100 to 5640.
      wash_z_position: 0 to 3270.
      minimum_height: the deepest the tips may go, 0 to 3270.
      minimum_traverse_height_at_beginning_of_a_command: 0 to 3270.
      wash_volume: 0 to 8760.
      wash_cycles: 0 to 99.
      surface_following_distance: 0 to 250.
      wash_speed: 3 to 2400.
      read_timeout: how long to wait for the answer, in s.
    """
    return await self._driver.send_command(
      module="C0",
      command="JG",
      subsystem=self.configuration.module,
      read_timeout=read_timeout,
      xs=f"{x_position:05}",
      xd=x_direction,
      yk=f"{y_position:04}",
      jt=f"{wash_z_position:04}",
      jm=f"{minimum_height:04}",
      jh=f"{surface_following_distance:03}",
      zf=f"{minimum_traverse_height_at_beginning_of_a_command:04}",
      jj=f"{wash_volume:05}",
      jk=f"{wash_cycles:02}",
      jn=f"{wash_speed:04}",
    )

  async def _unchecked_fw_empty_washed_tips(
    self,
    z_position: int,
    minimum_height_at_command_end: int,
    read_timeout: int = 120,
  ):
    """Empty the washed tips at the end of a wash, as it is given. `C0 JU`.

    Args:
      z_position: 0 to 3270.
      minimum_height_at_command_end: 0 to 3270.
      read_timeout: how long to wait for the answer, in s.
    """
    return await self._driver.send_command(
      module="C0",
      command="JU",
      subsystem=self.configuration.module,
      read_timeout=read_timeout,
      jt=f"{z_position:04}",
      zg=f"{minimum_height_at_command_end:04}",
    )
