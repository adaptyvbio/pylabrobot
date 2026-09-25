import dataclasses
import datetime
import json
import pathlib
import random
import tempfile
import unittest
from typing import Any, List, Optional, Tuple, cast
from unittest.mock import AsyncMock, patch

from pylabrobot.hamilton.protocol.text.framing import assemble_command
from pylabrobot.hamilton.star.device import (
  RECORDING_STAR,
  RECORDING_STAR_HEAD384,
  RECORDING_STARLET,
  RECORDING_STARLET_HEAD384,
)
from pylabrobot.hamilton.star.driver.configuration import read_configuration
from pylabrobot.hamilton.star.driver.errors import STARFirmwareError, check_fw_string_error
from pylabrobot.hamilton.star.driver.features.head96 import Head96, Head96Configuration
from pylabrobot.hamilton.star.driver.features.head384 import Head384, Head384Configuration
from pylabrobot.hamilton.star.driver.features.x_arm import XArm
from pylabrobot.hamilton.star.driver.master import STARDriver
from pylabrobot.hamilton.star.driver.simulator import STARSimulationDriver
from pylabrobot.resources.coordinate import Coordinate
from pylabrobot.resources.hamilton import STARDeck, STARLetDeck
from pylabrobot.resources.hamilton.tip_creators import hamilton_tip_50uL
from pylabrobot.resources.n_channel_pipettes import NChannelPipette
from pylabrobot.resources.tip_rack import TipRack, TipSpot
from pylabrobot.resources.utils import create_ordered_items_2d
from pylabrobot.serializer import serialize

# The 96-head on the device this package ships a recording of.
RECORDED_HEAD96 = cast(
  Head96Configuration, read_configuration(RECORDING_STAR)["arms"]["left"]["head96"]
)


def declaring(**parts: object) -> str:
  """The shipped recording with parts swapped out, written where it can be read back.

  A declaration is read from a file and nothing else, so a test that needs a device no recording
  describes writes one. Everything not named here stays as the recorded STAR has it.

  Args:
    parts: `device` for the device itself, or a feature name for something the left arm carries.

  Returns:
    The path it was written to.
  """
  tree = json.loads(pathlib.Path(RECORDING_STAR).read_text())
  for name, part in parts.items():
    assert dataclasses.is_dataclass(part) and not isinstance(part, type)
    if name == "device":
      tree["device"] = serialize(dataclasses.asdict(part))
    else:
      tree["arms"]["left"][name] = serialize(dataclasses.asdict(part))
  written = pathlib.Path(tempfile.mkdtemp()) / "declared.json"
  written.write_text(json.dumps(tree))
  return str(written)


class TestDriveDefaults(unittest.IsolatedAsyncioTestCase):
  """Where the value a move uses when the caller names none comes from: what the head reported,
  falling back to what its firmware documents."""

  async def test_the_defaults_are_what_the_head_reported(self):
    """Discovery reads the four Y and Z drive parameters off the head, and the defaults answer with
    them. Read from a head declaring values its firmware does not, so a default that ignored the
    head and computed the documented one instead could not pass. Four distinct values, so a read
    stored under the wrong name fails this too."""
    declared = dataclasses.replace(
      RECORDED_HEAD96,
      y_drive_speed_firmware_reported=200.0,
      y_drive_acceleration_firmware_reported=300.0,
      z_drive_speed_firmware_reported=50.0,
      z_drive_acceleration_firmware_reported=250.0,
    )
    driver = STARSimulationDriver(
      deck=STARDeck(), declared_configuration_json=declaring(head96=declared)
    )
    await driver.setup()

    c = cast(Head96, driver.x_arm.head96).configuration
    self.assertEqual(
      (
        c.y_drive_speed_default,
        c.y_drive_acceleration_default,
        c.z_drive_speed_default,
        c.z_drive_acceleration_default,
      ),
      (200.0, 300.0, 50.0, 250.0),
    )

  async def test_the_96_head_takes_its_dispensing_and_squeezer_defaults_too(self):
    """`Head96.discover` reads four drive parameters on top of the Y and Z ones every head shares,
    and its defaults answer with what it reported for them. Apart from the test above because it
    covers the override rather than the base: the 384-head adds no reads of its own.

    Compared to a tenth rather than exactly: the drive counts in increments, so a value that does
    not fall on one comes back as the nearest that does - 400.0 mm/s reads back as 400.01. That is
    what a head does, and what the simulated one does now that its answer crosses the link and is
    decoded rather than handed over whole."""
    declared = dataclasses.replace(
      RECORDED_HEAD96,
      dispensing_drive_speed_firmware_reported=400.0,
      dispensing_drive_acceleration_firmware_reported=9000.0,
      squeezer_drive_speed_firmware_reported=12.0,
      squeezer_drive_acceleration_firmware_reported=50.0,
    )
    driver = STARSimulationDriver(
      deck=STARDeck(), declared_configuration_json=declaring(head96=declared)
    )
    await driver.setup()

    c = cast(Head96, driver.x_arm.head96).configuration
    for read, declared_value in zip(
      (
        c.dispensing_drive_speed_default,
        c.dispensing_drive_acceleration_default,
        c.squeezer_drive_speed_default,
        c.squeezer_drive_acceleration_default,
      ),
      (400.0, 9000.0, 12.0, 50.0),
    ):
      self.assertAlmostEqual(read, declared_value, places=1)

  async def test_a_head_that_will_not_say_keeps_what_its_firmware_documents(self):
    """A head that refuses the read leaves discovery with nothing to record, and the defaults fall
    back to the increments its firmware documents rather than the read failing setup. Driven
    through `discover` alone: the rest of setup moves the head, and reads these same parameters to
    do it."""
    driver = STARSimulationDriver(deck=STARDeck(), declared_configuration_json=RECORDING_STAR)
    head = cast(Head96, cast(XArm, driver.left_x_arm).head96)

    async def refuse(parameter: str) -> float:
      raise RuntimeError("this head does not answer for its drives")

    head.request_drive_parameter = refuse  # type: ignore[method-assign]
    await head.discover()

    c = head.configuration
    self.assertEqual(
      (
        c.y_drive_speed_firmware_reported,
        c.y_drive_acceleration_firmware_reported,
        c.z_drive_speed_firmware_reported,
        c.z_drive_acceleration_firmware_reported,
      ),
      (None, None, None, None),
    )
    self.assertEqual(
      (
        c.y_drive_speed_default,
        c.y_drive_acceleration_default,
        c.z_drive_speed_default,
        c.z_drive_acceleration_default,
      ),
      (390.62, 546.88, 85.0, 400.0),
    )


class TestHead96Tips(unittest.IsolatedAsyncioTestCase):
  """As legacy's 96-head tip tests, on the layout they use: a 300 uL filter rack on a STARlet."""

  async def asyncSetUp(self):
    from pylabrobot.resources import set_tip_tracking
    from pylabrobot.resources.hamilton import TIP_CAR_480_A00, hamilton_96_tiprack_300uL_filter

    set_tip_tracking(True)
    self.addCleanup(set_tip_tracking, False)
    self.deck = STARLetDeck()
    self.driver = STARSimulationDriver(
      deck=self.deck, declared_configuration_json=RECORDING_STARLET
    )
    await self.driver.setup()
    self.head = cast(Head96, self.driver.head96)
    self.head_resource = cast(NChannelPipette, self.head.resource)
    tip_car = TIP_CAR_480_A00(name="tip carrier")
    tip_car[1] = self.tip_rack = hamilton_96_tiprack_300uL_filter(name="tip_rack_01")
    self.deck.assign_child_resource(tip_car, track=1)
    self.sent: List[str] = []
    log = self.driver._log_exchange

    def recorded(written: str, read: Optional[str]) -> None:
      if written[:4] in ("C0TT", "H0DQ", "C0EP", "C0ER"):
        self.sent.append(written)
      log(written, read)

    self.driver._log_exchange = recorded  # type: ignore[method-assign]

  async def test_pick_up_and_drop_send_what_legacy_sends(self):
    await self.head.pick_up_tips(self.tip_rack)
    await self.head.drop_tips(self.tip_rack)
    await self.head.pick_up_tips(self.tip_rack)
    await self.head.drop_tips(self.deck.get_trash_area96())
    self.assertEqual(
      self.sent,
      [
        "C0TTtt01tf1tl0519tv03600tg2tu0",
        "H0DQdq11281dv13500du00000dr900000dw15",
        "C0EPxs01179xd0yh2418tt01wu0za2164zh2450ze2450",
        "C0ERxs01179xd0yh2418za2164zh2450ze2450",
        "H0DQdq11281dv13500du00000dr900000dw15",
        "C0EPxs01179xd0yh2418tt01wu0za2164zh2450ze2450",
        "C0ERxs00420xd1yh1203za2164zh2450ze2450",
      ],
    )

  async def test_tips_missing_from_a_rack_are_missing_from_the_head(self):
    rng = random.Random(0)
    shafts = self.head_resource.get_all_items()
    for missing in (0, 1, 48, 95):
      with self.subTest(missing=missing):
        spots = self.tip_rack.get_all_items()
        empty = set(rng.sample(range(96), missing))
        for i in empty:
          spots[i].unassign_tip()
        tips = [spot.tip for spot in spots]

        await self.head.pick_up_tips(self.tip_rack)
        self.assertEqual([shaft.tip for shaft in shafts], tips)
        self.assertFalse(any(spot.has_tip() for spot in spots))

        await self.head.drop_tips(self.tip_rack)
        self.assertEqual([spot.tip for spot in spots], tips)
        self.assertFalse(any(shaft.has_tip() for shaft in shafts))
        self.tip_rack.fill()

  async def test_tips_dropped_in_the_trash_belong_to_nothing(self):
    await self.head.pick_up_tips(self.tip_rack)
    tips = [shaft.tip for shaft in self.head_resource.get_all_items()]
    await self.head.drop_tips(self.deck.get_trash_area96())
    self.assertFalse(any(shaft.has_tip() for shaft in self.head_resource.get_all_items()))
    self.assertFalse(any(spot.has_tip() for spot in self.tip_rack.get_all_items()))
    self.assertTrue(all(tip is not None and tip.parent is None for tip in tips))

  async def test_a_refused_pickup_moves_nothing(self):
    """An empty rack is refused before anything is sent; an unreachable one after the tip type."""
    with self.assertRaises(ValueError):
      await self.head.pick_up_tips(self.tip_rack, offset=Coordinate(0, 1000, 0))
    self.assertEqual(self.sent, ["C0TTtt01tf1tl0519tv03600tg2tu0"])
    self.tip_rack.empty()
    self.sent.clear()
    with self.assertRaises(ValueError):
      await self.head.pick_up_tips(self.tip_rack)
    self.assertEqual(self.sent, [])

  async def test_a_failed_command_leaves_the_tips_where_they_were(self):
    answer = self.driver._answer
    spots = self.tip_rack.get_all_items()
    shafts = self.head_resource.get_all_items()

    def failing(failed: str):
      async def answering(module: str, command: str, **kwargs: Any):
        if command == failed:
          check_fw_string_error(f"C0{failed}id0001er99/00")
        return await answer(module, command, **kwargs)

      return answering

    with patch.object(self.driver, "_answer", failing("EP")):
      with self.assertRaises(STARFirmwareError):
        await self.head.pick_up_tips(self.tip_rack)
    self.assertTrue(all(spot.has_tip() for spot in spots))
    self.assertFalse(any(shaft.has_tip() for shaft in shafts))

    await self.head.pick_up_tips(self.tip_rack)
    with patch.object(self.driver, "_answer", failing("ER")):
      with self.assertRaises(STARFirmwareError):
        await self.head.drop_tips(self.tip_rack)
    self.assertTrue(all(shaft.has_tip() for shaft in shafts))
    self.assertFalse(any(spot.has_tip() for spot in spots))


# What a real 384-head answered, keyed by the module and command that asked. `er00/00` is the
# master's own "no error" prefix, which the reads that go through it carry and the ones addressed
# to the head's own module do not.
HEAD384_REPLIES = {
  "D0QW": "D0QWid0001qw1",
  "D0QG": "D0QGid0001qg2",
  "D0RF": "D0RFid0001rf1.4S b 2015-10-07",
  "C0QK": "C0QKid0001er00/00qk1",
  "C0QJ": "C0QJid0001er00/00xs01157xd0yk3402je2450",
}

# The same head type read off the master, which answers `C0 QY` with what the head answers `D0 QG`
# with. This driver asks the head, and this is what the two are checked to agree on.
HEAD384_TYPE_FROM_THE_MASTER = "C0QYid0001er00/00qy2"


async def head384() -> Tuple[Head384, List[str]]:
  """The 384-head of a simulated device, and the list its commands are recorded in.

  Returns:
    The feature, and the list every command it sends is appended to.
  """
  driver = STARSimulationDriver(deck=STARDeck(), declared_configuration_json=RECORDING_STAR_HEAD384)
  await driver.setup()
  head = cast(Head384, driver.head384)

  sent: List[str] = []
  answer = driver.send_command

  async def recorded(
    module: str,
    command: str,
    fmt: Optional[Any] = None,
    subsystem: Optional[str] = None,
    read_timeout: Optional[int] = None,
    **kwargs: Any,
  ):
    # `fmt`, `subsystem` and `read_timeout` are the driver's own, not firmware parameters, so they
    # never reach the assembler.
    sent.append(assemble_command(module=module, command=command, id_=None, **kwargs))
    return await answer(
      module=module,
      command=command,
      fmt=fmt,
      subsystem=subsystem,
      read_timeout=read_timeout,
      **kwargs,
    )

  driver.send_command = recorded  # type: ignore[assignment]
  return head, sent


async def head384_answering_captures() -> Head384:
  """A 384-head whose every read is answered with what a real one answered.

  On a driver of its own rather than the simulated one, which answers the reads from its model and
  would stand between these replies and the decoding they are here to check. Nothing is sent, so
  the driver needs no link.

  Returns:
    The feature.
  """
  driver = STARDriver(io=AsyncMock())

  async def answering(module: str, command: str, fmt: Optional[Any] = None, **kwargs: Any):
    reply = HEAD384_REPLIES[f"{module}{command}"]
    # As `_send` finishes: parsed against the format the caller gave, raw where it gave none.
    return driver._parse_response(reply, fmt) if fmt is not None else reply

  driver._send = answering  # type: ignore[assignment]
  return Head384(driver)


class TestHead384Commands(unittest.IsolatedAsyncioTestCase):
  """What the 384-head is sent, against what a real one was sent."""

  async def test_initialization_ejects_where_it_is_pointed(self):
    """`C0 JI`. The capture's `xd1` comes of the position being left of the deck origin, which is
    where this device's waste sits; see the TODO on `Head384`."""
    head, sent = await head384()
    await head.initialize(
      tip_discard_location=Coordinate(-161.3, 116.9, 242.0),
      z_position_at_the_command_end=245.0,
    )
    self.assertEqual(sent[0], "C0JIxs01613xd1yk1169je2420zg2450")

  async def test_the_retract_is_this_heads_own(self):
    """`C0 JV`, where the 96-head's is `C0 EV`: the head states which, and the probe sends it."""
    head, sent = await head384()
    await head.probe_z_max()
    self.assertEqual(sent[0], "C0JV")

  async def test_an_all_axis_move_is_addressed_to_this_head(self):
    """`C0 EN`, the 384-head's version of the 96-head's `C0 EM`."""
    head, sent = await head384()
    await head._unchecked_fw_move_to_coordinate(
      Coordinate(115.7, 340.2, 245.0),
      minimum_height_at_beginning_of_a_command=245.0,
    )
    self.assertEqual(sent, ["C0ENxs01157xd0yk3402je2450zf2450"])

  async def test_an_all_axis_move_travels_at_what_the_command_accepts(self):
    """Named no height, the move travels at the top of the command's own field, which is lower
    than the top of the Z drive's window."""
    head, sent = await head384()
    await head._unchecked_fw_move_to_coordinate(Coordinate(36.5, 116.9, 320.0))
    self.assertEqual(sent, ["C0ENxs00365xd0yk1169je3200zf3270"])

  async def test_a_y_only_move_carries_y_and_a_height_and_nothing_else(self):
    """`C0 EY`: the master raises the head to the height given and then travels."""
    head, sent = await head384()
    await head._unchecked_fw_safety_move_to_y_position(
      244.2, minimum_height_at_beginning_of_a_command=320.0
    )
    self.assertEqual(sent, ["C0EYyk2442zf3200"])

  async def test_it_picks_up_a_rack_of_tips(self):
    """`C0 JB`, sent as the firmware takes it."""
    head, sent = await head384()
    await head._unchecked_fw_pick_up_tips(
      x_position=1157,
      x_direction=0,
      y_position=2442,
      tip_type_table_index=33,
      z_pick_up_position=2195,
      minimum_traverse_height_at_beginning_of_a_command=2450,
      minimum_height_at_command_end=2450,
      centering=False,
    )
    self.assertEqual(sent, ["C0JBxs01157xd0yk2442tt33iu0je2195zf2450zg2450ii0"])

  async def test_it_discards_tips_to_a_rack_and_to_the_waste(self):
    """`C0 JC` twice: returning a rack, and ejecting where `C0 JI` ejects."""
    head, sent = await head384()
    await head._unchecked_fw_discard_tips(
      x_position=1157,
      x_direction=0,
      y_position=2442,
      z_deposit_position=2195,
      minimum_traverse_height_at_beginning_of_a_command=2450,
      minimum_height_at_command_end=2450,
    )
    await head._unchecked_fw_discard_tips(
      x_position=1613,
      x_direction=1,
      y_position=1169,
      z_deposit_position=2420,
      minimum_traverse_height_at_beginning_of_a_command=2450,
      minimum_height_at_command_end=2450,
    )
    self.assertEqual(
      sent,
      [
        "C0JCxs01157xd0yk2442je2195zf2450zg2450jd0",
        "C0JCxs01613xd1yk1169je2420zf2450zg2450jd0",
      ],
    )

  async def test_it_aspirates(self):
    """`C0 JA`, which leaves out `ig` and `ih` where no capacitive LLD is asked for."""
    head, sent = await head384()
    await head._unchecked_fw_aspirate(
      x_position=2507,
      x_direction=0,
      y_position=3402,
      minimum_traverse_height_at_beginning_of_a_command=2450,
      minimum_height_at_command_end=2450,
      lld_search_height=2231,
      liquid_surface_no_lld=1901,
      minimum_height=1861,
      aspiration_volume=4800,
      aspiration_speed=500,
      transport_air_volume=300,
      blow_out_air_volume=300,
      lld_mode=0,
      swap_speed=20,
      settling_time=10,
      homogenization_speed=500,
      pull_out_distance_transport_air=100,
    )
    self.assertEqual(
      sent,
      [
        "C0JAja0xs02507xd0yk3402zf2450zg2450jz2231jt1901jm1861jw000jx0jh000jf04800"
        "jg0500ju0300jv00300jy00000jq0jp1js0020ji10jj00000jk00jl000jn0500zw0000zs00000"
        "mk000pq0100"
      ],
    )

  async def test_it_dispenses(self):
    """`C0 JD`, which leaves out `ig` and `ih` as the aspiration does."""
    head, sent = await head384()
    await head._unchecked_fw_dispense(
      dispensing_mode=1,
      x_position=2507,
      x_direction=0,
      y_position=1482,
      minimum_traverse_height_at_beginning_of_a_command=2450,
      minimum_height_at_command_end=2450,
      lld_search_height=2001,
      liquid_surface_no_lld=1876,
      minimum_height=1871,
      dispense_volume=4800,
      dispense_speed=500,
      cut_off_speed=200,
      transport_air_volume=300,
      blow_out_air_volume=300,
      lld_mode=0,
      swap_speed=20,
      settling_time=5,
      mix_speed=500,
      pull_out_distance_transport_air=100,
    )
    self.assertEqual(
      sent,
      [
        "C0JDjo1xs02507xd0yk1482jm1871jz2001jt1876jw000jx0jh000zf2450zg2450jb04800"
        "jc0500jr0200im0000ju0300jv00300jq0jp1js0020ji05jj00000jk00jl000jn0500zw0000ij00"
        "zs00000mk000pq0100"
      ],
    )

  async def test_the_capacitive_lld_gain_and_offset_are_appended_when_asked_for(self):
    """`ig` and `ih` go on the end of the aspiration and the dispense, in that order, and are
    left out where the caller names neither."""
    head, sent = await head384()
    where = dict(
      x_position=2507,
      x_direction=0,
      y_position=3402,
      minimum_traverse_height_at_beginning_of_a_command=2450,
      minimum_height_at_command_end=2450,
      liquid_surface_no_lld=1901,
      minimum_height=1861,
    )
    await head._unchecked_fw_aspirate(aspiration_volume=4800, **where)
    await head._unchecked_fw_aspirate(
      aspiration_volume=4800, capacitive_lld_gain=512, capacitive_lld_offset=64, **where
    )
    await head._unchecked_fw_dispense(
      dispense_volume=4800, capacitive_lld_gain=512, capacitive_lld_offset=64, **where
    )
    self.assertFalse(sent[0].endswith("ig0512ih0064"))
    self.assertTrue(sent[1].endswith("ig0512ih0064"))
    self.assertTrue(sent[2].endswith("ig0512ih0064"))

  async def test_it_washes_and_empties_the_tips(self):
    """`C0 JG` and `C0 JU`. Field widths only: no wash station was fitted to pin them against."""
    head, sent = await head384()
    await head._unchecked_fw_wash_tips(
      x_position=1157,
      x_direction=0,
      y_position=2442,
      wash_z_position=1800,
      minimum_height=1700,
      minimum_traverse_height_at_beginning_of_a_command=2450,
      wash_volume=5000,
      wash_cycles=3,
    )
    await head._unchecked_fw_empty_washed_tips(z_position=2450, minimum_height_at_command_end=2450)
    self.assertEqual(
      sent,
      [
        "C0JGxs01157xd0yk2442jt1800jm1700jh000zf2450jj05000jk03jn2000",
        "C0JUjt2450zg2450",
      ],
    )

  async def test_each_head_names_its_own_command_and_parameters(self):
    """The four constants the all-axis move turns on, which is why they are configuration. A head
    given the other's fails here before any test that would send one."""
    named = {
      name: (c.defined_position_command, c.y_parameter, c.z_parameter, c.traverse_z_parameter)
      for name, c in (("96", Head96Configuration()), ("384", Head384Configuration()))
    }
    self.assertEqual(named, {"96": ("EM", "yh", "za", "zh"), "384": ("EN", "yk", "je", "zf")})


class TestHead384Queries(unittest.IsolatedAsyncioTestCase):
  """What the 384-head answers, against what a real one replied."""

  async def test_it_reports_itself_initialized(self):
    """`D0 QW`, asked of the head's own module: the master's command table does not carry it."""
    head = await head384_answering_captures()
    self.assertTrue(await head._driver.request_initialization_status(module="D0"))

  async def test_it_reports_tips_mounted(self):
    """`C0 QK`, where the 96-head's is `C0 QH`."""
    head = await head384_answering_captures()
    self.assertTrue(await head.request_tip_presence())

  async def test_it_reports_where_channel_a1_is(self):
    """`C0 QJ`, in the parameter names this head's commands use throughout: `yk` and `je`."""
    head = await head384_answering_captures()
    self.assertEqual(await head.request_location(), Coordinate(115.7, 340.2, 245.0))

  async def test_it_reports_which_head_is_fitted(self):
    """`D0 QG`, decoded through this head's own table: code 2 is the shifted tip pickup head,
    where the same code on a 96-head means a 96 head II. The master's `C0 QY` agrees."""
    head = await head384_answering_captures()
    self.assertEqual(await head.request_head_type(), "STP head")
    from_the_master = int(HEAD384_TYPE_FROM_THE_MASTER.split("qy")[-1])
    self.assertEqual(head.configuration.head_types[from_the_master], "STP head")

  async def test_it_reports_its_own_firmware(self):
    """`D0 RF`. The version is kept whole, date included, as every other feature keeps it."""
    head = await head384_answering_captures()
    self.assertEqual(
      await head.request_firmware_version(),
      ("1.4S b 2015-10-07", datetime.date(2015, 10, 7)),
    )


def tip_rack_384(name: str, num_items_x: int = 24, num_items_y: int = 16) -> TipRack:
  """A 384 tip rack to place a command against.

  Built here rather than taken from the resource library, which carries no Hamilton 384 rack: the
  geometry below is a plate footprint on the head's own 4.5 mm pitch, which is what a tip command
  needs to be placed, and is not a measurement of any particular rack.

  Args:
    name: what to call it.
    num_items_x: how many columns, for building a rack of the wrong size.
    num_items_y: how many rows.

  Returns:
    The rack, filled with tips.
  """
  return TipRack(
    name=name,
    size_x=127.76,
    size_y=85.48,
    size_z=20.0,
    ordered_items=create_ordered_items_2d(
      TipSpot,
      num_items_x=num_items_x,
      num_items_y=num_items_y,
      dx=9.0,
      dy=6.8,
      dz=12.0,
      item_dx=4.5,
      item_dy=4.5,
      size_x=3.0,
      size_y=3.0,
      make_tip=hamilton_tip_50uL,
      name_prefix=name,
    ),
  )


class TestHead384Tips(unittest.IsolatedAsyncioTestCase):
  """Collecting and returning a rack on the 384-head, and what that does to the model."""

  async def asyncSetUp(self):
    from pylabrobot.resources import set_tip_tracking
    from pylabrobot.resources.hamilton import PLT_CAR_L5AC_A00

    set_tip_tracking(True)
    self.addCleanup(set_tip_tracking, False)
    self.deck = STARLetDeck()
    self.driver = STARSimulationDriver(
      deck=self.deck, declared_configuration_json=RECORDING_STARLET_HEAD384
    )
    await self.driver.setup()
    self.head = cast(Head384, self.driver.head384)
    self.head_resource = cast(NChannelPipette, self.head.resource)
    carrier = PLT_CAR_L5AC_A00(name="plate carrier")
    carrier[0] = self.tip_rack = tip_rack_384(name="tip_rack_384")
    self.deck.assign_child_resource(carrier, track=9)
    self.sent: List[str] = []
    log = self.driver._log_exchange

    def recorded(written: str, read: Optional[str]) -> None:
      if written[:4] in ("C0TT", "C0JB", "C0JC"):
        self.sent.append(written)
      log(written, read)

    self.driver._log_exchange = recorded  # type: ignore[method-assign]

  async def test_pick_up_and_drop_place_the_command_from_the_deck(self):
    await self.head.pick_up_tips(self.tip_rack)
    await self.head.drop_tips(self.tip_rack)
    await self.head.drop_tips(self.deck.get_trash_area96())
    self.assertEqual(
      self.sent,
      [
        "C0TTtt01tf0tl0424tv00650tg2tu0",
        "C0JBxs02945xd0yk1473tt01iu0je1982zf2450zg2450ii1",
        "C0JCxs02945xd0yk1473je1982zf2450zg2450jd0",
        "C0JCxs00465xd1yk1158je2164zf2450zg2450jd0",
      ],
    )

  async def test_the_rack_empties_onto_the_head_and_fills_again(self):
    spots = self.tip_rack.get_all_items()
    shafts = self.head_resource.get_all_items()
    tips = [spot.tip for spot in spots]

    await self.head.pick_up_tips(self.tip_rack)
    self.assertEqual([shaft.tip for shaft in shafts], tips)
    self.assertFalse(any(spot.has_tip() for spot in spots))

    await self.head.drop_tips(self.tip_rack)
    self.assertEqual([spot.tip for spot in spots], tips)
    self.assertFalse(any(shaft.has_tip() for shaft in shafts))

  async def test_tips_dropped_in_the_trash_belong_to_nothing(self):
    await self.head.pick_up_tips(self.tip_rack)
    tips = [shaft.tip for shaft in self.head_resource.get_all_items()]
    await self.head.drop_tips(self.deck.get_trash_area96())
    self.assertFalse(any(shaft.has_tip() for shaft in self.head_resource.get_all_items()))
    self.assertTrue(all(tip is not None and tip.parent is None for tip in tips))

  async def test_a_rack_of_the_wrong_size_is_refused_before_anything_is_sent(self):
    wrong = tip_rack_384(name="tip_rack_96", num_items_x=12, num_items_y=8)
    self.deck.assign_child_resource(wrong, location=Coordinate(700.0, 200.0, 100.0))
    with self.assertRaises(ValueError):
      await self.head.pick_up_tips(wrong)
    with self.assertRaises(ValueError):
      await self.head.drop_tips(wrong)
    self.assertEqual(self.sent, [])
