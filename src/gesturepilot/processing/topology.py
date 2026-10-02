"""Named indices for the standard 21-point hand topology.

The tracking layer deliberately hands landmarks over in the model's own order and
refuses to attach finger semantics to them, because that would couple it to the model
file. Landmark *processing* is the right place for that knowledge: it needs to know
which index is a knuckle, and it needs to know it in exactly one place.

This module is pure data. It imports nothing from the rest of GesturePilot, holds no
MediaPipe type, and never mentions the backend's class names. Only the integer layout
is mirrored, because that layout is the shared vocabulary between the model and every
later stage.

Topology (MediaPipe hand-landmarker, 21 points)::

        0  WRIST
        |
        |  4 THUMB_TIP
        |  /
        | 3 THUMB_IP
        | /
        | 2 THUMB_MCP                  5 INDEX_MCP   9 MIDDLE_MCP   13 RING_MCP   17 PINKY_MCP
        |/                              6 INDEX_PIP  10 MIDDLE_PIP  14 RING_PIP  18 PINKY_PIP
    1 THUMB_CMC                       7 INDEX_DIP  11 MIDDLE_DIP  15 RING_DIP  19 PINKY_DIP
                                     8 INDEX_TIP  12 MIDDLE_TIP  16 RING_TIP  20 PINKY_TIP
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum


class Finger(Enum):
    """The five digits, in a fixed order.

    The declaration order *is* the canonical order used by
    :attr:`~gesturepilot.processing.types.HandFeatures.fingers`, so callers can rely on
    ``tuple(Finger)`` to line up with that tuple without sorting.
    """

    THUMB = "thumb"
    INDEX = "index"
    MIDDLE = "middle"
    RING = "ring"
    PINKY = "pinky"


class LandmarkIndex(IntEnum):
    """Index of each landmark in the standard 21-point topology.

    ``IntEnum`` rather than bare integers so the values index directly — ``landmarks
    [LandmarkIndex.WRIST]`` — while still reading as a name at every call site. No
    magic numbers appear anywhere else in this package.
    """

    WRIST = 0

    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4

    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8

    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12

    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16

    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


@dataclass(frozen=True)
class FingerChain:
    """One finger as an ordered polyline of landmarks.

    A finger is modelled as ``root -> base -> middle -> distal -> tip``: five points
    forming four segments and three interior joints. The thumb is expressed the same
    way by rooting its chain at the wrist, so no finger needs a special case.

    Attributes:
        finger: Which digit this chain describes.
        root: The landmark the first segment starts from — the wrist for every finger,
            including the thumb, whose first bone runs from the wrist to the thumb
            carpometacarpal joint.
        joints: ``(base, middle, distal, tip)`` in proximal-to-distal order. The thumb
            uses ``(CMC, MCP, IP, TIP)``; the other fingers use ``(MCP, PIP, DIP, TIP)``.
    """

    finger: Finger
    root: LandmarkIndex
    joints: tuple[LandmarkIndex, LandmarkIndex, LandmarkIndex, LandmarkIndex]

    def __post_init__(self) -> None:
        if len(self.joints) != 4:
            raise ValueError(f"A finger chain needs exactly 4 joints, got {len(self.joints)}")
        if len(set(self.joints)) != 4:
            raise ValueError(f"Finger chain {self.finger.name} repeats a landmark index")
        if self.root in self.joints:
            raise ValueError(f"Finger chain {self.finger.name} roots on one of its own joints")

    @property
    def polyline(
        self,
    ) -> tuple[LandmarkIndex, LandmarkIndex, LandmarkIndex, LandmarkIndex, LandmarkIndex]:
        """The chain as ``(root, base, middle, distal, tip)``."""
        return (self.root, *self.joints)

    @property
    def segment_pairs(
        self,
    ) -> tuple[
        tuple[LandmarkIndex, LandmarkIndex],
        tuple[LandmarkIndex, LandmarkIndex],
        tuple[LandmarkIndex, LandmarkIndex],
        tuple[LandmarkIndex, LandmarkIndex],
    ]:
        """The four bone segments as ``(from, to)`` landmark pairs."""
        root, base, middle, distal, tip = self.polyline
        return ((root, base), (base, middle), (middle, distal), (distal, tip))

    @property
    def tip(self) -> LandmarkIndex:
        """The fingertip landmark."""
        return self.joints[3]


FINGER_CHAINS: tuple[FingerChain, ...] = (
    FingerChain(
        finger=Finger.THUMB,
        root=LandmarkIndex.WRIST,
        joints=(
            LandmarkIndex.THUMB_CMC,
            LandmarkIndex.THUMB_MCP,
            LandmarkIndex.THUMB_IP,
            LandmarkIndex.THUMB_TIP,
        ),
    ),
    FingerChain(
        finger=Finger.INDEX,
        root=LandmarkIndex.WRIST,
        joints=(
            LandmarkIndex.INDEX_MCP,
            LandmarkIndex.INDEX_PIP,
            LandmarkIndex.INDEX_DIP,
            LandmarkIndex.INDEX_TIP,
        ),
    ),
    FingerChain(
        finger=Finger.MIDDLE,
        root=LandmarkIndex.WRIST,
        joints=(
            LandmarkIndex.MIDDLE_MCP,
            LandmarkIndex.MIDDLE_PIP,
            LandmarkIndex.MIDDLE_DIP,
            LandmarkIndex.MIDDLE_TIP,
        ),
    ),
    FingerChain(
        finger=Finger.RING,
        root=LandmarkIndex.WRIST,
        joints=(
            LandmarkIndex.RING_MCP,
            LandmarkIndex.RING_PIP,
            LandmarkIndex.RING_DIP,
            LandmarkIndex.RING_TIP,
        ),
    ),
    FingerChain(
        finger=Finger.PINKY,
        root=LandmarkIndex.WRIST,
        joints=(
            LandmarkIndex.PINKY_MCP,
            LandmarkIndex.PINKY_PIP,
            LandmarkIndex.PINKY_DIP,
            LandmarkIndex.PINKY_TIP,
        ),
    ),
)

#: Fingers in canonical order, matching the order of :data:`FINGER_CHAINS`.
FINGERS: tuple[Finger, ...] = tuple(chain.finger for chain in FINGER_CHAINS)

#: Landmarks averaged to form the palm centre: the wrist and the four finger knuckles.
PALM_LANDMARKS: tuple[LandmarkIndex, ...] = (
    LandmarkIndex.WRIST,
    LandmarkIndex.INDEX_MCP,
    LandmarkIndex.MIDDLE_MCP,
    LandmarkIndex.RING_MCP,
    LandmarkIndex.PINKY_MCP,
)

#: The pair whose image-plane separation defines :func:`gesturepilot.processing.geometry.hand_scale`.
#:
#: The wrist-to-middle-knuckle span is preferred over a knuckle-to-knuckle width because it
#: is the longest stable bone in the palm. Finger *tips* move a great deal as a hand opens
#: and closes, so any scale reference built from them would drift with the very motion a
#: classifier is trying to measure.
HAND_SCALE_LANDMARKS: tuple[LandmarkIndex, LandmarkIndex] = (
    LandmarkIndex.WRIST,
    LandmarkIndex.MIDDLE_MCP,
)

__all__ = [
    "FINGER_CHAINS",
    "FINGERS",
    "HAND_SCALE_LANDMARKS",
    "PALM_LANDMARKS",
    "Finger",
    "FingerChain",
    "LandmarkIndex",
]
