"""Unit tests for the named 21-point hand topology.

The index mapping is a contract with the model, and it is the one place in the package
where model-specific numbers are allowed. If it drifts, every derived measurement is
subtly wrong, so it is pinned exactly rather than spot-checked.
"""

from __future__ import annotations

import dataclasses

import pytest

from gesturepilot.processing.topology import (
    FINGER_CHAINS,
    FINGERS,
    HAND_SCALE_LANDMARKS,
    PALM_LANDMARKS,
    Finger,
    FingerChain,
    LandmarkIndex,
)
from gesturepilot.tracking.types import LANDMARK_COUNT

#: The layout the whole package assumes. Written out in full on purpose: a test that
#: derived its expectations from the enum would agree with any change, including a wrong
#: one.
EXPECTED_INDICES = {
    "WRIST": 0,
    "THUMB_CMC": 1,
    "THUMB_MCP": 2,
    "THUMB_IP": 3,
    "THUMB_TIP": 4,
    "INDEX_MCP": 5,
    "INDEX_PIP": 6,
    "INDEX_DIP": 7,
    "INDEX_TIP": 8,
    "MIDDLE_MCP": 9,
    "MIDDLE_PIP": 10,
    "MIDDLE_DIP": 11,
    "MIDDLE_TIP": 12,
    "RING_MCP": 13,
    "RING_PIP": 14,
    "RING_DIP": 15,
    "RING_TIP": 16,
    "PINKY_MCP": 17,
    "PINKY_PIP": 18,
    "PINKY_DIP": 19,
    "PINKY_TIP": 20,
}


class TestLandmarkIndex:
    def test_every_landmark_has_the_expected_index(self) -> None:
        assert {member.name: int(member) for member in LandmarkIndex} == EXPECTED_INDICES

    def test_covers_the_whole_topology_without_gaps(self) -> None:
        assert sorted(int(member) for member in LandmarkIndex) == list(range(LANDMARK_COUNT))

    def test_indexes_a_landmark_sequence_directly(self) -> None:
        landmarks = tuple(object() for _ in range(LANDMARK_COUNT))
        assert landmarks[LandmarkIndex.MIDDLE_TIP] is landmarks[12]

    def test_members_compare_as_plain_integers(self) -> None:
        assert LandmarkIndex.WRIST == 0


class TestFinger:
    def test_declares_exactly_the_five_digits(self) -> None:
        assert {finger.name for finger in Finger} == {"THUMB", "INDEX", "MIDDLE", "RING", "PINKY"}

    def test_declaration_order_is_canonical(self) -> None:
        assert FINGERS == (
            Finger.THUMB,
            Finger.INDEX,
            Finger.MIDDLE,
            Finger.RING,
            Finger.PINKY,
        )


class TestFingerChain:
    def test_there_is_one_chain_per_finger_in_canonical_order(self) -> None:
        assert tuple(chain.finger for chain in FINGER_CHAINS) == FINGERS

    def test_every_finger_is_rooted_at_the_wrist(self) -> None:
        assert {chain.root for chain in FINGER_CHAINS} == {LandmarkIndex.WRIST}

    def test_index_chain_uses_the_expected_landmarks(self) -> None:
        index = next(chain for chain in FINGER_CHAINS if chain.finger is Finger.INDEX)
        assert index.joints == (
            LandmarkIndex.INDEX_MCP,
            LandmarkIndex.INDEX_PIP,
            LandmarkIndex.INDEX_DIP,
            LandmarkIndex.INDEX_TIP,
        )

    def test_thumb_chain_uses_the_expected_landmarks(self) -> None:
        thumb = next(chain for chain in FINGER_CHAINS if chain.finger is Finger.THUMB)
        assert thumb.joints == (
            LandmarkIndex.THUMB_CMC,
            LandmarkIndex.THUMB_MCP,
            LandmarkIndex.THUMB_IP,
            LandmarkIndex.THUMB_TIP,
        )

    def test_polylines_are_five_points_each(self) -> None:
        for chain in FINGER_CHAINS:
            assert len(chain.polyline) == 5

    def test_every_polyline_is_a_strictly_descending_path(self) -> None:
        """Index order is the topological order, so no chain may double back."""
        for chain in FINGER_CHAINS:
            polyline = [int(index) for index in chain.polyline]
            assert polyline == sorted(polyline), chain.finger

    def test_each_chain_defines_four_segments(self) -> None:
        for chain in FINGER_CHAINS:
            assert len(chain.segment_pairs) == 4

    def test_segments_connect_the_polyline_without_gaps(self) -> None:
        for chain in FINGER_CHAINS:
            expected = [(chain.polyline[i], chain.polyline[i + 1]) for i in range(4)]
            assert list(chain.segment_pairs) == expected

    def test_tip_is_the_last_joint(self) -> None:
        for chain in FINGER_CHAINS:
            assert chain.tip == chain.joints[3]

    def test_rejects_a_short_joint_tuple(self) -> None:
        with pytest.raises(ValueError, match="exactly 4 joints"):
            FingerChain(
                finger=Finger.INDEX,
                root=LandmarkIndex.WRIST,
                joints=(LandmarkIndex.INDEX_MCP, LandmarkIndex.INDEX_PIP),  # type: ignore[arg-type]
            )

    def test_rejects_a_repeated_joint(self) -> None:
        with pytest.raises(ValueError, match="repeats a landmark"):
            FingerChain(
                finger=Finger.INDEX,
                root=LandmarkIndex.WRIST,
                joints=(
                    LandmarkIndex.INDEX_MCP,
                    LandmarkIndex.INDEX_PIP,
                    LandmarkIndex.INDEX_PIP,
                    LandmarkIndex.INDEX_TIP,
                ),
            )

    def test_rejects_rooting_on_one_of_its_own_joints(self) -> None:
        with pytest.raises(ValueError, match="roots on one of its own joints"):
            FingerChain(
                finger=Finger.INDEX,
                root=LandmarkIndex.INDEX_MCP,
                joints=(
                    LandmarkIndex.INDEX_MCP,
                    LandmarkIndex.INDEX_PIP,
                    LandmarkIndex.INDEX_DIP,
                    LandmarkIndex.INDEX_TIP,
                ),
            )

    def test_is_frozen(self) -> None:
        chain = FINGER_CHAINS[1]
        with pytest.raises(dataclasses.FrozenInstanceError):
            chain.finger = Finger.THUMB  # type: ignore[misc]


class TestTopologyConstants:
    def test_palm_landmarks_are_the_wrist_and_four_knuckles(self) -> None:
        assert PALM_LANDMARKS == (
            LandmarkIndex.WRIST,
            LandmarkIndex.INDEX_MCP,
            LandmarkIndex.MIDDLE_MCP,
            LandmarkIndex.RING_MCP,
            LandmarkIndex.PINKY_MCP,
        )

    def test_hand_scale_is_measured_from_the_wrist_to_the_middle_knuckle(self) -> None:
        assert HAND_SCALE_LANDMARKS == (LandmarkIndex.WRIST, LandmarkIndex.MIDDLE_MCP)

    def test_all_indices_are_in_range(self) -> None:
        every = (*PALM_LANDMARKS, *HAND_SCALE_LANDMARKS)
        assert all(0 <= int(index) < LANDMARK_COUNT for index in every)
