"""Shared parameters for the heart CT tutorials.

Mirrors :mod:`parameters_lung_ct_dirlab` for the heart use cases, and carries
different values: the heart is registered with a much tighter mask than the
lungs, so its distance maps saturate over a correspondingly shorter radius.
That is why the heart has its own distance-map finetuning tutorial rather than
reusing the lung one's weights -- the two organs' distance maps do not share an
intensity distribution.

Paths stay out of this module: each tutorial owns its own inputs and outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from physiotwin4d import SegmentAnatomyBase, SegmentChestTotalSegmentator


@dataclass(frozen=True)
class ParametersHeartCTKCL:
    """Settings shared by the heart tutorials.

    Attributes:
        mask_dilation_mm: Dilation of the binary registration masks, in
            millimeters.  Tighter than the lungs': the heart is a compact organ
            whose neighbours are not part of the model.
        distancemap_squared_max: Saturation radius of every heart distance map,
            in squared millimeters.  Fixes their intensity distribution, so the
            finetuning tutorial and every tutorial that registers heart distance
            maps must use this one value.
        number_of_pca_components: PCA components retained when building the
            heart statistical model, and used when fitting it to a patient.
        number_of_pca_components_test: Same, under ``TestTools.running_as_test``.
        number_of_iterations_greedy: Greedy coarse-to-fine iteration schedule.
        number_of_iterations_greedy_test: Same, under
            ``TestTools.running_as_test``.
        segmenter_class: Segmenter every heart tutorial instantiates, so the
            surfaces they compare share a definition of "heart".
        anatomy_group: Anatomy group name that segmenter registers for the heart.
        interior_object_ids_totalsegmentator: Chamber labels in a
            TotalSegmentator labelmap.  The chambers are interior to the
            myocardium, so a distance map must not measure to them.
        interior_object_ids_simpleware: The same chambers in a Simpleware
            ASCardio labelmap, which the Duke-Heart-4DLabelmaps data was
            segmented with.  Which list applies is a property of the data a
            tutorial reads, not of any one segmenter class, so both live here.
    """

    mask_dilation_mm: float = 10.0
    distancemap_squared_max: float = (1.25 * 10.0) ** 2

    number_of_pca_components: int = 10
    number_of_pca_components_test: int = 5

    number_of_iterations_greedy: list[int] = field(
        default_factory=lambda: [30, 15, 7, 3]
    )
    number_of_iterations_greedy_test: list[int] = field(default_factory=lambda: [1, 0])

    segmenter_class: type[SegmentAnatomyBase] = SegmentChestTotalSegmentator
    anatomy_group: str = "heart"
    interior_object_ids_totalsegmentator: Optional[list[int]] = field(
        default_factory=lambda: [141, 142, 143, 144]
    )
    interior_object_ids_simpleware: list[int] = field(
        default_factory=lambda: [1, 2, 3, 4]
    )

    def pca_components(self, test_mode: bool) -> int:
        """Return the PCA component count for this run mode."""
        return (
            self.number_of_pca_components_test
            if test_mode
            else (self.number_of_pca_components)
        )

    def greedy_iterations(self, test_mode: bool) -> list[int]:
        """Return the Greedy iteration schedule for this run mode."""
        return list(
            self.number_of_iterations_greedy_test
            if test_mode
            else self.number_of_iterations_greedy
        )


#: The single instance every heart tutorial imports.
HEART_CT_KCL = ParametersHeartCTKCL()
