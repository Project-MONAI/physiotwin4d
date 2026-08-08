"""Shared parameters for the DIR-Lab 4D CT lung tutorials.

Every lung tutorial reads its settings from :data:`LUNG_CT_DIRLAB` so the
distance maps one tutorial finetunes ICON on are rasterized exactly the way the
tutorials that later register them rasterize theirs.  A saturation radius or a
dilation that drifts between two of these scripts silently trains on one image
distribution and infers on another.

Paths stay out of this module: each tutorial owns its own inputs and outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from physiotwin4d import SegmentAnatomyBase, SegmentNVSegmentCTMRI


@dataclass(frozen=True)
class ParametersLungCTDirLab:
    """Settings shared by the DIR-Lab lung tutorials.

    Attributes:
        mask_dilation_mm: Dilation of the binary registration masks, in
            millimeters.  Also sets how far outside the lung surface the
            registration is allowed to look.
        distancemap_squared_max: Saturation radius of every lung distance map,
            in squared millimeters.  Fixes their intensity distribution, so the
            finetuning tutorial and every tutorial that registers lung distance
            maps must use this one value.
        number_of_pca_components: PCA components retained when building the
            lung statistical model, and used when fitting it to a patient.
        number_of_pca_components_test: Same, under ``TestTools.running_as_test``.
        number_of_iterations_greedy: Greedy coarse-to-fine iteration schedule.
        number_of_iterations_greedy_test: Same, under
            ``TestTools.running_as_test``.
        segmenter_class: Segmenter every lung tutorial instantiates, so the
            surfaces they compare share a definition of "lung".
        anatomy_group: Anatomy group name that segmenter registers for lungs.

    There is no interior-structure list here, the counterpart of the heart's
    chamber ids: the lung labels are the lobes, and every one of them is on the
    surface a distance map is measured to.
    """

    mask_dilation_mm: float = 40.0
    distancemap_squared_max: float = (1.25 * 40.0) ** 2

    number_of_pca_components: int = 6
    number_of_pca_components_test: int = 5

    number_of_iterations_greedy: list[int] = field(
        default_factory=lambda: [30, 15, 7, 3]
    )
    number_of_iterations_greedy_test: list[int] = field(default_factory=lambda: [1, 0])

    segmenter_class: type[SegmentAnatomyBase] = SegmentNVSegmentCTMRI
    anatomy_group: str = "lung"

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


#: The single instance every lung tutorial imports.
LUNG_CT_DIRLAB = ParametersLungCTDirLab()
