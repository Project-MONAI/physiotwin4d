"""Composite registration: run multiple registrars in sequence.

This module provides the RegisterImagesChain class, which combines an
ordered list of RegisterImagesBase instances into a single multi-stage
registration pipeline.
"""

import logging
from typing import Optional, Union, cast

import itk

from .register_images_base import RegisterImagesBase


class RegisterImagesChain(RegisterImagesBase):
    """Run an ordered list of registrars in sequence, feeding each stage's
    forward_transform as the next stage's initial_forward_transform.

    Use this to combine independent registration backends into a multi-stage
    pipeline (e.g. a fast coarse registrar followed by a refinement stage).
    Every element of ``registrars`` must be a :class:`RegisterImagesBase`
    instance. ``registrars`` (plural, a list) is distinct from the singular
    ``registrar`` attribute used by classes like
    :class:`RegisterTimeSeriesImages`.

    See :class:`RegisterImagesGreedyICON` for a named 2-stage convenience
    subclass (Greedy followed by ICON refinement).

    Example:
        >>> chain = RegisterImagesChain([RegisterImagesGreedy(), RegisterImagesICON()])
        >>> chain.set_fixed_image(fixed_image)
        >>> result = chain.register(moving_image)
    """

    def __init__(
        self, registrars: list[RegisterImagesBase], log_level: int | str = logging.INFO
    ) -> None:
        """Initialize the registration chain.

        Args:
            registrars: Ordered, non-empty list of RegisterImagesBase
                instances to run in sequence.
            log_level: Logging level (default: logging.INFO)

        Raises:
            ValueError: If registrars is empty.
            TypeError: If any element of registrars is not a
                RegisterImagesBase instance.
        """
        super().__init__(log_level=log_level)

        if not registrars:
            raise ValueError("registrars must not be empty")
        for registrar in registrars:
            if not isinstance(registrar, RegisterImagesBase):
                raise TypeError(
                    "Every element of registrars must be a RegisterImagesBase "
                    f"instance, got {type(registrar).__name__}"
                )

        self.registrars = registrars

    def registration_method(
        self,
        moving_image: itk.Image,
        moving_mask: Optional[itk.Image] = None,
        moving_labelmap: Optional[itk.Image] = None,
        moving_image_pre: Optional[itk.Image] = None,
        initial_forward_transform: Optional[itk.Transform] = None,
    ) -> dict[str, Union[itk.Transform, float]]:
        """Run each registrar in ``self.registrars`` in order.

        Each stage's ``forward_transform`` becomes the next stage's
        ``initial_forward_transform``.

        Note:
            ``moving_image_pre`` is ignored: each stage may need different
            intensity preprocessing (e.g. ICON's uniGradICON preprocessing
            vs. Greedy's no-op), so every stage computes its own
            preprocessing from the raw ``moving_image`` rather than reuse a
            value computed for a different backend.

        Args:
            moving_image (itk.image): The 3D image to be registered
            moving_mask (itk.image, optional): Binary mask for moving image ROI
            moving_labelmap (itk.image, optional): Multi-label segmentation
                for the moving image
            moving_image_pre (itk.image, optional): Ignored - see Note above
            initial_forward_transform (itk.Transform, optional): Initial
                transformation from moving to fixed, used to initialize the
                first stage

        Returns:
            dict: The last stage's result dict (see :meth:`RegisterImagesBase.register`)
        """
        current_initial = initial_forward_transform
        result: dict[str, Union[itk.Transform, float]] = {}
        for registrar in self.registrars:
            self._delegate_to(registrar, moving_image, moving_mask, moving_labelmap)
            result = registrar.registration_method(
                moving_image,
                moving_mask=moving_mask,
                moving_labelmap=moving_labelmap,
                moving_image_pre=None,
                initial_forward_transform=current_initial,
            )
            self._capture_delegate_result(registrar, result)
            current_initial = cast(itk.Transform, result["forward_transform"])

        return result
