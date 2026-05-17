"""
PhysioMotion4D - Medical imaging package for generating anatomic models with
    physiological motion.

This package converts 4D CT scans (particularly heart and lung gated CT data) into
    dynamic 3D models
for visualization in NVIDIA Omniverse. It provides comprehensive tools for image
    processing,
segmentation, registration, and USD file generation.

Main Components:
    - WorkflowConvertImageToUSD: 4D CT image to USD workflow
    - Segmentation classes: Multiple AI-based chest segmentation implementations
    - Registration tools: Deep learning-based image registration
    - Transform utilities: Tools for image and contour transformations
    - USD tools: Utilities for Omniverse integration
    - PhysioMotion4DBase: Base class with standardized logging and debug settings
"""

__version__ = "2026.05.8"

import warnings as _warnings

try:
    import cupy as _cupy  # noqa: F401
except ImportError:
    _warnings.warn(
        "CuPy is not installed — GPU acceleration is unavailable and processing "
        "will be slow. Re-install with uv to get CuPy and CUDA-enabled PyTorch "
        "in one step (pip alone will not select the correct CUDA wheel):\n"
        "  uv pip install 'physiomotion4d[cuda13]'  # CUDA 13",
        UserWarning,
        stacklevel=2,
    )

from .anatomy_taxonomy import AnatomyGroup, AnatomyTaxonomy
from .contour_tools import ContourTools

# Data processing utilities
from .convert_image_4d_to_3d import ConvertImage4DTo3D
from .convert_vtk_to_usd import ConvertVTKToUSD
from .data_download_tools import DataDownloadTools

# Utility classes
from .image_tools import ImageTools

# Base classes
from .physiomotion4d_base import PhysioMotion4DBase
from .register_images_ants import RegisterImagesANTs
from .register_images_greedy import RegisterImagesGreedy

# Registration classes
from .register_images_base import RegisterImagesBase
from .register_images_icon import RegisterImagesICON
from .register_models_distance_maps import RegisterModelsDistanceMaps
from .register_models_icp import RegisterModelsICP
from .register_models_icp_itk import RegisterModelsICPITK
from .register_models_pca import RegisterModelsPCA
from .register_time_series_images import RegisterTimeSeriesImages

# Segmentation classes
from .segment_anatomy_base import SegmentAnatomyBase
from .segment_chest_total_segmentator import SegmentChestTotalSegmentator
from .segment_heart_simpleware import SegmentHeartSimpleware
from .test_tools import TestTools
from .transform_tools import TransformTools
from .usd_anatomy_tools import USDAnatomyTools
from .usd_tools import USDTools

# Core workflow processor
from .workflow_convert_image_to_vtk import WorkflowConvertImageToVTK
from .workflow_convert_image_to_usd import WorkflowConvertImageToUSD
from .workflow_convert_vtk_to_usd import WorkflowConvertVTKToUSD
from .workflow_reconstruct_highres_4d_ct import WorkflowReconstructHighres4DCT
from .workflow_create_statistical_model import WorkflowCreateStatisticalModel
from .workflow_fit_statistical_model_to_patient import (
    WorkflowFitStatisticalModelToPatient,
)

__all__ = [
    # Workflow classes
    "WorkflowConvertImageToVTK",
    "WorkflowConvertImageToUSD",
    "WorkflowConvertVTKToUSD",
    "WorkflowCreateStatisticalModel",
    "WorkflowReconstructHighres4DCT",
    "WorkflowFitStatisticalModelToPatient",
    # Segmentation classes
    "SegmentAnatomyBase",
    "SegmentChestTotalSegmentator",
    "SegmentHeartSimpleware",
    # Registration classes
    "RegisterImagesBase",
    "RegisterImagesICON",
    "RegisterImagesANTs",
    "RegisterImagesGreedy",
    "RegisterTimeSeriesImages",
    "RegisterModelsPCA",
    "RegisterModelsICP",
    "RegisterModelsICPITK",
    "RegisterModelsDistanceMaps",
    # Base classes
    "PhysioMotion4DBase",
    # Utility classes
    "ImageTools",
    "TestTools",
    "TransformTools",
    "USDTools",
    "ContourTools",
    "USDAnatomyTools",
    "DataDownloadTools",
    # Data processing utilities
    "ConvertImage4DTo3D",
    "ConvertVTKToUSD",
    # Anatomy taxonomy (shared between segmenters and USD renderer)
    "AnatomyTaxonomy",
    "AnatomyGroup",
]
