============
Architecture
============

PhysioTwin4D is organized around explicit workflow classes and smaller
registration, segmentation, geometry, and USD utilities that together create
personalized physiological digital twins from 3D medical images. Runtime
workflow classes inherit from :class:`PhysioTwin4DBase` for logging and common
runtime configuration.

Data Flow
=========

.. code-block:: text

   4D CT / time-series CT
          |
          v
   ConvertImage4DTo3D / ImageTools
          |
          v
   RegisterTimeSeriesImages
      |        |
      |        +--> RegisterImagesANTS / RegisterImagesGreedy / RegisterImagesICON
      |        +--> RegisterImagesGreedyICON / RegisterImagesChain (chained methods)
      |        +--> WorkflowFinetuneICONRegistration (finetune ICON on subject data)
      v
   SegmentChestTotalSegmentator / SegmentChestTotalSegmentatorWithContrast
   SegmentHeartSimpleware / SegmentHeartSimplewareTrimmedBranches
   SegmentNVSegmentCTMRI (CT + MRI)
          |
          v
   ContourTools + TransformTools
          |
          v
   WorkflowConvertImageToVTK / ConvertVTKToUSD / WorkflowConvertVTKToUSD
          |
          v
   OpenUSD assets for NVIDIA Omniverse

   Population meshes --> WorkflowCreateStatisticalModel (PCA shape model)
                              |
                              v
   Patient surfaces/image --> WorkflowFitStatisticalModelToPatient
                              |
                              v
   Fitted SSM propagated across gated phases (WorkflowReconstructHighres4DCT)
                              |
                              v
   PhysicsNeMo AI surrogate training (MeshGraphNet / MLP) and evaluation

Primary Workflows
=================

``WorkflowConvertImageToUSD``
   Converts a 4D cardiac CT file or 3D CT time series into registered anatomy
   contours and painted animated USD files.

``WorkflowConvertImageToVTK``
   Segments a 3D CT image and exports anatomy groups as VTK surfaces.

``WorkflowCreateMeanSurface``
   Builds an unbiased mean surface from a population, so a shape model can be
   built without privileging any one subject as the template.

``WorkflowCreateStatisticalModel``
   Aligns a population of meshes to a reference and builds a PCA statistical
   shape model.

``WorkflowFitStatisticalModelToPatient``
   Fits a template/statistical model to patient-specific surfaces with ICP,
   optional PCA fitting, labelmap-to-labelmap registration, and optional
   labelmap-to-image refinement.

``WorkflowReconstructHighres4DCT``
   Reconstructs higher-resolution 4D CT frames from a time series and a fixed
   high-resolution reference image.

``WorkflowFinetuneICONRegistration``
   Finetunes a uniGradICON checkpoint on subject-specific image/labelmap/
   landmark data and returns the path to the resulting weights, which
   :class:`RegisterImagesICON` can then load.

``WorkflowConvertVTKToUSD``
   Converts in-memory PyVista/VTK meshes to static or animated USD scenes
   through the supported workflow wrapper. The lower-level
   :mod:`physiotwin4d.vtk_to_usd` package exposes advanced file conversion
   primitives.

AI Surrogate Workflows (PhysicsNeMo)
=====================================

The final tier of tutorials (``tutorial_08`` through ``tutorial_10``) turns a
fitted statistical shape model into a trained AI physiological surrogate,
replacing the explicit per-phase registration solve with a learned model at
inference time:

``tutorial_08_lung_fit_model_to_4d_patients.py``
   Fits the lung PCA model to a DIR-Lab case (via
   ``WorkflowFitStatisticalModelToPatient``) and propagates the fitted surface
   through every respiratory phase using ICON-based registration
   (``WorkflowReconstructHighres4DCT``), producing the per-phase surfaces used
   as AI surrogate training data.

``tutorial_09_lung_train_physicsnemo_mgn.py``
   Trains a PhysicsNeMo surrogate with ``WorkflowTrainPhysicsNeMo``, driving a
   graph-based ``TrainPhysicsNeMoMGN`` (``MeshGraphNet``) method on the
   Tutorial 8 output. The tutorial writes each phase's training target — here
   the per-vertex displacement — into a mesh point-data array the manifest
   names, so the same stack trains on any per-point target of any width. It
   then evaluates the held-out cases with ``WorkflowInferPhysicsNeMo`` wrapped
   in ``WorkflowInferMovement``, turning predicted displacements back into
   surfaces without running registration — i.e. the AI surrogate stands in for
   ``WorkflowReconstructHighres4DCT`` at inference time. Requires the
   ``[physicsnemo]`` extra and ``torch-geometric``; Python >= 3.11.

``tutorial_10_lung_infer_physicsnemo_mgn.py``
   Loads that checkpoint and predicts the case's surface at a requested stage
   with ``WorkflowInferPhysicsNeMo`` plus ``WorkflowInferMovement``, then
   exports it as USD — one forward pass in place of the registration solve that
   produced the training data, and able to predict stages that were never
   acquired.

These tutorials are thin drivers over the ``WorkflowTrainPhysicsNeMo`` /
``WorkflowInferPhysicsNeMo`` workflow classes; each workflow owns the data side
(manifests, normalization, datasets, saving) and delegates the network to a
method class (``TrainPhysicsNeMoBase`` / ``InferPhysicsNeMoBase`` subclass).
The manifest is the extension point: it names the point-data array holding the
targets, which the stack reads verbatim, so the same code trains on
displacement or on any other per-point quantity. See
:doc:`api/physicsnemo/index`.

They follow the same fit -> propagate -> train -> predict pattern the rest of
the workflow layer uses, and are the intended template for future cardiac,
respiratory, and electrophysiology AI surrogates.

Component Boundaries
====================

Segmentation classes produce anatomy masks or labelmaps from ITK images.
``SegmentAnatomyBase`` subclasses (``SegmentChestTotalSegmentator``,
``SegmentChestTotalSegmentatorWithContrast``, ``SegmentHeartSimpleware``,
``SegmentHeartSimplewareTrimmedBranches``, ``SegmentNVSegmentCTMRI``) share the
same segment/taxonomy interface, so new segmentation methods or anatomy groups
slot in without touching the workflow layer.

Deriving from a base class propagates capability, not just interface. Each
``SegmentAnatomyBase`` subclass owns an :class:`AnatomyTaxonomy` instance and
declares its own group→organ label map by calling
``self.taxonomy.add_organ(group_name, label_id, organ_name)`` for every label
it produces — a new segmenter for a new organ or data type only has to
declare that map once. Everything downstream reads it rather than
special-casing the segmenter: ``ConvertVTKToUSD`` groups label-mode mesh
prims under per-anatomy-group Xforms (``/World/{name}/{group}/{organ}``)
straight from the taxonomy, and ``USDAnatomyTools`` looks up
:data:`DEFAULT_RENDER_PARAMS` by group name to assign the matching
OmniSurface material. A group without a registered look still renders (via
the ``"other"`` fallback), so a new segmentation class is usable end-to-end —
segmented, meshed, grouped, and materialized — before anyone writes a custom
render style for it.

Registration classes produce ITK transforms or transformed meshes.
``RegisterImagesBase`` subclasses (``RegisterImagesANTS``,
``RegisterImagesGreedy``, ``RegisterImagesICON``) implement a single
registration method; ``RegisterImagesChain`` and ``RegisterImagesGreedyICON``
compose two registrars into a coarse-to-fine chain. Model-to-model/image
registration (``RegisterModelsICP``, ``RegisterModelsICPITK``,
``RegisterModelsPCA``, ``RegisterModelsDistanceMaps``) shares an analogous
base-class boundary so new surface- or shape-based registration methods can
be added the same way.

Geometry utilities bridge ITK masks and PyVista meshes. USD tools are
responsible for OpenUSD stage creation, material assignment, coordinate
conversion, and time samples.

The high-risk boundary is the ITK-to-PyVista-to-USD path. Image data remains in
ITK's native LPS world space until contours are extracted. Meshes are
represented as PyVista objects (still in LPS) before USD export. The VTK-to-USD
layer applies the repository's LPS-to-USD-Y-up coordinate transform during USD
conversion.

CLI Boundary
============

The installed CLI commands in ``pyproject.toml`` are thin wrappers around the
workflow classes. They are the preferred examples for executable API usage:

* ``physiotwin4d-convert-image-4d-to-3d``
* ``physiotwin4d-convert-image-to-usd``
* ``physiotwin4d-convert-image-to-vtk``
* ``physiotwin4d-convert-vtk-to-usd``
* ``physiotwin4d-create-statistical-model``
* ``physiotwin4d-download-data``
* ``physiotwin4d-fit-statistical-model-to-patient``
* ``physiotwin4d-infer-physicsnemo``
* ``physiotwin4d-reconstruct-highres-4d-ct``
* ``physiotwin4d-train-physicsnemo``
* ``physiotwin4d-visualize-pca-modes``

``physiotwin4d-train-physicsnemo`` and ``physiotwin4d-infer-physicsnemo`` wrap
``WorkflowTrainPhysicsNeMo`` and ``WorkflowInferPhysicsNeMo`` and need the
optional ``[physicsnemo]`` extra. There is no CLI wrapper for
``WorkflowFinetuneICONRegistration``; it is used through the Python API and
tutorial scripts.
