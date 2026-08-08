=========
Tutorials
=========

.. raw:: html

   <section class="pt4d-hero">
     <div class="pt4d-hero__brand">
       <img src="_static/nvidia-logo.svg" alt="NVIDIA logo">
     </div>
     <p class="pt4d-kicker">PhysioTwin4D tutorials</p>
     <h1>From a CT scan to an animated digital twin</h1>
     <p>
       Ten numbered stages across 17 runnable Python scripts.
       Each one drives the real workflow classes end-to-end on downloadable
       data, shows what it produced, and ends with the handful of constants
       to change so it runs on your own scans.
     </p>
   </section>

Before You Start
================

**1. Get the scripts.** They ship with the source repository, not with the pip
package — ``pip install physiotwin4d`` gives you the library and the
``physiotwin4d-*`` commands but no ``tutorials/`` directory:

.. code-block:: bash

   git clone https://github.com/Project-MONAI/physiotwin4d.git
   cd physiotwin4d

See :doc:`quickstart` for version-matched clones and the release tarball link.

**2. Get the data**, running every download from the top level of the clone.
The tutorials resolve their inputs against the repository root
(``<repo>/data/<dataset>``), while the CLI writes to ``data/<dataset>``
relative to the current working directory:

.. code-block:: bash

   physiotwin4d-download-data Slicer-Heart-CT --directory data/Slicer-Heart-CT
   physiotwin4d-download-data KCL-Heart-Model --directory data/KCL-Heart-Model
   physiotwin4d-download-data Chest-CT --directory data/Chest-CT

That covers Heart Tutorials 1, 3, 4 and 6 (``Slicer-Heart-CT`` and
``KCL-Heart-Model``) and Lung Tutorial 7 (``Chest-CT``). ``DirLab-4DCT`` — used
by Lung Tutorials 1, 2, 3, 4, 6 and 8, and by Heart Tutorial 7 — is **not**
auto-downloaded: DIR-Lab distributes each case individually and may require
registration. Tutorials 5, 9 and 10 need no dataset of their own; they consume
the outputs of Tutorials 4, 8 and 9. See ``data/DirLab-4DCT/README.md``, and
:doc:`cli_scripts/download_data` for every dataset's size and source.

**3. Know where output lands.** Every tutorial writes to
``tutorials/output/<tutorial_name>/`` and reuses what it finds there, so a
second run is cheap and later tutorials pick up earlier results automatically.

.. raw:: html

   <section class="pt4d-card-grid" aria-label="Tutorial cards">
     <a class="pt4d-card" href="#tutorial-1-gated-4d-ct-to-animated-usd">
       <span class="pt4d-card__number">01</span>
       <h2>Gated 4D CT to Animated USD</h2>
       <p>Segment, register and assemble a 4D CT series into an animated OpenUSD scene.</p>
       <span class="pt4d-card__meta">Slicer-Heart-CT &middot; DIR-Lab</span>
     </a>
     <a class="pt4d-card" href="#tutorial-2-finetune-icon-registration">
       <span class="pt4d-card__number">02</span>
       <h2>Finetune ICON Registration</h2>
       <p>Adapt uniGradICON to your own cohort and measure what the finetuning bought you.</p>
       <span class="pt4d-card__meta">DIR-Lab</span>
     </a>
     <a class="pt4d-card" href="#tutorial-3-reconstruct-high-resolution-4d-ct">
       <span class="pt4d-card__number">03</span>
       <h2>Reconstruct High-Resolution 4D CT</h2>
       <p>Register every phase to one reference and reconstruct the series at its resolution.</p>
       <span class="pt4d-card__meta">Slicer-Heart-CT &middot; DIR-Lab</span>
     </a>
     <a class="pt4d-card" href="#tutorial-4-ct-segmentation-to-vtk-surfaces">
       <span class="pt4d-card__number">04</span>
       <h2>CT Segmentation to VTK Surfaces</h2>
       <p>Segment one CT phase and export patient anatomy as VTK PolyData surfaces.</p>
       <span class="pt4d-card__meta">Slicer-Heart-CT &middot; DIR-Lab</span>
     </a>
     <a class="pt4d-card" href="#tutorial-5-vtk-surfaces-to-animated-usd">
       <span class="pt4d-card__number">05</span>
       <h2>VTK Surfaces to Animated USD</h2>
       <p>Convert meshes into a time-sampled USD scene for Omniverse playback.</p>
       <span class="pt4d-card__meta">Tutorial 4 output</span>
     </a>
     <a class="pt4d-card" href="#tutorial-6-create-a-pca-shape-model">
       <span class="pt4d-card__number">06</span>
       <h2>Create a PCA Shape Model</h2>
       <p>Turn a population of meshes into a statistical shape model and its modes.</p>
       <span class="pt4d-card__meta">KCL-Heart-Model &middot; DIR-Lab</span>
     </a>
     <a class="pt4d-card" href="#tutorial-7-fit-the-shape-model-to-a-patient">
       <span class="pt4d-card__number">07</span>
       <h2>Fit the Shape Model to a Patient</h2>
       <p>Fit the shape model to one routine clinical scan, PCA coefficients and all.</p>
       <span class="pt4d-card__meta">Chest-CT &middot; Tutorial 6 output</span>
     </a>
     <a class="pt4d-card" href="#tutorial-8-propagate-the-shape-model-through-4d">
       <span class="pt4d-card__number">08</span>
       <h2>Propagate the Model Through 4D</h2>
       <p>Fit each case at its reference phase and carry the mesh through every phase.</p>
       <span class="pt4d-card__meta">DIR-Lab &middot; Tutorials 2 and 6</span>
     </a>
     <a class="pt4d-card" href="#tutorial-9-train-a-physicsnemo-surrogate">
       <span class="pt4d-card__number">09</span>
       <h2>Train a PhysicsNeMo Surrogate</h2>
       <p>Train a MeshGraphNet to predict per-vertex motion from shape and phase.</p>
       <span class="pt4d-card__meta">Tutorial 8 output</span>
     </a>
     <a class="pt4d-card" href="#tutorial-10-predict-motion-with-the-surrogate">
       <span class="pt4d-card__number">10</span>
       <h2>Predict Motion With the Surrogate</h2>
       <p>Replace the registration solve with one forward pass, then export to USD.</p>
       <span class="pt4d-card__meta">Tutorials 8 and 9 output</span>
     </a>
   </section>

Recommended Run Order
=====================

Tutorials are straightforward Python scripts: run one with
``python tutorials/tutorial_01_heart_gated_ct_to_usd.py``, or open it in your
editor and read it top to bottom. Numbers 1, 4 and 5 are the fastest way to see
the toolkit
work end-to-end; 6 through 10 build the statistical-model and AI-surrogate
pipeline on top.

1. **Tutorial 1** — after downloading Slicer-Heart-CT.
2. **Tutorial 2** — after obtaining DIR-Lab. It writes the finetuned ICON
   weights that Tutorials 3 (lung) and 8 use.
3. **Tutorial 3** — after Tutorial 2, whose weights it registers with.
4. **Tutorial 4** — after downloading Slicer-Heart-CT.
5. **Tutorial 5** — after Tutorial 4, whose surfaces it converts.
6. **Tutorial 6** — heart needs KCL-Heart-Model, lung needs DIR-Lab.
7. **Tutorial 7** — after Tutorial 6; the lung variant also needs Chest-CT.
8. **Tutorial 8** — after Tutorials 2 and 6 (lung).
9. **Tutorial 9** — after Tutorial 8, whose fitted meshes it trains on.
10. **Tutorial 10** — after Tutorial 9, whose checkpoint it loads.

Tutorial 1: Gated 4D CT to Animated USD
=======================================

Script
   ``tutorials/tutorial_01_heart_gated_ct_to_usd.py`` (Slicer-Heart-CT)

   ``tutorials/tutorial_01_lung_gated_ct_to_usd.py`` (DIR-Lab)

Workflow
   :class:`~physiotwin4d.WorkflowConvertImageToUSD`, driving
   :class:`~physiotwin4d.RegisterImagesGreedy` and a
   :class:`~physiotwin4d.SegmentAnatomyBase` subclass.

Dataset
   Slicer-Heart-CT (auto-download) for the heart, DIR-Lab (manual) for the
   lung. The phase roughly 70% through the series is the segmentation and
   registration reference.

Requirements
   Greedy registers every phase against the reference on the CPU; a GPU is
   still needed for segmentation.

Preview
   .. figure:: assets/tutorial_01_heart_4d.gif
      :alt: Animated cardiac USD produced by Tutorial 1
      :width: 90%

      The animated cardiac model, played back in Omniverse.

   .. figure:: assets/tutorial_01_lung_4d.gif
      :alt: Animated lung USD produced by Tutorial 1
      :width: 90%

      The same workflow on a DIR-Lab respiratory series.

Inner API usage
   .. code-block:: python

      workflow = WorkflowConvertImageToUSD(
          time_series_images=time_series_images,
          reference_image=reference_image,
          output_directory=str(output_dir),
          usd_project_name="cardiac_model",
          registration_method=registration_method,
          segmentation_method=segmentation_method,
          save_assets=True,
      )
      workflow_results = workflow.process()

Run
   .. code-block:: bash

      python tutorials/tutorial_01_heart_gated_ct_to_usd.py
      python tutorials/tutorial_01_lung_gated_ct_to_usd.py

Outputs
   The animated USD named after ``usd_project_name``, the per-phase registered
   volumes and labelmaps, and screenshots — all under
   ``tutorials/output/tutorial_01_{heart,lung}/``.

Adapt to your data
   Point ``data_dir`` and the file glob near the top of the script at your own
   series: any set of 3D volumes ITK can read (``.mha``, ``.nrrd``,
   ``.nii.gz``) in acquisition order, or a 4D ``.seq.nrrd`` split first with
   ``physiotwin4d-convert-image-4d-to-3d``. Choose the reference phase by
   changing the index expression, and swap ``segmentation_method`` for the one
   matching your anatomy and contrast — see :doc:`api/segmentation/index`. For
   command-line use without editing code, run
   ``physiotwin4d-convert-image-to-usd`` (:doc:`cli_scripts/heart_gated_ct`).

Tutorial 2: Finetune ICON Registration
======================================

Script
   ``tutorials/tutorial_02_lung_finetune_icon.py``

   ``tutorials/tutorial_02_lung_distancemap_finetune_icon.py`` — the lung
   distance-map variant, which finetunes on distance maps rather than image
   intensities so the labelmap-to-labelmap stage of Tutorials 7 and 8 has
   in-distribution weights.

   ``tutorials/tutorial_02_duke_heart_distancemap_finetune_icon.py`` — the same
   for the heart, on Duke-Heart-4DLabelmaps. The heart needs its own run because
   it registers with a much tighter mask than the lungs, so its distance maps
   saturate over a shorter radius and do not share an intensity distribution
   with lung ones. The per-organ values live in
   ``tutorials/parameters_lung_ct_dirlab.py`` and
   ``tutorials/parameters_heart_ct_kcl.py``. This is a ``duke_heart`` tutorial:
   Duke-Heart-4DLabelmaps is not publicly available yet, so it cannot be run —
   see ``data/Duke-Heart-4DLabelmaps/README.md``.

Workflow
   :class:`~physiotwin4d.WorkflowFinetuneICONRegistration`, then
   :class:`~physiotwin4d.RegisterImagesGreedy` and
   :class:`~physiotwin4d.RegisterImagesGreedyICON` to score the result, with
   :class:`~physiotwin4d.SegmentNVSegmentCTMRI` supplying the labelmaps.

Dataset
   DIR-Lab (manual). Every case except ``Case1Pack`` trains; ``Case1Pack`` is
   held out and registered three ways — Greedy alone with its defaults, then
   Greedy+ICON with the stock uniGradICON weights and with the finetuned ones —
   so the improvement is measured, not asserted.

Scoring
   The fixed image is segmented once, and each registered moving image is
   segmented again after warping. The table reports the mean, 5th percentile,
   median, 95th percentile, minimum and maximum of the per-class Dice scores,
   plus the mislabeled voxel count, with the unregistered moving image as a
   reference row. Segmenting each warped volume separately costs one GPU
   segmentation per method and folds segmentation variability into the scores.

Requirements
   GPU required. 100 epochs over nine cases: the longest-running tutorial
   before the AI-surrogate chain. The experiment directory is cleared on every
   run, so it does not resume.

Preview
   .. figure:: assets/tutorial_02_finetuning.png
      :alt: Registration accuracy table for the held-out case
      :width: 100%

      The held-out case scored per method — unregistered, Greedy, Greedy+ICON
      with the stock weights, and with the finetuned weights.

Inner API usage
   .. code-block:: python

      workflow = WorkflowFinetuneICONRegistration(
          subject_image_files=list(subject_image_files.values()),
          output_dir=weights_dir,
          finetune_name=finetune_name,
          subject_ids=list(subject_image_files.keys()),
          epochs=epochs,
          dice_loss_weight=0.0,
      )
      weights_path = workflow.process()

Run
   .. code-block:: bash

      python tutorials/tutorial_02_lung_finetune_icon.py
      python tutorials/tutorial_02_lung_distancemap_finetune_icon.py

Outputs
   The finetuned checkpoint under
   ``tutorials/network_weights/icon_dirlab_4dct/``, plus
   ``registration_summary.csv``, the fixed-minus-registered difference images
   (residual structure is what separates the methods), the fixed and warped
   labelmaps, and before/after screenshots in
   ``tutorials/output/tutorial_02_lung/``.

Adapt to your data
   Replace the training cohort glob with your own volumes and set ``epochs``
   to fit your budget — the workflow needs only a list of image files and
   matching subject ids. Raise ``dice_loss_weight`` above ``0.0`` when you also
   have labelmaps to supervise with. Load the resulting weights anywhere by
   passing them to :class:`~physiotwin4d.RegisterImagesICON`.

Tutorial 3: Reconstruct High-Resolution 4D CT
=============================================

Script
   ``tutorials/tutorial_03_heart_reconstruct_highres_4d_ct.py``

   ``tutorials/tutorial_03_lung_reconstruct_highres_4d_ct.py``

Workflow
   :class:`~physiotwin4d.WorkflowReconstructHighres4DCT` with
   :class:`~physiotwin4d.RegisterImagesGreedy`.

Dataset
   Slicer-Heart-CT for the heart; DIR-Lab for the lung, which reconstructs
   against its T70 (end-exhale) phase — the same reference Tutorial 8 fits to.

Requirements
   CPU is enough. One coarse-to-fine registration per phase, greedy schedule
   ``[30, 15, 7, 3]``.

Preview
   .. figure:: assets/Tutorial_03_heart_original.gif
      :alt: Acquired cardiac phases
      :width: 90%

      The acquired cardiac phases.

   .. figure:: assets/Tutorial_03_heart_recon.gif
      :alt: Cardiac phases reconstructed at the reference resolution
      :width: 90%

      The same phases reconstructed at the reference resolution.

   .. figure:: assets/tutorial_03_output_comparison.gif
      :alt: Acquired phase beside the reconstructed high-resolution phase
      :width: 90%

      Side by side on the lung series.

Inner API usage
   .. code-block:: python

      registration_method = RegisterImagesGreedy()
      registration_method.set_number_of_iterations([30, 15, 7, 3])

      workflow = WorkflowReconstructHighres4DCT(
          time_series_images=time_series,
          reference_image=reference_image,
          reference_time_frame=reference_time_frame,
          registration_method=registration_method,
      )
      workflow.set_modality("ct")
      result = workflow.process()

Run
   .. code-block:: bash

      python tutorials/tutorial_03_heart_reconstruct_highres_4d_ct.py
      python tutorials/tutorial_03_lung_reconstruct_highres_4d_ct.py

Outputs
   ``reconstructed_frame_<i>.mha`` plus forward and inverse transforms for
   every phase, and two screenshots, under
   ``tutorials/output/tutorial_03_{heart,lung}/``.

Adapt to your data
   Set ``case_glob`` and ``data_dir`` to your series and pick the reference
   with ``reference_time_frame``. If you have a separate breath-hold or
   contrast-enhanced volume, pass it as ``reference_image`` instead of one of
   the phases — that is what the workflow is really designed for. Tune
   ``number_of_iterations_greedy`` down for a fast smoke test. The saved
   ``.hdf`` transforms are reusable:
   :class:`~physiotwin4d.TransformTools` applies them to meshes and labelmaps.

Tutorial 4: CT Segmentation to VTK Surfaces
===========================================

Script
   ``tutorials/tutorial_04_heart_ct_to_vtk.py``

   ``tutorials/tutorial_04_lung_ct_to_vtk.py``

Workflow
   :class:`~physiotwin4d.WorkflowConvertImageToVTK` with
   :class:`~physiotwin4d.SegmentChestTotalSegmentatorWithContrast` (heart) or
   :class:`~physiotwin4d.SegmentChestTotalSegmentator` (lung).

Dataset
   One frame of Slicer-Heart-CT or DIR-Lab — a single static volume is enough.

Requirements
   GPU recommended for segmentation; no registration, so this is the quickest
   way to confirm your environment and model weights work.

Preview
   .. figure:: assets/tutorial_04_heart.gif
      :alt: Cardiac surfaces extracted from a CT phase
      :width: 90%

      Cardiac anatomy surfaces exported from one CT phase.

   .. figure:: assets/tutorial_04_lung.png
      :alt: Lung surfaces extracted from a CT phase
      :width: 90%

      The same workflow on a DIR-Lab respiratory case.

Inner API usage
   .. code-block:: python

      workflow = WorkflowConvertImageToVTK(
          segmentation_method=segmentation_method,
      )
      result = workflow.process(
          input_image=ct_image,
          surface_target_reduction=0.5,
          extract_label_surfaces=save_label_surfaces,
      )

Run
   .. code-block:: bash

      python tutorials/tutorial_04_heart_ct_to_vtk.py
      python tutorials/tutorial_04_lung_ct_to_vtk.py

Outputs
   ``patient_surfaces.vtp`` (all anatomy in one mesh, with a per-cell
   ``SegmentationLabelIds`` array so each cell still names the structure it came
   from), per-group and per-label ``.vtp`` files, ``patient_labelmap.mha`` and
   two screenshots, under ``tutorials/output/tutorial_04_{heart,lung}/``.

Adapt to your data
   Change the input volume path, then choose the segmenter matching your scan:
   contrast versus non-contrast CT, or
   :class:`~physiotwin4d.SegmentNVSegmentCTMRI` for CT **and** MRI. Raise
   ``surface_target_reduction`` toward ``1.0`` for lighter meshes. Every
   segmenter declares its own labels through
   :class:`~physiotwin4d.AnatomyTaxonomy`, so downstream grouping and USD
   materials follow automatically — see :doc:`api/segmentation/index`.

Tutorial 5: VTK Surfaces to Animated USD
========================================

Script
   ``tutorials/tutorial_05_heart_vtk_to_usd.py``

Workflow
   :class:`~physiotwin4d.WorkflowConvertVTKToUSD`.

Dataset
   Tutorial 4's per-structure ``patient_*.vtp`` surfaces — no image data, no
   download.

Requirements
   CPU only, seconds to run. The cheapest tutorial in the set.

Preview
   .. figure:: assets/tutorial_05_heart_vtk_to_usd.png
      :alt: Cardiac surfaces rendered from the exported USD scene
      :width: 90%

      The exported USD scene, split by anatomy and painted with OmniSurface
      materials.

Inner API usage
   .. code-block:: python

      workflow = WorkflowConvertVTKToUSD(
          input_meshes=meshes,
          usd_project_name=project_name,
          output_directory=output_dir,
          appearance="anatomy",
          static_merge=True,
          separate_by_connectivity=True,
      )
      results = workflow.process()

   Each input surface keeps the structure name that
   :class:`~physiotwin4d.WorkflowConvertImageToVTK` wrote into its
   ``field_data['SegmentationLabelNames']``. That name becomes the USD prim
   name and, with ``anatomy_type`` left unset, selects the prim's material —
   so the left chambers, right chambers, myocardium and great vessels each get
   their own look rather than one shared heart material.

Run
   .. code-block:: bash

      python tutorials/tutorial_05_heart_vtk_to_usd.py

Outputs
   The USD scene and a rendered screenshot under
   ``tutorials/output/tutorial_05_heart/``.

Adapt to your data
   ``input_meshes`` takes any list of PyVista meshes — pass one per time point,
   in order, for an animated scene instead of a static one (drop
   ``static_merge``), and set ``frames_per_second`` to control playback.
   ``appearance="anatomy"`` binds per-organ materials through
   :class:`~physiotwin4d.USDAnatomyTools`; set ``anatomy_type`` to force one
   palette onto every object, or ``object_names`` to name the prims yourself.
   For file-in, file-out conversion without Python, see
   :doc:`cli_scripts/vtk_to_usd`.

Tutorial 6: Create a PCA Shape Model
====================================

Script
   ``tutorials/tutorial_06_heart_create_statistical_model.py``

   ``tutorials/tutorial_06_lung_create_statistical_model.py``

Workflow
   :class:`~physiotwin4d.WorkflowCreateStatisticalModel`; the lung variant
   first builds an unbiased atlas with
   :class:`~physiotwin4d.WorkflowCreateMeanSurface`.

Dataset
   KCL-Heart-Model (auto-download) for the heart. The lung variant starts from
   raw DIR-Lab volumes, segmenting each case's T70 phase itself.

Requirements
   The heart variant is CPU-only and quick. **The lung variant is the slowest
   of Tutorials 1-7**: one GPU segmentation per case, then a deformable
   registration per case per atlas iteration. Every intermediate is cached, so
   a re-run costs almost nothing.

Preview
   .. figure:: assets/tutorial_06_heart_modes_of_variation.png
      :alt: Cardiac shape model modes of variation
      :width: 90%

      Heart model: the mean shape at ±2σ along its leading modes.

   .. figure:: assets/tutorial_06_lung_modes_of_variation.png
      :alt: Lung shape model modes of variation
      :width: 90%

      The same decomposition for the lung population.

Inner API usage
   .. code-block:: python

      mean_workflow = WorkflowCreateMeanSurface(surfaces=sample_surfaces)
      mean_workflow.set_number_of_iterations(mean_surface_iterations)
      reference_surface = mean_workflow.process()["mean_surface"]

      workflow = WorkflowCreateStatisticalModel(
          sample_meshes=sample_surfaces,
          reference_mesh=reference_surface,
          number_of_pca_components=number_of_pca_components,
      )
      result = workflow.process()

Run
   .. code-block:: bash

      python tutorials/tutorial_06_heart_create_statistical_model.py
      python tutorials/tutorial_06_lung_create_statistical_model.py

Outputs
   ``pca_model.json``, ``pca_mean_surface.vtp``, the ±2σ mode surfaces and
   their renders, under ``tutorials/output/tutorial_06_{heart,lung}/``. The
   lung variant also leaves its per-case segmentations there, which Tutorial 8
   reuses.

Adapt to your data
   The workflow wants a population of meshes plus one reference; point
   ``sample_meshes`` at your own cohort and let
   :class:`~physiotwin4d.WorkflowCreateMeanSurface` build the reference when no
   natural template exists. ``number_of_pca_components`` trades fidelity
   against cohort size — you need more subjects than modes. The saved
   ``pca_model.json`` is the portable artifact: Tutorials 7 and 8 and
   :doc:`cli_scripts/create_statistical_model` all speak it.

Tutorial 7: Fit the Shape Model to a Patient
============================================

Script
   ``tutorials/tutorial_07_heart_fit_statistical_model_to_patient.py``

   ``tutorials/tutorial_07_lung_fit_statistical_model_to_patient.py``

Workflow
   :class:`~physiotwin4d.WorkflowFitStatisticalModelToPatient`.

Dataset
   Tutorial 6's model plus one patient scan. The lung variant fits to
   ``Chest-CT`` — a routine, single-time-point clinical chest CT, which is the
   scan most adopters actually have.

Requirements
   One segmentation pass plus a PCA-constrained fit; GPU recommended for the
   segmentation, and no registration over time.

Preview
   .. figure:: assets/tutorial_07_heart_in_noncontrast_ct.gif
      :alt: Fitted heart model overlaid on a non-contrast CT
      :width: 90%

      The heart model fitted to a non-contrast scan.

   .. figure:: assets/tutorial_07_lung.gif
      :alt: Fitted lung model on the routine clinical Chest-CT scan
      :width: 90%

      The lung model fitted to the routine clinical ``Chest-CT`` volume.

Inner API usage
   .. code-block:: python

      workflow = WorkflowFitStatisticalModelToPatient(
          template_model=pca_mean,
          patient_models=[patient_surface],
          patient_image=patient_image,
          patient_labelmap=patient_labelmap,
      )
      workflow.set_use_pca_registration(
          use_pca_registration=True,
          pca_model=pca_model,
      )
      result = workflow.process()

Run
   .. code-block:: bash

      python tutorials/tutorial_07_heart_fit_statistical_model_to_patient.py
      python tutorials/tutorial_07_lung_fit_statistical_model_to_patient.py

Outputs
   The registered template surface, the fitted mesh, and — the piece the rest
   of the pipeline needs — ``*_registered_coefficients.json``, the patient's
   position in shape space. Under
   ``tutorials/output/tutorial_07_{heart,lung}/``.

Adapt to your data
   Set the patient image path and keep the segmenter consistent with the one
   that built the model. ``labelmap_interior_object_ids`` (heart) tells the fit
   which labels are interior structures — those ids are TotalSegmentator's
   chamber labels, so change them if you change segmenter. Turn
   ``set_use_pca_registration`` off to fall back to an unconstrained
   template-to-patient fit when you have no model. The CLI equivalent is
   :doc:`cli_scripts/fit_statistical_model_to_patient`.

Tutorial 8: Propagate the Shape Model Through 4D
================================================

Script
   ``tutorials/tutorial_08_lung_fit_model_to_4d_patients.py``

Workflow
   :class:`~physiotwin4d.WorkflowFitStatisticalModelToPatient` at the reference
   phase, then :class:`~physiotwin4d.WorkflowReconstructHighres4DCT` to carry
   the fitted surface through every other phase.

Dataset
   DIR-Lab, plus Tutorial 6 (lung)'s model. Tutorial 2's finetuned distance-map
   ICON weights are used by the model fit when present; without them the
   tutorial warns and fits with the stock uniGradICON weights.

Requirements
   GPU required, and the heaviest registration workload in the set: one
   segmentation and one fit per case, plus one registration per phase per case.

Preview
   .. figure:: assets/tutorial_08_lung.gif
      :alt: Fitted lung shape model carried through every respiratory phase
      :width: 90%

      The fitted shape-model surface propagated across the phases of a DIR-Lab
      case.

Inner API usage
   .. code-block:: python

      fit_workflow = WorkflowFitStatisticalModelToPatient(
          template_model=pca_mean_surface,
          patient_models=[lung_surface],
          patient_image=reference_image,
          patient_labelmap=lung_labelmap,
      )
      fit_workflow.set_use_pca_registration(True, pca_model=pca_model)

      reg_workflow = WorkflowReconstructHighres4DCT(
          time_series_images=time_series,
          reference_image=reference_image,
          reference_time_frame=phase_ids.index(reference_phase),
          register_reference_time_frame_to_reference_image=False,
          registration_method=registration_method,
      )

Run
   .. code-block:: bash

      python tutorials/tutorial_08_lung_fit_model_to_4d_patients.py

Outputs
   Per case, under ``tutorials/output/tutorial_08_lung/<case>/``: the fitted
   reference surface, its PCA coefficients, and one warped surface plus
   forward/inverse transform per phase. Those per-phase surfaces are exactly
   the training set Tutorial 9 consumes.

Adapt to your data
   Point ``data_dir`` at a directory of per-case 4D series and set
   ``reference_phase`` to the phase your model was built at; the case and phase
   file patterns are two globs near the top of the script. Everything is cached
   per case, so adding a subject re-runs only that subject.

Tutorial 9: Train a PhysicsNeMo Surrogate
=========================================

Script
   ``tutorials/tutorial_09_lung_train_physicsnemo_mgn.py``

Workflow
   :class:`~physiotwin4d.WorkflowTrainPhysicsNeMo` driving
   :class:`~physiotwin4d.TrainPhysicsNeMoMGN`, then
   :class:`~physiotwin4d.WorkflowInferPhysicsNeMo` and
   :class:`~physiotwin4d.WorkflowInferMovement` to score the held-out case. A
   fully connected :class:`~physiotwin4d.TrainPhysicsNeMoMLP` method is a
   drop-in replacement; no separate tutorial ships for it.

Dataset
   Tutorial 8's per-phase surfaces and Tutorial 6 (lung)'s mean surface. The
   tutorial writes one JSON manifest per case, plus the per-vertex displacement
   targets those manifests point at.

Requirements
   GPU, plus the optional extra::

      pip install "physiotwin4d[physicsnemo]"
      pip install torch-geometric

   Python >= 3.11. 1500 epochs by default.

Preview
   .. figure:: assets/example.gif
      :alt: Tutorial 9 output preview (capture pending)
      :width: 60%

      Capture pending — the tutorial writes ``predicted_surface.png`` and
      ``rmse_surface.png`` when it runs.

Inner API usage
   .. code-block:: python

      training_method = TrainPhysicsNeMoMGN()
      training_method.set_epochs(epochs)
      training_method.set_processor_size(processor_size)

      train_workflow = WorkflowTrainPhysicsNeMo(
          train_manifests=train_manifests,
          val_manifests=val_manifests,
          pca_mean_mesh=ssm_mean_surface_file,
          output_directory=output_dir,
          training_method=training_method,
      )
      train_result = train_workflow.process()

Run
   .. code-block:: bash

      python tutorials/tutorial_09_lung_train_physicsnemo_mgn.py

Outputs
   ``mgn_stage_model.pt``, its metadata and loss/RMSE logs, the per-case
   manifests, and the held-out evaluation under ``eval_mgn/`` — in the
   directory training used: ``tutorials/output/tutorial_09_lung_mgn/``
   normally, or a fresh sibling when resuming.

Adapt to your data
   The contract is the manifest, not the tutorial. Each JSON names a reference
   mesh, a PCA coefficient file, a ``target_array`` name and one entry per
   phase; the workflow reads that array verbatim, so the target can be
   displacement — as here — or any per-point quantity of any width. Produce
   manifests in that shape from your own pipeline and nothing else changes.
   See :doc:`api/physicsnemo/index` for the schema, and
   :doc:`cli_scripts/train_physicsnemo` for the command-line path.

Tutorial 10: Predict Motion With the Surrogate
==============================================

Script
   ``tutorials/tutorial_10_lung_infer_physicsnemo_mgn.py``

Workflow
   :class:`~physiotwin4d.WorkflowInferPhysicsNeMo` for the raw prediction,
   :class:`~physiotwin4d.WorkflowInferMovement` to turn it back into geometry,
   and :class:`~physiotwin4d.WorkflowConvertVTKToUSD` to export it.

Dataset
   Tutorial 8's fitted surfaces for one case, and Tutorial 9's checkpoint.

Requirements
   The ``[physicsnemo]`` extra; otherwise trivial — one forward pass replaces
   the per-phase registration solve that produced the training data.

Preview
   .. figure:: assets/example.gif
      :alt: Tutorial 10 output preview (capture pending)
      :width: 60%

      Capture pending — the tutorial writes ``predicted_surface.png`` and
      ``ground_truth_surface.png`` when it runs.

Inner API usage
   .. code-block:: python

      infer_workflow = WorkflowInferPhysicsNeMo(
          model_directory=model_dir,
          epoch=epoch,
      )
      infer_result = WorkflowInferMovement(infer_workflow).predict_single(
          shape_parameters=pca_file,
          stage=test_stage,
          reference_mesh=reference_file,
          ground_truth=ground_truth_file,
          output_directory=output_dir,
      )

Run
   .. code-block:: bash

      python tutorials/tutorial_10_lung_infer_physicsnemo_mgn.py

Outputs
   The predicted surface, its error statistics against the ground-truth phase
   in millimetres, and a USD scene, under
   ``tutorials/output/tutorial_09_lung_mgn/tutorial_10_lung_mgn/<case>/``.

Adapt to your data
   Change ``case_id`` and ``stage_fraction`` to predict a different subject, or
   a stage that was never acquired — which is the point of the surrogate. Omit
   ``reference_mesh`` to displace the mesh reconstructed from the PCA
   coefficients alone, needing no per-subject geometry at all. Use
   :class:`~physiotwin4d.WorkflowInferPhysicsNeMo` on its own to get the raw
   target array when your model predicts something other than displacement.

Where to Go Next
================

- :doc:`viewing_usd` — installing an Omniverse Kit application and opening the
  scenes these tutorials produce.
- :doc:`cli_scripts/byod_tutorials` — running the workflows on your own DICOM,
  NRRD or VTK data, including directory layout and conversion.
- :doc:`api/index` — every workflow, segmenter, registrar and utility class.
- :doc:`architecture` — how the workflow layer fits together and where to
  extend it.
- :doc:`testing` — ``tests/test_tutorials.py`` runs these scripts end-to-end
  behind the ``--run-tutorials`` flag.
