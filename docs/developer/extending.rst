====================================
Extending PhysioMotion4D
====================================

This guide shows how to extend PhysioMotion4D with custom functionality for your specific research or clinical applications.

Overview
========

PhysioMotion4D is designed for extension. You can:

* Create custom workflows for new anatomical regions
* Implement new segmentation or registration methods
* Add custom USD materials and visualizations
* Integrate with external tools and pipelines
* Contribute improvements back to the project

.. important::

   **Using Scripts and Experiments as References:**

   When extending PhysioMotion4D, use these repository resources appropriately:

   * **CLI Implementations (src/physiomotion4d/cli/)** ⭐ **START HERE** - Production-quality
     implementations showing proper class usage, error handling, and parameter specifications.
     Use these as templates for your extensions.

   * **experiments/** - Research prototypes demonstrating conceptual workflows that can inform
     adaptation to new digital twin models, anatomical regions, and imaging modalities. Study these
     for architectural inspiration, but always refer to the CLI implementations for proper API usage.

   **Key Principle:** Experiments show *what is possible*, CLI implementations show *how to do it correctly*.

Extension Patterns
==================

All extensions follow common patterns:

1. **Inherit from base classes** (:class:`PhysioMotion4DBase` or specialized bases - see :doc:`../api/base`)
2. **Override specific methods** to customize behavior
3. **Use existing utilities** for common operations (see :doc:`../api/utilities/index`)
4. **Follow consistent naming and documentation** conventions

Creating Custom Workflows
==========================

Basic Workflow Template
-----------------------

Start with this template for new workflows:

.. code-block:: python

   from physiomotion4d import PhysioMotion4DBase
   from physiomotion4d import SegmentChestTotalSegmentator
   from physiomotion4d import RegisterImagesICON
   from physiomotion4d import ConvertVTKToUSD

   class MyCustomWorkflow(PhysioMotion4DBase):
       """
       Custom workflow for [describe purpose].

       This workflow processes [input type] to produce [output type].
       """

       def __init__(
           self,
           input_file,
           output_directory="./results",
           project_name="custom_project",
           verbose=False
       ):
           """
           Initialize custom workflow.

           Args:
               input_file: Path to input file
               output_directory: Output directory path
               project_name: Project name for organization
               verbose: Enable verbose logging
           """
           super().__init__(verbose=verbose)

           # Validate inputs
           self.input_file = self.validate_file_exists(input_file)
           self.output_directory = self.ensure_directory(output_directory)
           self.project_name = project_name

           # Initialize components
           self._initialize_components()

           self.log("MyCustomWorkflow initialized", level="INFO")

       def _initialize_components(self):
           """Initialize processing components."""
           self.segmentator = SegmentChestTotalSegmentator(verbose=self.verbose)
           self.registrator = RegisterImagesICON(device="cuda:0")
           self.usd_converter_class = ConvertVTKToUSD

       def process(self):
           """Execute complete workflow."""
           self.log("Starting processing...", level="INFO")

           try:
               # Step 1: Load data
               data = self._load_data()

               # Step 2: Process
               processed = self._process_data(data)

               # Step 3: Generate output
               output = self._generate_output(processed)

               self.log("Processing complete!", level="INFO")
               return output

           except Exception as e:
               self.log(f"Processing failed: {e}", level="ERROR")
               raise

       def _load_data(self):
           """Load and validate input data."""
           self.log("Loading data...")
           # Implement data loading
           pass

       def _process_data(self, data):
           """Process loaded data."""
           self.log("Processing data...")
           # Implement processing steps
           pass

       def _generate_output(self, processed):
           """Generate final output."""
           self.log("Generating output...")
           # Implement output generation
           pass

Example: Brain Vessel Workflow
-------------------------------

Complete example of a custom workflow:

.. code-block:: python

   from physiomotion4d import PhysioMotion4DBase
   from physiomotion4d.image_tools import read_image, write_image
   from physiomotion4d.contour_tools import extract_surface_mesh
   from physiomotion4d import ConvertVTKToUSD
   import SimpleITK as sitk

   class BrainVesselWorkflow(PhysioMotion4DBase):
       """Extract and visualize brain vasculature."""

       def __init__(
           self,
           angiography_file,
           output_directory="./results",
           vessel_threshold=200,
           verbose=False
       ):
           super().__init__(verbose=verbose)

           self.angiography_file = self.validate_file_exists(angiography_file)
           self.output_directory = self.ensure_directory(output_directory)
           self.vessel_threshold = vessel_threshold

       def process(self):
           """Process brain vasculature."""
           # Load angiography
           image = read_image(self.angiography_file)
           self.log(f"Loaded image: {image.GetSize()}")

           # Enhance vessels
           enhanced = self._enhance_vessels(image)

           # Segment vessels
           vessel_mask = self._segment_vessels(enhanced)

           # Extract centerlines
           centerlines = self._extract_centerlines(vessel_mask)

           # Create surface mesh
           vessel_mesh = extract_surface_mesh(
               vessel_mask,
               label_value=1,
               smooth_iterations=30
           )

           # Convert to USD
           usd_file = f"{self.output_directory}/brain_vessels.usd"
           converter = ConvertVTKToUSD(
               data_basename="brain_vessels",
               input_polydata=[vessel_mesh],
           )
           converter.convert(usd_file)

           self.log(f"Brain vessel model created: {usd_file}", level="INFO")
           return usd_file

       def _enhance_vessels(self, image):
           """Enhance vessels using Hessian filter."""
           self.log("Enhancing vessels...")

           # Apply Hessian-based vessel enhancement
           hessian_filter = sitk.HessianRecursiveGaussianImageFilter()
           hessian_filter.SetSigma(1.0)
           hessian = hessian_filter.Execute(image)

           # Compute vesselness
           vesselness_filter = sitk.Hessian3DToVesselnessImageFilter()
           vesselness = vesselness_filter.Execute(hessian)

           return vesselness

       def _segment_vessels(self, enhanced):
           """Segment vessels from enhanced image."""
           self.log("Segmenting vessels...")

           # Threshold
           vessel_mask = sitk.BinaryThreshold(
               enhanced,
               lowerThreshold=self.vessel_threshold,
               upperThreshold=1000,
               insideValue=1,
               outsideValue=0
           )

           # Clean up
           vessel_mask = sitk.BinaryMorphologicalClosing(
               vessel_mask,
               kernelRadius=[2, 2, 2]
           )

           return vessel_mask

       def _extract_centerlines(self, vessel_mask):
           """Extract vessel centerlines."""
           self.log("Extracting centerlines...")
           # Implement centerline extraction
           # Could use ITK-TubeTK or custom algorithm
           pass

Custom Segmentation Methods
============================

Adding New Segmentation Algorithms
-----------------------------------

Extend :class:`SegmentAnatomyBase`:

.. code-block:: python

   from physiomotion4d import SegmentAnatomyBase
   import torch

   class MyCustomSegmentator(SegmentAnatomyBase):
       """Custom deep learning segmentation."""

       def __init__(
           self,
           model_path,
           device="cuda:0",
           confidence_threshold=0.5,
           verbose=False
       ):
           super().__init__(verbose=verbose)

           self.model_path = model_path
           self.device = device
           self.confidence_threshold = confidence_threshold

           # Load model
           self.model = self._load_model()

       def _load_model(self):
           """Load custom segmentation model."""
           self.log(f"Loading model from {self.model_path}")
           model = torch.load(self.model_path)
           model.to(self.device)
           model.eval()
           return model

       def segment(self, image_path):
           """
           Segment image using custom model.

           Args:
               image_path: Path to input image

           Returns:
               Segmentation labelmap
           """
           # Load image
           image = self.load_image(image_path)

           # Preprocess
           input_tensor = self._preprocess(image)

           # Run inference
           with torch.no_grad():
               output = self.model(input_tensor)
               probabilities = torch.softmax(output, dim=1)

           # Post-process
           labelmap = self._postprocess(probabilities)

           return labelmap

       def _preprocess(self, image):
           """Preprocess image for model."""
           # Convert to tensor, normalize, etc.
           pass

       def _postprocess(self, probabilities):
           """Convert probabilities to labelmap."""
           # Threshold, apply morphological operations, etc.
           pass

Custom Registration Methods
============================

Implementing New Registration Algorithms
-----------------------------------------

Extend :class:`RegisterImagesBase`:

.. code-block:: python

   from physiomotion4d import RegisterImagesBase
   import numpy as np

   class MyCustomRegistrator(RegisterImagesBase):
       """Custom registration algorithm."""

       def __init__(
           self,
           similarity_metric='ncc',
           optimization_method='gd',
           max_iterations=100,
           verbose=False
       ):
           super().__init__(verbose=verbose)

           self.similarity_metric = similarity_metric
           self.optimization_method = optimization_method
           self.max_iterations = max_iterations

       def register(self, fixed_image_path, moving_image_path):
           """
           Register two images.

           Args:
               fixed_image_path: Path to fixed image
               moving_image_path: Path to moving image

           Returns:
               Transform object
           """
           # Load images
           fixed = self.load_image(fixed_image_path)
           moving = self.load_image(moving_image_path)

           # Initialize transform
           transform = self._initialize_transform(fixed, moving)

           # Optimize
           optimized_transform = self._optimize(
               fixed,
               moving,
               transform
           )

           return optimized_transform

       def _initialize_transform(self, fixed, moving):
           """Initialize transformation parameters."""
           # Implement initialization (e.g., center of mass alignment)
           pass

       def _optimize(self, fixed, moving, initial_transform):
           """Optimize transformation parameters."""
           self.log("Starting optimization...")

           transform = initial_transform

           for iteration in range(self.max_iterations):
               # Compute similarity
               similarity = self._compute_similarity(fixed, moving, transform)

               # Update transform
               gradient = self._compute_gradient(fixed, moving, transform)
               transform = self._update_transform(transform, gradient)

               # Log progress
               if iteration % 10 == 0:
                   self.log(f"Iteration {iteration}: similarity = {similarity}")

           return transform

       def _compute_similarity(self, fixed, moving, transform):
           """Compute similarity metric."""
           # Implement similarity computation
           pass

       def _compute_gradient(self, fixed, moving, transform):
           """Compute gradient of similarity."""
           # Implement gradient computation
           pass

       def _update_transform(self, transform, gradient):
           """Update transform parameters."""
           # Implement parameter update
           pass

Custom USD Materials
====================

Adding Anatomical Materials
----------------------------

Extend the material library:

.. code-block:: python

   from physiomotion4d.usd_anatomy_tools import USDAnatomyTools
   from pxr import Usd, UsdShade, Sdf

   class CustomAnatomyMaterials(USDAnatomyTools):
       """Custom anatomical materials."""

       def __init__(self):
           super().__init__()

           # Add custom materials
           self.custom_materials = {
               'pathological_tissue': {
                   'base_color': [0.6, 0.1, 0.1],
                   'metallic': 0.0,
                   'roughness': 0.9,
                   'opacity': 0.85,
                   'emission_color': [0.2, 0.0, 0.0],
                   'emission_strength': 0.3
               },
               'implant_material': {
                   'base_color': [0.7, 0.7, 0.8],
                   'metallic': 0.8,
                   'roughness': 0.2,
                   'opacity': 1.0,
                   'ior': 2.5
               }
           }

       def create_pathological_material(self, stage, material_path):
           """Create material for pathological tissue."""
           material = UsdShade.Material.Define(stage, material_path)

           # Create PBR shader
           shader = UsdShade.Shader.Define(
               stage,
               material_path + "/PBRShader"
           )
           shader.CreateIdAttr("UsdPreviewSurface")

           # Set properties
           props = self.custom_materials['pathological_tissue']
           shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
               tuple(props['base_color'])
           )
           shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
               props['metallic']
           )
           shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
               props['roughness']
           )
           shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(
               props['opacity']
           )
           shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
               tuple(props['emission_color'])
           )

           # Connect shader to material
           material.CreateSurfaceOutput().ConnectToSource(
               shader.ConnectableAPI(),
               "surface"
           )

           return material

Integration with External Tools
================================

Connecting to External Libraries
---------------------------------

Integrate PhysioMotion4D with external tools:

.. code-block:: python

   from physiomotion4d import WorkflowConvertHeartGatedCTToUSD
   import external_tool

   class IntegratedWorkflow(WorkflowConvertHeartGatedCTToUSD):
       """Workflow integrated with external tool."""

       def __init__(self, *args, use_external_segmentation=False, **kwargs):
           super().__init__(*args, **kwargs)
           self.use_external_segmentation = use_external_segmentation

       def segment_reference(self):
           """Override segmentation to use external tool."""
           if self.use_external_segmentation:
               self.log("Using external segmentation tool...")

               # Call external tool
               segmentation = external_tool.segment(
                   self.reference_image_path
               )

               # Convert to PhysioMotion4D format
               labelmap = self._convert_external_format(segmentation)

               return labelmap
           else:
               # Use default PhysioMotion4D segmentation
               return super().segment_reference()

Creating Command-Line Tools
============================

CLI Wrapper for Custom Workflow
--------------------------------

Create a CLI script for your custom workflow:

.. code-block:: python

   #!/usr/bin/env python
   """CLI for custom brain vessel workflow."""

   import argparse
   from pathlib import Path
   from my_extensions import BrainVesselWorkflow

   def main():
       parser = argparse.ArgumentParser(
           description="Extract and visualize brain vasculature",
           formatter_class=argparse.RawDescriptionHelpFormatter,
           epilog="""
   Examples:
     # Basic usage
     %(prog)s angiography.nrrd --output-dir ./results

     # With custom threshold
     %(prog)s cta.nrrd --vessel-threshold 250 --output-dir ./output
           """
       )

       parser.add_argument(
           "input_file",
           help="Path to angiography image file"
       )
       parser.add_argument(
           "--output-dir",
           default="./results",
           help="Output directory (default: ./results)"
       )
       parser.add_argument(
           "--vessel-threshold",
           type=int,
           default=200,
           help="Vessel intensity threshold (default: 200)"
       )
       parser.add_argument(
           "--verbose",
           action="store_true",
           help="Enable verbose output"
       )

       args = parser.parse_args()

       # Validate input
       input_file = Path(args.input_file)
       if not input_file.exists():
           print(f"Error: Input file not found: {input_file}")
           return 1

       # Run workflow
       print(f"Processing {input_file.name}...")

       try:
           workflow = BrainVesselWorkflow(
               angiography_file=str(input_file),
               output_directory=args.output_dir,
               vessel_threshold=args.vessel_threshold,
               verbose=args.verbose
           )

           result = workflow.process()
           print(f"Success! Output: {result}")
           return 0

       except Exception as e:
           print(f"Error: {e}")
           return 1

   if __name__ == "__main__":
       exit(main())

Installation as Package Entry Point
------------------------------------

Make your CLI available after package installation:

.. code-block:: python

   # In setup.py or pyproject.toml

   [project.scripts]
   my-brain-vessel-tool = "my_extensions.cli_brain_vessel:main"

Testing Extensions
==================

Unit Tests for Custom Code
---------------------------

Write tests for your extensions:

.. code-block:: python

   import pytest
   from my_extensions import BrainVesselWorkflow

   def test_brain_vessel_workflow():
       """Test brain vessel workflow."""
       workflow = BrainVesselWorkflow(
           angiography_file="test_data/angiography.nrrd",
           output_directory="./test_output"
       )

       result = workflow.process()

       assert result is not None
       assert Path(result).exists()

   def test_vessel_enhancement():
       """Test vessel enhancement step."""
       workflow = BrainVesselWorkflow("test_data/angiography.nrrd")

       image = workflow.load_image("test_data/angiography.nrrd")
       enhanced = workflow._enhance_vessels(image)

       assert enhanced is not None
       # Add more assertions

Contributing Back
=================

Sharing Your Extensions
-----------------------

To contribute your extensions to PhysioMotion4D:

1. **Fork the repository** on GitHub
2. **Create a feature branch**: ``git checkout -b feature/my-extension``
3. **Implement your extension** following code style
4. **Add tests** for new functionality
5. **Update documentation** with examples
6. **Submit pull request** with description

Code Style Guidelines
---------------------

Follow these conventions:

.. code-block:: python

   # Use Google-style docstrings
   def my_function(arg1, arg2):
       """
       Brief description.

       Detailed description of what the function does.

       Args:
           arg1: Description of arg1
           arg2: Description of arg2

       Returns:
           Description of return value

       Raises:
           ValueError: When invalid input
       """
       pass

   # Use type hints
   def process_image(image_path: str, threshold: float = 0.5) -> np.ndarray:
       """Process image with threshold."""
       pass

   # Follow PEP 8 naming
   class MyClassName:  # CamelCase for classes
       pass

   def my_function_name():  # snake_case for functions
       pass

   MY_CONSTANT = 42  # UPPERCASE for constants

Documentation Standards
-----------------------

Document your extensions:

.. code-block:: rst

   ====================================
   My Custom Extension
   ====================================

   Brief description of your extension.

   Overview
   ========

   Detailed overview...

   Usage
   =====

   .. code-block:: python

      from my_extension import MyClass

      # Example usage
      obj = MyClass()
      result = obj.process()

Best Practices
==============

Extension Design
----------------

* **Keep it modular**: Create focused, reusable components
* **Follow existing patterns**: Inherit from appropriate base classes
* **Provide clear interfaces**: Document inputs, outputs, parameters
* **Handle errors gracefully**: Validate inputs, provide useful error messages
* **Log appropriately**: Use the logging system from base class

Performance
-----------

* **Profile your code**: Identify bottlenecks
* **Use GPU when available**: Check for CUDA and fall back to CPU
* **Cache expensive operations**: Save intermediate results
* **Consider memory usage**: Be aware of large datasets

See Also
========

* :doc:`architecture` - System architecture
* :doc:`workflows` - Workflow patterns
* :doc:`core` - Base class documentation
* :doc:`../contributing` - Contributing guidelines
