"""
This module contains the USDAnatomyTools class, which is used to enhance
the anatomy meshes in a USD file.

Extensibility
-------------
The default OmniSurface look for each anatomy group/organ lives in the
module-level :data:`DEFAULT_RENDER_PARAMS` dict. A new segmenter that
introduces a new group (e.g. ``"brain"``, ``"tumor"``) can register a
matching look in one of three ways:

1. **Globally**, before instantiating any ``USDAnatomyTools``::

       from physiotwin4d.usd_anatomy_tools import DEFAULT_RENDER_PARAMS
       DEFAULT_RENDER_PARAMS["brain"] = {"name": "Brain", ...}

   Every subsequent ``USDAnatomyTools`` instance picks up the new entry.

2. **Per-instance**, after construction::

       tools = USDAnatomyTools(stage)
       tools.render_params["brain"] = {"name": "Brain", ...}

3. **By subclassing**, overriding ``__init__`` to populate
   ``self.render_params`` with project-specific defaults.

Group lookup falls back to ``render_params["other"]`` when a group has no
registered entry, so any group present in the segmenter's
:class:`physiotwin4d.AnatomyTaxonomy` will still render *something*.
"""

import logging
from typing import Any, Mapping

from pxr import Sdf, UsdGeom, UsdShade

from .physiotwin4d_base import PhysioTwin4DBase

# Default OmniSurface render parameters keyed by group name (matching
# :class:`physiotwin4d.AnatomyTaxonomy.group_names`) and by organ-level
# overrides (e.g. ``liver``, ``spleen``, ``kidney_left``). ``enhance_meshes``
# consults an organ-level entry first, then falls back to the containing
# group's entry, and finally to ``"other"``. Module-level so CLIs and tests
# can enumerate the supported types without constructing a USD stage.
DEFAULT_RENDER_PARAMS: dict[str, dict[str, Any]] = {
    "heart": {
        "name": "Heart",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.25,
        "diffuse_reflection_color": (0.2, 0.01, 0.01),
        "diffuse_reflection_roughness": 0.5,
        "metalness": 0.0,
        "specular_reflection_weight": 0.5,
        "specular_reflection_roughness": 0.5,
        "subsurface_transmission_color": (0.8, 0.4, 0.4),
        "subsurface_scattering_color": (0.8, 0.4, 0.4),
        "subsurface_weight": 0.1,
        "subsurface_scale": 2.0,
        "coat_weight": 0.0,
    },
    "lung": {
        "name": "Lung",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.125,
        "diffuse_reflection_color": (0.34, 0.0, 0.0),
        "diffuse_reflection_roughness": 0.6,
        "metalness": 0.0,
        "specular_reflection_weight": 0.5,
        "specular_reflection_roughness": 0.5,
        "subsurface_transmission_color": (0.9, 0.7, 0.7),
        "subsurface_scattering_color": (0.9, 0.7, 0.7),
        "subsurface_weight": 0.1,
        "subsurface_scale": 0.2,
        "coat_weight": 0.0,
    },
    "bone": {
        "name": "Bone",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.2,
        "diffuse_reflection_color": (0.8, 0.8, 0.9),
        "diffuse_reflection_roughness": 0.3,
        "metalness": 0.0,
        "specular_reflection_weight": 0.5,
        "specular_reflection_roughness": 0.5,
        "subsurface_transmission_color": (0.95, 0.9, 0.7),
        "subsurface_scattering_color": (0.95, 0.9, 0.7),
        "subsurface_weight": 0.03,
        "subsurface_scale": 0.05,
        "coat_weight": 0.0,
    },
    "major_vessels": {
        "name": "Major_Vessels",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.35,
        "diffuse_reflection_color": (0.2, 0.01, 0.01),
        "diffuse_reflection_roughness": 0.3,
        "metalness": 0.0,
        "specular_reflection_weight": 0.5,
        "specular_reflection_roughness": 0.5,
        "subsurface_transmission_color": (0.7, 0.15, 0.18),
        "subsurface_scattering_color": (0.7, 0.15, 0.18),
        "subsurface_weight": 0.09,
        "subsurface_scale": 0.22,
        "coat_weight": 0.12,
    },
    "contrast": {
        "name": "Contrast",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.25,
        "diffuse_reflection_color": (0.2, 0.01, 0.01),
        "diffuse_reflection_roughness": 0.5,
        "metalness": 0.0,
        "specular_reflection_weight": 0.5,
        "specular_reflection_roughness": 0.5,
        "subsurface_transmission_color": (0.9, 0.4, 0.4),
        "subsurface_scattering_color": (0.9, 0.4, 0.4),
        "subsurface_weight": 0.1,
        "subsurface_scale": 2.0,
        "coat_weight": 0.0,
    },
    "soft_tissue": {
        "name": "Soft_Tissue",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.17,
        "diffuse_reflection_color": (0.7, 0.5, 0.4),
        "diffuse_reflection_roughness": 0.5,
        "metalness": 0.0,
        "specular_reflection_weight": 0.5,
        "specular_reflection_roughness": 0.5,
        "subsurface_transmission_color": (0.95, 0.9, 0.9),
        "subsurface_scattering_color": (0.95, 0.9, 0.9),
        "subsurface_weight": 0.08,
        "subsurface_scale": 0.3,
        "coat_weight": 0.12,
    },
    "other": {
        "name": "Other",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.1,
        "diffuse_reflection_color": (0.7, 0.5, 0.4),
        "diffuse_reflection_roughness": 0.5,
        "metalness": 0.0,
        "specular_reflection_weight": 0.5,
        "specular_reflection_roughness": 0.5,
        "subsurface_transmission_color": (0.95, 0.5, 0.4),
        "subsurface_scattering_color": (0.95, 0.5, 0.4),
        "subsurface_weight": 0.1,
        "subsurface_scale": 0.3,
        "coat_weight": 0.12,
    },
    # Organ-level overrides. enhance_meshes consults these by organ
    # name *before* falling back to the containing group's params, so
    # liver/spleen/kidney get their dedicated look despite being in
    # the soft_tissue group of the taxonomy.
    "liver": {
        "name": "Liver",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.1,
        "diffuse_reflection_color": (0.16, 0.01, 0.01),
        "diffuse_reflection_roughness": 0.4,
        "metalness": 0.0,
        "specular_reflection_weight": 0.01,
        "specular_reflection_roughness": 0.5,
        "subsurface_transmission_color": (0.7, 0.2, 0.15),
        "subsurface_scattering_color": (0.7, 0.2, 0.15),
        "subsurface_weight": 0.08,
        "subsurface_scale": 0.15,
        "coat_weight": 0.01,
    },
    "spleen": {
        "name": "Spleen",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.1,
        "diffuse_reflection_color": (0.45, 0.08, 0.15),
        "diffuse_reflection_roughness": 0.4,
        "metalness": 0.0,
        "specular_reflection_weight": 0.5,
        "specular_reflection_roughness": 0.5,
        "subsurface_transmission_color": (0.6, 0.15, 0.22),
        "subsurface_scattering_color": (0.6, 0.15, 0.22),
        "subsurface_weight": 0.08,
        "subsurface_scale": 0.15,
        "coat_weight": 0.1,
    },
    "kidney_right": {
        "name": "Kidney",
        "enable_diffuse_transmission": True,
        "diffuse_reflection_weight": 0.1,
        "diffuse_reflection_color": (0.45, 0.13, 0.12),
        "diffuse_reflection_roughness": 0.35,
        "metalness": 0.0,
        "specular_reflection_weight": 0.5,
        "specular_reflection_roughness": 0.5,
        "subsurface_transmission_color": (0.6, 0.18, 0.15),
        "subsurface_scattering_color": (0.6, 0.18, 0.15),
        "subsurface_weight": 0.085,
        "subsurface_scale": 0.18,
        "coat_weight": 0.1,
    },
}
# kidney_left and kidney share the same look as kidney_right ("kidney" is a
# convenience alias kept for the CLI and other generic callers).
DEFAULT_RENDER_PARAMS["kidney_left"] = DEFAULT_RENDER_PARAMS["kidney_right"]
DEFAULT_RENDER_PARAMS["kidney"] = DEFAULT_RENDER_PARAMS["kidney_right"]


class USDAnatomyTools(PhysioTwin4DBase):
    """Apply OmniSurface materials to anatomy mesh prims in a USD stage.

    The instance attribute :attr:`render_params` is initialized from the
    module-level :data:`DEFAULT_RENDER_PARAMS` (deep copy per instance, so
    in-place edits stay local). See the module docstring for how to add new
    groups/organs.
    """

    def __init__(self, stage: Any, log_level: int | str = logging.INFO) -> None:
        """Initialize USDAnatomyTools.

        Args:
            stage: USD stage to work with. May be ``None`` when the instance
                is only used for color look-ups (e.g. via
                :meth:`get_anatomy_diffuse_color`).
            log_level: Logging level (default: logging.INFO).
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)
        self.stage = stage
        # Per-instance copy so per-instance mutations don't leak into other
        # USDAnatomyTools instances.
        self.render_params: dict[str, dict[str, Any]] = {
            key: dict(params) for key, params in DEFAULT_RENDER_PARAMS.items()
        }

    def get_anatomy_types(self) -> list[str]:
        """Return list of registered render-param keys (groups + organ overrides)."""
        return list(self.render_params.keys())

    def get_anatomy_diffuse_color(
        self, anatomy_type: str
    ) -> tuple[float, float, float]:
        """Return the diffuse reflection RGB color for the given group/organ.

        This accessor does not require a USD stage and may be called on an instance
        created with ``stage=None`` purely for color look-up purposes.

        Args:
            anatomy_type: A registered render-params key (e.g. ``"heart"``,
                ``"lung"``, ``"liver"``).

        Returns:
            RGB tuple of floats in ``[0, 1]``.

        Raises:
            ValueError: If *anatomy_type* is not registered.
        """
        params = self.render_params.get(anatomy_type.lower())
        if params is None:
            raise ValueError(
                f"Unknown anatomy_type '{anatomy_type}'. "
                f"Supported: {', '.join(self.get_anatomy_types())}"
            )
        color = params["diffuse_reflection_color"]
        return (float(color[0]), float(color[1]), float(color[2]))

    def apply_anatomy_material_to_mesh(self, mesh_path: str, anatomy_type: str) -> None:
        """Apply an anatomic OmniSurface material to a single mesh prim by type.

        Args:
            mesh_path: USD path to the mesh prim (e.g. "/World/Meshes/MyMesh").
            anatomy_type: A registered render-params key (group or organ
                override). See :meth:`get_anatomy_types`.

        Raises:
            ValueError: If mesh_path is invalid or anatomy_type is not registered.
        """
        params = self.render_params.get(anatomy_type.lower())
        if params is None:
            raise ValueError(
                f"Unknown anatomy_type '{anatomy_type}'. "
                f"Supported: {', '.join(self.get_anatomy_types())}"
            )
        prim = self.stage.GetPrimAtPath(mesh_path)
        if not prim.IsValid():
            raise ValueError(f"Invalid prim at path: {mesh_path}")
        if not prim.IsA(UsdGeom.Mesh):
            raise ValueError(f"Prim at {mesh_path} is not a Mesh")
        self.apply_anatomy_material_to_prim(prim, params)

    def apply_anatomy_material_to_prim(
        self, prim: Any, material_params: Mapping[str, Any]
    ) -> None:
        """Corrected material application with Omniverse-specific fixes"""

        # 1. Unique material path using prim's full path hierarchy
        material_path = Sdf.Path(f"/World/Looks/OmniSurface_{material_params['name']}")
        shader_path = material_path.AppendPath("Shader")

        material = UsdShade.Material.Get(self.stage, material_path)
        if not material or not material.GetPrim().IsValid():
            # prim.CreateDisplayColorAttr().Set(material_params["diffuse_reflection_color"])

            material = UsdShade.Material.Define(self.stage, material_path)
            shader = UsdShade.Shader.Define(self.stage, shader_path)

            # 2. MDL-Context Shader Definition (REQUIRED for Omniverse)
            shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
            shader.SetSourceAsset("OmniSurface.mdl", "mdl")
            shader.SetSourceAssetSubIdentifier("OmniSurface", "mdl")

            # 3. Set the parameters
            shader.CreateInput(
                "enable_diffuse_transmission", Sdf.ValueTypeNames.Bool
            ).Set(material_params["enable_diffuse_transmission"])
            shader.CreateInput(
                "diffuse_reflection_weight", Sdf.ValueTypeNames.Float
            ).Set(material_params["diffuse_reflection_weight"])
            shader.CreateInput(
                "diffuse_reflection_color", Sdf.ValueTypeNames.Color3f
            ).Set(material_params["diffuse_reflection_color"])
            shader.CreateInput(
                "diffuse_reflection_roughness", Sdf.ValueTypeNames.Float
            ).Set(material_params["diffuse_reflection_roughness"])
            shader.CreateInput("metalness", Sdf.ValueTypeNames.Float).Set(
                material_params["metalness"]
            )
            shader.CreateInput(
                "specular_reflection_weight", Sdf.ValueTypeNames.Float
            ).Set(material_params["specular_reflection_weight"])
            shader.CreateInput(
                "specular_reflection_roughness", Sdf.ValueTypeNames.Float
            ).Set(material_params["specular_reflection_roughness"])
            shader.CreateInput(
                "subsurface_transmission_color", Sdf.ValueTypeNames.Color3f
            ).Set(material_params["subsurface_transmission_color"])
            shader.CreateInput(
                "subsurface_scattering_color", Sdf.ValueTypeNames.Color3f
            ).Set(material_params["subsurface_scattering_color"])
            shader.CreateInput("subsurface_weight", Sdf.ValueTypeNames.Float).Set(
                material_params["subsurface_weight"]
            )
            shader.CreateInput("subsurface_scale", Sdf.ValueTypeNames.Float).Set(
                material_params["subsurface_scale"]
            )
            shader.CreateInput("coat_weight", Sdf.ValueTypeNames.Float).Set(
                material_params["coat_weight"]
            )

            # 4. Connect the shader's output to the material's surface output for the MDL render context.
            material.CreateSurfaceOutput("mdl").ConnectToSource(
                shader.ConnectableAPI(), "out"
            )
            material.CreateDisplacementOutput("mdl").ConnectToSource(
                shader.ConnectableAPI(), "out"
            )

        # 5. Bind the material to the mesh primitive.
        binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
        binding_api.Bind(material)

    def enhance_meshes(self, segmentator: Any) -> None:
        """Apply per-organ OmniSurface materials to every matching mesh prim.

        Walks the segmenter's :class:`AnatomyTaxonomy` and applies a material
        to each mesh prim whose leaf name matches an organ name in any group.
        An organ-level entry in :attr:`render_params` (e.g. ``"liver"``,
        ``"spleen"``, ``"kidney_left"``) takes precedence over the entry for
        the containing group.

        Anatomy grouping is performed upstream by ConvertVTKToUSD, which
        writes labeled prims under ``/World/{basename}/{type}/{label_name}``.
        This method only needs to apply materials; it does not move prims.

        Args:
            segmentator: A :class:`SegmentAnatomyBase` instance whose
                ``taxonomy`` attribute holds the group/organ structure.
        """
        taxonomy = segmentator.taxonomy

        # Build organ_name -> render params dict in one pass. Organ-level
        # overrides win over the containing group's params; if neither is
        # registered, fall back to the "other" entry (always present in
        # DEFAULT_RENDER_PARAMS, so the lookup is safe).
        organ_params: dict[str, dict[str, Any]] = {}
        default_params: dict[str, Any] = self.render_params["other"]
        for group_name in taxonomy.group_names():
            group_params = self.render_params.get(group_name, default_params)
            for organ_name in taxonomy.labels_in_group(group_name).values():
                organ_params[organ_name] = self.render_params.get(
                    organ_name, group_params
                )

        for prim in self.stage.Traverse():
            mesh_prim = UsdGeom.Mesh(prim)
            if not mesh_prim:
                continue
            prim_name = prim.GetName()
            # ConvertVTKToUSD may prefix prim names with an index like
            # "frame0_<organ>"; accept the suffix as a match too.
            prim_sub_name = "_".join(prim_name.split("_")[1:])
            params = organ_params.get(prim_name) or organ_params.get(prim_sub_name)
            if params is None:
                continue
            self.apply_anatomy_material_to_prim(prim, params)
