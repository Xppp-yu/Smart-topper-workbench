# SLP Canonical Sample v0.1 — Examples and Schema Reference

This document accompanies the A05 stage report. It contains:

* the canonical sample JSON schema (also on disk at
  `configs/annotations/slp_canonical_sample_v0.1.schema.json`);
* a worked provenance example for a healthy danaLab sample;
* a worked quarantine example for a simLab cover2 sample whose `depthRaw`
  modality is structurally absent;
* a short glossary of the A04 fields the adapter preserves verbatim.

The examples below are extracted from a real SLP run on
`E:/TeamProjects/datasets/smart-topper/SLP2022/SLP` via
`scripts/build_slp_canonical_samples.py`. The full files are kept in:

* `docs/stage_reports/examples/slp_canonical_provenance_example_v0.1.json`
* `docs/stage_reports/examples/slp_canonical_quarantine_example_v0.1.json`

## 1. JSON Schema Highlights

* `sample_id` follows the A03 contract
  `slp::<setting>::<subject_id>::<cover_condition>::<frame_index:06d>`.
* `coordinate_frame` is the constant
  `raw_dataset_pixel_coordinates_no_offset` until A04 confirms otherwise.
* `coordinate_origin_status` is the constant
  `UNRESOLVED_RAW_DATASET_COORDINATES_NO_OFFSET_APPLIED`.
* `frame` carries only per-frame modality URIs and the A03 missing /
  expected-missing / ambiguous signals. It does not carry labels.
* `joint.j1_status` is the constant
  `not_generated_A04_direction_unresolved_see_homography_contract`.
* `joint.homography_contracts.<modality>.direction_status` must match
  `^(BLOCKED_|UNRESOLVED_).+`. A confirmed direction is not encodable in
  this schema.
* `region.annotation_count == 0` and
  `region.can_be_used_as_training_truth == false` are both required. A05
  does not generate region truth.
* `provenance.subject_split_applied`,
  `provenance.review_status_applied`,
  `provenance.model_prediction_applied`,
  `provenance.semantic_direction_auto_selected`,
  `provenance.coordinate_origin_auto_shifted` and
  `provenance.silent_imputation` are all required to be `false`.

## 2. Provenance Example (healthy danaLab sample)

Sample: `slp::danaLab::00001::uncover::000001` (RGB, IR, depth, PM, depthRaw
all present, J0 source mat files present, all three A04 homography
matrices invertible, direction still marked `UNRESOLVED_*` because A04 has
not formally confirmed it). The full record is in
`slp_canonical_provenance_example_v0.1.json`.

The key fields, abbreviated:

```json
{
  "sample_id": "slp::danaLab::00001::uncover::000001",
  "setting": "danaLab",
  "subject_id": "00001",
  "cover_condition": "uncover",
  "frame_index": 1,
  "coordinate_frame": "raw_dataset_pixel_coordinates_no_offset",
  "coordinate_origin_status": "UNRESOLVED_RAW_DATASET_COORDINATES_NO_OFFSET_APPLIED",
  "frame": {
    "modality_uris": {
      "RGB": "danaLab/00001/RGB/uncover/image_000001.png",
      "IR": "danaLab/00001/IR/uncover/image_000001.png",
      "IRraw": "danaLab/00001/IRraw/uncover/000001.npy",
      "depth": "danaLab/00001/depth/uncover/image_000001.png",
      "depthRaw": "danaLab/00001/depthRaw/uncover/000001.npy",
      "PM": "danaLab/00001/PM/uncover/image_000001.png"
    },
    "missing_modalities": [],
    "expected_missing_modalities": [],
    "ambiguous_modalities": [],
    "uri_existence_flags": {
      "RGB": "present", "IR": "present", "IRraw": "present",
      "depth": "present", "depthRaw": "present", "PM": "present"
    }
  },
  "joint": {
    "j0_source_uris": {
      "RGB": "danaLab/00001/joints_gt_RGB.mat",
      "IR": "danaLab/00001/joints_gt_IR.mat"
    },
    "j0_present": {"RGB": true, "IR": true},
    "j0_artifact_count": 14,
    "joint_provenance_status": "j0_only_j1_pending_a04_direction_resolution",
    "j1_status": "not_generated_A04_direction_unresolved_see_homography_contract",
    "homography_contracts": {
      "RGB": {
        "modality": "RGB",
        "matrix_uri": "danaLab/00001/align_PTr_RGB.npy",
        "matrix_present": true,
        "invertible": true,
        "direction_status": "UNRESOLVED_REQUIRES_DOCUMENT_AND_OVERLAY_REVIEW",
        "coordinate_origin_status": "UNRESOLVED_RAW_DATASET_COORDINATES_NO_OFFSET_APPLIED",
        "probe_roundtrip_max_error": 1.15e-13,
        "direct_joint_in_bounds_rate": 0.9952,
        "inverse_joint_in_bounds_rate": 0.0,
        "error_codes": [],
        "blocked": false,
        "unresolved_direction": true
      }
    }
  },
  "region": {
    "schema_version": "slp_region_annotation_v0.1",
    "placeholder_status": "A10_to_A17_pending_no_training_truth_generated_by_a05",
    "annotation_count": 0,
    "annotations": [],
    "can_be_used_as_training_truth": false
  },
  "provenance": {
    "task_id": "TASK-SLP-A05-CANONICAL-ADAPTER-v0.1",
    "adapter_version": "slp_canonical_adapter_v0.1",
    "canonical_schema_version": "slp_canonical_sample_v0.1",
    "pairing_method": "explicit_frame_index_join_via_a03",
    "semantic_direction_auto_selected": false,
    "coordinate_origin_auto_shifted": false,
    "silent_imputation": false,
    "subject_split_applied": false,
    "review_status_applied": false,
    "model_prediction_applied": false
  },
  "quality_flags": [
    "coordinate_origin_unresolved",
    "homography_unresolved_IR",
    "homography_unresolved_RGB",
    "homography_unresolved_depth",
    "region_placeholder_only"
  ],
  "quarantine": false,
  "quarantine_reasons": []
}
```

Notes:

* The "soft" warnings (`homography_unresolved_*`, `coordinate_origin_unresolved`,
  `region_placeholder_only`) are surfaced on every sample. They are NOT
  quarantine triggers, because A04 has not blocked the sample — it has only
  refused to confirm the semantic direction.
* The A04 audit row for `RGB` reports `direct_joint_in_bounds_rate = 0.9952`
  with `inverse_joint_in_bounds_rate = 0.0`. The A05 adapter carries those
  numbers verbatim; it does not pick a direction.

## 3. Quarantine Example (simLab `cover2` `depthRaw` missing)

Sample: `slp::simLab::00003::cover2::000001`. The full record is in
`slp_canonical_quarantine_example_v0.1.json`. Abbreviated key fields:

```json
{
  "sample_id": "slp::simLab::00003::cover2::000001",
  "setting": "simLab",
  "subject_id": "00003",
  "cover_condition": "cover2",
  "frame_index": 1,
  "frame": {
    "modality_uris": {
      "RGB": "simLab/00003/RGB/cover2/image_000001.png",
      "IR":  "simLab/00003/IR/cover2/image_000001.png",
      "IRraw": "simLab/00003/IRraw/cover2/000001.npy",
      "depth": "simLab/00003/depth/cover2/image_000001.png",
      "depthRaw": "",
      "PM": ""
    },
    "missing_modalities": ["depthRaw"],
    "expected_missing_modalities": ["PM"],
    "ambiguous_modalities": [],
    "uri_existence_flags": {
      "RGB": "present", "IR": "present", "IRraw": "present",
      "depth": "present", "depthRaw": "absent", "PM": "absent"
    }
  },
  "joint": {
    "j0_source_uris": {
      "RGB": "simLab/00003/joints_gt_RGB.mat",
      "IR":  "simLab/00003/joints_gt_IR.mat"
    },
    "j0_present": {"RGB": true, "IR": true}
  },
  "quality_flags": [
    "coordinate_origin_unresolved",
    "homography_unresolved_IR",
    "homography_unresolved_RGB",
    "homography_unresolved_depth",
    "missing_depthRaw",
    "region_placeholder_only"
  ],
  "quarantine": true,
  "quarantine_reasons": ["missing_modality:depthRaw"]
}
```

Notes:

* `PM` is `expected_missing_modalities` (simLab has no PM); it does NOT
  trigger quarantine. `depthRaw` is `missing_modalities` (the structural
  cover2 absence from the S0 inventory); it DOES trigger quarantine and
  is the only hard reason.
* The J0 mat files exist; `j0_present` is `{RGB: true, IR: true}`. There is
  no "missing J0" reason, so this sample is quarantined only on the depthRaw
  slot.
* The A04 homography contracts are still attached and remain
  `unresolved_direction: true`. They are soft warnings, not quarantine
  triggers.

## 4. A04 Field Glossary (carried verbatim by the adapter)

| A04 audit field | Canonical field | Notes |
|---|---|---|
| `direction_status` | `joint.homography_contracts.<modality>.direction_status` | Kept as `UNRESOLVED_*` / `BLOCKED_*`. Adapter never picks. |
| `invertible` | `joint.homography_contracts.<modality>.invertible` and `.blocked` | `blocked` is derived as `not invertible` or direction `BLOCKED_*`. |
| `probe_roundtrip_max_error` | `joint.homography_contracts.<modality>.probe_roundtrip_max_error` | Verbatim, may be `null`. |
| `direct_joint_in_bounds_rate` | `joint.homography_contracts.<modality>.direct_joint_in_bounds_rate` | Verbatim, may be `null`. |
| `inverse_joint_in_bounds_rate` | `joint.homography_contracts.<modality>.inverse_joint_in_bounds_rate` | Verbatim, may be `null`. |
| `coordinate_origin_status` | top-level `coordinate_origin_status` and per-contract field | Always `UNRESOLVED_RAW_DATASET_COORDINATES_NO_OFFSET_APPLIED` until A04 confirms. |
| `error_codes` | `joint.homography_contracts.<modality>.error_codes` | `;`-split into a list. |
| `matrix_uri` | `joint.homography_contracts.<modality>.matrix_uri` | Subject-level relative URI. |
| `matrix_present` | `joint.homography_contracts.<modality>.matrix_present` | Verbatim. |
