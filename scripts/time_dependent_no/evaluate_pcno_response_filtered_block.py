#!/usr/bin/env python3
"""Calibrate the W26-L5 fixed response-filtered SP19 two-state block."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.time_dependent_no import (
    evaluate_pcno_cross_resolution_teacher_forced as collection,
)
from scripts.time_dependent_no import (
    evaluate_pcno_fine_discrepancy_correction as parent,
)
from utility.time_dependent_no.pcno_artifacts import (
    atomic_write_json_with_paths as atomic_write_json,
)
from utility.time_dependent_no.pcno_artifacts import sha256_file, sha256_files
from utility.time_dependent_no.pcno_artifacts import (
    write_csv_with_paths as write_csv,
)
from utility.time_dependent_no.pcno_cross_resolution_correction import (
    prepare_common_native_inputs,
)
from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
    CALIBRATION_CASE_IDS,
    CALIBRATION_GROUPS,
    EVALUATION_CASE_IDS,
    EVALUATION_STRENGTHS,
    FRONT_CONTROL_KEYS,
    INTEGRAL_COMPONENT_NAMES,
    POSITION_SUFFIXES,
    verify_payload_sha256,
    with_payload_sha256,
)
from utility.time_dependent_no.pcno_euler2d import PCNOEuler2DShardStore
from utility.time_dependent_no.pcno_resolution_transfer import (
    conservative_admissibility_summary,
    load_resolution_reference,
    predict_resolution_sample,
    reference_at_resolution,
    weighted_scaled_rms,
)
from utility.time_dependent_no.pcno_response_filtered_block import (
    FROZEN_POSITION_BUFFER_THRESHOLD,
    FROZEN_POSITION_TRUST_THRESHOLD,
    TransverseVelocityPositionDescriptor,
    buffered_frozen_offset_position_route,
    fixed_late_terminal_ramp_gain,
    monotone_persistence_gain,
    phase_triggered_terminal_ramp_gain,
    position_trusts_cross_resolution,
    projected_shadow_tether,
    recurrent_response_filtered_blocks,
    relaxed_projected_shadow_tether,
    slew_limited_projected_shadow_tether,
    synchronized_persistence_probe,
    synchronized_response_filtered_block,
    synchronized_shadow_anchored_block,
    target_free_front_branch_audit,
    transverse_velocity_position_descriptor,
)
from utility.time_dependent_no.pcno_sparse_modal_correction import (
    FROZEN_ACTIVE_CELLS,
    synchronized_sparse_modal_step,
)
from utility.time_dependent_no.shock_vortex_family import (
    REFERENCE_ARTIFACT_SCHEMA,
    family_case_provenance,
    load_shock_vortex_family_manifest,
)

WORKING_ID = "W26-L5-P6-RFB19-A2-CAL"
RESCORE_WORKING_ID = "W26-L5-P6-RFB19-A2-R1"
EVALUATION_WORKING_ID = "W26-L5-P6-RFB19-A3-EVAL"
ROLLOUT_WORKING_ID = "W26-L5-P6-RFB19-A4-ROLL30"
SHADOW_WORKING_ID = "W26-L5-P6-RFB19-A5-SHADOW30"
SHADOW_RESCORE_WORKING_ID = "W26-L5-P6-RFB19-A5-R1"
STRENGTH_OOD_WORKING_ID = "W26-L5-P6-RFB19-A6-SOOD-E13"
STRUCTURAL_GATE_WORKING_ID = "W26-L5-P6-RFB19-A7-STRUCT-E13"
PROSPECTIVE_STRUCTURAL_WORKING_ID = "W26-L5-P6-RFB19-A8-PROS-E12"
SHADOW_CANDIDATE_WORKING_ID = "W26-L5-P6-RFB19-A9-SHADOW-DIAG-E12"
SHADOW_CANDIDATE_RESCORE_WORKING_ID = "W26-L5-P6-RFB19-A9-R1"
WARM_START_WORKING_ID = "W26-L5-P6-RFB19-A10-WARM5-E12"
PROJECTED_COAST_WORKING_ID = "W26-L5-P6-RFB19-A11-PROJCOAST-E12"
FROZEN_OFFSET_WORKING_ID = "W26-L5-P6-RFB19-A12-FROZENOFFSET-E12"
BUFFERED_OFFSET_RESCORE_WORKING_ID = "W26-L5-P6-RFB19-A12-R1-BUFFER"
BUFFERED_OFFSET_WORKING_ID = "W26-L5-P6-RFB19-A13-BUFFEREDOFFSET-E12"
PROSPECTIVE_BUFFERED_OFFSET_WORKING_ID = "W26-L5-P6-RFB19-A14-PROS-BUFFEREDOFFSET-E14"
PERSISTENCE_GAIN_WORKING_ID = "W26-L5-P6-RFB19-A15-PERSISTENCE-E12"
TERMINAL_RAMP_WORKING_ID = "W26-L5-P6-RFB19-A16-TERMINAL-RAMP-E12"
FIXED_LATE_RAMP_WORKING_ID = "W26-L5-P6-RFB19-A17-FIXED-LATE-RAMP-E12"
FIXED_LATE_RAMP_RESCORE_WORKING_ID = "W26-L5-P6-RFB19-A17-R1"
SLEW_LIMITED_TETHER_WORKING_ID = "W26-L5-P6-RFB19-A18-SLEW-TETHER-E12"
RELAXED_TETHER_WORKING_ID = "W26-L5-P6-RFB19-A19-RELAX-TETHER-E12"
BUFFERED_RELAXED_TETHER_RESCORE_WORKING_ID = (
    "W26-L5-P6-RFB19-A22-BUFFERED-RELAX-RESCORE-E12"
)
BUFFERED_RELAXED_TETHER_REPLAY_WORKING_ID = (
    "W26-L5-P6-RFB19-A22-R1-BUFFERED-RELAX-REPLAY-E12"
)
BUFFERED_RELAXED_TETHER_E14_WORKING_ID = (
    "W26-L5-P6-RFB19-A23-FROZEN-BUFFERED-RELAX-E14"
)
PREFLIGHT_SCHEMA = "pcno_response_filtered_block_preflight_v1"
EVALUATION_PREFLIGHT_SCHEMA = "pcno_response_filtered_block_evaluation_preflight_v1"
RESULT_SCHEMA = "pcno_response_filtered_block_calibration_v1"
RESCORE_SCHEMA = "pcno_response_filtered_block_rescore_v1"
EVALUATION_SCHEMA = "pcno_response_filtered_block_evaluation_v1"
ROLLOUT_PREFLIGHT_SCHEMA = "pcno_response_filtered_block_rollout_preflight_v1"
ROLLOUT_SCHEMA = "pcno_response_filtered_block_rollout_v1"
SHADOW_PREFLIGHT_SCHEMA = "pcno_response_filtered_block_shadow_preflight_v1"
SHADOW_SCHEMA = "pcno_response_filtered_block_shadow_rollout_v1"
SHADOW_RESCORE_SCHEMA = "pcno_response_filtered_block_shadow_rescore_v1"
STRENGTH_OOD_PREFLIGHT_SCHEMA = "pcno_response_filtered_block_strength_ood_preflight_v1"
STRENGTH_OOD_SCHEMA = "pcno_response_filtered_block_strength_ood_rollout_v1"
STRUCTURAL_GATE_PREFLIGHT_SCHEMA = (
    "pcno_response_filtered_block_structural_gate_preflight_v1"
)
STRUCTURAL_GATE_SCHEMA = "pcno_response_filtered_block_structural_gate_rollout_v1"
PROSPECTIVE_STRUCTURAL_PREFLIGHT_SCHEMA = (
    "pcno_response_filtered_block_prospective_structural_preflight_v1"
)
PROSPECTIVE_STRUCTURAL_SCHEMA = (
    "pcno_response_filtered_block_prospective_structural_rollout_v1"
)
SHADOW_CANDIDATE_PREFLIGHT_SCHEMA = (
    "pcno_response_filtered_block_shadow_candidate_preflight_v1"
)
SHADOW_CANDIDATE_SCHEMA = "pcno_response_filtered_block_shadow_candidate_v1"
SHADOW_CANDIDATE_RESCORE_SCHEMA = (
    "pcno_response_filtered_block_shadow_candidate_rescore_v1"
)
WARM_START_PREFLIGHT_SCHEMA = "pcno_response_filtered_block_warm_start_preflight_v1"
WARM_START_SCHEMA = "pcno_response_filtered_block_warm_start_rollout_v1"
PROJECTED_COAST_PREFLIGHT_SCHEMA = (
    "pcno_response_filtered_block_projected_coast_preflight_v1"
)
PROJECTED_COAST_SCHEMA = "pcno_response_filtered_block_projected_coast_rollout_v1"
FROZEN_OFFSET_PREFLIGHT_SCHEMA = (
    "pcno_response_filtered_block_frozen_offset_preflight_v1"
)
FROZEN_OFFSET_SCHEMA = "pcno_response_filtered_block_frozen_offset_rollout_v1"
BUFFERED_OFFSET_RESCORE_SCHEMA = (
    "pcno_response_filtered_block_buffered_offset_rescore_v1"
)
BUFFERED_OFFSET_PREFLIGHT_SCHEMA = (
    "pcno_response_filtered_block_buffered_offset_preflight_v1"
)
BUFFERED_OFFSET_SCHEMA = "pcno_response_filtered_block_buffered_offset_rollout_v1"
PROSPECTIVE_BUFFERED_OFFSET_PREFLIGHT_SCHEMA = (
    "pcno_response_filtered_block_prospective_buffered_offset_preflight_v1"
)
PROSPECTIVE_BUFFERED_OFFSET_SCHEMA = (
    "pcno_response_filtered_block_prospective_buffered_offset_rollout_v1"
)
PERSISTENCE_GAIN_PREFLIGHT_SCHEMA = (
    "pcno_response_filtered_block_persistence_gain_preflight_v1"
)
PERSISTENCE_GAIN_SCHEMA = "pcno_response_filtered_block_persistence_gain_rollout_v1"
TERMINAL_RAMP_PREFLIGHT_SCHEMA = (
    "pcno_response_filtered_block_terminal_ramp_preflight_v1"
)
TERMINAL_RAMP_SCHEMA = "pcno_response_filtered_block_terminal_ramp_rollout_v1"
FIXED_LATE_RAMP_PREFLIGHT_SCHEMA = (
    "pcno_response_filtered_block_fixed_late_ramp_preflight_v1"
)
FIXED_LATE_RAMP_SCHEMA = "pcno_response_filtered_block_fixed_late_ramp_rollout_v1"
FIXED_LATE_RAMP_RESCORE_SCHEMA = (
    "pcno_response_filtered_block_fixed_late_ramp_rescore_v1"
)
SLEW_LIMITED_TETHER_PREFLIGHT_SCHEMA = (
    "pcno_response_filtered_block_slew_limited_tether_preflight_v1"
)
SLEW_LIMITED_TETHER_SCHEMA = (
    "pcno_response_filtered_block_slew_limited_tether_rollout_v1"
)
RELAXED_TETHER_PREFLIGHT_SCHEMA = (
    "pcno_response_filtered_block_relaxed_tether_preflight_v1"
)
RELAXED_TETHER_SCHEMA = "pcno_response_filtered_block_relaxed_tether_rollout_v1"
BUFFERED_RELAXED_TETHER_RESCORE_SCHEMA = (
    "pcno_response_filtered_block_buffered_relaxed_tether_rescore_v1"
)
BUFFERED_RELAXED_TETHER_REPLAY_PREFLIGHT_SCHEMA = (
    "pcno_response_filtered_block_buffered_relaxed_tether_replay_preflight_v1"
)
BUFFERED_RELAXED_TETHER_REPLAY_SCHEMA = (
    "pcno_response_filtered_block_buffered_relaxed_tether_replay_v1"
)
BUFFERED_RELAXED_TETHER_E14_PREFLIGHT_SCHEMA = (
    "pcno_response_filtered_block_buffered_relaxed_tether_e14_preflight_v1"
)
BUFFERED_RELAXED_TETHER_E14_SCHEMA = (
    "pcno_response_filtered_block_buffered_relaxed_tether_e14_rollout_v1"
)
SHARD_NATIVE_REFERENCE_SCHEMA = "pcno_shard_native_reference_v1"
SHARD_NATIVE_REFERENCE_AUDIT_SCHEMA = "pcno_shard_native_reference_audit_v1"
SHARD_NATIVE_REFERENCE_AUDIT_WORKING_ID = "W26-L5-P6-RFB19-A14-R0-SHARDTRUTH"
SHARD_NATIVE_A13_REPLAY_ABSOLUTE_TOLERANCE = 1.0e-5
EXPECTED_P6_SHARD_NATIVE_REFERENCE_AUDIT_SHA256 = (
    "5917345022d25c9e81a2a18f13f05d0c3cfe55b85945fb0d536e5f4b82174af3"
)
EXPECTED_P6_SHARD_NATIVE_REFERENCE_AUDIT_PAYLOAD_SHA256 = (
    "02986c4f4596726448aab62aab24be1ca73e2aa1df96a58c3d75baf53b69767b"
)
INPUT_CALLS = tuple(range(29))
EVALUATION_GROUPS = {
    f"e{strength:02d}": tuple(
        f"sv_e{strength:02d}_{suffix}" for suffix in POSITION_SUFFIXES
    )
    for strength in EVALUATION_STRENGTHS
}
NATIVE_RESOLUTION = parent.NATIVE_RESOLUTION
STRENGTH_OOD_GROUP_ID = "strength_ood_e13"
STRENGTH_OOD_CASE_IDS = tuple(f"sv_e13_y{index:02d}" for index in range(9))
STRENGTH_OOD_ANIMATION_CASE_IDS = (
    "sv_e13_y00",
    "sv_e13_y04",
    "sv_e13_y08",
)
STRENGTH_OOD_ANIMATION_CALLS = tuple(range(0, 31, 2))
STRENGTH_OOD_ANIMATION_CONTRACT = {
    "purpose": "post_run_diagnostic_only_not_a_selector_or_gate",
    "case_ids": list(STRENGTH_OOD_ANIMATION_CASE_IDS),
    "output_calls": list(STRENGTH_OOD_ANIMATION_CALLS),
    "physical_times": [0.02 * call for call in STRENGTH_OOD_ANIMATION_CALLS],
    "native_resolution": list(NATIVE_RESOLUTION),
    "scored_output_calls": list(range(31)),
    "visualization_only_frame_stride": 2,
    "gamma": 1.4,
    "field_limits": {
        "density": [0.75, 1.25],
        "pressure": [0.65, 1.35],
    },
    "absolute_error_limits": [1.0e-5, 0.25],
    "relative_improvement_limits": [-1.0, 1.0],
    "relative_improvement_denominator_floor": 1.0e-5,
}
STRUCTURAL_GATE_ANIMATION_CONTRACT = {
    **STRENGTH_OOD_ANIMATION_CONTRACT,
    "purpose": "post_run_retrospective_structural_gate_visualization_only",
    "candidate_label": "structural-gated correction",
    "output_stem": "structural_gated_h30",
}
PROSPECTIVE_STRUCTURAL_GROUP_ID = "strength_ood_e12"
PROSPECTIVE_STRUCTURAL_CASE_IDS = tuple(f"sv_e12_y{index:02d}" for index in range(9))
PROSPECTIVE_STRUCTURAL_INTERIOR_CASE_IDS = tuple(
    f"sv_e12_y{index:02d}" for index in range(1, 8)
)
PROSPECTIVE_STRUCTURAL_ANIMATION_CASE_IDS = (
    "sv_e12_y00",
    "sv_e12_y04",
    "sv_e12_y08",
)
PROSPECTIVE_STRUCTURAL_ANIMATION_CONTRACT = {
    **STRUCTURAL_GATE_ANIMATION_CONTRACT,
    "purpose": "post_run_prospective_structural_gate_visualization_only",
    "case_ids": list(PROSPECTIVE_STRUCTURAL_ANIMATION_CASE_IDS),
    "output_stem": "prospective_structural_gated_h30",
}
WARM_START_BLOCKS = 5
WARM_START_ANIMATION_CONTRACT = {
    **PROSPECTIVE_STRUCTURAL_ANIMATION_CONTRACT,
    "purpose": "post_run_open_e12_warm_start_calibration_visualization_only",
    "candidate_label": "warm-start then native coast",
    "output_stem": "warm5_native_coast_h30",
}
PROJECTED_COAST_ANIMATION_CONTRACT = {
    **WARM_START_ANIMATION_CONTRACT,
    "purpose": "post_run_open_e12_projected_coast_calibration_visualization_only",
    "candidate_label": "warm-start then projected coast",
    "output_stem": "warm5_projected_coast_h30",
}
FROZEN_OFFSET_WARM_BLOCKS = 4
FROZEN_OFFSET_ANIMATION_CONTRACT = {
    **PROJECTED_COAST_ANIMATION_CONTRACT,
    "purpose": "post_run_open_e12_frozen_offset_calibration_visualization_only",
    "candidate_label": "four-block warm-start then frozen SP19 output offset",
    "output_stem": "warm4_frozen_sp19_offset_h30",
}
BUFFERED_OFFSET_ANIMATION_CASE_IDS = (
    "sv_e12_y00",
    "sv_e12_y01",
    "sv_e12_y04",
    "sv_e12_y08",
)
BUFFERED_OFFSET_ANIMATION_CONTRACT = {
    **FROZEN_OFFSET_ANIMATION_CONTRACT,
    "purpose": "post_run_open_e12_buffered_offset_calibration_visualization_only",
    "case_ids": list(BUFFERED_OFFSET_ANIMATION_CASE_IDS),
    "candidate_label": "target-free buffered four-block frozen SP19 output offset",
    "output_stem": "buffered_warm4_frozen_sp19_offset_h30",
}
PROSPECTIVE_BUFFERED_OFFSET_GROUP_ID = "strength_ood_e14"
PROSPECTIVE_BUFFERED_OFFSET_CASE_IDS = tuple(
    f"sv_e14_y{index:02d}" for index in range(9)
)
PROSPECTIVE_BUFFERED_OFFSET_ANIMATION_CASE_IDS = (
    "sv_e14_y00",
    "sv_e14_y01",
    "sv_e14_y04",
    "sv_e14_y08",
)
PROSPECTIVE_BUFFERED_OFFSET_ANIMATION_CONTRACT = {
    **BUFFERED_OFFSET_ANIMATION_CONTRACT,
    "purpose": "post_run_prospective_e14_buffered_offset_visualization_only",
    "case_ids": list(PROSPECTIVE_BUFFERED_OFFSET_ANIMATION_CASE_IDS),
    "candidate_label": "frozen A13 target-free buffered output offset",
    "output_stem": "prospective_buffered_warm4_frozen_sp19_offset_h30",
}
PERSISTENCE_GAIN_ANIMATION_CONTRACT = {
    **BUFFERED_OFFSET_ANIMATION_CONTRACT,
    "purpose": "post_run_open_e12_target_free_persistence_visualization_only",
    "candidate_label": "blockwise target-free persistence-gated output offset",
    "output_stem": "persistence_gated_warm4_sp19_offset_h30",
}
TERMINAL_RAMP_ANIMATION_CONTRACT = {
    **PERSISTENCE_GAIN_ANIMATION_CONTRACT,
    "purpose": "post_run_open_e12_phase_triggered_terminal_ramp_visualization_only",
    "candidate_label": "target-free phase-triggered terminal-ramp output offset",
    "output_stem": "phase_triggered_terminal_ramp_sp19_offset_h30",
}
FIXED_LATE_RAMP_ANIMATION_CONTRACT = {
    **BUFFERED_OFFSET_ANIMATION_CONTRACT,
    "purpose": "post_run_open_e12_fixed_late_ramp_visualization_only",
    "candidate_label": "truth-selected fixed late-ramp output offset",
    "output_stem": "fixed_block9_terminal_ramp_sp19_offset_h30",
}
SLEW_LIMITED_TETHER_RELATIVE_CHANGE_LIMIT = 0.10
RELAXED_TETHER_RATE = 0.10
SLEW_LIMITED_TETHER_ANIMATION_CONTRACT = {
    **BUFFERED_OFFSET_ANIMATION_CONTRACT,
    "purpose": "post_run_open_e12_slew_limited_tether_visualization_only",
    "candidate_label": "four-block warm start then recurrent slew-limited SP19 tether",
    "output_stem": "warm4_slew_limited_sp19_tether_h30",
}
RELAXED_TETHER_ANIMATION_CONTRACT = {
    **SLEW_LIMITED_TETHER_ANIMATION_CONTRACT,
    "purpose": "post_run_open_e12_relaxed_tether_visualization_only",
    "candidate_label": "four-block warm start then relaxed recurrent SP19 tether",
    "output_stem": "warm4_relaxed_sp19_tether_h30",
}
BUFFERED_RELAXED_TETHER_REPLAY_ANIMATION_CONTRACT = {
    **RELAXED_TETHER_ANIMATION_CONTRACT,
    "purpose": "post_run_open_e12_buffered_relaxed_tether_replay_visualization_only",
    "candidate_label": "buffered four-block warm start then relaxed recurrent SP19 tether",
    "output_stem": "buffered_warm4_relaxed_sp19_tether_h30",
}
BUFFERED_RELAXED_TETHER_E14_ANIMATION_CONTRACT = {
    **BUFFERED_RELAXED_TETHER_REPLAY_ANIMATION_CONTRACT,
    "purpose": "post_run_open_e14_buffered_relaxed_tether_transfer_visualization_only",
    "case_ids": list(PROSPECTIVE_BUFFERED_OFFSET_ANIMATION_CASE_IDS),
    "candidate_label": "frozen E12-buffered relaxed recurrent SP19 tether",
    "output_stem": "frozen_e12_buffered_warm4_relaxed_sp19_tether_h30",
}
A9_DENOMINATOR_FLOOR = 1.0e-12
A9_FEATURE_NAMES = (
    "initial_normalized_wall_distance",
    "first_correction_rms",
    "correction_to_native_increment",
    "full_response_rms",
    "filtered_response_rms",
    "retained_response_fraction",
    "full_response_to_first_intervention",
    "filtered_response_to_first_intervention",
    "second_to_first_intervention",
    "first_intervention_to_shadow_increment",
    "second_intervention_to_shadow_increment",
    "first_correction_filtered_response_cosine",
    "first_second_intervention_cosine",
    "full_filtered_response_cosine",
    "front_branch_changed",
)
A9_TARGET_NAMES = (
    "block_state_sse_skill",
    "block_increment_sse_skill",
    "endpoint_state_log_ratio",
)
A9_CONTROL_KEYS = (
    "rank8_state_error",
    "boundary_state_error",
    "shock_state_error",
    "vortex_state_error",
    "smooth_state_error",
    "front_position",
    "front_strength_log_ratio",
    "front_thickness_log_ratio",
    *(f"component_{name}_state_error" for name in INTEGRAL_COMPONENT_NAMES),
)
EXPECTED_P5_SHA256 = "cb5c3d856ca8d6a02b6685b880827b4c131747634e4a9efcb460c102eff55b23"
EXPECTED_P6_CALIBRATION_SHA256 = (
    "1cf4c4d2312c59037c5bf86723c01db7b49ec226d836d09f91d345f83d4452c6"
)
EXPECTED_P6_ARTIFACT_SHA256 = {
    "control_ratios.csv": (
        "6568174610408179711cf7b8e013f5cde5aadd4e93ad2880c35aed117c880638"
    ),
    "reference_checks.csv": (
        "d476847f07969e1956295a9dc986c4b01c15383fa6935e428ecb07792778b557"
    ),
    "sample_records.csv": (
        "d9880a2438a1f9697bf85aac7ae8b73cf8d8d1585b434051041a3577454710e4"
    ),
    "source_manifest.json": (
        "b6e9bdfc21565d50202ddb9b4b4b5d42847457a7408ffbb7de14ca83dd694efd"
    ),
}
EXPECTED_P6_RESCORE_SHA256 = (
    "914178d3dee21016055aee46d91d2015c4d4e762400efb9674866d9444fd3200"
)
EXPECTED_P6_EVALUATION_SHA256 = (
    "e08d7472c1e36e41bed6c849d8dab5630a88b2886745b6161c09d196ee0b2bd0"
)
EXPECTED_P6_A4_ROLLOUT_SHA256 = (
    "2742f6daafe63576b8697f83b930c12b8c1223f34039f015869742afffd189ec"
)
EXPECTED_P6_A4_ARTIFACT_SHA256 = {
    "reference_checks.csv": (
        "131ab437d714b95e4beb4245382530b3b970db25b4d445f3fa44621fbd49e631"
    ),
    "rollout_call_metrics.csv": (
        "4effbb4414b81e25319bdb8624cd906b3e85cca8bda84ee98c5fd507308dbb1f"
    ),
    "rollout_case_metrics.csv": (
        "7f010b85bb5408f038121c1a3a91f16d011c0be3934d4c6995610f182fe1672a"
    ),
    "rollout_closure.csv": (
        "fe38104d68c81e21984be7b6c4dee53dada2e5f9ea535bbb5bec8e378d02045d"
    ),
    "rollout_controls.csv": (
        "dd50c72d98ff2d2d1f305369f563d488bab338a3a1ea967ba2f37597a4ed73b9"
    ),
    "rollout_execution.csv": (
        "c2ae8183002bbe0edb43002f40992e6f2eb1312a41cceb18e4a36768694819b8"
    ),
    "source_manifest.json": (
        "ce26aceaf7f4578b0f7737b7edd6920671ec303250cc95c4b4295eb38751fd58"
    ),
}
EXPECTED_P6_A5_ROLLOUT_SHA256 = (
    "6e1494fdd1a6f00a5421bf0664524fa1290a49291957bf788856fd88c1a26ec2"
)
EXPECTED_P6_A5_ARTIFACT_SHA256 = {
    "anchor_audits.csv": (
        "d1cabd8e2bc6fc6f18d337bdfee1bd9e006e3486b657b8fcceae160b8425ca0c"
    ),
    "reference_checks.csv": (
        "f685f5b01462a609e3bf590f6997566803016c50143cc5a3dcabbac7ff73eeeb"
    ),
    "rollout_call_metrics.csv": (
        "6a78f88926446c0c60e4e5f87c1d84898f1e4a59fa14ab6404870280bb686007"
    ),
    "rollout_case_metrics.csv": (
        "7efbfde839b2837f1ba77242da7de76c9bbc18653ab4276110b7e53a3b7be311"
    ),
    "rollout_closure.csv": (
        "d51d4b2302f9bdef6e3a034e2d1110117fed57ef8210f501c1f1fd525837b9ad"
    ),
    "rollout_controls.csv": (
        "a700624c91d5f5c44bea48beea017982d77787ba14156fc8a5deca13f46202ba"
    ),
    "rollout_execution.csv": (
        "8747c6dd7043e5d4435490328735dcf4342d6d9f8b6ac17daf94a955b727df43"
    ),
    "source_manifest.json": (
        "b742023b9083c809d2df28735469ea2452b4fbe02e070f904b1182ed39b24ae9"
    ),
}
EXPECTED_P6_A5_RESCORE_SHA256 = (
    "66ede749add43ac0ed44bd0944ca7905de86aa58c98242a5faeeac9f9007aaab"
)
EXPECTED_P6_A5_RESCORE_ARTIFACT_SHA256 = {
    "explicit_call_checks.csv": (
        "a41884102569d6d6a53a21daf2cbd674e0b8502bea235868b2f27a2be8bcef8a"
    ),
    "rollout_case_metrics.csv": (
        "7efbfde839b2837f1ba77242da7de76c9bbc18653ab4276110b7e53a3b7be311"
    ),
    "rollout_controls.csv": (
        "a700624c91d5f5c44bea48beea017982d77787ba14156fc8a5deca13f46202ba"
    ),
    "source_manifest.json": (
        "f34667965be20055866e13cf221abe9709014b66e26073d1db7a9f50a4d301a2"
    ),
}
EXPECTED_P6_A6_ROLLOUT_SHA256 = (
    "e3cfaa796cefeceab7b33e1315c7f59e7d7b9b12babac269796decccf045e3d4"
)
EXPECTED_P6_A6_ARTIFACT_SHA256 = {
    "anchor_audits.csv": (
        "570e73402f0f97f130d768ad91afecc96142e588bc94f041226185107130e9fa"
    ),
    "reference_checks.csv": (
        "c3f7108239c436cd81d3ecfcdd45eea7bc0eeb0957b8606c8e65809a4f6ae1ee"
    ),
    "rollout_call_metrics.csv": (
        "2637a5fec3e9c4f882108e612c484c3a2deb972eac650355f3d83630b632e2f0"
    ),
    "rollout_case_metrics.csv": (
        "2e6ec095b869afd7e19ed7c4f2d68e1c90e3995f8f687473abfaebd36b4768ca"
    ),
    "rollout_closure.csv": (
        "a07a5e4b24af21816987a75007ce9042c40a6ea0d107f4742a170d9e2c5d536f"
    ),
    "rollout_controls.csv": (
        "1e6a5b07f62421f33e522cb165667c0c6e96eaec9b88e48aa4d8702ce2344ef1"
    ),
    "rollout_execution.csv": (
        "ed928687da184014b93d3a7f42cf6bed18a51faac673c19fa5819e97f5f6f7f1"
    ),
    "source_manifest.json": (
        "905083c81d779348daf538e5bff471fee19619f73d27765070ffb77f8036d152"
    ),
}
EXPECTED_P6_A7_ROLLOUT_SHA256 = (
    "c6dade89b5966281f0d771f543c331f7aec76840223a87ebf6efd5f6432e072b"
)
EXPECTED_P6_A7_ARTIFACT_SHA256 = {
    "anchor_audits.csv": (
        "8a0ca9f20c3c0453ebf919aaf09bffe04a89a90acd6f82c2e803ed7202c5832a"
    ),
    "front_branch_audits.csv": (
        "db5dd5f3449bc558f9b2e3e639ef9bc22af41e5e8f7ac0cd95f3ccc8b51ac1bc"
    ),
    "position_decisions.csv": (
        "cf93cb78d6c435fba82a58a0362c9b80b7b953807665bb9598df22af54ab5a54"
    ),
    "reference_checks.csv": (
        "c3f7108239c436cd81d3ecfcdd45eea7bc0eeb0957b8606c8e65809a4f6ae1ee"
    ),
    "rollout_call_metrics.csv": (
        "f3945e5465586c1ff237cc34764e461b198a64625075084e22d06bd679ad132e"
    ),
    "rollout_case_metrics.csv": (
        "48656695c4365bee610cf52255163110105644fdab1b3a4d598aa407a9ef3444"
    ),
    "rollout_closure.csv": (
        "6ea7bc8781b78270598cc7cddbfb9992b72e7bb60c814ff0a7d1f6b2855c784b"
    ),
    "rollout_controls.csv": (
        "06b373bfe6193a1dc7668d2d302adf1ac59ae40d353b2ff0a09f33689565a3a9"
    ),
    "rollout_execution.csv": (
        "81e404b9c5d914ecd3fac729d352ad541d9d199da559aa9682612ac35a9d419a"
    ),
    "source_manifest.json": (
        "eda19ab6e20fad880fa3d8761620ff5cf3c4202e2801de07c781d2a2fe67146e"
    ),
}
EXPECTED_P6_A7_ANIMATION_BUNDLE_MANIFEST_SHA256 = (
    "b46ca37fb0a472aa675aa5536542ebabcd0cc9e3eae8aba2ceac42a3d39c57ed"
)
EXPECTED_P6_A8_ROLLOUT_SHA256 = (
    "50493045b097e78221347b0cc9140de798e7a49bf67056f91f609b721461a6b9"
)
EXPECTED_P6_A8_ARTIFACT_SHA256 = {
    "anchor_audits.csv": (
        "72384c54ae65bece1349b7b70cc59afc3ffd9fd30e09b1cf0763fcb322b74634"
    ),
    "front_branch_audits.csv": (
        "47991133f984efe91d0b7f7a0791391bb95d81f971b7f91ddfbba7815c1bea0d"
    ),
    "position_decisions.csv": (
        "8ff1994235ab2554d6f6de7ac97c4ac777655e9ee3c2ec2755f5e1ca31056e04"
    ),
    "reference_checks.csv": (
        "cf4b5292e182b5e1fb9eedde2773cfe73cea5f64fd6d204a9e8022d9413c1a84"
    ),
    "rollout_call_metrics.csv": (
        "33afa110bc634ffb9cdcdd59a484e48ce67ca58fd1951a09177217264257e4b5"
    ),
    "rollout_case_metrics.csv": (
        "a93456b0cef20e54152c88735bf5015dfcac8fa8e7a97cfa8e51b8e97ae0b97c"
    ),
    "rollout_closure.csv": (
        "a9f1d15eb18c6deae1a78ff3696c72ba173c93864ce511c52a501ed8477c620e"
    ),
    "rollout_controls.csv": (
        "91f9bfc832845337e7b766ba68692a445089957e5f898abc04c753fc248515f3"
    ),
    "rollout_execution.csv": (
        "e7e83b2dda12a4043d0b3c0d7322a31fa6d08cbeca5fb0c3c866e5c40bef814a"
    ),
    "source_manifest.json": (
        "b8fa1d828e207497ec928bde906ad8bb9462e15b27d0368fecbc6146e1bc8d42"
    ),
}
EXPECTED_P6_A8_ANIMATION_BUNDLE_MANIFEST_SHA256 = (
    "06162c6199ef2bc10fa7917e00bd786dba37c18fc51a1f4316d0575c72e87e85"
)
EXPECTED_P6_A9_DIAGNOSTIC_SHA256 = (
    "384a72fd2f181365df2681127d8a3e28754e10afad16fc8117dd5f81e1153fbe"
)
A9_MISWIRED_CALL_CHECKS = {
    "shadow_call_inventory_exact",
    "shadow_call_counts_exact",
    "shadow_call_order_and_recurrence_exact",
    "accepted_post_fp32_floor_at_most_1e_6",
    "candidate_common_source_exact",
    "candidate_lookahead_inputs_exact",
}
EXPECTED_P6_A9_RESCORE_SHA256 = (
    "a17674ec9c982ddfaa299971533d9a1099951e1148f4a9399c2e0c870febdd87"
)
EXPECTED_P6_A9_RESCORE_ARTIFACT_SHA256 = {
    "contract_checks.csv": (
        "dfdae4589f4343f76933735e3e52c80b7ff4ae1070838a65196df949db163f6c"
    ),
    "source_manifest.json": (
        "557363a734b93071db59c6e95459f4da2458297c304c8ba8902db622f6ef26c2"
    ),
}
EXPECTED_P6_A10_ROLLOUT_SHA256 = (
    "7b7fe4e04ce25069770302c8465d09b149498a4509cd969891b7778dc9d70e96"
)
EXPECTED_P6_A10_ARTIFACT_SHA256 = {
    "anchor_audits.csv": (
        "5f3739048fac6a2b1c9af90626769a58f6516f20c378006ee46a18131e59f86a"
    ),
    "front_branch_audits.csv": (
        "0e7355f9e15ef9f3715641fa273f57a751cd314e552a4ba81dc26ab7decd7c53"
    ),
    "position_decisions.csv": (
        "f8a837cf7c7ae15231b6638828118bc1f3041f249fd3598fe03eb3ae6180c29e"
    ),
    "reference_checks.csv": (
        "cf4b5292e182b5e1fb9eedde2773cfe73cea5f64fd6d204a9e8022d9413c1a84"
    ),
    "rollout_call_metrics.csv": (
        "68e9d7096cfad620bf7afaf062123c24cd8eb1e7c1dfdb9c2375a81d36eded0d"
    ),
    "rollout_case_metrics.csv": (
        "4b1b51972864b544c1ef0aa98c9a36fec20cdad7916925de3ee1793a02f941e7"
    ),
    "rollout_closure.csv": (
        "25fb0986b9dc7be49e56f936c4132113fc3a6963d1d92f4b6b244a0f951ec8bb"
    ),
    "rollout_controls.csv": (
        "2db3f9aab89f4eafdce96ab15ec0a2eec66a7912f18596ca800926c65d2f7c86"
    ),
    "rollout_execution.csv": (
        "24dd99e7710eb4221d6b4e2796144f6be6104520161183d7f9e3223f0a41b942"
    ),
    "source_manifest.json": (
        "6bf49ba29b5d8a99b63733ae9df2660bd74f2bfc7043c0d4dd0afa147341a4b5"
    ),
}
EXPECTED_P6_A10_ANIMATION_BUNDLE_MANIFEST_SHA256 = (
    "837681f2db58b389040a1b920e11abd59bfdb2f5b13d6984da59fded499bc75c"
)
EXPECTED_P6_A11_ROLLOUT_SHA256 = (
    "10c560232a41cfeeb56fc293fb3202edad037e9f9437de3d1947e87d81c42fa6"
)
EXPECTED_P6_A11_ARTIFACT_SHA256 = {
    "anchor_audits.csv": (
        "5f3739048fac6a2b1c9af90626769a58f6516f20c378006ee46a18131e59f86a"
    ),
    "front_branch_audits.csv": (
        "d5bb767c8931ba005f13a6b5da42992561b26f5830cc6a58012053b4bd5a457a"
    ),
    "position_decisions.csv": (
        "f8a837cf7c7ae15231b6638828118bc1f3041f249fd3598fe03eb3ae6180c29e"
    ),
    "projected_coast_audits.csv": (
        "01c3a1c8ea939b987740e24df20f3e0fd2d52c6e2f712b5a01e5e831d0c24e74"
    ),
    "reference_checks.csv": (
        "cf4b5292e182b5e1fb9eedde2773cfe73cea5f64fd6d204a9e8022d9413c1a84"
    ),
    "rollout_call_metrics.csv": (
        "5247f0741ec06cd1a9e41abe1a4b27852d4b8f462622a21b361485d7786e0dec"
    ),
    "rollout_case_metrics.csv": (
        "1381f4d99e2294f903f32ecbe0c7d24b8314898489c5afe7d9633ca78c9e1b7e"
    ),
    "rollout_closure.csv": (
        "e7f2db0bbb5468cb44988769595c862f661c2e962b6deec0a0c81c9a56fa8fd5"
    ),
    "rollout_controls.csv": (
        "2c3630937ed5801646e268e8770e268971b3550c133a73c206baefc00e3a90bd"
    ),
    "rollout_execution.csv": (
        "0545b8391e9d0292271c0272be769d1e42adb1089ab633662e90c4eaa23faabc"
    ),
    "source_manifest.json": (
        "5e44284a98577241cd3193053fc14b7cd874030935f42c869878da3c71302324"
    ),
}
EXPECTED_P6_A11_ANIMATION_BUNDLE_MANIFEST_SHA256 = (
    "86360ee0e28ee5ee721b137e42935550256e0f499a4327a9562303deb79722d7"
)
EXPECTED_P6_A12_ROLLOUT_SHA256 = (
    "4418234c6fe50cca28c48607b8b7f09a1a2965a0f317242d29901a49a632f1a2"
)
EXPECTED_P6_A12_ARTIFACT_SHA256 = {
    "anchor_audits.csv": (
        "275fb0d47f321c3285f73a9c6f6106bd89c6af0bc650865d5e502d250829e035"
    ),
    "front_branch_audits.csv": (
        "fb5fdd7c32cab290fe2090fb622f7cca7d18fd57e6061c5f42ba25832fa95b95"
    ),
    "frozen_offset_audits.csv": (
        "aef34ae4a857ccaeae21a6b7fe50eed03aefb4369538dd3dec56ac8f35116fc5"
    ),
    "position_decisions.csv": (
        "f8a837cf7c7ae15231b6638828118bc1f3041f249fd3598fe03eb3ae6180c29e"
    ),
    "reference_checks.csv": (
        "cf4b5292e182b5e1fb9eedde2773cfe73cea5f64fd6d204a9e8022d9413c1a84"
    ),
    "rollout_call_metrics.csv": (
        "7f02641ab5769eb8408cee883f7faf988382e0b5136dd2f94f6e3a8442fbc391"
    ),
    "rollout_case_metrics.csv": (
        "2ec3471b3fafb3d06462be62b6d3cced5c7a3329b494006e424712a52d296c78"
    ),
    "rollout_closure.csv": (
        "14eb270f35f2fa499524c92d906dc346d0228a0b180d2cb462e232b5931bf34e"
    ),
    "rollout_controls.csv": (
        "f6938c3ce42a578f3b097dafedb2b1b2a2331ee445d417c1e5c2b0e3352503dd"
    ),
    "rollout_execution.csv": (
        "9368a2f7bb3b7110bbaef1101aa63fead0a60cd8b7e881ca4b9b9d0a31b6c6d7"
    ),
    "source_manifest.json": (
        "4654f997242ccfb2117d3107e18920fec5eff1087d2474f8077ab30ca6ce878e"
    ),
}
EXPECTED_P6_A12_ANIMATION_BUNDLE_MANIFEST_SHA256 = (
    "04219ac1c3fa5691da14b5aa4ebc7469704c19e12543dbd780b3639eaea96c1d"
)
EXPECTED_P6_A12_BUFFER_RESCORE_SHA256 = (
    "512362367b0a274b0a8da5a34a8f5dbad7e0a25b86838cbbf72b7af4f1a0f8de"
)
EXPECTED_P6_A12_BUFFER_RESCORE_ARTIFACT_SHA256 = {
    "gate_checks.csv": (
        "1b72ee49ac093047f9790d2cdac26b0ed09a213c1963be6590a9fe7372be2940"
    ),
    "position_routes.csv": (
        "5afc3cb51ad19bf3c0f2671cbec54c01ea02d50a8cbff96c6d99b49b2635fab8"
    ),
    "rollout_case_metrics.csv": (
        "796b43a41e6c3bfd53b24d02d8957d09e0a993dfae3a0cd01b56500c997e8b87"
    ),
    "rollout_controls.csv": (
        "3a80be03eaeaa0062d9d7c65cf657624188ffe5bc7d056b44279de18ea136efa"
    ),
    "source_manifest.json": (
        "509b7dafb9971e7680494800376bce97c6b1fd9cb2286909524e127ed8320fae"
    ),
}
EXPECTED_P6_A13_ROLLOUT_SHA256 = (
    "ee2e7f7c7497cacf281064bfcf79a92bebda817932125d6ef52a96d1d9580926"
)
EXPECTED_P6_A13_ARTIFACT_SHA256 = {
    "anchor_audits.csv": (
        "b06ffbd47c7577af01057ba2bab74b5fb802aaa8fadd4b454140c6a11d20788a"
    ),
    "front_branch_audits.csv": (
        "1dc12593d1bf884a2092739575dd30d60df1b4d5b13bdfd1f7fb644bb2856b6b"
    ),
    "frozen_offset_audits.csv": (
        "85f8555a2bcb5d84bc91e644ab0f518b8595fec2b035aefa92f352fe218c0a26"
    ),
    "position_decisions.csv": (
        "81abcb85e0dd5c7e85f1fdcda6910179bc8e7d74a498dc9ee6a7c2d42b9b7965"
    ),
    "reference_checks.csv": (
        "cf4b5292e182b5e1fb9eedde2773cfe73cea5f64fd6d204a9e8022d9413c1a84"
    ),
    "rollout_call_metrics.csv": (
        "6c4c2caef8972f1aa4c45af9e3eed7b2ced9c3533590c2df5cbe62510f14692d"
    ),
    "rollout_case_metrics.csv": (
        "796b43a41e6c3bfd53b24d02d8957d09e0a993dfae3a0cd01b56500c997e8b87"
    ),
    "rollout_closure.csv": (
        "85b39e8814f066fa2852d9291ac9bbb83671cee4436d8e7f1a9403e26c976750"
    ),
    "rollout_controls.csv": (
        "3a80be03eaeaa0062d9d7c65cf657624188ffe5bc7d056b44279de18ea136efa"
    ),
    "rollout_execution.csv": (
        "3ee51529c752248dab2df22a7f15c98f0ed7df7ac59f0a61fca043e6dc472c2a"
    ),
    "source_manifest.json": (
        "0dd41c2174935df9d2ff43f02183fdb6c96dcc5214a85fed3151252f197e7b1b"
    ),
}
EXPECTED_P6_A13_ANIMATION_BUNDLE_MANIFEST_SHA256 = (
    "65b3cd2dcbd0292914f5fae06ffe739f7dfdb9e584885a264a877af5beb5787e"
)
EXPECTED_P6_A13_SOURCE_SHA256 = {
    "docs/time_dependent_no/W26_L5_RESPONSE_FILTERED_BLOCK_PREREGISTRATION.md": (
        "a940a2d7fe0f219e97d742e30bbf1d352687f0b63ccbd4cdcb258b3b1028021b"
    ),
    "scripts/time_dependent_no/evaluate_pcno_response_filtered_block.py": (
        "0b2126226745ca2c5253ac3bd23e4c128b2271da4f0ace861c3c8b7d1360790a"
    ),
    "scripts/time_dependent_no/visualize_pcno_response_filtered_block.py": (
        "6866d67f75d74948666f614d5ed4d6a9da649d10cec78175601e16f4b9ddc901"
    ),
    "tests/time_dependent_no/test_pcno_response_filtered_block.py": (
        "3b923bd6232bcfdfe44f6dbde6ae18cef7024c77b16f311565a0396ab842310a"
    ),
    "utility/time_dependent_no/pcno_response_filtered_block.py": (
        "fd63cc9d3ddc207cb0df86060924facc092ff047ceb6497466ed966569c52f5c"
    ),
}
EXPECTED_P6_A14_ROLLOUT_SHA256 = (
    "72694e9976a93f699167d1ef6d23566e0c7ba89bf96e6586b05717fec21b90f3"
)
EXPECTED_P6_A14_PAYLOAD_SHA256 = (
    "7a373e4bee3d3c8fff8f773fc1a345b9c71d6cc829a01378a19af824850ded06"
)
EXPECTED_P6_A14_SOURCE_MANIFEST_SHA256 = (
    "6c1d161358fe45d3ea849f5b8b9b955a5e114c6c2768b40569bf003a67710a47"
)
EXPECTED_P6_A15_ROLLOUT_SHA256 = (
    "a60252a13039488cd1480d0817856005f790a4d7bb491484dc72dc1e1080eadc"
)
EXPECTED_P6_A15_PAYLOAD_SHA256 = (
    "c460cbd37b3a406dd3c964f038aa9e58d1b9ec96072b161aadbc0dd28b472ad2"
)
EXPECTED_P6_A15_SOURCE_MANIFEST_SHA256 = (
    "a734c1556147171dc69bde71ab65e11d3e73a5bb8df4717d20e2c5b956c30714"
)
EXPECTED_P6_A16_ROLLOUT_SHA256 = (
    "3352dfc934b73eb16782016bd233df485efb693eb7e90ad82fb5f441feba6174"
)
EXPECTED_P6_A16_PAYLOAD_SHA256 = (
    "0efa1f267865455957e522fa34e4a04774d7a2484ee0e104242b79ffc7ebdf96"
)
EXPECTED_P6_A16_SOURCE_MANIFEST_SHA256 = (
    "c3ee227607c11149e3486f3840a78096e2749fdc42b1cb10d39356c0fb75e941"
)
EXPECTED_P6_A17_PREFLIGHT_SHA256 = (
    "a965758071a8c4a5b9c20f79d6ad1520b5265c1c01b23475eb6991748e755cf1"
)
EXPECTED_P6_A17_PREFLIGHT_PAYLOAD_SHA256 = (
    "5d147899fc05159ea876d4c70366b49b58299689b88ad72b42125e365a3e2356"
)
EXPECTED_P6_A17_ROLLOUT_SHA256 = (
    "85fd88f4ddcb0ffe0ddd5e113c09ab0779880c5a3ae59de8799f70c7111b9fa9"
)
EXPECTED_P6_A17_PAYLOAD_SHA256 = (
    "3953746430748ecb954c8a0a69c8a6e48251e07c2d95a16a29e711ed56d3b063"
)
EXPECTED_P6_A17_SOURCE_MANIFEST_SHA256 = (
    "ca7758cb85bab1183ba2760c0d9ca15c4959920741a2ac55017e144086280ad2"
)
EXPECTED_P6_A17_ANIMATION_BUNDLE_MANIFEST_SHA256 = (
    "9ec4f75ef5b68aecd37f3ef92520803d38a09c85414eacb222bea5f4c7e126d7"
)
EXPECTED_P6_A17_ARTIFACT_SHA256 = {
    "anchor_audits.csv": (
        "b06ffbd47c7577af01057ba2bab74b5fb802aaa8fadd4b454140c6a11d20788a"
    ),
    "front_branch_audits.csv": (
        "7f2289d061f93783db817d10db8fab9ecbbb54917d6d796e8be50fa257371c61"
    ),
    "frozen_offset_audits.csv": (
        "85f8555a2bcb5d84bc91e644ab0f518b8595fec2b035aefa92f352fe218c0a26"
    ),
    "persistence_probe_audits.csv": (
        "164741a1f0c738f0604b5df812f4411882521c374d155e24b7895ec82424894c"
    ),
    "position_decisions.csv": (
        "81abcb85e0dd5c7e85f1fdcda6910179bc8e7d74a498dc9ee6a7c2d42b9b7965"
    ),
    "reference_checks.csv": (
        "cf4b5292e182b5e1fb9eedde2773cfe73cea5f64fd6d204a9e8022d9413c1a84"
    ),
    "rollout_call_metrics.csv": (
        "e16f1b0aadb4d581f7b7c2e95149cd955e0c5fe771a0dab34e45ffa0c3d9bc06"
    ),
    "rollout_case_metrics.csv": (
        "582d2cae8b99f2de3747dab7622568d78c71e8a916c9a350fc6c9c2e5d28ae68"
    ),
    "rollout_closure.csv": (
        "8bd2ab11d5b528394f3700a702e226425808575acc0b7621e7fab8fcc47b1004"
    ),
    "rollout_controls.csv": (
        "e11b4f907b680415f132d34788f4b0de6573d9897d722e07991bb3a121c8a14e"
    ),
    "rollout_execution.csv": (
        "5a2ccbd5f7d4f4eace9f3fba1b826a64c5a7bab7ade29b694e00b464f4a6da39"
    ),
    "source_manifest.json": EXPECTED_P6_A17_SOURCE_MANIFEST_SHA256,
}
EXPECTED_P6_A17_RESCORE_SHA256 = (
    "6cf01c380fe84281150cddc348fb3bb247a8694a77f7530c6bbc55c8d5a6a812"
)
EXPECTED_P6_A17_RESCORE_PAYLOAD_SHA256 = (
    "345e174570846788339aeea9a75927370699db512b7f6915802920b9abf03246"
)
EXPECTED_P6_A17_RESCORE_ARTIFACT_SHA256 = {
    "gate_checks.csv": (
        "f324736d96e6c4d721ce35e5df4cfc89ca0cf6b60cc73a58e7615de492333566"
    ),
    "source_manifest.json": (
        "5ea1bdde692f99a9c160ba7d4c7136b0981df0f01b4c7a79499b3b4827a32e15"
    ),
}
EXPECTED_P6_A18_ROLLOUT_SHA256 = (
    "3c534f0631a08d65ce2111960eeea1b79330c084f0f64afeeeef0abb01f96803"
)
EXPECTED_P6_A18_PAYLOAD_SHA256 = (
    "1c88bf6d80d6bc240ecaac17daf42e493fc0685e806762b422ef2b5ff4ca7c32"
)
EXPECTED_P6_A18_SOURCE_MANIFEST_SHA256 = (
    "268ed4242cb23b8144599e90d5533eefc91ab06faca35aa2e013e998d1bf36b2"
)
EXPECTED_P6_A18_ANIMATION_BUNDLE_MANIFEST_SHA256 = (
    "b572b0c25f7adf3b682d9d80cd9dab9c7275d821667effdb9a9d5be44a9d739c"
)
EXPECTED_P6_A18_ARTIFACT_SHA256 = {
    "anchor_audits.csv": (
        "275fb0d47f321c3285f73a9c6f6106bd89c6af0bc650865d5e502d250829e035"
    ),
    "front_branch_audits.csv": (
        "ffe0513413237da5b99c6842916b59bec614a046c3801f108bb5f11e9ac51540"
    ),
    "position_decisions.csv": (
        "9a51e7723121ce32aced6b9649de2cfa962905847ae811e3fb78237a81d7d150"
    ),
    "reference_checks.csv": (
        "cf4b5292e182b5e1fb9eedde2773cfe73cea5f64fd6d204a9e8022d9413c1a84"
    ),
    "rollout_call_metrics.csv": (
        "91067f757315f49dd1d4fe7d1d2884585796144c42404d9efe37efc1b9b90952"
    ),
    "rollout_case_metrics.csv": (
        "ed7f3d9276ac387982dd348eb0182c1a6b439c416bf30ef664fbfaccc6376d90"
    ),
    "rollout_closure.csv": (
        "67337ad3a083b8ef461d99f398652fa2b81ee1d7e825186c58055964dac9b708"
    ),
    "rollout_controls.csv": (
        "0fdc18a71d117b776fd10f336e7b2642240f11408baaa54a12dbec3a15d77d8c"
    ),
    "rollout_execution.csv": (
        "3bcb668ece23533b80fad6df4f6df10676cd4f5d7a71aecec2bcb3bd440bc971"
    ),
    "slew_limited_tether_audits.csv": (
        "a58d0343d8f3c9f738038c5b4115fc7e3fae6257b0a76290a096cab0609bd173"
    ),
    "source_manifest.json": EXPECTED_P6_A18_SOURCE_MANIFEST_SHA256,
}
SOURCE_PATHS = (
    "docs/time_dependent_no/W26_L5_RESPONSE_FILTERED_BLOCK_PREREGISTRATION.md",
    "utility/time_dependent_no/pcno_response_filtered_block.py",
    "scripts/time_dependent_no/evaluate_pcno_response_filtered_block.py",
    "scripts/time_dependent_no/visualize_pcno_response_filtered_block.py",
    "tests/time_dependent_no/test_pcno_response_filtered_block.py",
)

EXPECTED_P6_A19_ROLLOUT_SHA256 = (
    "c6675dbb5481aea31a16fbd6aef8b7a71d7403a0fb199cd0120ff2cf343b868f"
)
EXPECTED_P6_A19_PAYLOAD_SHA256 = (
    "bbe9abffbf1b4b699499eab72add357acf4ee6955a0fae9f1603579f0cd7f5dd"
)
EXPECTED_P6_A19_SOURCE_MANIFEST_SHA256 = (
    "6be32d4f6ead4578b357847ffd0912c63a8738eec68c3bd53446b473c1f062c2"
)
EXPECTED_P6_A19_ARTIFACT_SHA256 = {
    "anchor_audits.csv": (
        "275fb0d47f321c3285f73a9c6f6106bd89c6af0bc650865d5e502d250829e035"
    ),
    "front_branch_audits.csv": (
        "42d35c9258ad99cfcdaf31dd4d68293f72b4b20e559782a0ebf80ff70befa39f"
    ),
    "position_decisions.csv": (
        "9a51e7723121ce32aced6b9649de2cfa962905847ae811e3fb78237a81d7d150"
    ),
    "reference_checks.csv": (
        "cf4b5292e182b5e1fb9eedde2773cfe73cea5f64fd6d204a9e8022d9413c1a84"
    ),
    "relaxed_tether_audits.csv": (
        "8542fe2606cb87508c73a31ce33582c9621a8810eeeebce8874e79746ae884f1"
    ),
    "rollout_call_metrics.csv": (
        "6c736eb815987bd94a57e7a969e4b34d5b1706ca17d544c85b76e9a95a8c6f89"
    ),
    "rollout_case_metrics.csv": (
        "b9000d9190e7a0be32e1ef300598feba232c80b14a46c2836cbde4c0a565946a"
    ),
    "rollout_closure.csv": (
        "3fa40263aa14ebdc012bb8a81f569bc89654b5dbb96b8a0e739e78f726b16ebd"
    ),
    "rollout_controls.csv": (
        "4250e19ec7bd3b2c9a1452079eec895fbed5f8a13a43fe226299c84d5e2851d3"
    ),
    "rollout_execution.csv": (
        "0a0cb1abb9850fae1d556263a8ee11937f0f2696ba5c58e7cd2fb1507d595a28"
    ),
    "source_manifest.json": EXPECTED_P6_A19_SOURCE_MANIFEST_SHA256,
}
EXPECTED_P6_A22_RESCORE_SHA256 = (
    "0e4975e441cc608f665ae7f6b708a5c663c7e6f4687156078309036e475b3a00"
)
EXPECTED_P6_A22_RESCORE_PAYLOAD_SHA256 = (
    "af7f5e9fea3ab553284aee117871d2ea7eaabacbf876668baa840fd6ef518803"
)
EXPECTED_P6_A22_RESCORE_ARTIFACT_SHA256 = {
    "gate_checks.csv": (
        "b168c8b60feed6eda669ffcf83843f3102103addb18de35b0cd3a8192306a059"
    ),
    "position_routes.csv": (
        "5afc3cb51ad19bf3c0f2671cbec54c01ea02d50a8cbff96c6d99b49b2635fab8"
    ),
    "rollout_case_metrics.csv": (
        "86cef98c7b8d4db6827e45f24f9916398337847f4d00ec5b99ad302abe88434a"
    ),
    "rollout_controls.csv": (
        "e47f8155865771ccee84bb9fc60499612d955952e64555834855908f8581d21d"
    ),
    "selected_call_metrics.csv": (
        "2ee78222516d0640df52ec16ce36222b637999bcb8a16474b56a2105bc43d308"
    ),
    "source_manifest.json": (
        "75004d2800e6dcbd0ac9ad080dba6d1453fb2d206d6f2a993f43ca095e467098"
    ),
}
EXPECTED_P6_A22_R1_ROLLOUT_SHA256 = (
    "eb964b01dbb07da16a4ca95fe96f5497f1a87c2a68b137e7ecc2b75def521fc7"
)
EXPECTED_P6_A22_R1_PAYLOAD_SHA256 = (
    "b3f798ece9662ae54678fe59bef99b01ecaf6985b6e350ea082a865447f51125"
)
EXPECTED_P6_A22_R1_SOURCE_MANIFEST_SHA256 = (
    "b9f077bb85462364a16bc3b71a7245469d5eafc4647e0eab0ad6c7d32bc6cf4f"
)
EXPECTED_P6_A22_R1_ANIMATION_BUNDLE_MANIFEST_SHA256 = (
    "575b0878b900ba6fc73e1d026e706a1b42f5c91ccd51e268ec9d1e8bb42c8aa7"
)
EXPECTED_P6_A22_R1_ARTIFACT_SHA256 = {
    "anchor_audits.csv": (
        "b06ffbd47c7577af01057ba2bab74b5fb802aaa8fadd4b454140c6a11d20788a"
    ),
    "front_branch_audits.csv": (
        "329ac5a95406c0736861c4256026f88cb3d9c9fd7c4bb91973d1df312213706a"
    ),
    "position_decisions.csv": (
        "81abcb85e0dd5c7e85f1fdcda6910179bc8e7d74a498dc9ee6a7c2d42b9b7965"
    ),
    "reference_checks.csv": (
        "cf4b5292e182b5e1fb9eedde2773cfe73cea5f64fd6d204a9e8022d9413c1a84"
    ),
    "relaxed_tether_audits.csv": (
        "051766d565cbb6ff614890f358e0fb8426840d699c737293327176e36c9575e0"
    ),
    "rollout_call_metrics.csv": (
        "28cd26c4a9ea03fb1e33e4f0b3f32346a849dd708a0b96999d40528beefad8ca"
    ),
    "rollout_case_metrics.csv": (
        "86cef98c7b8d4db6827e45f24f9916398337847f4d00ec5b99ad302abe88434a"
    ),
    "rollout_closure.csv": (
        "fb36e6ef2cd749f80d7a375ba49aad6d39d37304c4c07fc883cf210453121971"
    ),
    "rollout_controls.csv": (
        "e47f8155865771ccee84bb9fc60499612d955952e64555834855908f8581d21d"
    ),
    "rollout_execution.csv": (
        "72b5e8978dd77c876ae429e93e8e27c75cf9cb57f124ab2dbc6ae9f44f4d4996"
    ),
    "source_manifest.json": EXPECTED_P6_A22_R1_SOURCE_MANIFEST_SHA256,
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON payload must be an object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _typed_rollout_rows(
    rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    text_columns = {"case_id", "policy", "correction_status"}
    integer_columns = {"input_call", "output_call"}
    boolean_columns = {"finite", "admissible", "cap_active"}
    typed_rows: list[dict[str, Any]] = []
    for row in rows:
        typed: dict[str, Any] = {}
        for key, value in row.items():
            if key in text_columns:
                typed[key] = value
            elif key in integer_columns:
                typed[key] = int(value)
            elif key in boolean_columns:
                if value not in {"True", "False"}:
                    raise ValueError(f"invalid rollout boolean {key}={value!r}")
                typed[key] = value == "True"
            else:
                typed[key] = None if value == "" else float(value)
        typed_rows.append(typed)
    return typed_rows


def _git_status_short(paths: Sequence[str]) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "status", "--short", "--", *paths],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ["unknown"]
    return result.stdout.splitlines() if result.returncode == 0 else ["unknown"]


def _source_hashes() -> dict[str, str]:
    return sha256_files(SOURCE_PATHS, root=ROOT)


def _sha256_array_payload(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _load_shard_native_reference(
    runtime: Any, case_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Expose the immutable checkpoint-bound native trajectory as scoring truth."""

    entry = runtime.store.entry(case_id)
    expected_arrays = entry.get("array_sha256")
    required = ("states_conservative", "physical_times", "nodes", "node_measures")
    if not isinstance(expected_arrays, Mapping) or any(
        name not in expected_arrays for name in required
    ):
        raise ValueError("shard-native reference lacks required array digests")
    arrays = {}
    for name in required:
        path = runtime.store.root / str(entry["folder"]) / f"{name}.npy"
        if sha256_file(path) != expected_arrays[name]:
            raise ValueError(f"shard-native reference array digest mismatch: {name}")
        arrays[name] = np.asarray(runtime.store.array(case_id, name))
    states = arrays["states_conservative"]
    times = arrays["physical_times"]
    nodes = arrays["nodes"]
    measures = arrays["node_measures"]
    expected_nodes = NATIVE_RESOLUTION[0] * NATIVE_RESOLUTION[1]
    if states.dtype != np.float32 or states.shape != (61, expected_nodes, 4):
        raise ValueError("shard-native reference state contract differs")
    expected_times = np.arange(61, dtype=np.float64) * 0.01
    if (
        times.shape != (61,)
        or not np.all(np.diff(times) > 0.0)
        or not np.allclose(times, expected_times, rtol=0.0, atol=1.0e-15)
    ):
        raise ValueError("shard-native reference time contract differs")
    if nodes.shape != (expected_nodes, 2) or measures.shape not in {
        (expected_nodes,),
        (expected_nodes, 1),
    }:
        raise ValueError("shard-native reference geometry contract differs")
    if _sha256_array_payload(states) != entry.get("state_digest"):
        raise ValueError("shard-native reference state digest mismatch")
    native_geometry = runtime.geometry_by_resolution[NATIVE_RESOLUTION]
    geometry_nodes = np.asarray(native_geometry.nodes, dtype=nodes.dtype)
    geometry_measures = np.asarray(native_geometry.node_measures).reshape(-1)
    if not np.array_equal(nodes, geometry_nodes) or not np.array_equal(
        measures.reshape(-1), geometry_measures.astype(measures.dtype, copy=False)
    ):
        raise ValueError("shard-native reference and runtime geometry differ")
    reference = {
        "conservative_states": np.asarray(states, dtype=np.float64),
        "physical_times": np.asarray(times, dtype=np.float64),
        "retained_resolution": NATIVE_RESOLUTION,
    }
    check = {
        "case_id": case_id,
        "reference_schema": SHARD_NATIVE_REFERENCE_SCHEMA,
        "reference_source": "checkpoint_bound_shard_states_conservative",
        "source_reference_sha256": entry.get("source_reference_sha256"),
        "shard_manifest_sha256": runtime.store.manifest_digest,
        "shard_states_array_sha256": expected_arrays["states_conservative"],
        "shard_state_digest": entry.get("state_digest"),
        "retained_resolution": "250x100",
        "state_dtype": str(states.dtype),
        "state_shape": list(states.shape),
        "restriction_crosscheck_max_abs": 0.0,
        "serialization_floor": "exact_original_reference_after_float32_cast",
    }
    return reference, check


def _verify_shard_native_reference_audit(
    path: Path, args: argparse.Namespace
) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    checks = payload.get("checks", {})
    a23_mode = getattr(args, "command", "") in {
        "buffered-relaxed-tether-e14-preflight",
        "buffered-relaxed-tether-e14-rollout",
    }
    source_identity_exact = (
        sha256_file(path) == EXPECTED_P6_SHARD_NATIVE_REFERENCE_AUDIT_SHA256
        and payload.get("payload_sha256")
        == EXPECTED_P6_SHARD_NATIVE_REFERENCE_AUDIT_PAYLOAD_SHA256
        if a23_mode
        else payload.get("source_sha256") == _source_hashes()
    )
    if (
        payload.get("schema") != SHARD_NATIVE_REFERENCE_AUDIT_SCHEMA
        or payload.get("working_id") != SHARD_NATIVE_REFERENCE_AUDIT_WORKING_ID
        or payload.get("status") != "passed"
        or payload.get("case_ids") != list(PROSPECTIVE_STRUCTURAL_CASE_IDS)
        or not source_identity_exact
        or payload.get("family_manifest_sha256")
        != sha256_file(args.family_root / "family_manifest.json")
        or payload.get("shard_manifest_sha256")
        != sha256_file(args.data_dir / "manifest.json")
        or not checks
        or not all(checks.values())
    ):
        raise ValueError("shard-native reference audit differs from contract")
    return payload


def _native_scoring_truth_contract(args: argparse.Namespace) -> dict[str, Any]:
    path = getattr(args, "shard_native_reference_audit", None)
    if path is None:
        return {
            "mode": "frozen_reference_npz_float64",
            "recovery_audit": None,
        }
    audit = _verify_shard_native_reference_audit(path, args)
    replay_path = getattr(args, "shard_native_a13_replay", None)
    replay = None
    if getattr(args, "command", "") in {
        "prospective-buffered-offset-preflight",
        "prospective-buffered-offset-rollout",
    }:
        if replay_path is None:
            raise ValueError("prospective shard-native scoring requires A13 replay")
        replay = _verify_shard_native_a13_replay(
            replay_path,
            audit=audit,
            audit_sha256=sha256_file(path),
            original_path=args.a13_rollout,
        )
    elif getattr(args, "command", "") in {
        "buffered-relaxed-tether-e14-preflight",
        "buffered-relaxed-tether-e14-rollout",
    }:
        if (
            replay_path is not None
            or sha256_file(path) != EXPECTED_P6_SHARD_NATIVE_REFERENCE_AUDIT_SHA256
            or audit.get("payload_sha256")
            != EXPECTED_P6_SHARD_NATIVE_REFERENCE_AUDIT_PAYLOAD_SHA256
        ):
            raise ValueError("A23 requires the exact R0 audit and no A13 replay")
    return {
        "mode": "checkpoint_bound_shard_states_float32",
        "recovery_audit": {
            "path_sha256": sha256_file(path),
            "payload_sha256": audit["payload_sha256"],
            "working_id": audit["working_id"],
            "calibration_population": list(PROSPECTIVE_STRUCTURAL_CASE_IDS),
            "exact_after_float32_cast": True,
            "maximum_calibration_quantization_abs": audit[
                "maximum_float64_quantization_abs"
            ],
            "maximum_calibration_quantization_rms": audit[
                "maximum_float64_quantization_rms"
            ],
            "qualified_a13_shard_native_replay": (
                None
                if replay is None
                else {
                    "path_sha256": sha256_file(replay_path),
                    "payload_sha256": replay["payload_sha256"],
                    "status": replay["status"],
                }
            ),
        },
    }


def _verify_shard_native_a13_replay(
    path: Path,
    *,
    audit: Mapping[str, Any],
    audit_sha256: str,
    original_path: Path,
) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    source_path = path.parent / "source_manifest.json"
    source = _read_json(source_path)
    verify_payload_sha256(source)
    truth = source.get("population", {}).get("native_scoring_truth_contract", {})
    recovery = truth.get("recovery_audit", {})
    original = _verify_a13_rollout(original_path)
    replay_population = payload.get("population", {})
    original_population = original.get("population", {})
    ratios = (
        "aggregate_state_rms_ratio",
        "aggregate_increment_defect_rms_ratio",
        "median_endpoint_cumulative_defect_ratio",
        "maximum_endpoint_state_ratio",
    )
    ratios_close = all(
        abs(float(replay_population[name]) - float(original_population[name]))
        <= SHARD_NATIVE_A13_REPLAY_ABSOLUTE_TOLERANCE
        for name in ratios
    )
    if (
        payload.get("schema") != BUFFERED_OFFSET_SCHEMA
        or payload.get("working_id") != BUFFERED_OFFSET_WORKING_ID
        or payload.get("status") != "qualified_calibration"
        or not all(payload.get("calibration_gate", {}).get("checks", {}).values())
        or payload.get("source_manifest_sha256") != sha256_file(source_path)
        or payload.get("source_manifest_payload_sha256") != source.get("payload_sha256")
        or truth.get("mode") != "checkpoint_bound_shard_states_float32"
        or recovery.get("payload_sha256") != audit.get("payload_sha256")
        or recovery.get("path_sha256") != audit_sha256
        or payload.get("population", {}).get("strict_interior_trajectory_win_count") < 5
        or not ratios_close
        or replay_population.get("endpoint_win_count")
        != original_population.get("endpoint_win_count")
        or replay_population.get("strict_interior_trajectory_win_count")
        != original_population.get("strict_interior_trajectory_win_count")
        or payload.get("cost", {}).get("logical_model_calls") != 470
        or payload.get("cost", {}).get("native_logical_calls") != 420
        or payload.get("cost", {}).get("fine_logical_calls") != 50
    ):
        raise ValueError("shard-native A13 replay differs from contract")
    return payload


def _load_native_scoring_reference(
    runtime: Any, case_id: str, *, shard_native_reference: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    if shard_native_reference:
        audit_path = getattr(runtime.args, "shard_native_reference_audit", None)
        if audit_path is None:
            raise ValueError("shard-native scoring requires its recovery audit")
        audit = _verify_shard_native_reference_audit(audit_path, runtime.args)
        reference, check = _load_shard_native_reference(runtime, case_id)
        check.update(
            {
                "shard_native_reference_audit_sha256": sha256_file(audit_path),
                "shard_native_reference_audit_payload_sha256": audit["payload_sha256"],
            }
        )
        return reference, check
    return load_resolution_reference(
        runtime.args.family_root,
        None,
        runtime.store,
        runtime.manifest,
        case_id,
        training_resolution=NATIVE_RESOLUTION,
    )


def _native_reference_check_exact(
    check: Mapping[str, Any], *, truth_contract: Mapping[str, Any]
) -> bool:
    if truth_contract.get("mode") == "checkpoint_bound_shard_states_float32":
        audit = truth_contract.get("recovery_audit")
        return bool(
            isinstance(audit, Mapping)
            and check.get("reference_schema") == SHARD_NATIVE_REFERENCE_SCHEMA
            and check.get("reference_source")
            == "checkpoint_bound_shard_states_conservative"
            and check.get("retained_resolution") == "250x100"
            and check.get("state_dtype") == "float32"
            and check.get("state_shape") == [61, 25000, 4]
            and check.get("restriction_crosscheck_max_abs") == 0.0
            and check.get("serialization_floor")
            == "exact_original_reference_after_float32_cast"
            and check.get("shard_native_reference_audit_sha256")
            == audit.get("path_sha256")
            and check.get("shard_native_reference_audit_payload_sha256")
            == audit.get("payload_sha256")
        )
    return bool(
        check.get("retained_resolution") == "250x100"
        and check.get("active_reference_artifact_sha256")
        == check.get("frozen_training_reference_sha256")
        and check.get("restriction_crosscheck_max_abs") == 0.0
    )


def run_shard_native_reference_audit(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.family_root / "family_manifest.json"
    manifest = load_shock_vortex_family_manifest(manifest_path)
    store = PCNOEuler2DShardStore(args.data_dir, max_cached_trajectories=1)
    rows = []
    try:
        for case_id in PROSPECTIVE_STRUCTURAL_CASE_IDS:
            provenance = family_case_provenance(manifest, case_id)
            if (
                provenance["split_group_id"] != PROSPECTIVE_STRUCTURAL_GROUP_ID
                or case_id not in store.keys
            ):
                raise ValueError("recovery audit case is outside opened e12")
            entry = store.entry(case_id)
            reference_path = args.family_root / case_id / "reference.npz"
            reference_sha = sha256_file(reference_path)
            if reference_sha != entry.get("source_reference_sha256"):
                raise ValueError("recovery audit reference digest mismatch")
            with np.load(reference_path, allow_pickle=False) as artifact:
                if artifact["schema"].item() != REFERENCE_ARTIFACT_SCHEMA:
                    raise ValueError("recovery audit reference schema mismatch")
                if json.loads(artifact["family_contract_json"].item()) != provenance:
                    raise ValueError("recovery audit reference provenance mismatch")
                reference_states = np.asarray(artifact["conservative_states"])
                reference_times = np.asarray(artifact["physical_times"])
                reference_nodes = np.asarray(artifact["cell_centers"])
                reference_measures = np.asarray(artifact["cell_volume"])
            comparisons = {
                "states_conservative": reference_states,
                "physical_times": reference_times,
                "nodes": reference_nodes,
                "node_measures": reference_measures,
            }
            exact = {}
            for name, reference_array in comparisons.items():
                entry_hash = entry.get("array_sha256", {}).get(name)
                array_path = store.root / str(entry["folder"]) / f"{name}.npy"
                if entry_hash is None or sha256_file(array_path) != entry_hash:
                    raise ValueError(f"recovery audit shard hash mismatch: {name}")
                shard_array = np.asarray(store.array(case_id, name))
                reference_cast = reference_array.astype(shard_array.dtype, copy=False)
                if reference_cast.shape != shard_array.shape:
                    if reference_cast.size != shard_array.size:
                        raise ValueError(f"recovery audit array shape mismatch: {name}")
                    reference_cast = reference_cast.reshape(shard_array.shape)
                exact[name] = bool(np.array_equal(reference_cast, shard_array))
            shard_states = np.asarray(store.states(case_id))
            if _sha256_array_payload(shard_states) != entry.get("state_digest"):
                raise ValueError("recovery audit state digest mismatch")
            delta = shard_states.astype(np.float64) - reference_states.astype(
                np.float64
            )
            rows.append(
                {
                    "case_id": case_id,
                    "source_reference_sha256": reference_sha,
                    "shard_states_array_sha256": entry["array_sha256"][
                        "states_conservative"
                    ],
                    "shard_state_digest": entry["state_digest"],
                    "exact_after_stored_dtype_cast": exact,
                    "maximum_float64_quantization_abs": float(np.max(np.abs(delta))),
                    "float64_quantization_rms": float(
                        np.sqrt(np.mean(np.square(delta)))
                    ),
                }
            )
    finally:
        store.close()
    checks = {
        "opened_e12_inventory_exact": [row["case_id"] for row in rows]
        == list(PROSPECTIVE_STRUCTURAL_CASE_IDS),
        "all_relevant_arrays_exact_after_stored_dtype_cast": all(
            all(row["exact_after_stored_dtype_cast"].values()) for row in rows
        ),
        "maximum_float64_quantization_abs_at_most_2p4e_7": max(
            row["maximum_float64_quantization_abs"] for row in rows
        )
        <= 2.4e-7,
        "maximum_float64_quantization_rms_at_most_4p6e_8": max(
            row["float64_quantization_rms"] for row in rows
        )
        <= 4.6e-8,
        "e14_reference_not_accessed": True,
        "checkpoint_model_not_built": True,
        "recurrence_not_executed": True,
    }
    payload = with_payload_sha256(
        {
            "schema": SHARD_NATIVE_REFERENCE_AUDIT_SCHEMA,
            "working_id": SHARD_NATIVE_REFERENCE_AUDIT_WORKING_ID,
            "status": "passed" if all(checks.values()) else "failed",
            "source_sha256": _source_hashes(),
            "family_manifest_sha256": sha256_file(manifest_path),
            "shard_manifest_sha256": sha256_file(args.data_dir / "manifest.json"),
            "case_ids": list(PROSPECTIVE_STRUCTURAL_CASE_IDS),
            "case_rows": rows,
            "maximum_float64_quantization_abs": max(
                row["maximum_float64_quantization_abs"] for row in rows
            ),
            "maximum_float64_quantization_rms": max(
                row["float64_quantization_rms"] for row in rows
            ),
            "checks": checks,
            "claim_boundary": (
                "Already-open e12 serialization audit only. It qualifies an explicit "
                "checkpoint-bound FP32 scoring-truth recovery path but does not "
                "supply FP64 truth, authorize e14 selection, or alter inference."
            ),
        }
    )
    atomic_write_json(args.output, payload)
    return payload


def _verify_p5(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_P5_SHA256:
        raise ValueError("P5 calibration artifact differs from the preregistration")
    payload = _read_json(path)
    verify_payload_sha256(payload)
    if (
        payload.get("schema") != "pcno_propagated_sensitivity_calibration_v1"
        or payload.get("status") != "stopped"
        or payload.get("threshold_crossfit", {}).get("selector_counts") != {"never": 9}
        or payload.get("evaluation_executed") is not False
        or payload.get("recurrence_executed") is not False
    ):
        raise ValueError("P5 artifact is not the frozen stopped mechanism result")
    return payload


def _verify_rescore(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_P6_RESCORE_SHA256:
        raise ValueError("P6 re-score differs from the preregistered A3 prerequisite")
    payload = _read_json(path)
    verify_payload_sha256(payload)
    if (
        payload.get("schema") != RESCORE_SCHEMA
        or payload.get("status") != "qualified"
        or payload.get("prospective_gate", {}).get("evaluation_authorized") is not True
        or payload.get("model_calls") != 0
        or payload.get("predictions_recomputed") is not False
        or payload.get("prior_calibration_sha256") != EXPECTED_P6_CALIBRATION_SHA256
        or payload.get("evaluation_executed") is not False
        or payload.get("recurrence_executed") is not False
    ):
        raise ValueError("P6 re-score is not the qualified scorer-only prerequisite")
    for name, expected in payload.get("artifact_hashes", {}).items():
        if sha256_file(path.parent / name) != expected:
            raise ValueError(f"P6 re-score artifact differs: {name}")
    return payload


def _verify_evaluation(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_P6_EVALUATION_SHA256:
        raise ValueError("P6 A3 evaluation differs from the preregistered A4 input")
    payload = _read_json(path)
    verify_payload_sha256(payload)
    if (
        payload.get("schema") != EVALUATION_SCHEMA
        or payload.get("status") != "qualified"
        or payload.get("teacher_gate", {}).get("recurrent_pilot_authorized") is not True
        or payload.get("qualified_rescore_sha256") != EXPECTED_P6_RESCORE_SHA256
        or payload.get("evaluation_executed") is not True
        or payload.get("recurrence_executed") is not False
        or payload.get("sealed_population_opened") is not False
    ):
        raise ValueError("P6 A3 evaluation is not the qualified recurrent prerequisite")
    for name, expected in payload.get("artifact_hashes", {}).items():
        if sha256_file(path.parent / name) != expected:
            raise ValueError(f"P6 A3 evaluation artifact differs: {name}")
    return payload


def _verify_a4_rollout(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_P6_A4_ROLLOUT_SHA256:
        raise ValueError("P6 A4 rollout differs from the preregistered A5 input")
    payload = _read_json(path)
    verify_payload_sha256(payload)
    failed = {
        key
        for key, value in payload.get("recurrent_gate", {}).get("checks", {}).items()
        if not value
    }
    if (
        payload.get("schema") != ROLLOUT_SCHEMA
        or payload.get("status") != "stopped"
        or failed != {"all_filtered_controls_no_harm"}
        or payload.get("artifact_hashes") != EXPECTED_P6_A4_ARTIFACT_SHA256
        or payload.get("recurrence_executed") is not True
        or payload.get("sealed_population_opened") is not False
        or len(payload.get("failed_controls", {}).get("block_filtered", ())) != 4
    ):
        raise ValueError("P6 A4 is not the registered stopped integral-drift result")
    for name, expected in EXPECTED_P6_A4_ARTIFACT_SHA256.items():
        if sha256_file(path.parent / name) != expected:
            raise ValueError(f"P6 A4 artifact differs: {name}")
    return payload


def _verify_a5_rollout(
    path: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, str]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    if sha256_file(path) != EXPECTED_P6_A5_ROLLOUT_SHA256:
        raise ValueError("P6 A5 rollout differs from the registered scorer input")
    payload = _read_json(path)
    verify_payload_sha256(payload)
    failed = {
        key
        for key, value in payload.get("recurrent_gate", {}).get("checks", {}).items()
        if not value
    }
    if (
        payload.get("schema") != SHADOW_SCHEMA
        or payload.get("status") != "stopped"
        or failed != {"six_call_shadow_and_candidate_contract"}
        or payload.get("artifact_hashes") != EXPECTED_P6_A5_ARTIFACT_SHA256
        or payload.get("failed_controls") != []
        or payload.get("recurrence_executed") is not True
        or payload.get("sealed_population_opened") is not False
    ):
        raise ValueError("P6 A5 is not the registered bookkeeping-only stop")
    root = path.parent
    for name, expected in EXPECTED_P6_A5_ARTIFACT_SHA256.items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"P6 A5 artifact differs: {name}")
    rows = _typed_rollout_rows(_read_csv(root / "rollout_call_metrics.csv"))
    expected_inventory = {
        (policy, case_id, input_call)
        for policy in ("zero", "shadow_anchored")
        for case_id in EVALUATION_CASE_IDS
        for input_call in range(30)
    }
    actual_inventory = {
        (row["policy"], row["case_id"], int(row["input_call"])) for row in rows
    }
    if len(rows) != len(expected_inventory) or actual_inventory != expected_inventory:
        raise ValueError("P6 A5 call inventory differs from the registered cell")
    case_rows, controls, population = parent._paired_rollout_controls(rows)
    for key, value in population.items():
        if not math.isclose(
            float(value), float(payload["population"][key]), abs_tol=1.0e-14
        ):
            raise ValueError(f"P6 A5 population does not replay from CSV: {key}")
    if not controls or not all(parent._control_passed(row) for row in controls):
        raise ValueError("P6 A5 reconstructed controls do not all pass")
    return payload, rows, case_rows, controls, population


def _verify_a5_rescore(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_P6_A5_RESCORE_SHA256:
        raise ValueError("P6 A5-R1 differs from the frozen strength-OOD prerequisite")
    payload = _read_json(path)
    verify_payload_sha256(payload)
    if (
        payload.get("schema") != SHADOW_RESCORE_SCHEMA
        or payload.get("working_id") != SHADOW_RESCORE_WORKING_ID
        or payload.get("status") != "qualified"
        or payload.get("model_calls") != 0
        or payload.get("predictions_recomputed") is not False
        or payload.get("thresholds_changed") is not False
        or payload.get("recurrence_reexecuted") is not False
        or payload.get("sealed_population_opened") is not False
        or payload.get("failed_controls") != []
        or not all(payload.get("recurrent_gate", {}).get("checks", {}).values())
        or payload.get("artifact_hashes") != EXPECTED_P6_A5_RESCORE_ARTIFACT_SHA256
    ):
        raise ValueError("P6 A5-R1 is not the registered qualified prerequisite")
    for name, expected in EXPECTED_P6_A5_RESCORE_ARTIFACT_SHA256.items():
        if sha256_file(path.parent / name) != expected:
            raise ValueError(f"P6 A5-R1 artifact differs: {name}")
    return payload


def _verify_a6_rollout(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_P6_A6_ROLLOUT_SHA256:
        raise ValueError("P6 A6 differs from the frozen structural-gate calibration")
    payload = _read_json(path)
    verify_payload_sha256(payload)
    if (
        payload.get("schema") != STRENGTH_OOD_SCHEMA
        or payload.get("working_id") != STRENGTH_OOD_WORKING_ID
        or payload.get("status") != "stopped"
        or payload.get("opened_group") != STRENGTH_OOD_GROUP_ID
        or payload.get("opened_case_ids") != list(STRENGTH_OOD_CASE_IDS)
        or payload.get("still_sealed_groups")
        != ["strength_ood_e12", "strength_ood_e14"]
        or payload.get("recurrence_executed") is not True
        or payload.get("artifact_hashes") != EXPECTED_P6_A6_ARTIFACT_SHA256
    ):
        raise ValueError("P6 A6 is not the exact stopped e13 calibration artifact")
    for name, expected in EXPECTED_P6_A6_ARTIFACT_SHA256.items():
        if sha256_file(path.parent / name) != expected:
            raise ValueError(f"P6 A6 artifact differs: {name}")
    return payload


def _verify_a7_rollout(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_P6_A7_ROLLOUT_SHA256:
        raise ValueError("P6 A7 differs from the frozen prospective calibration")
    payload = _read_json(path)
    verify_payload_sha256(payload)
    gate = payload.get("retrospective_gate", {})
    if (
        payload.get("schema") != STRUCTURAL_GATE_SCHEMA
        or payload.get("working_id") != STRUCTURAL_GATE_WORKING_ID
        or payload.get("status") != "qualified_retrospective"
        or gate.get("status") != "qualified_retrospective"
        or gate.get("prospective_claim_authorized") is not False
        or not all(gate.get("checks", {}).values())
        or payload.get("population_status")
        != "already_open_e13_retrospective_calibration"
        or payload.get("new_sealed_population_opened") is not False
        or payload.get("still_sealed_groups")
        != ["strength_ood_e12", "strength_ood_e14"]
        or payload.get("recurrence_executed") is not True
        or payload.get("failed_controls") != []
        or payload.get("artifact_hashes") != EXPECTED_P6_A7_ARTIFACT_SHA256
        or payload.get("animation_bundle_manifest_sha256")
        != EXPECTED_P6_A7_ANIMATION_BUNDLE_MANIFEST_SHA256
    ):
        raise ValueError("P6 A7 is not the exact qualified e13 calibration artifact")
    for name, expected in EXPECTED_P6_A7_ARTIFACT_SHA256.items():
        if sha256_file(path.parent / name) != expected:
            raise ValueError(f"P6 A7 artifact differs: {name}")
    bundle_manifest = path.parent / "animation_bundles" / "bundle_manifest.json"
    if sha256_file(bundle_manifest) != EXPECTED_P6_A7_ANIMATION_BUNDLE_MANIFEST_SHA256:
        raise ValueError("P6 A7 animation bundle manifest differs")
    return payload


def _verify_a8_rollout(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_P6_A8_ROLLOUT_SHA256:
        raise ValueError("P6 A8 differs from the frozen A9 diagnostic prerequisite")
    payload = _read_json(path)
    verify_payload_sha256(payload)
    gate = payload.get("prospective_gate", {})
    if (
        payload.get("schema") != PROSPECTIVE_STRUCTURAL_SCHEMA
        or payload.get("working_id") != PROSPECTIVE_STRUCTURAL_WORKING_ID
        or payload.get("status") != "qualified_prospective"
        or gate.get("status") != "qualified_prospective"
        or gate.get("prospective_claim_authorized") is not True
        or not all(gate.get("checks", {}).values())
        or payload.get("new_sealed_population_opened")
        != PROSPECTIVE_STRUCTURAL_GROUP_ID
        or payload.get("still_sealed_groups") != ["strength_ood_e14"]
        or payload.get("recurrence_executed") is not True
        or payload.get("failed_controls") != []
        or payload.get("artifact_hashes") != EXPECTED_P6_A8_ARTIFACT_SHA256
        or payload.get("animation_bundle_manifest_sha256")
        != EXPECTED_P6_A8_ANIMATION_BUNDLE_MANIFEST_SHA256
    ):
        raise ValueError("P6 A8 is not the exact qualified e12 prerequisite")
    for name, expected in EXPECTED_P6_A8_ARTIFACT_SHA256.items():
        if sha256_file(path.parent / name) != expected:
            raise ValueError(f"P6 A8 artifact differs: {name}")
    bundle_manifest = path.parent / "animation_bundles" / "bundle_manifest.json"
    if sha256_file(bundle_manifest) != EXPECTED_P6_A8_ANIMATION_BUNDLE_MANIFEST_SHA256:
        raise ValueError("P6 A8 animation bundle manifest differs")
    return payload


def _verify_a9_diagnostic(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_P6_A9_DIAGNOSTIC_SHA256:
        raise ValueError("P6 A9 diagnostic differs from the registered rescore input")
    payload = _read_json(path)
    verify_payload_sha256(payload)
    checks = payload.get("contract_checks", {})
    failed = {key for key, value in checks.items() if not value}
    if (
        payload.get("schema") != SHADOW_CANDIDATE_SCHEMA
        or payload.get("working_id") != SHADOW_CANDIDATE_WORKING_ID
        or payload.get("status") != "invalid_diagnostic"
        or failed != A9_MISWIRED_CALL_CHECKS
        or payload.get("population", {}).get("block_count") != 135
        or payload.get("cost", {}).get("logical_model_calls") != 810
        or payload.get("candidate_accepted_into_recurrence") is not False
        or payload.get("feature_or_threshold_selection_executed") is not False
        or payload.get("coefficient_refit") is not False
        or payload.get("new_sealed_population_opened") is not False
        or payload.get("still_sealed_groups") != ["strength_ood_e14"]
    ):
        raise ValueError("P6 A9 is not the exact call-inventory-only invalid result")
    for name, expected in payload.get("artifact_hashes", {}).items():
        if sha256_file(path.parent / name) != expected:
            raise ValueError(f"P6 A9 artifact differs: {name}")
    return payload


def _verify_a9_rescore(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_P6_A9_RESCORE_SHA256:
        raise ValueError("P6 A9-R1 differs from the frozen A10 prerequisite")
    payload = _read_json(path)
    verify_payload_sha256(payload)
    if (
        payload.get("schema") != SHADOW_CANDIDATE_RESCORE_SCHEMA
        or payload.get("working_id") != SHADOW_CANDIDATE_RESCORE_WORKING_ID
        or payload.get("status") != "completed_diagnostic_rescore"
        or not all(payload.get("contract_checks", {}).values())
        or payload.get("model_calls") != 0
        or payload.get("predictions_recomputed") is not False
        or payload.get("recurrence_reexecuted") is not False
        or payload.get("threshold_or_coefficient_changed") is not False
        or payload.get("new_sealed_population_opened") is not False
        or payload.get("still_sealed_groups") != ["strength_ood_e14"]
        or payload.get("artifact_hashes") != EXPECTED_P6_A9_RESCORE_ARTIFACT_SHA256
    ):
        raise ValueError("P6 A9-R1 is not the exact completed rescore prerequisite")
    for name, expected in EXPECTED_P6_A9_RESCORE_ARTIFACT_SHA256.items():
        if sha256_file(path.parent / name) != expected:
            raise ValueError(f"P6 A9-R1 artifact differs: {name}")
    return payload


def _verify_a10_rollout(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_P6_A10_ROLLOUT_SHA256:
        raise ValueError("P6 A10 differs from the frozen A11 prerequisite")
    payload = _read_json(path)
    verify_payload_sha256(payload)
    checks = payload.get("calibration_gate", {}).get("checks", {})
    failed = {key for key, value in checks.items() if not value}
    if (
        payload.get("schema") != WARM_START_SCHEMA
        or payload.get("working_id") != WARM_START_WORKING_ID
        or payload.get("status") != "stopped_calibration"
        or failed
        != {
            "aggregate_increment_defect_rms_ratio_at_most_a8",
            "all_registered_controls_no_harm",
            "maximum_endpoint_state_ratio_at_most_one",
        }
        or payload.get("population", {}).get("aggregate_state_rms_ratio")
        != 0.9857532117505919
        or payload.get("population", {}).get("aggregate_increment_defect_rms_ratio")
        != 0.9971467177851852
        or payload.get("population", {}).get("strict_interior_trajectory_win_count")
        != 6
        or len(payload.get("failed_controls", ())) != 3
        or payload.get("artifact_hashes") != EXPECTED_P6_A10_ARTIFACT_SHA256
        or payload.get("new_sealed_population_opened") is not False
        or payload.get("opened_case_ids") != []
        or payload.get("still_sealed_groups") != ["strength_ood_e14"]
    ):
        raise ValueError("P6 A10 is not the exact stopped calibration prerequisite")
    for name, expected in EXPECTED_P6_A10_ARTIFACT_SHA256.items():
        if sha256_file(path.parent / name) != expected:
            raise ValueError(f"P6 A10 artifact differs: {name}")
    bundle_manifest = path.parent / "animation_bundles" / "bundle_manifest.json"
    if (
        payload.get("animation_bundle_manifest_sha256")
        != EXPECTED_P6_A10_ANIMATION_BUNDLE_MANIFEST_SHA256
        or sha256_file(bundle_manifest)
        != EXPECTED_P6_A10_ANIMATION_BUNDLE_MANIFEST_SHA256
    ):
        raise ValueError("P6 A10 animation bundle manifest differs")
    return payload


def _verify_a11_rollout(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_P6_A11_ROLLOUT_SHA256:
        raise ValueError("P6 A11 differs from the frozen A12 prerequisite")
    payload = _read_json(path)
    verify_payload_sha256(payload)
    checks = payload.get("calibration_gate", {}).get("checks", {})
    failed = {key for key, value in checks.items() if not value}
    if (
        payload.get("schema") != PROJECTED_COAST_SCHEMA
        or payload.get("working_id") != PROJECTED_COAST_WORKING_ID
        or payload.get("status") != "stopped_calibration"
        or failed != {"aggregate_increment_defect_rms_ratio_at_most_a8"}
        or payload.get("population", {}).get("aggregate_state_rms_ratio")
        != 0.9875045214578189
        or payload.get("population", {}).get("aggregate_increment_defect_rms_ratio")
        != 0.9970535081808779
        or payload.get("population", {}).get("strict_interior_trajectory_win_count")
        != 7
        or payload.get("failed_controls") != []
        or payload.get("artifact_hashes") != EXPECTED_P6_A11_ARTIFACT_SHA256
        or payload.get("new_sealed_population_opened") is not False
        or payload.get("opened_case_ids") != []
        or payload.get("still_sealed_groups") != ["strength_ood_e14"]
    ):
        raise ValueError("P6 A11 is not the exact stopped calibration prerequisite")
    for name, expected in EXPECTED_P6_A11_ARTIFACT_SHA256.items():
        if sha256_file(path.parent / name) != expected:
            raise ValueError(f"P6 A11 artifact differs: {name}")
    bundle_manifest = path.parent / "animation_bundles" / "bundle_manifest.json"
    if (
        payload.get("animation_bundle_manifest_sha256")
        != EXPECTED_P6_A11_ANIMATION_BUNDLE_MANIFEST_SHA256
        or sha256_file(bundle_manifest)
        != EXPECTED_P6_A11_ANIMATION_BUNDLE_MANIFEST_SHA256
    ):
        raise ValueError("P6 A11 animation bundle manifest differs")
    return payload


def _verify_a12_rollout(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_P6_A12_ROLLOUT_SHA256:
        raise ValueError("P6 A12 differs from the frozen A13 prerequisite")
    payload = _read_json(path)
    verify_payload_sha256(payload)
    checks = payload.get("calibration_gate", {}).get("checks", {})
    failed = {key for key, value in checks.items() if not value}
    if (
        payload.get("schema") != FROZEN_OFFSET_SCHEMA
        or payload.get("working_id") != FROZEN_OFFSET_WORKING_ID
        or payload.get("status") != "stopped_calibration"
        or failed != {"maximum_endpoint_state_ratio_at_most_one"}
        or payload.get("population", {}).get("aggregate_state_rms_ratio")
        != 0.9863726035243584
        or payload.get("population", {}).get("aggregate_increment_defect_rms_ratio")
        != 0.9970078752243214
        or payload.get("population", {}).get("strict_interior_trajectory_win_count")
        != 7
        or payload.get("failed_controls") != []
        or payload.get("front_branch_vetoes") != []
        or payload.get("artifact_hashes") != EXPECTED_P6_A12_ARTIFACT_SHA256
        or payload.get("new_sealed_population_opened") is not False
        or payload.get("opened_case_ids") != []
        or payload.get("still_sealed_groups") != ["strength_ood_e14"]
    ):
        raise ValueError("P6 A12 is not the exact stopped calibration prerequisite")
    for name, expected in EXPECTED_P6_A12_ARTIFACT_SHA256.items():
        if sha256_file(path.parent / name) != expected:
            raise ValueError(f"P6 A12 artifact differs: {name}")
    bundle_manifest = path.parent / "animation_bundles" / "bundle_manifest.json"
    if (
        payload.get("animation_bundle_manifest_sha256")
        != EXPECTED_P6_A12_ANIMATION_BUNDLE_MANIFEST_SHA256
        or sha256_file(bundle_manifest)
        != EXPECTED_P6_A12_ANIMATION_BUNDLE_MANIFEST_SHA256
    ):
        raise ValueError("P6 A12 animation bundle manifest differs")
    return payload


def _verify_a12_buffer_rescore(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_P6_A12_BUFFER_RESCORE_SHA256:
        raise ValueError("P6 A12-R1 differs from the frozen A13 selector")
    payload = _read_json(path)
    verify_payload_sha256(payload)
    if (
        payload.get("schema") != BUFFERED_OFFSET_RESCORE_SCHEMA
        or payload.get("working_id") != BUFFERED_OFFSET_RESCORE_WORKING_ID
        or payload.get("status") != "qualified_calibration_rescore"
        or not all(payload.get("calibration_gate", {}).get("checks", {}).values())
        or payload.get("model_calls") != 0
        or payload.get("predictions_recomputed") is not False
        or payload.get("recurrence_reexecuted") is not False
        or payload.get("truth_used_for_selector") is not True
        or payload.get("new_sealed_population_opened") is not False
        or payload.get("still_sealed_groups") != ["strength_ood_e14"]
        or payload.get("artifact_hashes")
        != EXPECTED_P6_A12_BUFFER_RESCORE_ARTIFACT_SHA256
    ):
        raise ValueError("P6 A12-R1 is not the exact qualified selector")
    for name, expected in EXPECTED_P6_A12_BUFFER_RESCORE_ARTIFACT_SHA256.items():
        if sha256_file(path.parent / name) != expected:
            raise ValueError(f"P6 A12-R1 artifact differs: {name}")
    return payload


def _verify_a13_rollout(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_P6_A13_ROLLOUT_SHA256:
        raise ValueError("P6 A13 differs from the frozen prospective prerequisite")
    payload = _read_json(path)
    verify_payload_sha256(payload)
    gate = payload.get("calibration_gate", {})
    if (
        payload.get("schema") != BUFFERED_OFFSET_SCHEMA
        or payload.get("working_id") != BUFFERED_OFFSET_WORKING_ID
        or payload.get("status") != "qualified_calibration"
        or gate.get("status") != "qualified_calibration"
        or not all(gate.get("checks", {}).values())
        or gate.get("prospective_claim_authorized") is not False
        or payload.get("population_status") != "already_open_e12_recurrent_calibration"
        or payload.get("population", {}).get("maximum_endpoint_state_ratio") != 1.0
        or payload.get("population", {}).get("strict_interior_trajectory_win_count")
        != 5
        or payload.get("new_sealed_population_opened") is not False
        or payload.get("opened_case_ids") != []
        or payload.get("still_sealed_groups") != ["strength_ood_e14"]
        or payload.get("artifact_hashes") != EXPECTED_P6_A13_ARTIFACT_SHA256
    ):
        raise ValueError("P6 A13 is not the exact qualified e12 prerequisite")
    for name, expected in EXPECTED_P6_A13_ARTIFACT_SHA256.items():
        if sha256_file(path.parent / name) != expected:
            raise ValueError(f"P6 A13 artifact differs: {name}")
    bundle_manifest = path.parent / "animation_bundles" / "bundle_manifest.json"
    if (
        payload.get("animation_bundle_manifest_sha256")
        != EXPECTED_P6_A13_ANIMATION_BUNDLE_MANIFEST_SHA256
        or sha256_file(bundle_manifest)
        != EXPECTED_P6_A13_ANIMATION_BUNDLE_MANIFEST_SHA256
    ):
        raise ValueError("P6 A13 animation bundle manifest differs")
    source = _read_json(path.parent / "source_manifest.json")
    verify_payload_sha256(source)
    if (
        source.get("source_sha256") != EXPECTED_P6_A13_SOURCE_SHA256
        or source.get("protocol", {}).get("coefficient") != -0.5
        or source.get("protocol", {}).get("position_trust_threshold")
        != FROZEN_POSITION_TRUST_THRESHOLD
        or source.get("protocol", {}).get("position_buffer_threshold")
        != FROZEN_POSITION_BUFFER_THRESHOLD
        or source.get("protocol", {}).get("interior_warm_blocks")
        != FROZEN_OFFSET_WARM_BLOCKS
        or source.get("protocol", {}).get("coast_projection_active_cells")
        != [list(cell) for cell in FROZEN_ACTIVE_CELLS]
        or source.get("protocol", {}).get("coast_offset_feedback") is not False
        or source.get("protocol", {}).get("true_error_or_reference_at_inference")
        is not False
    ):
        raise ValueError("P6 A13 frozen source or protocol differs")
    return payload


def _verify_a14_rollout(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_P6_A14_ROLLOUT_SHA256:
        raise ValueError("P6 A14 differs from the frozen A15 mechanism prerequisite")
    payload = _read_json(path)
    verify_payload_sha256(payload)
    gate = payload.get("prospective_gate", {})
    checks = gate.get("checks", {})
    failed = [name for name, passed in checks.items() if not bool(passed)]
    if (
        payload.get("schema") != PROSPECTIVE_BUFFERED_OFFSET_SCHEMA
        or payload.get("working_id") != PROSPECTIVE_BUFFERED_OFFSET_WORKING_ID
        or payload.get("payload_sha256") != EXPECTED_P6_A14_PAYLOAD_SHA256
        or payload.get("status") != "stopped_prospective"
        or gate.get("status") != "stopped_prospective"
        or gate.get("prospective_claim_authorized") is not False
        or failed != ["maximum_endpoint_state_ratio_at_most_one"]
        or payload.get("population", {}).get("strict_interior_trajectory_win_count")
        != 5
        or payload.get("new_sealed_population_opened")
        != PROSPECTIVE_BUFFERED_OFFSET_GROUP_ID
        or payload.get("still_sealed_groups") != []
        or payload.get("source_manifest_sha256")
        != EXPECTED_P6_A14_SOURCE_MANIFEST_SHA256
        or sha256_file(path.parent / "source_manifest.json")
        != EXPECTED_P6_A14_SOURCE_MANIFEST_SHA256
    ):
        raise ValueError("P6 A14 is not the exact stopped persistence prerequisite")
    return payload


def _verify_a15_rollout(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_P6_A15_ROLLOUT_SHA256:
        raise ValueError("P6 A15 differs from the frozen A16 prerequisite")
    payload = _read_json(path)
    verify_payload_sha256(payload)
    gate = payload.get("calibration_gate", {})
    checks = gate.get("checks", {})
    failed = [name for name, passed in checks.items() if not bool(passed)]
    if (
        payload.get("schema") != PERSISTENCE_GAIN_SCHEMA
        or payload.get("working_id") != PERSISTENCE_GAIN_WORKING_ID
        or payload.get("payload_sha256") != EXPECTED_P6_A15_PAYLOAD_SHA256
        or payload.get("status") != "stopped_calibration"
        or gate.get("prospective_claim_authorized") is not False
        or failed != ["aggregate_increment_defect_rms_ratio_at_most_a8"]
        or payload.get("population", {}).get("maximum_endpoint_state_ratio") != 1.0
        or payload.get("population", {}).get("strict_interior_trajectory_win_count")
        != 5
        or payload.get("source_manifest_sha256")
        != EXPECTED_P6_A15_SOURCE_MANIFEST_SHA256
        or sha256_file(path.parent / "source_manifest.json")
        != EXPECTED_P6_A15_SOURCE_MANIFEST_SHA256
    ):
        raise ValueError("P6 A15 is not the exact stopped A16 prerequisite")
    return payload


def _verify_a16_rollout(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_P6_A16_ROLLOUT_SHA256:
        raise ValueError("P6 A16 differs from the frozen A17 prerequisite")
    payload = _read_json(path)
    verify_payload_sha256(payload)
    gate = payload.get("calibration_gate", {})
    checks = gate.get("checks", {})
    failed = [name for name, passed in checks.items() if not bool(passed)]
    if (
        payload.get("schema") != TERMINAL_RAMP_SCHEMA
        or payload.get("working_id") != TERMINAL_RAMP_WORKING_ID
        or payload.get("payload_sha256") != EXPECTED_P6_A16_PAYLOAD_SHA256
        or payload.get("status") != "stopped_calibration"
        or failed != ["aggregate_increment_defect_rms_ratio_at_most_a8"]
        or payload.get("population", {}).get("aggregate_state_rms_ratio")
        != 0.9875181513147335
        or payload.get("population", {}).get(
            "aggregate_increment_defect_rms_ratio"
        )
        != 0.9970143802908475
        or payload.get("population", {}).get("maximum_endpoint_state_ratio") != 1.0
        or payload.get("population", {}).get("strict_interior_trajectory_win_count")
        != 5
        or payload.get("source_manifest_sha256")
        != EXPECTED_P6_A16_SOURCE_MANIFEST_SHA256
        or sha256_file(path.parent / "source_manifest.json")
        != EXPECTED_P6_A16_SOURCE_MANIFEST_SHA256
    ):
        raise ValueError("P6 A16 is not the exact stopped A17 prerequisite")
    return payload


def _verify_a17_rollout_for_rescore(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if sha256_file(path) != EXPECTED_P6_A17_ROLLOUT_SHA256:
        raise ValueError("P6 A17 differs from the frozen A17-R1 rescore input")
    payload = _read_json(path)
    verify_payload_sha256(payload)
    gate = payload.get("calibration_gate", {})
    checks = gate.get("checks", {})
    failed = [name for name, passed in checks.items() if not bool(passed)]
    root = path.parent
    preflight_path = root.parent / "a17_preflight.json"
    source_path = root / "source_manifest.json"
    bundle_path = root / "animation_bundles" / "bundle_manifest.json"
    if (
        payload.get("schema") != FIXED_LATE_RAMP_SCHEMA
        or payload.get("working_id") != FIXED_LATE_RAMP_WORKING_ID
        or payload.get("payload_sha256") != EXPECTED_P6_A17_PAYLOAD_SHA256
        or payload.get("status") != "stopped_calibration"
        or gate.get("prospective_claim_authorized") is not False
        or failed
        != [
            "fixed_late_ramp_edge_and_interior_contract_exact",
            "no_new_population_opened",
        ]
        or payload.get("population", {}).get("aggregate_state_rms_ratio")
        != 0.9873397026522671
        or payload.get("population", {}).get(
            "aggregate_increment_defect_rms_ratio"
        )
        != 0.9970089757436883
        or payload.get("population", {}).get("strict_interior_trajectory_win_count")
        != 5
        or payload.get("population", {}).get("maximum_endpoint_state_ratio")
        != 1.0
        or payload.get("new_sealed_population_opened") is not False
        or payload.get("opened_case_ids") != []
        or payload.get("still_sealed_groups") != []
        or payload.get("preflight_sha256") != EXPECTED_P6_A17_PREFLIGHT_SHA256
        or payload.get("preflight_payload_sha256")
        != EXPECTED_P6_A17_PREFLIGHT_PAYLOAD_SHA256
        or payload.get("source_manifest_sha256")
        != EXPECTED_P6_A17_SOURCE_MANIFEST_SHA256
        or payload.get("animation_bundle_manifest_sha256")
        != EXPECTED_P6_A17_ANIMATION_BUNDLE_MANIFEST_SHA256
        or payload.get("artifact_hashes") != EXPECTED_P6_A17_ARTIFACT_SHA256
        or sha256_file(preflight_path) != EXPECTED_P6_A17_PREFLIGHT_SHA256
        or sha256_file(source_path) != EXPECTED_P6_A17_SOURCE_MANIFEST_SHA256
        or sha256_file(bundle_path)
        != EXPECTED_P6_A17_ANIMATION_BUNDLE_MANIFEST_SHA256
        or any(
            sha256_file(root / name) != expected
            for name, expected in EXPECTED_P6_A17_ARTIFACT_SHA256.items()
        )
    ):
        raise ValueError("P6 A17 is not the exact frozen A17-R1 rescore input")

    preflight = _read_json(preflight_path)
    source = _read_json(source_path)
    bundle = _read_json(bundle_path)
    verify_payload_sha256(preflight)
    verify_payload_sha256(source)
    verify_payload_sha256(bundle)
    if (
        preflight.get("schema") != FIXED_LATE_RAMP_PREFLIGHT_SCHEMA
        or preflight.get("working_id") != FIXED_LATE_RAMP_WORKING_ID
        or preflight.get("payload_sha256")
        != EXPECTED_P6_A17_PREFLIGHT_PAYLOAD_SHA256
        or preflight.get("status") != "passed"
        or source.get("working_id") != FIXED_LATE_RAMP_WORKING_ID
        or bundle.get("working_id") != FIXED_LATE_RAMP_WORKING_ID
        or bundle.get("inference_or_gate_input") is not False
    ):
        raise ValueError("P6 A17 auxiliary provenance is not exact")
    return payload, preflight, source


def _verify_a17_rescore(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_P6_A17_RESCORE_SHA256:
        raise ValueError("P6 A17-R1 differs from the frozen A18 prerequisite")
    payload = _read_json(path)
    verify_payload_sha256(payload)
    checks = payload.get("calibration_gate", {}).get("checks", {})
    if (
        payload.get("schema") != FIXED_LATE_RAMP_RESCORE_SCHEMA
        or payload.get("working_id") != FIXED_LATE_RAMP_RESCORE_WORKING_ID
        or payload.get("payload_sha256")
        != EXPECTED_P6_A17_RESCORE_PAYLOAD_SHA256
        or payload.get("status") != "qualified_calibration_rescore"
        or not checks
        or not all(bool(value) for value in checks.values())
        or payload.get("model_calls") != 0
        or payload.get("predictions_recomputed") is not False
        or payload.get("recurrence_reexecuted") is not False
        or payload.get("artifact_hashes")
        != EXPECTED_P6_A17_RESCORE_ARTIFACT_SHA256
        or any(
            sha256_file(path.parent / name) != expected
            for name, expected in EXPECTED_P6_A17_RESCORE_ARTIFACT_SHA256.items()
        )
    ):
        raise ValueError("P6 A17-R1 is not the exact qualified A18 prerequisite")
    return payload


def _verify_a18_rollout(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_P6_A18_ROLLOUT_SHA256:
        raise ValueError("P6 A18 differs from the frozen A19 prerequisite")
    payload = _read_json(path)
    verify_payload_sha256(payload)
    gate = payload.get("calibration_gate", {})
    failed = {
        name
        for name, passed in gate.get("checks", {}).items()
        if not bool(passed)
    }
    root = path.parent
    if (
        payload.get("schema") != SLEW_LIMITED_TETHER_SCHEMA
        or payload.get("working_id") != SLEW_LIMITED_TETHER_WORKING_ID
        or payload.get("payload_sha256") != EXPECTED_P6_A18_PAYLOAD_SHA256
        or payload.get("status") != "stopped_calibration"
        or failed
        != {
            "aggregate_increment_defect_rms_ratio_at_most_a8",
            "coefficient_trust_threshold_and_window_not_refit",
        }
        or payload.get("population", {}).get("aggregate_state_rms_ratio")
        != 0.9876892273388457
        or payload.get("population", {}).get(
            "aggregate_increment_defect_rms_ratio"
        )
        != 0.9970256010493702
        or payload.get("population", {}).get(
            "strict_interior_trajectory_win_count"
        )
        != 7
        or payload.get("source_manifest_sha256")
        != EXPECTED_P6_A18_SOURCE_MANIFEST_SHA256
        or payload.get("animation_bundle_manifest_sha256")
        != EXPECTED_P6_A18_ANIMATION_BUNDLE_MANIFEST_SHA256
        or payload.get("artifact_hashes") != EXPECTED_P6_A18_ARTIFACT_SHA256
        or sha256_file(root / "source_manifest.json")
        != EXPECTED_P6_A18_SOURCE_MANIFEST_SHA256
        or sha256_file(root / "animation_bundles" / "bundle_manifest.json")
        != EXPECTED_P6_A18_ANIMATION_BUNDLE_MANIFEST_SHA256
        or any(
            sha256_file(root / name) != expected
            for name, expected in EXPECTED_P6_A18_ARTIFACT_SHA256.items()
        )
    ):
        raise ValueError("P6 A18 is not the exact stopped A19 prerequisite")
    return payload


def _source_manifest(
    args: argparse.Namespace,
    *,
    parent_source: Mapping[str, Any],
    p5: Mapping[str, Any],
) -> dict[str, Any]:
    return with_payload_sha256(
        {
            "schema": "pcno_response_filtered_block_source_manifest_v1",
            "working_id": WORKING_ID,
            "source_sha256": _source_hashes(),
            "source_status": _git_status_short(SOURCE_PATHS),
            "parent_source_manifest": dict(parent_source),
            "p5_calibration_sha256": sha256_file(args.p5_calibration),
            "p5_calibration_payload_sha256": p5["payload_sha256"],
            "population": {
                "family": "dynamic_fv_shock_vortex",
                "case_ids": list(CALIBRATION_CASE_IDS),
                "groups": {
                    key: list(value) for key, value in CALIBRATION_GROUPS.items()
                },
                "input_calls": list(INPUT_CALLS),
                "evaluation_cases": "closed",
                "strength_ood_and_test": "sealed",
            },
            "candidate": {
                "first_correction": "sp19_fine_away_half",
                "second_response": "fixed_sp19_projection",
                "active_cells": [list(cell) for cell in FROZEN_ACTIVE_CELLS],
                "true_error_at_inference": False,
                "logical_calls_per_two_state_block": 4,
            },
        }
    )


def _evaluation_source_manifest(
    args: argparse.Namespace,
    *,
    parent_source: Mapping[str, Any],
    p5: Mapping[str, Any],
    rescore: Mapping[str, Any],
) -> dict[str, Any]:
    return with_payload_sha256(
        {
            "schema": "pcno_response_filtered_block_evaluation_source_v1",
            "working_id": EVALUATION_WORKING_ID,
            "source_sha256": _source_hashes(),
            "source_status": _git_status_short(SOURCE_PATHS),
            "parent_source_manifest": dict(parent_source),
            "p5_calibration_sha256": sha256_file(args.p5_calibration),
            "p5_calibration_payload_sha256": p5["payload_sha256"],
            "qualified_rescore_sha256": sha256_file(args.rescore),
            "qualified_rescore_payload_sha256": rescore["payload_sha256"],
            "population": {
                "family": "dynamic_fv_shock_vortex",
                "case_ids": list(EVALUATION_CASE_IDS),
                "groups": {
                    key: list(value) for key, value in EVALUATION_GROUPS.items()
                },
                "input_calls": list(INPUT_CALLS),
                "calibration_cases_excluded": list(CALIBRATION_CASE_IDS),
                "strength_ood_and_test": "sealed",
            },
            "candidate": {
                "first_correction": "sp19_fine_away_half",
                "second_response": "fixed_sp19_projection",
                "active_cells": [list(cell) for cell in FROZEN_ACTIVE_CELLS],
                "true_error_at_inference": False,
                "logical_calls_per_two_state_block": 4,
                "coefficients_fitted_on_evaluation": False,
            },
        }
    )


def _rollout_source_manifest(
    args: argparse.Namespace,
    *,
    evaluation_source: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    return with_payload_sha256(
        {
            "schema": "pcno_response_filtered_block_rollout_source_v1",
            "working_id": ROLLOUT_WORKING_ID,
            "source_sha256": _source_hashes(),
            "source_status": _git_status_short(SOURCE_PATHS),
            "evaluation_source_manifest": dict(evaluation_source),
            "qualified_evaluation_sha256": sha256_file(args.evaluation),
            "qualified_evaluation_payload_sha256": evaluation["payload_sha256"],
            "population": {
                "family": "dynamic_fv_shock_vortex",
                "case_ids": list(EVALUATION_CASE_IDS),
                "initial_state": "exact_native_call_0_only",
                "output_calls": list(range(1, 31)),
                "truth_during_inference": False,
                "strength_ood_and_test": "sealed",
            },
            "arms": {
                "zero": {"native_calls": 30, "fine_calls": 0},
                "sp19_always": {"native_calls": 30, "fine_calls": 30},
                "block_full": {"native_calls": 45, "fine_calls": 15},
                "block_filtered": {"native_calls": 45, "fine_calls": 15},
            },
            "candidate": {
                "response_mode": "filtered",
                "block_count": 15,
                "emitted_states_per_block": 2,
                "active_cells": [list(cell) for cell in FROZEN_ACTIVE_CELLS],
                "first_correction_gain": -0.5,
                "true_error_at_inference": False,
            },
        }
    )


def _shadow_source_manifest(
    args: argparse.Namespace,
    *,
    evaluation_source: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    a4_rollout: Mapping[str, Any],
) -> dict[str, Any]:
    return with_payload_sha256(
        {
            "schema": "pcno_response_filtered_block_shadow_source_v1",
            "working_id": SHADOW_WORKING_ID,
            "source_sha256": _source_hashes(),
            "source_status": _git_status_short(SOURCE_PATHS),
            "evaluation_source_manifest": dict(evaluation_source),
            "qualified_evaluation_sha256": sha256_file(args.evaluation),
            "qualified_evaluation_payload_sha256": evaluation["payload_sha256"],
            "stopped_a4_rollout_sha256": sha256_file(args.a4_rollout),
            "stopped_a4_rollout_payload_sha256": a4_rollout["payload_sha256"],
            "population": {
                "family": "dynamic_fv_shock_vortex",
                "case_ids": list(EVALUATION_CASE_IDS),
                "initial_state": "exact_native_call_0_for_accepted_and_shadow",
                "output_calls": list(range(1, 31)),
                "truth_during_inference": False,
                "strength_ood_and_test": "sealed",
            },
            "protocol": {
                "accepted_state": "sp19_response_filtered_with_integral_anchor",
                "shadow_state": "independent_raw_native",
                "blocks": 15,
                "calls_per_block": 6,
                "native_calls_per_block": 5,
                "fine_calls_per_block": 1,
                "anchor": "minimum_volume_weighted_l2_boundary_zero",
                "active_cells": [list(cell) for cell in FROZEN_ACTIVE_CELLS],
                "first_correction_gain": -0.5,
                "true_error_at_inference": False,
            },
        }
    )


def _strength_ood_source_manifest(
    args: argparse.Namespace,
    *,
    adaptive_shadow_source: Mapping[str, Any],
    a5_rescore: Mapping[str, Any],
) -> dict[str, Any]:
    return with_payload_sha256(
        {
            "schema": "pcno_response_filtered_block_strength_ood_source_v1",
            "working_id": STRENGTH_OOD_WORKING_ID,
            "source_sha256": _source_hashes(),
            "source_status": _git_status_short(SOURCE_PATHS),
            "adaptive_open_lineage": dict(adaptive_shadow_source),
            "qualified_a5_rescore_sha256": sha256_file(args.a5_rescore),
            "qualified_a5_rescore_payload_sha256": a5_rescore["payload_sha256"],
            "qualified_a5_rescore_artifact_sha256": (
                EXPECTED_P6_A5_RESCORE_ARTIFACT_SHA256
            ),
            "population": {
                "family": "dynamic_fv_shock_vortex",
                "split": "test",
                "split_group_id": STRENGTH_OOD_GROUP_ID,
                "case_ids": list(STRENGTH_OOD_CASE_IDS),
                "epsilon_index": 13,
                "vortex_epsilon": 0.375,
                "vortex_y": [0.35 + 0.0375 * index for index in range(9)],
                "initial_state": "exact_native_call_0_for_accepted_and_shadow",
                "output_calls": list(range(1, 31)),
                "truth_during_inference": False,
                "truth_use": "offline_scoring_and_post_run_visualization_only",
                "opened_group": STRENGTH_OOD_GROUP_ID,
                "still_sealed_groups": ["strength_ood_e12", "strength_ood_e14"],
                "truth_reference": (
                    "checkpoint_bound_evolved_native_250x100_family_reference"
                ),
                "fine_reference_truth": "not_required_or_loaded",
            },
            "protocol": {
                "accepted_state": "sp19_response_filtered_with_integral_anchor",
                "shadow_state": "independent_raw_native",
                "blocks": 15,
                "calls_per_block": 6,
                "native_calls_per_block": 5,
                "fine_calls_per_block": 1,
                "anchor": "minimum_volume_weighted_l2_boundary_zero",
                "active_cells": [list(cell) for cell in FROZEN_ACTIVE_CELLS],
                "first_correction_gain": -0.5,
                "true_error_at_inference": False,
                "coefficient_or_threshold_refit": False,
                "fine_branch_role": (
                    "prediction_only_then_conservative_restriction_to_native"
                ),
                "transfer_floor": "inherited_D063_no_A6_refit",
            },
            "prospective_gate": {
                "median_endpoint_state_ratio_at_most": 0.98,
                "minimum_endpoint_wins": 6,
                "maximum_endpoint_state_ratio_at_most": 1.02,
                "aggregate_state_rms_ratio_at_most": 0.99,
                "aggregate_increment_defect_rms_ratio_at_most": 1.0,
                "median_endpoint_cumulative_defect_ratio_at_most": 1.0,
                "all_registered_controls_ratio_at_most": 1.05,
                "closure_and_admissibility": "exact_A5_thresholds",
            },
            "animation_contract": STRENGTH_OOD_ANIMATION_CONTRACT,
        }
    )


def _structural_gate_source_manifest(
    args: argparse.Namespace,
    *,
    strength_ood_source: Mapping[str, Any],
    a6_rollout: Mapping[str, Any],
) -> dict[str, Any]:
    return with_payload_sha256(
        {
            "schema": "pcno_response_filtered_block_structural_gate_source_v1",
            "working_id": STRUCTURAL_GATE_WORKING_ID,
            "source_sha256": _source_hashes(),
            "source_status": _git_status_short(SOURCE_PATHS),
            "strength_ood_lineage": dict(strength_ood_source),
            "stopped_a6_rollout_sha256": sha256_file(args.a6_rollout),
            "stopped_a6_rollout_payload_sha256": a6_rollout["payload_sha256"],
            "stopped_a6_artifact_sha256": EXPECTED_P6_A6_ARTIFACT_SHA256,
            "population": {
                "family": "dynamic_fv_shock_vortex",
                "split": "test",
                "split_group_id": STRENGTH_OOD_GROUP_ID,
                "case_ids": list(STRENGTH_OOD_CASE_IDS),
                "population_role": "already_open_retrospective_protocol_calibration",
                "output_calls": list(range(1, 31)),
                "truth_during_inference": False,
                "truth_use": "offline_scoring_and_post_run_visualization_only",
                "new_sealed_population_opened": False,
                "still_sealed_groups": ["strength_ood_e12", "strength_ood_e14"],
                "fine_reference_truth": "not_required_or_loaded",
            },
            "protocol": {
                "initial_descriptor": (
                    "physical_volume_weighted_abs_transverse_velocity_centroid"
                ),
                "normalized_wall_distance": (
                    "2*min(y_centroid-y_min,y_max-y_centroid)/(y_max-y_min)"
                ),
                "position_trust_threshold": FROZEN_POSITION_TRUST_THRESHOLD,
                "threshold_origin": (
                    "midpoint_between_open_e13_y00_y08_and_y01_y07_position_bands"
                ),
                "initial_reject_action": "raw_native_only_for_complete_trajectory",
                "trusted_block": "exact_A5_six_call_shadow_anchored_block",
                "front_veto": (
                    "reject_both_block_outputs_on_any_candidate_shadow_"
                    "pressure_front_thickness_cell_branch_change"
                ),
                "front_veto_threshold": "exact_integer_thickness_mismatch",
                "veto_latch": "permanent_raw_shadow_fallback",
                "decision_latency": "one_synchronized_two_output_block",
                "reactivation": False,
                "true_error_or_reference_at_inference": False,
                "coefficient": -0.5,
                "active_cells": [list(cell) for cell in FROZEN_ACTIVE_CELLS],
                "anchor": "minimum_volume_weighted_l2_boundary_zero",
            },
            "retrospective_scoring_gate": {
                "aggregate_state_rms_ratio_at_most": 0.995,
                "aggregate_increment_defect_rms_ratio_at_most": 1.0,
                "median_endpoint_cumulative_defect_ratio_at_most": 1.0,
                "maximum_endpoint_state_ratio_at_most": 1.0,
                "all_registered_controls_ratio_at_most": 1.05,
                "all_rollouts_finite_admissible": True,
                "all_target_free_decisions_replay_counterfactual": True,
                "prospective_claim_authorized": False,
            },
            "animation_contract": STRUCTURAL_GATE_ANIMATION_CONTRACT,
        }
    )


def _prospective_structural_source_manifest(
    args: argparse.Namespace,
    *,
    strength_ood_source: Mapping[str, Any],
    a6_rollout: Mapping[str, Any],
    a7_rollout: Mapping[str, Any],
) -> dict[str, Any]:
    return with_payload_sha256(
        {
            "schema": "pcno_response_filtered_block_prospective_structural_source_v1",
            "working_id": PROSPECTIVE_STRUCTURAL_WORKING_ID,
            "source_sha256": _source_hashes(),
            "source_status": _git_status_short(SOURCE_PATHS),
            "strength_ood_lineage": dict(strength_ood_source),
            "stopped_a6_rollout_sha256": sha256_file(args.a6_rollout),
            "stopped_a6_rollout_payload_sha256": a6_rollout["payload_sha256"],
            "qualified_a7_rollout_sha256": sha256_file(args.a7_rollout),
            "qualified_a7_rollout_payload_sha256": a7_rollout["payload_sha256"],
            "qualified_a7_artifact_sha256": EXPECTED_P6_A7_ARTIFACT_SHA256,
            "qualified_a7_animation_bundle_manifest_sha256": (
                EXPECTED_P6_A7_ANIMATION_BUNDLE_MANIFEST_SHA256
            ),
            "population": {
                "family": "dynamic_fv_shock_vortex",
                "split": "test",
                "split_group_id": PROSPECTIVE_STRUCTURAL_GROUP_ID,
                "case_ids": list(PROSPECTIVE_STRUCTURAL_CASE_IDS),
                "population_role": "prospectively_named_complete_strength_group",
                "output_calls": list(range(1, 31)),
                "truth_during_inference": False,
                "truth_use": "offline_scoring_and_post_run_visualization_only",
                "new_sealed_population_opened": PROSPECTIVE_STRUCTURAL_GROUP_ID,
                "still_sealed_groups": ["strength_ood_e14"],
                "fine_reference_truth": "not_required_or_loaded",
            },
            "protocol": {
                "calibration_identity": STRUCTURAL_GATE_WORKING_ID,
                "initial_descriptor": (
                    "physical_volume_weighted_abs_transverse_velocity_centroid"
                ),
                "normalized_wall_distance": (
                    "2*min(y_centroid-y_min,y_max-y_centroid)/(y_max-y_min)"
                ),
                "position_trust_threshold": FROZEN_POSITION_TRUST_THRESHOLD,
                "initial_reject_action": "raw_native_only_for_complete_trajectory",
                "trusted_block": "exact_A5_six_call_shadow_anchored_block",
                "front_veto": (
                    "reject_both_block_outputs_on_any_candidate_shadow_"
                    "pressure_front_thickness_cell_branch_change"
                ),
                "front_veto_threshold": "exact_integer_thickness_mismatch",
                "veto_latch": "permanent_raw_shadow_fallback",
                "decision_latency": "one_synchronized_two_output_block",
                "reactivation": False,
                "true_error_or_reference_at_inference": False,
                "coefficient": -0.5,
                "active_cells": [list(cell) for cell in FROZEN_ACTIVE_CELLS],
                "anchor": "minimum_volume_weighted_l2_boundary_zero",
                "coefficient_or_threshold_refit": False,
            },
            "prospective_scoring_gate": {
                "aggregate_state_rms_ratio_at_most": 0.995,
                "aggregate_increment_defect_rms_ratio_at_most": 1.0,
                "median_endpoint_cumulative_defect_ratio_at_most": 1.0,
                "maximum_endpoint_state_ratio_at_most": 1.0,
                "all_registered_controls_ratio_at_most": 1.05,
                "all_rollouts_finite_admissible": True,
                "all_target_free_decisions_replay_counterfactual": True,
                "single_use_no_refit_on_failure": True,
            },
            "animation_contract": PROSPECTIVE_STRUCTURAL_ANIMATION_CONTRACT,
        }
    )


def _shadow_candidate_source_manifest(
    args: argparse.Namespace,
    *,
    prospective_source: Mapping[str, Any],
    a8_rollout: Mapping[str, Any],
) -> dict[str, Any]:
    return with_payload_sha256(
        {
            "schema": "pcno_response_filtered_block_shadow_candidate_source_v1",
            "working_id": SHADOW_CANDIDATE_WORKING_ID,
            "source_sha256": _source_hashes(),
            "source_status": _git_status_short(SOURCE_PATHS),
            "a8_runtime_lineage": dict(prospective_source),
            "qualified_a8_rollout_sha256": sha256_file(args.a8_rollout),
            "qualified_a8_rollout_payload_sha256": a8_rollout["payload_sha256"],
            "qualified_a8_artifact_sha256": EXPECTED_P6_A8_ARTIFACT_SHA256,
            "qualified_a8_animation_bundle_manifest_sha256": (
                EXPECTED_P6_A8_ANIMATION_BUNDLE_MANIFEST_SHA256
            ),
            "population": {
                "family": "dynamic_fv_shock_vortex",
                "split": "test",
                "split_group_id": PROSPECTIVE_STRUCTURAL_GROUP_ID,
                "case_ids": list(PROSPECTIVE_STRUCTURAL_CASE_IDS),
                "population_role": "already_open_e12_mechanism_diagnostic",
                "output_calls": list(range(1, 31)),
                "new_sealed_population_opened": False,
                "still_sealed_groups": ["strength_ood_e14"],
                "truth_during_prediction": False,
                "truth_use": "offline_block_labels_only",
                "fine_reference_truth": "not_required_or_loaded",
            },
            "protocol": {
                "accepted_recurrence": "raw_native_only",
                "candidate_role": "synchronized_two_output_counterfactual_then_discard",
                "blocks_per_case": 15,
                "calls_per_block": 6,
                "native_calls_per_block": 5,
                "fine_calls_per_block": 1,
                "coefficient": -0.5,
                "active_cells": [list(cell) for cell in FROZEN_ACTIVE_CELLS],
                "anchor": "minimum_volume_weighted_l2_boundary_zero",
                "front_branch_audit": "record_only_never_vetoes_raw_recurrence",
                "denominator_floor": A9_DENOMINATOR_FLOOR,
                "features": list(A9_FEATURE_NAMES),
                "targets": list(A9_TARGET_NAMES),
                "feature_or_threshold_selection": False,
                "coefficient_or_threshold_refit": False,
                "candidate_accepted_into_recurrence": False,
            },
            "analysis": {
                "population_association": ["pearson", "spearman"],
                "case_association": "per_case_spearman_and_population_sign_agreement",
                "prediction": "leave_one_case_out_affine_vs_zero_skill",
                "multiple_testing_claim": False,
                "deployment_decision": False,
                "per_block_oracle": "diagnostic_only_not_deployable",
            },
        }
    )


def _warm_start_source_manifest(
    args: argparse.Namespace,
    *,
    prospective_source: Mapping[str, Any],
    a8_rollout: Mapping[str, Any],
    a9_rescore: Mapping[str, Any],
) -> dict[str, Any]:
    return with_payload_sha256(
        {
            "schema": "pcno_response_filtered_block_warm_start_source_v1",
            "working_id": WARM_START_WORKING_ID,
            "source_sha256": _source_hashes(),
            "source_status": _git_status_short(SOURCE_PATHS),
            "a8_runtime_lineage": dict(prospective_source),
            "qualified_a8_rollout_sha256": sha256_file(args.a8_rollout),
            "qualified_a8_rollout_payload_sha256": a8_rollout["payload_sha256"],
            "completed_a9_rescore_sha256": sha256_file(args.a9_rescore),
            "completed_a9_rescore_payload_sha256": a9_rescore["payload_sha256"],
            "completed_a9_rescore_artifact_sha256": (
                EXPECTED_P6_A9_RESCORE_ARTIFACT_SHA256
            ),
            "population": {
                "family": "dynamic_fv_shock_vortex",
                "split": "test",
                "split_group_id": PROSPECTIVE_STRUCTURAL_GROUP_ID,
                "case_ids": list(PROSPECTIVE_STRUCTURAL_CASE_IDS),
                "population_role": "already_open_e12_recurrent_calibration",
                "output_calls": list(range(1, 31)),
                "new_sealed_population_opened": False,
                "still_sealed_groups": ["strength_ood_e14"],
                "truth_during_inference": False,
                "truth_use": "offline_scoring_and_post_run_visualization_only",
                "fine_reference_truth": "not_required_or_loaded",
            },
            "protocol": {
                "edge_selector": ("initial_normalized_wall_distance_at_most_0p7375"),
                "position_trust_threshold": FROZEN_POSITION_TRUST_THRESHOLD,
                "edge_policy": "exact_A8_candidate_for_all_15_blocks",
                "interior_warm_blocks": WARM_START_BLOCKS,
                "interior_warm_policy": "exact_A8_candidate_blocks_0_through_4",
                "interior_coast_policy": (
                    "raw_native_recurrence_from_accepted_state_blocks_5_through_14"
                ),
                "coast_reset_to_shadow": False,
                "shadow_role_during_coast": "diagnostic_comparator_only",
                "front_veto": (
                    "warm_block_branch_change_rejects_block_and_latches_raw_shadow"
                ),
                "coefficient": -0.5,
                "active_cells": [list(cell) for cell in FROZEN_ACTIVE_CELLS],
                "anchor": "minimum_volume_weighted_l2_boundary_zero",
                "true_error_or_reference_at_inference": False,
                "coefficient_or_threshold_refit": False,
            },
            "calibration_gate": {
                "aggregate_state_rms_ratio_strictly_below_a8": a8_rollout["population"][
                    "aggregate_state_rms_ratio"
                ],
                "aggregate_increment_defect_rms_ratio_at_most_a8": a8_rollout[
                    "population"
                ]["aggregate_increment_defect_rms_ratio"],
                "minimum_strict_interior_trajectory_wins": 1,
                "median_endpoint_cumulative_defect_ratio_at_most": 1.0,
                "maximum_endpoint_state_ratio_at_most": 1.0,
                "all_registered_controls_ratio_at_most": 1.05,
                "prospective_e14_authorized": False,
            },
            "animation_contract": WARM_START_ANIMATION_CONTRACT,
        }
    )


def _projected_coast_source_manifest(
    args: argparse.Namespace,
    *,
    warm_start_source: Mapping[str, Any],
    a10_rollout: Mapping[str, Any],
) -> dict[str, Any]:
    return with_payload_sha256(
        {
            "schema": "pcno_response_filtered_block_projected_coast_source_v1",
            "working_id": PROJECTED_COAST_WORKING_ID,
            "source_sha256": _source_hashes(),
            "source_status": _git_status_short(SOURCE_PATHS),
            "a10_runtime_lineage": dict(warm_start_source),
            "stopped_a10_rollout_sha256": sha256_file(args.a10_rollout),
            "stopped_a10_rollout_payload_sha256": a10_rollout["payload_sha256"],
            "stopped_a10_artifact_sha256": EXPECTED_P6_A10_ARTIFACT_SHA256,
            "population": {
                "family": "dynamic_fv_shock_vortex",
                "split": "test",
                "split_group_id": PROSPECTIVE_STRUCTURAL_GROUP_ID,
                "case_ids": list(PROSPECTIVE_STRUCTURAL_CASE_IDS),
                "population_role": "already_open_e12_recurrent_calibration",
                "output_calls": list(range(1, 31)),
                "new_sealed_population_opened": False,
                "still_sealed_groups": ["strength_ood_e14"],
                "truth_during_inference": False,
                "truth_use": "offline_scoring_and_post_run_visualization_only",
                "fine_reference_truth": "not_required_or_loaded",
            },
            "protocol": {
                "edge_selector": ("initial_normalized_wall_distance_at_most_0p7375"),
                "position_trust_threshold": FROZEN_POSITION_TRUST_THRESHOLD,
                "edge_policy": "exact_A8_candidate_for_all_15_blocks",
                "interior_warm_blocks": WARM_START_BLOCKS,
                "interior_warm_policy": "exact_A10_candidate_blocks_0_through_4",
                "interior_coast_policy": (
                    "native_accepted_and_shadow_then_per_output_fixed_SP19_"
                    "accepted_minus_shadow_tether"
                ),
                "coast_reset_to_shadow": False,
                "coast_projection_rank": 8,
                "coast_projection_active_cells": [
                    list(cell) for cell in FROZEN_ACTIVE_CELLS
                ],
                "coast_projection_nonconstant_only": True,
                "front_veto": (
                    "any_warm_or_projected_coast_block_branch_change_rejects_"
                    "block_and_latches_raw_shadow"
                ),
                "coefficient": -0.5,
                "anchor": "minimum_volume_weighted_l2_boundary_zero_warm_only",
                "true_error_or_reference_at_inference": False,
                "coefficient_or_threshold_refit": False,
            },
            "calibration_gate": {
                "aggregate_state_rms_ratio_strictly_below_a8": 0.9884779121207531,
                "aggregate_increment_defect_rms_ratio_at_most_a8": (0.9970100807663629),
                "minimum_strict_interior_trajectory_wins": 5,
                "median_endpoint_cumulative_defect_ratio_at_most": 1.0,
                "maximum_endpoint_state_ratio_at_most": 1.0,
                "all_registered_controls_ratio_at_most": 1.05,
                "prospective_e14_authorized": False,
            },
            "animation_contract": PROJECTED_COAST_ANIMATION_CONTRACT,
        }
    )


def _slew_limited_tether_source_manifest(
    args: argparse.Namespace,
    *,
    projected_coast_source: Mapping[str, Any],
    a11_rollout: Mapping[str, Any],
    a17_rescore: Mapping[str, Any],
) -> dict[str, Any]:
    return with_payload_sha256(
        {
            "schema": "pcno_response_filtered_block_slew_limited_tether_source_v1",
            "working_id": SLEW_LIMITED_TETHER_WORKING_ID,
            "source_sha256": _source_hashes(),
            "source_status": _git_status_short(SOURCE_PATHS),
            "a11_runtime_lineage": dict(projected_coast_source),
            "stopped_a11_rollout_sha256": sha256_file(args.a11_rollout),
            "stopped_a11_rollout_payload_sha256": a11_rollout["payload_sha256"],
            "qualified_a17_rescore_sha256": sha256_file(args.a17_rescore),
            "qualified_a17_rescore_payload_sha256": a17_rescore["payload_sha256"],
            "population": {
                "family": "dynamic_fv_shock_vortex",
                "split": "test",
                "split_group_id": PROSPECTIVE_STRUCTURAL_GROUP_ID,
                "case_ids": list(PROSPECTIVE_STRUCTURAL_CASE_IDS),
                "population_role": "already_open_e12_recurrent_mechanism_calibration",
                "output_calls": list(range(1, 31)),
                "new_sealed_population_opened": False,
                "still_sealed_groups": [],
                "truth_during_inference": False,
                "truth_use": "offline_scoring_and_post_run_visualization_only",
                "fine_reference_truth": "not_required_or_loaded",
            },
            "protocol": {
                "position_trust_threshold": FROZEN_POSITION_TRUST_THRESHOLD,
                "edge_policy": "exact_A8_candidate_for_all_15_blocks",
                "interior_warm_blocks": FROZEN_OFFSET_WARM_BLOCKS,
                "interior_warm_policy": "exact_candidate_blocks_0_through_3",
                "interior_coast_policy": (
                    "native_accepted_and_shadow_then_per_output_SP19_target_"
                    "with_displacement_change_slew_limit"
                ),
                "coast_reset_to_shadow": False,
                "coast_projection_rank": 8,
                "coast_projection_active_cells": [
                    list(cell) for cell in FROZEN_ACTIVE_CELLS
                ],
                "coast_projection_nonconstant_target": True,
                "relative_displacement_change_limit": (
                    SLEW_LIMITED_TETHER_RELATIVE_CHANGE_LIMIT
                ),
                "limit_reference_norm": "raw_shadow_increment_weighted_state_rms",
                "exact_increment_identity": (
                    "accepted_increment_minus_shadow_increment_equals_"
                    "accepted_shadow_displacement_change"
                ),
                "front_veto": (
                    "any_warm_or_slew_coast_block_branch_change_rejects_block_"
                    "and_latches_raw_shadow"
                ),
                "coefficient": -0.5,
                "anchor": "minimum_volume_weighted_l2_boundary_zero_warm_only",
                "true_error_or_reference_at_inference": False,
                "coefficient_or_trust_threshold_refit": False,
                "single_unswept_relative_limit": True,
                "a11_a17_truth_used_for_mechanism_choice": True,
            },
            "calibration_gate": {
                "aggregate_state_rms_ratio_strictly_below_a8": 0.9884779121207531,
                "aggregate_increment_defect_rms_ratio_at_most_a8": (
                    0.9970100807663629
                ),
                "minimum_strict_interior_trajectory_wins": 5,
                "median_endpoint_cumulative_defect_ratio_at_most": 1.0,
                "maximum_endpoint_state_ratio_at_most": 1.0,
                "all_registered_controls_ratio_at_most": 1.05,
                "slew_bound_and_closure_required": True,
                "prospective_or_cross_family_authorized": False,
            },
            "animation_contract": SLEW_LIMITED_TETHER_ANIMATION_CONTRACT,
        }
    )


def _relaxed_tether_source_manifest(
    args: argparse.Namespace,
    *,
    slew_source: Mapping[str, Any],
    a18_rollout: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = dict(slew_source["protocol"])
    protocol.update(
        {
            "interior_coast_policy": (
                "native_accepted_and_shadow_then_fixed_relaxation_toward_"
                "per_output_SP19_target"
            ),
            "relaxation_rate": RELAXED_TETHER_RATE,
            "relaxation_reference": "projected_target_displacement_gap",
            "relative_displacement_change_limit": None,
            "single_unswept_relative_limit": False,
            "single_unswept_relaxation_rate": True,
            "a18_truth_used_for_transition_localization_only": True,
        }
    )
    return with_payload_sha256(
        {
            "schema": "pcno_response_filtered_block_relaxed_tether_source_v1",
            "working_id": RELAXED_TETHER_WORKING_ID,
            "source_sha256": _source_hashes(),
            "source_status": _git_status_short(SOURCE_PATHS),
            "a18_runtime_lineage": dict(slew_source),
            "stopped_a18_rollout_sha256": sha256_file(args.a18_rollout),
            "stopped_a18_rollout_payload_sha256": a18_rollout["payload_sha256"],
            "stopped_a18_artifact_sha256": EXPECTED_P6_A18_ARTIFACT_SHA256,
            "population": dict(slew_source["population"]),
            "protocol": protocol,
            "calibration_gate": dict(slew_source["calibration_gate"]),
            "animation_contract": RELAXED_TETHER_ANIMATION_CONTRACT,
        }
    )


def _verify_a22_rescore(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_P6_A22_RESCORE_SHA256:
        raise ValueError("A22-R1 requires the exact qualified A22 rescore")
    result = _read_json(path)
    verify_payload_sha256(result)
    if (
        result.get("schema") != BUFFERED_RELAXED_TETHER_RESCORE_SCHEMA
        or result.get("working_id")
        != BUFFERED_RELAXED_TETHER_RESCORE_WORKING_ID
        or result.get("status") != "qualified_calibration_rescore"
        or result.get("payload_sha256") != EXPECTED_P6_A22_RESCORE_PAYLOAD_SHA256
        or result.get("artifact_hashes")
        != EXPECTED_P6_A22_RESCORE_ARTIFACT_SHA256
        or result.get("model_calls") != 0
        or result.get("recurrence_reexecuted") is not False
        or result.get("calibration_gate", {}).get("one_exact_replay_authorized")
        is not True
        or not all(result.get("calibration_gate", {}).get("checks", {}).values())
    ):
        raise ValueError("A22 rescore identity or qualification mismatch")
    root = path.parent
    actual = {
        name: sha256_file(root / name)
        for name in EXPECTED_P6_A22_RESCORE_ARTIFACT_SHA256
    }
    if actual != EXPECTED_P6_A22_RESCORE_ARTIFACT_SHA256:
        raise ValueError("A22 rescore side-artifact hash mismatch")
    return result


def _verify_a22_r1_rollout(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_P6_A22_R1_ROLLOUT_SHA256:
        raise ValueError("A23 requires the exact qualified A22-R1 replay")
    result = _read_json(path)
    verify_payload_sha256(result)
    if (
        result.get("schema") != BUFFERED_RELAXED_TETHER_REPLAY_SCHEMA
        or result.get("working_id") != BUFFERED_RELAXED_TETHER_REPLAY_WORKING_ID
        or result.get("status") != "qualified_calibration_replay"
        or result.get("payload_sha256") != EXPECTED_P6_A22_R1_PAYLOAD_SHA256
        or result.get("artifact_hashes") != EXPECTED_P6_A22_R1_ARTIFACT_SHA256
        or result.get("recurrence_executed") is not True
        or not all(result.get("calibration_gate", {}).get("checks", {}).values())
        or result.get("cost", {}).get("logical_model_calls") != 580
        or result.get("cost", {}).get("native_logical_calls") != 530
        or result.get("cost", {}).get("fine_logical_calls") != 50
        or result.get("population", {}).get("strict_interior_trajectory_win_count")
        != 5
    ):
        raise ValueError("A22-R1 identity or qualification mismatch")
    root = path.parent
    actual = {
        name: sha256_file(root / name)
        for name in EXPECTED_P6_A22_R1_ARTIFACT_SHA256
    }
    if actual != EXPECTED_P6_A22_R1_ARTIFACT_SHA256:
        raise ValueError("A22-R1 side-artifact hash mismatch")
    bundle_manifest = root / "animation_bundles" / "bundle_manifest.json"
    if (
        result.get("animation_bundle_manifest_sha256")
        != EXPECTED_P6_A22_R1_ANIMATION_BUNDLE_MANIFEST_SHA256
        or sha256_file(bundle_manifest)
        != EXPECTED_P6_A22_R1_ANIMATION_BUNDLE_MANIFEST_SHA256
    ):
        raise ValueError("A22-R1 animation-bundle identity mismatch")
    return result


def _buffered_relaxed_tether_replay_source_manifest(
    args: argparse.Namespace,
    *,
    relaxed_source: Mapping[str, Any],
    a22_rescore: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = dict(relaxed_source["protocol"])
    protocol.update(
        {
            "position_buffer_threshold": FROZEN_POSITION_BUFFER_THRESHOLD,
            "position_route": (
                "edge_candidate_at_most_0p7375_raw_buffer_through_0p8125_"
                "deep_interior_relaxed_tether"
            ),
            "buffer_policy": "raw_native_all_30_outputs_without_candidate_calls",
            "unresolved_position_policy": "raw_native",
            "buffer_threshold_selected_from_a12_truth": True,
            "route_or_numeric_parameter_refit_after_a22": False,
        }
    )
    return with_payload_sha256(
        {
            "schema": (
                "pcno_response_filtered_block_buffered_relaxed_tether_"
                "replay_source_v1"
            ),
            "working_id": BUFFERED_RELAXED_TETHER_REPLAY_WORKING_ID,
            "source_sha256": _source_hashes(),
            "source_status": _git_status_short(SOURCE_PATHS),
            "a19_runtime_lineage": dict(relaxed_source),
            "qualified_a22_rescore_sha256": sha256_file(args.a22_rescore),
            "qualified_a22_rescore_payload_sha256": a22_rescore["payload_sha256"],
            "qualified_a22_rescore_artifact_sha256": (
                EXPECTED_P6_A22_RESCORE_ARTIFACT_SHA256
            ),
            "population": dict(relaxed_source["population"]),
            "protocol": protocol,
            "calibration_gate": dict(relaxed_source["calibration_gate"]),
            "animation_contract": BUFFERED_RELAXED_TETHER_REPLAY_ANIMATION_CONTRACT,
        }
    )


def _buffered_relaxed_tether_e14_source_manifest(
    args: argparse.Namespace,
    *,
    replay_source: Mapping[str, Any],
    a22_r1_rollout: Mapping[str, Any],
    a14_rollout: Mapping[str, Any],
) -> dict[str, Any]:
    truth_contract = _native_scoring_truth_contract(args)
    return with_payload_sha256(
        {
            "schema": (
                "pcno_response_filtered_block_buffered_relaxed_tether_"
                "e14_source_v1"
            ),
            "working_id": BUFFERED_RELAXED_TETHER_E14_WORKING_ID,
            "source_sha256": _source_hashes(),
            "source_status": _git_status_short(SOURCE_PATHS),
            "a22_r1_runtime_lineage": dict(replay_source),
            "qualified_a22_r1_rollout_sha256": sha256_file(args.a22_r1_rollout),
            "qualified_a22_r1_rollout_payload_sha256": a22_r1_rollout[
                "payload_sha256"
            ],
            "qualified_a22_r1_artifact_sha256": (
                EXPECTED_P6_A22_R1_ARTIFACT_SHA256
            ),
            "stopped_a14_comparator_sha256": sha256_file(args.a14_rollout),
            "stopped_a14_comparator_payload_sha256": a14_rollout["payload_sha256"],
            "population": {
                "family": "dynamic_fv_shock_vortex",
                "split": "test",
                "split_group_id": PROSPECTIVE_BUFFERED_OFFSET_GROUP_ID,
                "case_ids": list(PROSPECTIVE_BUFFERED_OFFSET_CASE_IDS),
                "epsilon_index": 14,
                "vortex_epsilon": 0.3875,
                "vortex_y": [0.35 + 0.0375 * index for index in range(9)],
                "population_role": "already_open_e14_frozen_protocol_transfer",
                "output_calls": list(range(1, 31)),
                "new_sealed_population_opened": False,
                "still_sealed_groups": [],
                "truth_during_inference": False,
                "truth_use": "offline_scoring_and_post_run_visualization_only",
                "fine_reference_truth": "not_required_or_loaded",
                "native_scoring_truth_contract": truth_contract,
            },
            "protocol": {
                **dict(replay_source["protocol"]),
                "frozen_from": BUFFERED_RELAXED_TETHER_REPLAY_WORKING_ID,
                "e14_outcome_used_for_protocol_selection": False,
                "coefficient_threshold_rate_cells_or_window_refit": False,
            },
            "calibration_gate": dict(replay_source["calibration_gate"]),
            "animation_contract": BUFFERED_RELAXED_TETHER_E14_ANIMATION_CONTRACT,
        }
    )


def _frozen_offset_source_manifest(
    args: argparse.Namespace,
    *,
    projected_coast_source: Mapping[str, Any],
    a11_rollout: Mapping[str, Any],
) -> dict[str, Any]:
    return with_payload_sha256(
        {
            "schema": "pcno_response_filtered_block_frozen_offset_source_v1",
            "working_id": FROZEN_OFFSET_WORKING_ID,
            "source_sha256": _source_hashes(),
            "source_status": _git_status_short(SOURCE_PATHS),
            "a11_runtime_lineage": dict(projected_coast_source),
            "stopped_a11_rollout_sha256": sha256_file(args.a11_rollout),
            "stopped_a11_rollout_payload_sha256": a11_rollout["payload_sha256"],
            "stopped_a11_artifact_sha256": EXPECTED_P6_A11_ARTIFACT_SHA256,
            "population": {
                "family": "dynamic_fv_shock_vortex",
                "split": "test",
                "split_group_id": PROSPECTIVE_STRUCTURAL_GROUP_ID,
                "case_ids": list(PROSPECTIVE_STRUCTURAL_CASE_IDS),
                "population_role": "already_open_e12_recurrent_calibration",
                "output_calls": list(range(1, 31)),
                "new_sealed_population_opened": False,
                "still_sealed_groups": ["strength_ood_e14"],
                "truth_during_inference": False,
                "truth_use": "offline_scoring_and_post_run_visualization_only",
                "fine_reference_truth": "not_required_or_loaded",
            },
            "protocol": {
                "edge_selector": "initial_normalized_wall_distance_at_most_0p7375",
                "position_trust_threshold": FROZEN_POSITION_TRUST_THRESHOLD,
                "edge_policy": "exact_A8_candidate_for_all_15_blocks",
                "interior_warm_blocks": FROZEN_OFFSET_WARM_BLOCKS,
                "interior_warm_policy": "exact_A8_candidate_blocks_0_through_3",
                "handoff": "fixed_SP19_projection_of_accepted_minus_shadow_at_call_8",
                "interior_coast_policy": "raw_native_recurrence_plus_frozen_output_offset",
                "coast_model_input": "raw_shadow_only",
                "coast_offset_feedback": False,
                "coast_projection_rank": 8,
                "coast_projection_active_cells": [
                    list(cell) for cell in FROZEN_ACTIVE_CELLS
                ],
                "coast_projection_nonconstant_only": True,
                "front_veto": (
                    "any_warm_handoff_or_coast_block_branch_change_rejects_block_"
                    "and_latches_raw_shadow"
                ),
                "coefficient": -0.5,
                "anchor": "minimum_volume_weighted_l2_boundary_zero_warm_only",
                "warm_window_selection": (
                    "longest_A11_zero_model_prefix_bound_below_A8_increment_ratio"
                ),
                "true_error_or_reference_at_inference": False,
                "coefficient_or_threshold_refit": False,
            },
            "calibration_gate": {
                "aggregate_state_rms_ratio_strictly_below_a8": 0.9884779121207531,
                "aggregate_increment_defect_rms_ratio_at_most_a8": 0.9970100807663629,
                "minimum_strict_interior_trajectory_wins": 5,
                "median_endpoint_cumulative_defect_ratio_at_most": 1.0,
                "maximum_endpoint_state_ratio_at_most": 1.0,
                "all_registered_controls_ratio_at_most": 1.05,
                "prospective_e14_authorized": False,
            },
            "animation_contract": FROZEN_OFFSET_ANIMATION_CONTRACT,
        }
    )


def _buffered_offset_source_manifest(
    args: argparse.Namespace,
    *,
    frozen_offset_source: Mapping[str, Any],
    a12_rollout: Mapping[str, Any],
    a12_buffer_rescore: Mapping[str, Any],
) -> dict[str, Any]:
    return with_payload_sha256(
        {
            "schema": "pcno_response_filtered_block_buffered_offset_source_v1",
            "working_id": BUFFERED_OFFSET_WORKING_ID,
            "source_sha256": _source_hashes(),
            "source_status": _git_status_short(SOURCE_PATHS),
            "a12_runtime_lineage": dict(frozen_offset_source),
            "stopped_a12_rollout_sha256": sha256_file(args.a12_rollout),
            "stopped_a12_rollout_payload_sha256": a12_rollout["payload_sha256"],
            "stopped_a12_artifact_sha256": EXPECTED_P6_A12_ARTIFACT_SHA256,
            "qualified_a12_buffer_rescore_sha256": sha256_file(args.a12_buffer_rescore),
            "qualified_a12_buffer_rescore_payload_sha256": a12_buffer_rescore[
                "payload_sha256"
            ],
            "qualified_a12_buffer_rescore_artifact_sha256": (
                EXPECTED_P6_A12_BUFFER_RESCORE_ARTIFACT_SHA256
            ),
            "population": {
                "family": "dynamic_fv_shock_vortex",
                "split": "test",
                "split_group_id": PROSPECTIVE_STRUCTURAL_GROUP_ID,
                "case_ids": list(PROSPECTIVE_STRUCTURAL_CASE_IDS),
                "population_role": "already_open_e12_recurrent_calibration",
                "output_calls": list(range(1, 31)),
                "new_sealed_population_opened": False,
                "still_sealed_groups": ["strength_ood_e14"],
                "truth_during_inference": False,
                "truth_use": "offline_scoring_and_post_run_visualization_only",
                "fine_reference_truth": "not_required_or_loaded",
            },
            "protocol": {
                "descriptor": "initial_normalized_wall_distance",
                "position_trust_threshold": FROZEN_POSITION_TRUST_THRESHOLD,
                "position_buffer_threshold": FROZEN_POSITION_BUFFER_THRESHOLD,
                "edge_candidate_at_most": FROZEN_POSITION_TRUST_THRESHOLD,
                "raw_uncertainty_buffer_at_most": FROZEN_POSITION_BUFFER_THRESHOLD,
                "interior_frozen_offset_above": FROZEN_POSITION_BUFFER_THRESHOLD,
                "unresolved_descriptor_action": "raw_native_complete_trajectory",
                "edge_policy": "exact_A8_candidate_for_all_15_blocks",
                "buffer_policy": "raw_native_complete_trajectory",
                "interior_warm_blocks": FROZEN_OFFSET_WARM_BLOCKS,
                "interior_warm_policy": "exact_A8_candidate_blocks_0_through_3",
                "handoff": "fixed_SP19_projection_of_accepted_minus_shadow_at_call_8",
                "interior_coast_policy": "raw_native_recurrence_plus_frozen_output_offset",
                "coast_model_input": "raw_shadow_only",
                "coast_offset_feedback": False,
                "coast_projection_rank": 8,
                "coast_projection_active_cells": [
                    list(cell) for cell in FROZEN_ACTIVE_CELLS
                ],
                "coast_projection_nonconstant_only": True,
                "front_veto": (
                    "any_candidate_handoff_or_coast_branch_change_latches_raw_shadow"
                ),
                "coefficient": -0.5,
                "active_cells": [list(cell) for cell in FROZEN_ACTIVE_CELLS],
                "buffer_selection": (
                    "symmetric_e12_truth_informed_no_forward_hybrid_rescore"
                ),
                "true_error_or_reference_at_inference": False,
                "coefficient_refit": False,
                "coefficient_or_trust_threshold_refit": False,
                "buffer_threshold_selected_from_a12_truth": True,
            },
            "calibration_gate": {
                "aggregate_state_rms_ratio_strictly_below_a8": 0.9884779121207531,
                "aggregate_increment_defect_rms_ratio_at_most_a8": 0.9970100807663629,
                "minimum_strict_corrected_interior_trajectory_wins": 5,
                "median_endpoint_cumulative_defect_ratio_at_most": 1.0,
                "maximum_endpoint_state_ratio_at_most": 1.0,
                "all_registered_controls_ratio_at_most": 1.05,
                "prospective_e14_authorized": False,
            },
            "animation_contract": BUFFERED_OFFSET_ANIMATION_CONTRACT,
        }
    )


def _prospective_buffered_offset_source_manifest(
    args: argparse.Namespace,
    *,
    buffered_offset_source: Mapping[str, Any],
    a13_rollout: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = dict(buffered_offset_source["protocol"])
    truth_contract = _native_scoring_truth_contract(args)
    return with_payload_sha256(
        {
            "schema": "pcno_response_filtered_block_prospective_buffered_offset_source_v1",
            "working_id": PROSPECTIVE_BUFFERED_OFFSET_WORKING_ID,
            "source_sha256": _source_hashes(),
            "source_status": _git_status_short(SOURCE_PATHS),
            "a13_runtime_lineage": dict(buffered_offset_source),
            "qualified_a13_rollout_sha256": sha256_file(args.a13_rollout),
            "qualified_a13_rollout_payload_sha256": a13_rollout["payload_sha256"],
            "qualified_a13_artifact_sha256": EXPECTED_P6_A13_ARTIFACT_SHA256,
            "qualified_a13_frozen_source_sha256": EXPECTED_P6_A13_SOURCE_SHA256,
            "population": {
                "family": "dynamic_fv_shock_vortex",
                "split": "test",
                "split_group_id": PROSPECTIVE_BUFFERED_OFFSET_GROUP_ID,
                "case_ids": list(PROSPECTIVE_BUFFERED_OFFSET_CASE_IDS),
                "epsilon_index": 14,
                "vortex_epsilon": 0.3875,
                "vortex_y": [0.35 + 0.0375 * index for index in range(9)],
                "population_role": "single_prospectively_named_e14_confirmation",
                "output_calls": list(range(1, 31)),
                "new_sealed_population_opened": PROSPECTIVE_BUFFERED_OFFSET_GROUP_ID,
                "still_sealed_groups": [],
                "truth_during_inference": False,
                "truth_use": "offline_scoring_and_post_run_visualization_only",
                "fine_reference_truth": "not_required_or_loaded",
                "native_scoring_truth_contract": truth_contract,
            },
            "protocol": {
                **protocol,
                "frozen_from": BUFFERED_OFFSET_WORKING_ID,
                "coefficient_refit": False,
                "coefficient_or_trust_threshold_refit": False,
                "additional_e12_selection": False,
            },
            "prospective_gate": {
                "aggregate_state_rms_ratio_strictly_below_a8": 0.9884779121207531,
                "aggregate_increment_defect_rms_ratio_at_most_a8": (0.9970100807663629),
                "minimum_strict_corrected_interior_trajectory_wins": 5,
                "median_endpoint_cumulative_defect_ratio_at_most": 1.0,
                "maximum_endpoint_state_ratio_at_most": 1.0,
                "all_registered_controls_ratio_at_most": 1.05,
                "all_existing_closure_and_admissibility_checks": True,
            },
            "animation_contract": PROSPECTIVE_BUFFERED_OFFSET_ANIMATION_CONTRACT,
        }
    )


def _persistence_gain_source_manifest(
    args: argparse.Namespace,
    *,
    buffered_offset_source: Mapping[str, Any],
    a13_rollout: Mapping[str, Any],
    a14_rollout: Mapping[str, Any],
) -> dict[str, Any]:
    base_protocol = dict(buffered_offset_source["protocol"])
    return with_payload_sha256(
        {
            "schema": "pcno_response_filtered_block_persistence_gain_source_v1",
            "working_id": PERSISTENCE_GAIN_WORKING_ID,
            "source_sha256": _source_hashes(),
            "source_status": _git_status_short(SOURCE_PATHS),
            "a13_runtime_lineage": dict(buffered_offset_source),
            "qualified_a13_rollout_sha256": sha256_file(args.a13_rollout),
            "qualified_a13_rollout_payload_sha256": a13_rollout["payload_sha256"],
            "stopped_a14_rollout_sha256": sha256_file(args.a14_rollout),
            "stopped_a14_rollout_payload_sha256": a14_rollout["payload_sha256"],
            "population": {
                "family": "dynamic_fv_shock_vortex",
                "split": "test",
                "split_group_id": PROSPECTIVE_STRUCTURAL_GROUP_ID,
                "case_ids": list(PROSPECTIVE_STRUCTURAL_CASE_IDS),
                "population_role": "already_open_e12_persistence_calibration",
                "output_calls": list(range(1, 31)),
                "new_sealed_population_opened": False,
                "still_sealed_groups": [],
                "truth_during_inference": False,
                "truth_use": (
                    "offline_scoring_and_probe_error_structure_diagnostics_only"
                ),
                "fine_reference_truth": "not_required_or_loaded",
                "e14_truth_use": "none_selection_and_none_execution",
            },
            "protocol": {
                **base_protocol,
                "frozen_from": BUFFERED_OFFSET_WORKING_ID,
                "edge_policy": "exact_A13_edge_candidate_all_15_blocks",
                "buffer_policy": "exact_A13_raw_native_complete_trajectory",
                "interior_warm_blocks": FROZEN_OFFSET_WARM_BLOCKS,
                "interior_warm_policy": "exact_A13_candidate_blocks_0_through_3",
                "handoff": "exact_A13_SP19_output_offset_at_call_8",
                "coast_model_input": "raw_shadow_only",
                "coast_offset_feedback": False,
                "persistence_probe_input": "same_raw_shadow_state_as_native_prediction",
                "persistence_probe_frequency": "one_fine_call_per_two_output_coast_block",
                "persistence_probe_policy": "SP19_fine_away_half",
                "persistence_probe_coefficient": -0.5,
                "persistence_probe_active_cells": [
                    list(cell) for cell in FROZEN_ACTIVE_CELLS
                ],
                "persistence_alignment": (
                    "physical_volume_state_scale_inner_product_with_handoff_offset"
                ),
                "baseline_alignment": "first_coast_probe_call_9",
                "gain": "clip_current_alignment_over_baseline_to_0_1_then_cummin",
                "unresolved_or_nonpositive_baseline_action": "gain_zero",
                "gain_shared_between": "the_two_outputs_of_each_coast_block",
                "true_error_or_reference_at_inference": False,
                "learned_coefficient": False,
                "coefficient_cell_route_or_window_refit": False,
                "a14_truth_used_for_selection": False,
                "front_veto": (
                    "any_candidate_handoff_or_persistence_coast_branch_change_"
                    "latches_raw_shadow"
                ),
            },
            "calibration_gate": {
                "aggregate_state_rms_ratio_strictly_below_one": 1.0,
                "aggregate_increment_defect_rms_ratio_at_most_a8": 0.9970100807663629,
                "minimum_strict_corrected_interior_trajectory_wins": 5,
                "median_endpoint_cumulative_defect_ratio_at_most": 1.0,
                "maximum_endpoint_state_ratio_at_most": 1.0,
                "all_registered_controls_ratio_at_most": 1.05,
                "all_gains_nonincreasing_and_bounded": True,
                "prospective_claim_authorized": False,
            },
            "diagnostics": {
                "probe_vs_signed_raw_state_error": (
                    "offline_weighted_cosine_and_rank8_component_region_decomposition"
                ),
                "probe_vs_handoff_offset": (
                    "signed_alignment_cosine_norm_and_applied_gain_per_block"
                ),
                "predictive_claim": "none_same_population_mechanism_diagnostic",
            },
            "cost_contract": {
                "a13_scored_calls": 470,
                "additional_fine_probe_calls": 55,
                "a15_scored_calls": 525,
                "deterministic_prefix_calls": 12,
            },
            "animation_contract": PERSISTENCE_GAIN_ANIMATION_CONTRACT,
        }
    )


def _terminal_ramp_source_manifest(
    args: argparse.Namespace,
    *,
    persistence_source: Mapping[str, Any],
    a15_rollout: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = dict(persistence_source["protocol"])
    return with_payload_sha256(
        {
            "schema": "pcno_response_filtered_block_terminal_ramp_source_v1",
            "working_id": TERMINAL_RAMP_WORKING_ID,
            "source_sha256": _source_hashes(),
            "source_status": _git_status_short(SOURCE_PATHS),
            "a15_runtime_lineage": dict(persistence_source),
            "stopped_a15_rollout_sha256": sha256_file(args.a15_rollout),
            "stopped_a15_rollout_payload_sha256": a15_rollout["payload_sha256"],
            "population": dict(persistence_source["population"]),
            "protocol": {
                **protocol,
                "gain": "unit_until_first_nonpositive_cosine_then_linear_to_zero",
                "phase_trigger": "probe_handoff_offset_cosine_at_most_zero_or_unresolved",
                "ramp_start_gain": 1.0,
                "ramp_terminal_block": 14,
                "ramp_terminal_gain": 0.0,
                "gain_rate_parameter": "derived_only_from_trigger_and_remaining_blocks",
                "a15_truth_used_for_rule": True,
                "coefficient_cell_route_or_numeric_rate_fit": False,
            },
            "calibration_gate": dict(persistence_source["calibration_gate"]),
            "diagnostics": {
                "a15_zero_model_rationale": (
                    "abrupt_gain_changes_localized_increment_gate_miss_and_"
                    "cosine_sign_marks_phase_loss"
                ),
                "truth_or_oracle_at_inference": False,
            },
            "cost_contract": dict(persistence_source["cost_contract"]),
            "animation_contract": TERMINAL_RAMP_ANIMATION_CONTRACT,
        }
    )


def _fixed_late_ramp_source_manifest(
    args: argparse.Namespace,
    *,
    buffered_source: Mapping[str, Any],
    a13_rollout: Mapping[str, Any],
    a16_rollout: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = dict(buffered_source["protocol"])
    return with_payload_sha256(
        {
            "schema": "pcno_response_filtered_block_fixed_late_ramp_source_v1",
            "working_id": FIXED_LATE_RAMP_WORKING_ID,
            "source_sha256": _source_hashes(),
            "source_status": _git_status_short(SOURCE_PATHS),
            "buffered_offset_runtime_lineage": dict(buffered_source),
            "qualified_a13_rollout_sha256": sha256_file(args.a13_rollout),
            "qualified_a13_rollout_payload_sha256": a13_rollout["payload_sha256"],
            "stopped_a16_rollout_sha256": sha256_file(args.a16_rollout),
            "stopped_a16_rollout_payload_sha256": a16_rollout["payload_sha256"],
            "population": {
                **dict(buffered_source["population"]),
                "still_sealed_groups": [],
            },
            "protocol": {
                **protocol,
                "gain": "unit_through_block9_then_fixed_linear_to_zero",
                "gain_by_coast_block_4_to_14": [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.8,
                    0.6,
                    0.4,
                    0.2,
                    0.0,
                ],
                "ramp_start_block": 9,
                "ramp_terminal_block": 14,
                "persistence_probe": False,
                "additional_fine_probe_calls": 0,
                "a13_a15_a16_truth_used_for_schedule_selection": True,
                "true_error_or_reference_at_inference": False,
                "coefficient_cell_route_or_schedule_refit_at_inference": False,
            },
            "calibration_gate": {
                "aggregate_state_rms_ratio_strictly_below_one": 1.0,
                "aggregate_increment_defect_rms_ratio_at_most_a8": (
                    0.9970100807663629
                ),
                "minimum_strict_corrected_interior_trajectory_wins": 5,
                "median_endpoint_cumulative_defect_ratio_at_most": 1.0,
                "maximum_endpoint_state_ratio_at_most": 1.0,
                "all_registered_controls_ratio_at_most": 1.05,
                "prospective_claim_authorized": False,
            },
            "offline_prediction": {
                "method": "exact_saved_trajectory_quadratic_in_gain_and_gain_change",
                "aggregate_state_rms_ratio": 0.9873397026522671,
                "aggregate_increment_defect_rms_ratio": 0.9970089757436883,
                "truth_used": True,
                "model_calls": 0,
            },
            "cost_contract": {
                "a13_scored_calls": 470,
                "additional_fine_probe_calls": 0,
                "a17_scored_calls": 470,
                "deterministic_prefix_calls": 12,
            },
            "animation_contract": FIXED_LATE_RAMP_ANIMATION_CONTRACT,
        }
    )


def _buffered_offset_source_with_truth_contract(
    source: Mapping[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    payload = {key: value for key, value in source.items() if key != "payload_sha256"}
    payload["population"] = {
        **dict(payload["population"]),
        "native_scoring_truth_contract": _native_scoring_truth_contract(args),
    }
    return with_payload_sha256(payload)


def _open_contract(args: argparse.Namespace):
    p5 = _verify_p5(args.p5_calibration)
    checkpoint, manifest, store, parent_source = parent._open_contract(args)
    try:
        source = _source_manifest(args, parent_source=parent_source, p5=p5)
    except Exception:
        store.close()
        raise
    return checkpoint, manifest, store, source, p5


def _build_runtime(args: argparse.Namespace):
    p5 = _verify_p5(args.p5_calibration)
    runtime, parent_source = parent._build_runtime(args)
    try:
        source = _source_manifest(args, parent_source=parent_source, p5=p5)
    except Exception:
        collection._close_runtime(runtime)
        raise
    return runtime, source, p5


def _open_evaluation_contract(args: argparse.Namespace):
    p5 = _verify_p5(args.p5_calibration)
    rescore = _verify_rescore(args.rescore)
    checkpoint, manifest, store, parent_source = parent._open_contract(args)
    try:
        source = _evaluation_source_manifest(
            args, parent_source=parent_source, p5=p5, rescore=rescore
        )
    except Exception:
        store.close()
        raise
    return checkpoint, manifest, store, source, p5, rescore


def _build_evaluation_runtime(args: argparse.Namespace):
    p5 = _verify_p5(args.p5_calibration)
    rescore = _verify_rescore(args.rescore)
    runtime, parent_source = parent._build_runtime(args)
    try:
        source = _evaluation_source_manifest(
            args, parent_source=parent_source, p5=p5, rescore=rescore
        )
    except Exception:
        collection._close_runtime(runtime)
        raise
    return runtime, source, p5, rescore


def _open_rollout_contract(args: argparse.Namespace):
    evaluation = _verify_evaluation(args.evaluation)
    checkpoint, manifest, store, evaluation_source, p5, rescore = (
        _open_evaluation_contract(args)
    )
    try:
        source = _rollout_source_manifest(
            args, evaluation_source=evaluation_source, evaluation=evaluation
        )
    except Exception:
        store.close()
        raise
    return checkpoint, manifest, store, source, p5, rescore, evaluation


def _build_rollout_runtime(args: argparse.Namespace):
    evaluation = _verify_evaluation(args.evaluation)
    runtime, evaluation_source, p5, rescore = _build_evaluation_runtime(args)
    try:
        source = _rollout_source_manifest(
            args, evaluation_source=evaluation_source, evaluation=evaluation
        )
    except Exception:
        collection._close_runtime(runtime)
        raise
    return runtime, source, p5, rescore, evaluation


def _open_shadow_contract(args: argparse.Namespace):
    evaluation = _verify_evaluation(args.evaluation)
    a4_rollout = _verify_a4_rollout(args.a4_rollout)
    checkpoint, manifest, store, evaluation_source, p5, rescore = (
        _open_evaluation_contract(args)
    )
    try:
        source = _shadow_source_manifest(
            args,
            evaluation_source=evaluation_source,
            evaluation=evaluation,
            a4_rollout=a4_rollout,
        )
    except Exception:
        store.close()
        raise
    return checkpoint, manifest, store, source, p5, rescore, evaluation, a4_rollout


def _build_shadow_runtime(args: argparse.Namespace):
    evaluation = _verify_evaluation(args.evaluation)
    a4_rollout = _verify_a4_rollout(args.a4_rollout)
    runtime, evaluation_source, p5, rescore = _build_evaluation_runtime(args)
    try:
        source = _shadow_source_manifest(
            args,
            evaluation_source=evaluation_source,
            evaluation=evaluation,
            a4_rollout=a4_rollout,
        )
    except Exception:
        collection._close_runtime(runtime)
        raise
    return runtime, source, p5, rescore, evaluation, a4_rollout


def _open_strength_ood_contract(args: argparse.Namespace):
    a5_rescore = _verify_a5_rescore(args.a5_rescore)
    (
        checkpoint,
        manifest,
        store,
        adaptive_shadow_source,
        _,
        _,
        _,
        _,
    ) = _open_shadow_contract(args)
    try:
        source = _strength_ood_source_manifest(
            args,
            adaptive_shadow_source=adaptive_shadow_source,
            a5_rescore=a5_rescore,
        )
    except Exception:
        store.close()
        raise
    return checkpoint, manifest, store, source, a5_rescore


def _build_strength_ood_runtime(args: argparse.Namespace):
    a5_rescore = _verify_a5_rescore(args.a5_rescore)
    runtime, adaptive_shadow_source, _, _, _, _ = _build_shadow_runtime(args)
    try:
        source = _strength_ood_source_manifest(
            args,
            adaptive_shadow_source=adaptive_shadow_source,
            a5_rescore=a5_rescore,
        )
    except Exception:
        collection._close_runtime(runtime)
        raise
    return runtime, source, a5_rescore


def _open_structural_gate_contract(args: argparse.Namespace):
    a6_rollout = _verify_a6_rollout(args.a6_rollout)
    checkpoint, manifest, store, strength_source, a5_rescore = (
        _open_strength_ood_contract(args)
    )
    try:
        source = _structural_gate_source_manifest(
            args,
            strength_ood_source=strength_source,
            a6_rollout=a6_rollout,
        )
    except Exception:
        store.close()
        raise
    return checkpoint, manifest, store, source, a5_rescore, a6_rollout


def _build_structural_gate_runtime(args: argparse.Namespace):
    a6_rollout = _verify_a6_rollout(args.a6_rollout)
    runtime, strength_source, a5_rescore = _build_strength_ood_runtime(args)
    try:
        source = _structural_gate_source_manifest(
            args,
            strength_ood_source=strength_source,
            a6_rollout=a6_rollout,
        )
    except Exception:
        collection._close_runtime(runtime)
        raise
    return runtime, source, a5_rescore, a6_rollout


def _open_prospective_structural_contract(args: argparse.Namespace):
    a6_rollout = _verify_a6_rollout(args.a6_rollout)
    a7_rollout = _verify_a7_rollout(args.a7_rollout)
    checkpoint, manifest, store, strength_source, a5_rescore = (
        _open_strength_ood_contract(args)
    )
    try:
        source = _prospective_structural_source_manifest(
            args,
            strength_ood_source=strength_source,
            a6_rollout=a6_rollout,
            a7_rollout=a7_rollout,
        )
    except Exception:
        store.close()
        raise
    return (
        checkpoint,
        manifest,
        store,
        source,
        a5_rescore,
        a6_rollout,
        a7_rollout,
    )


def _build_prospective_structural_runtime(args: argparse.Namespace):
    a6_rollout = _verify_a6_rollout(args.a6_rollout)
    a7_rollout = _verify_a7_rollout(args.a7_rollout)
    runtime, strength_source, a5_rescore = _build_strength_ood_runtime(args)
    try:
        source = _prospective_structural_source_manifest(
            args,
            strength_ood_source=strength_source,
            a6_rollout=a6_rollout,
            a7_rollout=a7_rollout,
        )
    except Exception:
        collection._close_runtime(runtime)
        raise
    return runtime, source, a5_rescore, a6_rollout, a7_rollout


def _open_shadow_candidate_contract(args: argparse.Namespace):
    a8_rollout = _verify_a8_rollout(args.a8_rollout)
    (
        checkpoint,
        manifest,
        store,
        prospective_source,
        a5_rescore,
        a6_rollout,
        a7_rollout,
    ) = _open_prospective_structural_contract(args)
    try:
        source = _shadow_candidate_source_manifest(
            args,
            prospective_source=prospective_source,
            a8_rollout=a8_rollout,
        )
    except Exception:
        store.close()
        raise
    return (
        checkpoint,
        manifest,
        store,
        source,
        a5_rescore,
        a6_rollout,
        a7_rollout,
        a8_rollout,
    )


def _build_shadow_candidate_runtime(args: argparse.Namespace):
    a8_rollout = _verify_a8_rollout(args.a8_rollout)
    runtime, prospective_source, a5_rescore, a6_rollout, a7_rollout = (
        _build_prospective_structural_runtime(args)
    )
    try:
        source = _shadow_candidate_source_manifest(
            args,
            prospective_source=prospective_source,
            a8_rollout=a8_rollout,
        )
    except Exception:
        collection._close_runtime(runtime)
        raise
    return runtime, source, a5_rescore, a6_rollout, a7_rollout, a8_rollout


def _open_warm_start_contract(args: argparse.Namespace):
    a8_rollout = _verify_a8_rollout(args.a8_rollout)
    a9_rescore = _verify_a9_rescore(args.a9_rescore)
    (
        checkpoint,
        manifest,
        store,
        prospective_source,
        a5_rescore,
        a6_rollout,
        a7_rollout,
    ) = _open_prospective_structural_contract(args)
    try:
        source = _warm_start_source_manifest(
            args,
            prospective_source=prospective_source,
            a8_rollout=a8_rollout,
            a9_rescore=a9_rescore,
        )
    except Exception:
        store.close()
        raise
    return (
        checkpoint,
        manifest,
        store,
        source,
        a5_rescore,
        a6_rollout,
        a7_rollout,
        a8_rollout,
        a9_rescore,
    )


def _build_warm_start_runtime(args: argparse.Namespace):
    a8_rollout = _verify_a8_rollout(args.a8_rollout)
    a9_rescore = _verify_a9_rescore(args.a9_rescore)
    runtime, prospective_source, a5_rescore, a6_rollout, a7_rollout = (
        _build_prospective_structural_runtime(args)
    )
    try:
        source = _warm_start_source_manifest(
            args,
            prospective_source=prospective_source,
            a8_rollout=a8_rollout,
            a9_rescore=a9_rescore,
        )
    except Exception:
        collection._close_runtime(runtime)
        raise
    return (
        runtime,
        source,
        a5_rescore,
        a6_rollout,
        a7_rollout,
        a8_rollout,
        a9_rescore,
    )


def _open_projected_coast_contract(args: argparse.Namespace):
    a10_rollout = _verify_a10_rollout(args.a10_rollout)
    (
        checkpoint,
        manifest,
        store,
        warm_start_source,
        a5_rescore,
        a6_rollout,
        a7_rollout,
        a8_rollout,
        a9_rescore,
    ) = _open_warm_start_contract(args)
    try:
        source = _projected_coast_source_manifest(
            args,
            warm_start_source=warm_start_source,
            a10_rollout=a10_rollout,
        )
    except Exception:
        store.close()
        raise
    return (
        checkpoint,
        manifest,
        store,
        source,
        a5_rescore,
        a6_rollout,
        a7_rollout,
        a8_rollout,
        a9_rescore,
        a10_rollout,
    )


def _build_projected_coast_runtime(args: argparse.Namespace):
    a10_rollout = _verify_a10_rollout(args.a10_rollout)
    (
        runtime,
        warm_start_source,
        a5_rescore,
        a6_rollout,
        a7_rollout,
        a8_rollout,
        a9_rescore,
    ) = _build_warm_start_runtime(args)
    try:
        source = _projected_coast_source_manifest(
            args,
            warm_start_source=warm_start_source,
            a10_rollout=a10_rollout,
        )
    except Exception:
        collection._close_runtime(runtime)
        raise
    return (
        runtime,
        source,
        a5_rescore,
        a6_rollout,
        a7_rollout,
        a8_rollout,
        a9_rescore,
        a10_rollout,
    )


def _open_slew_limited_tether_contract(args: argparse.Namespace):
    a11_rollout = _verify_a11_rollout(args.a11_rollout)
    a17_rescore = _verify_a17_rescore(args.a17_rescore)
    opened = _open_projected_coast_contract(args)
    checkpoint, manifest, store, projected_source, *lineage = opened
    try:
        source = _slew_limited_tether_source_manifest(
            args,
            projected_coast_source=projected_source,
            a11_rollout=a11_rollout,
            a17_rescore=a17_rescore,
        )
    except Exception:
        store.close()
        raise
    return (
        checkpoint,
        manifest,
        store,
        source,
        *lineage,
        a11_rollout,
        a17_rescore,
    )


def _build_slew_limited_tether_runtime(args: argparse.Namespace):
    a11_rollout = _verify_a11_rollout(args.a11_rollout)
    a17_rescore = _verify_a17_rescore(args.a17_rescore)
    built = _build_projected_coast_runtime(args)
    runtime, projected_source, *lineage = built
    try:
        source = _slew_limited_tether_source_manifest(
            args,
            projected_coast_source=projected_source,
            a11_rollout=a11_rollout,
            a17_rescore=a17_rescore,
        )
    except Exception:
        collection._close_runtime(runtime)
        raise
    return runtime, source, *lineage, a11_rollout, a17_rescore


def _open_relaxed_tether_contract(args: argparse.Namespace):
    a18_rollout = _verify_a18_rollout(args.a18_rollout)
    opened = _open_slew_limited_tether_contract(args)
    checkpoint, manifest, store, slew_source, *lineage = opened
    try:
        source = _relaxed_tether_source_manifest(
            args,
            slew_source=slew_source,
            a18_rollout=a18_rollout,
        )
    except Exception:
        store.close()
        raise
    return checkpoint, manifest, store, source, *lineage, a18_rollout


def _build_relaxed_tether_runtime(args: argparse.Namespace):
    a18_rollout = _verify_a18_rollout(args.a18_rollout)
    built = _build_slew_limited_tether_runtime(args)
    runtime, slew_source, *lineage = built
    try:
        source = _relaxed_tether_source_manifest(
            args,
            slew_source=slew_source,
            a18_rollout=a18_rollout,
        )
    except Exception:
        collection._close_runtime(runtime)
        raise
    return runtime, source, *lineage, a18_rollout


def _open_buffered_relaxed_tether_replay_contract(args: argparse.Namespace):
    a22_rescore = _verify_a22_rescore(args.a22_rescore)
    opened = _open_relaxed_tether_contract(args)
    checkpoint, manifest, store, relaxed_source, *lineage = opened
    try:
        source = _buffered_relaxed_tether_replay_source_manifest(
            args,
            relaxed_source=relaxed_source,
            a22_rescore=a22_rescore,
        )
    except Exception:
        store.close()
        raise
    return checkpoint, manifest, store, source, *lineage, a22_rescore


def _build_buffered_relaxed_tether_replay_runtime(args: argparse.Namespace):
    a22_rescore = _verify_a22_rescore(args.a22_rescore)
    built = _build_relaxed_tether_runtime(args)
    runtime, relaxed_source, *lineage = built
    try:
        source = _buffered_relaxed_tether_replay_source_manifest(
            args,
            relaxed_source=relaxed_source,
            a22_rescore=a22_rescore,
        )
    except Exception:
        collection._close_runtime(runtime)
        raise
    return runtime, source, *lineage, a22_rescore


def _open_buffered_relaxed_tether_e14_contract(args: argparse.Namespace):
    a22_r1_rollout = _verify_a22_r1_rollout(args.a22_r1_rollout)
    a14_rollout = _verify_a14_rollout(args.a14_rollout)
    opened = _open_buffered_relaxed_tether_replay_contract(args)
    checkpoint, manifest, store, replay_source, *lineage = opened
    try:
        source = _buffered_relaxed_tether_e14_source_manifest(
            args,
            replay_source=replay_source,
            a22_r1_rollout=a22_r1_rollout,
            a14_rollout=a14_rollout,
        )
    except Exception:
        store.close()
        raise
    return (
        checkpoint,
        manifest,
        store,
        source,
        *lineage,
        a22_r1_rollout,
        a14_rollout,
    )


def _build_buffered_relaxed_tether_e14_runtime(args: argparse.Namespace):
    a22_r1_rollout = _verify_a22_r1_rollout(args.a22_r1_rollout)
    a14_rollout = _verify_a14_rollout(args.a14_rollout)
    built = _build_buffered_relaxed_tether_replay_runtime(args)
    runtime, replay_source, *lineage = built
    try:
        source = _buffered_relaxed_tether_e14_source_manifest(
            args,
            replay_source=replay_source,
            a22_r1_rollout=a22_r1_rollout,
            a14_rollout=a14_rollout,
        )
    except Exception:
        collection._close_runtime(runtime)
        raise
    return runtime, source, *lineage, a22_r1_rollout, a14_rollout


def _open_frozen_offset_contract(args: argparse.Namespace):
    a11_rollout = _verify_a11_rollout(args.a11_rollout)
    (
        checkpoint,
        manifest,
        store,
        projected_coast_source,
        a5_rescore,
        a6_rollout,
        a7_rollout,
        a8_rollout,
        a9_rescore,
        a10_rollout,
    ) = _open_projected_coast_contract(args)
    try:
        source = _frozen_offset_source_manifest(
            args,
            projected_coast_source=projected_coast_source,
            a11_rollout=a11_rollout,
        )
    except Exception:
        store.close()
        raise
    return (
        checkpoint,
        manifest,
        store,
        source,
        a5_rescore,
        a6_rollout,
        a7_rollout,
        a8_rollout,
        a9_rescore,
        a10_rollout,
        a11_rollout,
    )


def _build_frozen_offset_runtime(args: argparse.Namespace):
    a11_rollout = _verify_a11_rollout(args.a11_rollout)
    (
        runtime,
        projected_coast_source,
        a5_rescore,
        a6_rollout,
        a7_rollout,
        a8_rollout,
        a9_rescore,
        a10_rollout,
    ) = _build_projected_coast_runtime(args)
    try:
        source = _frozen_offset_source_manifest(
            args,
            projected_coast_source=projected_coast_source,
            a11_rollout=a11_rollout,
        )
    except Exception:
        collection._close_runtime(runtime)
        raise
    return (
        runtime,
        source,
        a5_rescore,
        a6_rollout,
        a7_rollout,
        a8_rollout,
        a9_rescore,
        a10_rollout,
        a11_rollout,
    )


def _open_buffered_offset_contract(args: argparse.Namespace):
    a12_rollout = _verify_a12_rollout(args.a12_rollout)
    a12_buffer_rescore = _verify_a12_buffer_rescore(args.a12_buffer_rescore)
    opened = _open_frozen_offset_contract(args)
    checkpoint, manifest, store, frozen_source, *lineage = opened
    try:
        source = _buffered_offset_source_manifest(
            args,
            frozen_offset_source=frozen_source,
            a12_rollout=a12_rollout,
            a12_buffer_rescore=a12_buffer_rescore,
        )
        if getattr(args, "shard_native_reference_audit", None) is not None:
            source = _buffered_offset_source_with_truth_contract(source, args)
    except Exception:
        store.close()
        raise
    return (
        checkpoint,
        manifest,
        store,
        source,
        *lineage,
        a12_rollout,
        a12_buffer_rescore,
    )


def _build_buffered_offset_runtime(args: argparse.Namespace):
    a12_rollout = _verify_a12_rollout(args.a12_rollout)
    a12_buffer_rescore = _verify_a12_buffer_rescore(args.a12_buffer_rescore)
    built = _build_frozen_offset_runtime(args)
    runtime, frozen_source, *lineage = built
    try:
        source = _buffered_offset_source_manifest(
            args,
            frozen_offset_source=frozen_source,
            a12_rollout=a12_rollout,
            a12_buffer_rescore=a12_buffer_rescore,
        )
        if getattr(args, "shard_native_reference_audit", None) is not None:
            source = _buffered_offset_source_with_truth_contract(source, args)
    except Exception:
        collection._close_runtime(runtime)
        raise
    return runtime, source, *lineage, a12_rollout, a12_buffer_rescore


def _open_prospective_buffered_offset_contract(args: argparse.Namespace):
    a13_rollout = _verify_a13_rollout(args.a13_rollout)
    opened = _open_buffered_offset_contract(args)
    checkpoint, manifest, store, buffered_source, *lineage = opened
    try:
        source = _prospective_buffered_offset_source_manifest(
            args,
            buffered_offset_source=buffered_source,
            a13_rollout=a13_rollout,
        )
    except Exception:
        store.close()
        raise
    return checkpoint, manifest, store, source, *lineage, a13_rollout


def _build_prospective_buffered_offset_runtime(args: argparse.Namespace):
    a13_rollout = _verify_a13_rollout(args.a13_rollout)
    built = _build_buffered_offset_runtime(args)
    runtime, buffered_source, *lineage = built
    try:
        source = _prospective_buffered_offset_source_manifest(
            args,
            buffered_offset_source=buffered_source,
            a13_rollout=a13_rollout,
        )
    except Exception:
        collection._close_runtime(runtime)
        raise
    return runtime, source, *lineage, a13_rollout


def _open_persistence_gain_contract(args: argparse.Namespace):
    a13_rollout = _verify_a13_rollout(args.a13_rollout)
    a14_rollout = _verify_a14_rollout(args.a14_rollout)
    opened = _open_buffered_offset_contract(args)
    checkpoint, manifest, store, buffered_source, *lineage = opened
    try:
        source = _persistence_gain_source_manifest(
            args,
            buffered_offset_source=buffered_source,
            a13_rollout=a13_rollout,
            a14_rollout=a14_rollout,
        )
    except Exception:
        store.close()
        raise
    return (
        checkpoint,
        manifest,
        store,
        source,
        *lineage,
        a13_rollout,
        a14_rollout,
    )


def _build_persistence_gain_runtime(args: argparse.Namespace):
    a13_rollout = _verify_a13_rollout(args.a13_rollout)
    a14_rollout = _verify_a14_rollout(args.a14_rollout)
    built = _build_buffered_offset_runtime(args)
    runtime, buffered_source, *lineage = built
    try:
        source = _persistence_gain_source_manifest(
            args,
            buffered_offset_source=buffered_source,
            a13_rollout=a13_rollout,
            a14_rollout=a14_rollout,
        )
    except Exception:
        collection._close_runtime(runtime)
        raise
    return runtime, source, *lineage, a13_rollout, a14_rollout


def _open_terminal_ramp_contract(args: argparse.Namespace):
    a15_rollout = _verify_a15_rollout(args.a15_rollout)
    opened = _open_persistence_gain_contract(args)
    checkpoint, manifest, store, persistence_source, *lineage = opened
    try:
        source = _terminal_ramp_source_manifest(
            args,
            persistence_source=persistence_source,
            a15_rollout=a15_rollout,
        )
    except Exception:
        store.close()
        raise
    return checkpoint, manifest, store, source, *lineage, a15_rollout


def _build_terminal_ramp_runtime(args: argparse.Namespace):
    a15_rollout = _verify_a15_rollout(args.a15_rollout)
    built = _build_persistence_gain_runtime(args)
    runtime, persistence_source, *lineage = built
    try:
        source = _terminal_ramp_source_manifest(
            args,
            persistence_source=persistence_source,
            a15_rollout=a15_rollout,
        )
    except Exception:
        collection._close_runtime(runtime)
        raise
    return runtime, source, *lineage, a15_rollout


def _open_fixed_late_ramp_contract(args: argparse.Namespace):
    a13_rollout = _verify_a13_rollout(args.a13_rollout)
    a16_rollout = _verify_a16_rollout(args.a16_rollout)
    opened = _open_buffered_offset_contract(args)
    checkpoint, manifest, store, buffered_source, *lineage = opened
    try:
        source = _fixed_late_ramp_source_manifest(
            args,
            buffered_source=buffered_source,
            a13_rollout=a13_rollout,
            a16_rollout=a16_rollout,
        )
    except Exception:
        store.close()
        raise
    return (
        checkpoint,
        manifest,
        store,
        source,
        *lineage,
        a13_rollout,
        a16_rollout,
    )


def _build_fixed_late_ramp_runtime(args: argparse.Namespace):
    a13_rollout = _verify_a13_rollout(args.a13_rollout)
    a16_rollout = _verify_a16_rollout(args.a16_rollout)
    built = _build_buffered_offset_runtime(args)
    runtime, buffered_source, *lineage = built
    try:
        source = _fixed_late_ramp_source_manifest(
            args,
            buffered_source=buffered_source,
            a13_rollout=a13_rollout,
            a16_rollout=a16_rollout,
        )
    except Exception:
        collection._close_runtime(runtime)
        raise
    return runtime, source, *lineage, a13_rollout, a16_rollout


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    _, manifest, store, source, _ = _open_contract(args)
    try:
        splits = {
            case_id: collection.family_case_provenance(manifest, case_id)["split"]
            for case_id in CALIBRATION_CASE_IDS
        }
        payload = with_payload_sha256(
            {
                "schema": PREFLIGHT_SCHEMA,
                "working_id": WORKING_ID,
                "status": "passed",
                "source_manifest": source,
                "case_inventory_present": all(
                    case_id in store.keys for case_id in CALIBRATION_CASE_IDS
                ),
                "all_cases_open_validation": all(
                    value == "validation" for value in splits.values()
                ),
                "checkpoint_model_built": False,
                "reference_arrays_loaded": False,
                "evaluation_cases_opened": False,
            }
        )
    finally:
        store.close()
    if (
        not payload["case_inventory_present"]
        or not payload["all_cases_open_validation"]
    ):
        raise ValueError("response-filtered calibration population is unavailable")
    atomic_write_json(args.output, payload)
    return payload


def _verify_preflight(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    if payload.get("schema") != PREFLIGHT_SCHEMA or payload.get("status") != "passed":
        raise ValueError("response-filtered preflight did not pass")
    source = payload.get("source_manifest", {})
    if source.get("source_sha256") != _source_hashes():
        raise ValueError("response-filtered source differs from the preflight")
    if source.get("p5_calibration_sha256") != sha256_file(args.p5_calibration):
        raise ValueError("P5 artifact differs from the preflight")
    return payload


def run_evaluation_preflight(args: argparse.Namespace) -> dict[str, Any]:
    _, manifest, store, source, _, rescore = _open_evaluation_contract(args)
    try:
        splits = {
            case_id: collection.family_case_provenance(manifest, case_id)["split"]
            for case_id in EVALUATION_CASE_IDS
        }
        payload = with_payload_sha256(
            {
                "schema": EVALUATION_PREFLIGHT_SCHEMA,
                "working_id": EVALUATION_WORKING_ID,
                "status": "passed",
                "population_status": "adaptive_open_validation_evaluation",
                "source_manifest": source,
                "qualified_rescore_payload_sha256": rescore["payload_sha256"],
                "case_inventory_present": all(
                    case_id in store.keys for case_id in EVALUATION_CASE_IDS
                ),
                "all_cases_open_validation": all(
                    value == "validation" for value in splits.values()
                ),
                "calibration_and_evaluation_disjoint": set(
                    CALIBRATION_CASE_IDS
                ).isdisjoint(EVALUATION_CASE_IDS),
                "checkpoint_model_built": False,
                "reference_arrays_loaded": False,
                "strength_ood_test_or_recurrence_opened": False,
            }
        )
    finally:
        store.close()
    required = (
        "case_inventory_present",
        "all_cases_open_validation",
        "calibration_and_evaluation_disjoint",
    )
    if not all(payload[key] for key in required):
        raise ValueError("response-filtered evaluation population is unavailable")
    atomic_write_json(args.output, payload)
    return payload


def _verify_evaluation_preflight(
    path: Path, args: argparse.Namespace
) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    if (
        payload.get("schema") != EVALUATION_PREFLIGHT_SCHEMA
        or payload.get("status") != "passed"
    ):
        raise ValueError("response-filtered evaluation preflight did not pass")
    source = payload.get("source_manifest", {})
    if source.get("source_sha256") != _source_hashes():
        raise ValueError("response-filtered source differs from evaluation preflight")
    if source.get("p5_calibration_sha256") != sha256_file(args.p5_calibration):
        raise ValueError("P5 artifact differs from evaluation preflight")
    if source.get("qualified_rescore_sha256") != sha256_file(args.rescore):
        raise ValueError("P6 re-score differs from evaluation preflight")
    return payload


def run_rollout_preflight(args: argparse.Namespace) -> dict[str, Any]:
    _, manifest, store, source, _, _, evaluation = _open_rollout_contract(args)
    try:
        splits = {
            case_id: collection.family_case_provenance(manifest, case_id)["split"]
            for case_id in EVALUATION_CASE_IDS
        }
        payload = with_payload_sha256(
            {
                "schema": ROLLOUT_PREFLIGHT_SCHEMA,
                "working_id": ROLLOUT_WORKING_ID,
                "status": "passed",
                "population_status": "adaptive_open_validation_recurrent_pilot",
                "source_manifest": source,
                "qualified_evaluation_payload_sha256": evaluation["payload_sha256"],
                "case_inventory_present": all(
                    case_id in store.keys for case_id in EVALUATION_CASE_IDS
                ),
                "all_cases_open_validation": all(
                    value == "validation" for value in splits.values()
                ),
                "checkpoint_model_built": False,
                "reference_arrays_loaded": False,
                "recurrence_executed": False,
                "strength_ood_or_test_opened": False,
            }
        )
    finally:
        store.close()
    if (
        not payload["case_inventory_present"]
        or not payload["all_cases_open_validation"]
    ):
        raise ValueError("response-filtered rollout population is unavailable")
    atomic_write_json(args.output, payload)
    return payload


def _verify_rollout_preflight(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    if (
        payload.get("schema") != ROLLOUT_PREFLIGHT_SCHEMA
        or payload.get("status") != "passed"
        or payload.get("recurrence_executed") is not False
    ):
        raise ValueError("response-filtered rollout preflight did not pass")
    source = payload.get("source_manifest", {})
    if source.get("source_sha256") != _source_hashes():
        raise ValueError("response-filtered source differs from rollout preflight")
    if source.get("qualified_evaluation_sha256") != sha256_file(args.evaluation):
        raise ValueError("P6 A3 evaluation differs from rollout preflight")
    return payload


def run_shadow_preflight(args: argparse.Namespace) -> dict[str, Any]:
    _, manifest, store, source, _, _, evaluation, a4_rollout = _open_shadow_contract(
        args
    )
    try:
        splits = {
            case_id: collection.family_case_provenance(manifest, case_id)["split"]
            for case_id in EVALUATION_CASE_IDS
        }
        payload = with_payload_sha256(
            {
                "schema": SHADOW_PREFLIGHT_SCHEMA,
                "working_id": SHADOW_WORKING_ID,
                "status": "passed",
                "population_status": "adaptive_open_validation_shadow_pilot",
                "source_manifest": source,
                "qualified_evaluation_payload_sha256": evaluation["payload_sha256"],
                "stopped_a4_payload_sha256": a4_rollout["payload_sha256"],
                "case_inventory_present": all(
                    case_id in store.keys for case_id in EVALUATION_CASE_IDS
                ),
                "all_cases_open_validation": all(
                    value == "validation" for value in splits.values()
                ),
                "checkpoint_model_built": False,
                "reference_arrays_loaded": False,
                "shadow_recurrence_executed": False,
                "strength_ood_or_test_opened": False,
            }
        )
    finally:
        store.close()
    if (
        not payload["case_inventory_present"]
        or not payload["all_cases_open_validation"]
    ):
        raise ValueError("raw-shadow rollout population is unavailable")
    atomic_write_json(args.output, payload)
    return payload


def _verify_shadow_preflight(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    if (
        payload.get("schema") != SHADOW_PREFLIGHT_SCHEMA
        or payload.get("status") != "passed"
        or payload.get("shadow_recurrence_executed") is not False
    ):
        raise ValueError("raw-shadow rollout preflight did not pass")
    source = payload.get("source_manifest", {})
    if source.get("source_sha256") != _source_hashes():
        raise ValueError("raw-shadow source differs from preflight")
    if source.get("qualified_evaluation_sha256") != sha256_file(args.evaluation):
        raise ValueError("P6 A3 evaluation differs from raw-shadow preflight")
    if source.get("stopped_a4_rollout_sha256") != sha256_file(args.a4_rollout):
        raise ValueError("P6 A4 rollout differs from raw-shadow preflight")
    return payload


def run_strength_ood_preflight(args: argparse.Namespace) -> dict[str, Any]:
    _, manifest, store, source, a5_rescore = _open_strength_ood_contract(args)
    try:
        provenance = {
            case_id: collection.family_case_provenance(manifest, case_id)
            for case_id in STRENGTH_OOD_CASE_IDS
        }
        exact_group = all(
            row["split"] == "test"
            and row["split_group_id"] == STRENGTH_OOD_GROUP_ID
            and math.isclose(
                float(row["parameters"]["vortex_epsilon"]),
                0.375,
                abs_tol=0.0,
            )
            for row in provenance.values()
        )
        payload = with_payload_sha256(
            {
                "schema": STRENGTH_OOD_PREFLIGHT_SCHEMA,
                "working_id": STRENGTH_OOD_WORKING_ID,
                "status": "passed",
                "population_status": "sealed_strength_ood_e13_named_confirmation",
                "source_manifest": source,
                "qualified_a5_rescore_payload_sha256": a5_rescore["payload_sha256"],
                "case_inventory_present": all(
                    case_id in store.keys for case_id in STRENGTH_OOD_CASE_IDS
                ),
                "case_inventory_exact": len(provenance) == len(STRENGTH_OOD_CASE_IDS),
                "all_cases_in_exact_named_group": exact_group,
                "checkpoint_model_built": False,
                "reference_arrays_loaded": False,
                "recurrence_executed": False,
                "opened_group": None,
                "still_sealed_groups": ["strength_ood_e12", "strength_ood_e14"],
                "animation_contract": STRENGTH_OOD_ANIMATION_CONTRACT,
            }
        )
    finally:
        store.close()
    if not (
        payload["case_inventory_present"]
        and payload["case_inventory_exact"]
        and payload["all_cases_in_exact_named_group"]
    ):
        raise ValueError("named strength-OOD group is unavailable or mismatched")
    atomic_write_json(args.output, payload)
    return payload


def _verify_strength_ood_preflight(
    path: Path, args: argparse.Namespace
) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    source = payload.get("source_manifest", {})
    if (
        payload.get("schema") != STRENGTH_OOD_PREFLIGHT_SCHEMA
        or payload.get("working_id") != STRENGTH_OOD_WORKING_ID
        or payload.get("status") != "passed"
        or payload.get("reference_arrays_loaded") is not False
        or payload.get("recurrence_executed") is not False
        or payload.get("opened_group") is not None
        or payload.get("all_cases_in_exact_named_group") is not True
        or source.get("source_sha256") != _source_hashes()
        or source.get("qualified_a5_rescore_sha256") != sha256_file(args.a5_rescore)
        or source.get("population", {}).get("split_group_id") != STRENGTH_OOD_GROUP_ID
        or source.get("population", {}).get("case_ids") != list(STRENGTH_OOD_CASE_IDS)
    ):
        raise ValueError("strength-OOD preflight does not match the frozen contract")
    return payload


def run_structural_gate_preflight(args: argparse.Namespace) -> dict[str, Any]:
    _, manifest, store, source, _, a6_rollout = _open_structural_gate_contract(args)
    try:
        provenance = {
            case_id: collection.family_case_provenance(manifest, case_id)
            for case_id in STRENGTH_OOD_CASE_IDS
        }
        payload = with_payload_sha256(
            {
                "schema": STRUCTURAL_GATE_PREFLIGHT_SCHEMA,
                "working_id": STRUCTURAL_GATE_WORKING_ID,
                "status": "passed",
                "population_status": "already_open_e13_retrospective_calibration",
                "source_manifest": source,
                "stopped_a6_payload_sha256": a6_rollout["payload_sha256"],
                "case_inventory_present": all(
                    case_id in store.keys for case_id in STRENGTH_OOD_CASE_IDS
                ),
                "case_inventory_exact": set(provenance) == set(STRENGTH_OOD_CASE_IDS),
                "all_cases_in_opened_e13_group": all(
                    row["split_group_id"] == STRENGTH_OOD_GROUP_ID
                    for row in provenance.values()
                ),
                "checkpoint_model_built": False,
                "reference_arrays_loaded": False,
                "recurrence_executed": False,
                "new_sealed_population_opened": False,
                "still_sealed_groups": ["strength_ood_e12", "strength_ood_e14"],
                "position_trust_threshold": FROZEN_POSITION_TRUST_THRESHOLD,
                "front_veto": "exact_integer_thickness_mismatch",
                "animation_contract": STRUCTURAL_GATE_ANIMATION_CONTRACT,
            }
        )
    finally:
        store.close()
    if not (
        payload["case_inventory_present"]
        and payload["case_inventory_exact"]
        and payload["all_cases_in_opened_e13_group"]
    ):
        raise ValueError("A7 retrospective e13 inventory is unavailable or mismatched")
    atomic_write_json(args.output, payload)
    return payload


def _verify_structural_gate_preflight(
    path: Path, args: argparse.Namespace
) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    source = payload.get("source_manifest", {})
    if (
        payload.get("schema") != STRUCTURAL_GATE_PREFLIGHT_SCHEMA
        or payload.get("working_id") != STRUCTURAL_GATE_WORKING_ID
        or payload.get("status") != "passed"
        or payload.get("reference_arrays_loaded") is not False
        or payload.get("recurrence_executed") is not False
        or payload.get("new_sealed_population_opened") is not False
        or payload.get("all_cases_in_opened_e13_group") is not True
        or payload.get("position_trust_threshold") != FROZEN_POSITION_TRUST_THRESHOLD
        or source.get("source_sha256") != _source_hashes()
        or source.get("stopped_a6_rollout_sha256") != sha256_file(args.a6_rollout)
        or source.get("population", {}).get("case_ids") != list(STRENGTH_OOD_CASE_IDS)
    ):
        raise ValueError("A7 structural-gate preflight differs from the frozen replay")
    return payload


def run_prospective_structural_preflight(
    args: argparse.Namespace,
) -> dict[str, Any]:
    _, manifest, store, source, _, _, a7_rollout = (
        _open_prospective_structural_contract(args)
    )
    try:
        provenance = {
            case_id: collection.family_case_provenance(manifest, case_id)
            for case_id in PROSPECTIVE_STRUCTURAL_CASE_IDS
        }
        payload = with_payload_sha256(
            {
                "schema": PROSPECTIVE_STRUCTURAL_PREFLIGHT_SCHEMA,
                "working_id": PROSPECTIVE_STRUCTURAL_WORKING_ID,
                "status": "passed",
                "population_status": "prospectively_named_e12_confirmation",
                "source_manifest": source,
                "qualified_a7_payload_sha256": a7_rollout["payload_sha256"],
                "case_inventory_present": all(
                    case_id in store.keys for case_id in PROSPECTIVE_STRUCTURAL_CASE_IDS
                ),
                "case_inventory_exact": set(provenance)
                == set(PROSPECTIVE_STRUCTURAL_CASE_IDS),
                "all_cases_in_named_e12_group": all(
                    row["split_group_id"] == PROSPECTIVE_STRUCTURAL_GROUP_ID
                    for row in provenance.values()
                ),
                "checkpoint_model_built": False,
                "reference_arrays_loaded": False,
                "recurrence_executed": False,
                "new_sealed_population_opened": PROSPECTIVE_STRUCTURAL_GROUP_ID,
                "still_sealed_groups": ["strength_ood_e14"],
                "position_trust_threshold": FROZEN_POSITION_TRUST_THRESHOLD,
                "front_veto": "exact_integer_thickness_mismatch",
                "coefficient_or_threshold_refit": False,
                "animation_contract": PROSPECTIVE_STRUCTURAL_ANIMATION_CONTRACT,
            }
        )
    finally:
        store.close()
    if not (
        payload["case_inventory_present"]
        and payload["case_inventory_exact"]
        and payload["all_cases_in_named_e12_group"]
    ):
        raise ValueError("A8 prospective e12 inventory is unavailable or mismatched")
    atomic_write_json(args.output, payload)
    return payload


def _verify_prospective_structural_preflight(
    path: Path, args: argparse.Namespace
) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    source = payload.get("source_manifest", {})
    if (
        payload.get("schema") != PROSPECTIVE_STRUCTURAL_PREFLIGHT_SCHEMA
        or payload.get("working_id") != PROSPECTIVE_STRUCTURAL_WORKING_ID
        or payload.get("status") != "passed"
        or payload.get("reference_arrays_loaded") is not False
        or payload.get("recurrence_executed") is not False
        or payload.get("new_sealed_population_opened")
        != PROSPECTIVE_STRUCTURAL_GROUP_ID
        or payload.get("still_sealed_groups") != ["strength_ood_e14"]
        or payload.get("all_cases_in_named_e12_group") is not True
        or payload.get("coefficient_or_threshold_refit") is not False
        or payload.get("position_trust_threshold") != FROZEN_POSITION_TRUST_THRESHOLD
        or source.get("source_sha256") != _source_hashes()
        or source.get("qualified_a7_rollout_sha256") != sha256_file(args.a7_rollout)
        or source.get("population", {}).get("case_ids")
        != list(PROSPECTIVE_STRUCTURAL_CASE_IDS)
        or source.get("population", {}).get("split_group_id")
        != PROSPECTIVE_STRUCTURAL_GROUP_ID
    ):
        raise ValueError("A8 prospective structural preflight differs from contract")
    return payload


def run_shadow_candidate_preflight(args: argparse.Namespace) -> dict[str, Any]:
    _, manifest, store, source, _, _, _, a8_rollout = _open_shadow_candidate_contract(
        args
    )
    try:
        provenance = {
            case_id: collection.family_case_provenance(manifest, case_id)
            for case_id in PROSPECTIVE_STRUCTURAL_CASE_IDS
        }
        payload = with_payload_sha256(
            {
                "schema": SHADOW_CANDIDATE_PREFLIGHT_SCHEMA,
                "working_id": SHADOW_CANDIDATE_WORKING_ID,
                "status": "passed",
                "population_status": "already_open_e12_mechanism_diagnostic",
                "source_manifest": source,
                "qualified_a8_payload_sha256": a8_rollout["payload_sha256"],
                "case_inventory_present": all(
                    case_id in store.keys for case_id in PROSPECTIVE_STRUCTURAL_CASE_IDS
                ),
                "case_inventory_exact": set(provenance)
                == set(PROSPECTIVE_STRUCTURAL_CASE_IDS),
                "all_cases_in_opened_e12_group": all(
                    row["split_group_id"] == PROSPECTIVE_STRUCTURAL_GROUP_ID
                    for row in provenance.values()
                ),
                "checkpoint_model_built": False,
                "reference_arrays_loaded": False,
                "recurrence_executed": False,
                "new_sealed_population_opened": False,
                "still_sealed_groups": ["strength_ood_e14"],
                "feature_or_threshold_selection": False,
                "coefficient_or_threshold_refit": False,
            }
        )
    finally:
        store.close()
    if not (
        payload["case_inventory_present"]
        and payload["case_inventory_exact"]
        and payload["all_cases_in_opened_e12_group"]
    ):
        raise ValueError(
            "A9 open-e12 diagnostic inventory is unavailable or mismatched"
        )
    atomic_write_json(args.output, payload)
    return payload


def _verify_shadow_candidate_preflight(
    path: Path, args: argparse.Namespace
) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    source = payload.get("source_manifest", {})
    if (
        payload.get("schema") != SHADOW_CANDIDATE_PREFLIGHT_SCHEMA
        or payload.get("working_id") != SHADOW_CANDIDATE_WORKING_ID
        or payload.get("status") != "passed"
        or payload.get("reference_arrays_loaded") is not False
        or payload.get("recurrence_executed") is not False
        or payload.get("new_sealed_population_opened") is not False
        or payload.get("still_sealed_groups") != ["strength_ood_e14"]
        or payload.get("all_cases_in_opened_e12_group") is not True
        or payload.get("feature_or_threshold_selection") is not False
        or payload.get("coefficient_or_threshold_refit") is not False
        or source.get("source_sha256") != _source_hashes()
        or source.get("qualified_a8_rollout_sha256") != sha256_file(args.a8_rollout)
        or source.get("population", {}).get("case_ids")
        != list(PROSPECTIVE_STRUCTURAL_CASE_IDS)
        or source.get("protocol", {}).get("features") != list(A9_FEATURE_NAMES)
        or source.get("protocol", {}).get("targets") != list(A9_TARGET_NAMES)
    ):
        raise ValueError("A9 shadow-candidate preflight differs from contract")
    return payload


def run_warm_start_preflight(args: argparse.Namespace) -> dict[str, Any]:
    _, manifest, store, source, _, _, _, a8_rollout, a9_rescore = (
        _open_warm_start_contract(args)
    )
    try:
        provenance = {
            case_id: collection.family_case_provenance(manifest, case_id)
            for case_id in PROSPECTIVE_STRUCTURAL_CASE_IDS
        }
        payload = with_payload_sha256(
            {
                "schema": WARM_START_PREFLIGHT_SCHEMA,
                "working_id": WARM_START_WORKING_ID,
                "status": "passed",
                "population_status": "already_open_e12_recurrent_calibration",
                "source_manifest": source,
                "qualified_a8_payload_sha256": a8_rollout["payload_sha256"],
                "completed_a9_rescore_payload_sha256": a9_rescore["payload_sha256"],
                "case_inventory_present": all(
                    case_id in store.keys for case_id in PROSPECTIVE_STRUCTURAL_CASE_IDS
                ),
                "case_inventory_exact": set(provenance)
                == set(PROSPECTIVE_STRUCTURAL_CASE_IDS),
                "all_cases_in_opened_e12_group": all(
                    row["split_group_id"] == PROSPECTIVE_STRUCTURAL_GROUP_ID
                    for row in provenance.values()
                ),
                "checkpoint_model_built": False,
                "reference_arrays_loaded": False,
                "recurrence_executed": False,
                "new_sealed_population_opened": False,
                "still_sealed_groups": ["strength_ood_e14"],
                "warm_start_blocks": WARM_START_BLOCKS,
                "coast_reset_to_shadow": False,
                "coefficient_or_threshold_refit": False,
                "animation_contract": WARM_START_ANIMATION_CONTRACT,
            }
        )
    finally:
        store.close()
    if not (
        payload["case_inventory_present"]
        and payload["case_inventory_exact"]
        and payload["all_cases_in_opened_e12_group"]
    ):
        raise ValueError("A10 open-e12 calibration inventory is unavailable")
    atomic_write_json(args.output, payload)
    return payload


def _verify_warm_start_preflight(
    path: Path, args: argparse.Namespace
) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    source = payload.get("source_manifest", {})
    if (
        payload.get("schema") != WARM_START_PREFLIGHT_SCHEMA
        or payload.get("working_id") != WARM_START_WORKING_ID
        or payload.get("status") != "passed"
        or payload.get("reference_arrays_loaded") is not False
        or payload.get("recurrence_executed") is not False
        or payload.get("new_sealed_population_opened") is not False
        or payload.get("still_sealed_groups") != ["strength_ood_e14"]
        or payload.get("all_cases_in_opened_e12_group") is not True
        or payload.get("warm_start_blocks") != WARM_START_BLOCKS
        or payload.get("coast_reset_to_shadow") is not False
        or payload.get("coefficient_or_threshold_refit") is not False
        or source.get("source_sha256") != _source_hashes()
        or source.get("qualified_a8_rollout_sha256") != sha256_file(args.a8_rollout)
        or source.get("completed_a9_rescore_sha256") != sha256_file(args.a9_rescore)
        or source.get("population", {}).get("case_ids")
        != list(PROSPECTIVE_STRUCTURAL_CASE_IDS)
    ):
        raise ValueError("A10 warm-start preflight differs from contract")
    return payload


def run_projected_coast_preflight(args: argparse.Namespace) -> dict[str, Any]:
    _, manifest, store, source, _, _, _, _, _, a10_rollout = (
        _open_projected_coast_contract(args)
    )
    try:
        provenance = {
            case_id: collection.family_case_provenance(manifest, case_id)
            for case_id in PROSPECTIVE_STRUCTURAL_CASE_IDS
        }
        payload = with_payload_sha256(
            {
                "schema": PROJECTED_COAST_PREFLIGHT_SCHEMA,
                "working_id": PROJECTED_COAST_WORKING_ID,
                "status": "passed",
                "population_status": "already_open_e12_recurrent_calibration",
                "source_manifest": source,
                "stopped_a10_payload_sha256": a10_rollout["payload_sha256"],
                "case_inventory_present": all(
                    case_id in store.keys for case_id in PROSPECTIVE_STRUCTURAL_CASE_IDS
                ),
                "case_inventory_exact": set(provenance)
                == set(PROSPECTIVE_STRUCTURAL_CASE_IDS),
                "all_cases_in_opened_e12_group": all(
                    row["split_group_id"] == PROSPECTIVE_STRUCTURAL_GROUP_ID
                    for row in provenance.values()
                ),
                "checkpoint_model_built": False,
                "reference_arrays_loaded": False,
                "recurrence_executed": False,
                "new_sealed_population_opened": False,
                "still_sealed_groups": ["strength_ood_e14"],
                "warm_start_blocks": WARM_START_BLOCKS,
                "projected_coast": True,
                "coast_projection_active_cells": [
                    list(cell) for cell in FROZEN_ACTIVE_CELLS
                ],
                "coefficient_or_threshold_refit": False,
                "animation_contract": PROJECTED_COAST_ANIMATION_CONTRACT,
            }
        )
    finally:
        store.close()
    if not (
        payload["case_inventory_present"]
        and payload["case_inventory_exact"]
        and payload["all_cases_in_opened_e12_group"]
    ):
        raise ValueError("A11 open-e12 calibration inventory is unavailable")
    atomic_write_json(args.output, payload)
    return payload


def _verify_projected_coast_preflight(
    path: Path, args: argparse.Namespace
) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    source = payload.get("source_manifest", {})
    if (
        payload.get("schema") != PROJECTED_COAST_PREFLIGHT_SCHEMA
        or payload.get("working_id") != PROJECTED_COAST_WORKING_ID
        or payload.get("status") != "passed"
        or payload.get("reference_arrays_loaded") is not False
        or payload.get("recurrence_executed") is not False
        or payload.get("new_sealed_population_opened") is not False
        or payload.get("still_sealed_groups") != ["strength_ood_e14"]
        or payload.get("all_cases_in_opened_e12_group") is not True
        or payload.get("warm_start_blocks") != WARM_START_BLOCKS
        or payload.get("projected_coast") is not True
        or payload.get("coast_projection_active_cells")
        != [list(cell) for cell in FROZEN_ACTIVE_CELLS]
        or payload.get("coefficient_or_threshold_refit") is not False
        or source.get("source_sha256") != _source_hashes()
        or source.get("stopped_a10_rollout_sha256") != sha256_file(args.a10_rollout)
        or source.get("population", {}).get("case_ids")
        != list(PROSPECTIVE_STRUCTURAL_CASE_IDS)
    ):
        raise ValueError("A11 projected-coast preflight differs from contract")
    return payload


def run_slew_limited_tether_preflight(args: argparse.Namespace) -> dict[str, Any]:
    opened = _open_slew_limited_tether_contract(args)
    _, manifest, store, source, *_, a11_rollout, a17_rescore = opened
    try:
        provenance = {
            case_id: collection.family_case_provenance(manifest, case_id)
            for case_id in PROSPECTIVE_STRUCTURAL_CASE_IDS
        }
        payload = with_payload_sha256(
            {
                "schema": SLEW_LIMITED_TETHER_PREFLIGHT_SCHEMA,
                "working_id": SLEW_LIMITED_TETHER_WORKING_ID,
                "status": "passed",
                "population_status": (
                    "already_open_e12_recurrent_mechanism_calibration"
                ),
                "source_manifest": source,
                "stopped_a11_payload_sha256": a11_rollout["payload_sha256"],
                "qualified_a17_rescore_payload_sha256": a17_rescore[
                    "payload_sha256"
                ],
                "case_inventory_present": all(
                    case_id in store.keys
                    for case_id in PROSPECTIVE_STRUCTURAL_CASE_IDS
                ),
                "case_inventory_exact": set(provenance)
                == set(PROSPECTIVE_STRUCTURAL_CASE_IDS),
                "all_cases_in_opened_e12_group": all(
                    row["split_group_id"] == PROSPECTIVE_STRUCTURAL_GROUP_ID
                    for row in provenance.values()
                ),
                "checkpoint_model_built": False,
                "reference_arrays_loaded": False,
                "recurrence_executed": False,
                "new_population_opened": False,
                "warm_start_blocks": FROZEN_OFFSET_WARM_BLOCKS,
                "slew_limited_tether": True,
                "relative_displacement_change_limit": (
                    SLEW_LIMITED_TETHER_RELATIVE_CHANGE_LIMIT
                ),
                "single_unswept_limit": True,
                "truth_or_reference_used_at_inference": False,
                "animation_contract": SLEW_LIMITED_TETHER_ANIMATION_CONTRACT,
            }
        )
    finally:
        store.close()
    if not (
        payload["case_inventory_present"]
        and payload["case_inventory_exact"]
        and payload["all_cases_in_opened_e12_group"]
    ):
        raise ValueError("A18 open-e12 calibration inventory is unavailable")
    atomic_write_json(args.output, payload)
    return payload


def _verify_slew_limited_tether_preflight(
    path: Path, args: argparse.Namespace
) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    source = payload.get("source_manifest", {})
    if (
        payload.get("schema") != SLEW_LIMITED_TETHER_PREFLIGHT_SCHEMA
        or payload.get("working_id") != SLEW_LIMITED_TETHER_WORKING_ID
        or payload.get("status") != "passed"
        or payload.get("reference_arrays_loaded") is not False
        or payload.get("recurrence_executed") is not False
        or payload.get("new_population_opened") is not False
        or payload.get("warm_start_blocks") != FROZEN_OFFSET_WARM_BLOCKS
        or payload.get("slew_limited_tether") is not True
        or payload.get("relative_displacement_change_limit")
        != SLEW_LIMITED_TETHER_RELATIVE_CHANGE_LIMIT
        or payload.get("single_unswept_limit") is not True
        or payload.get("truth_or_reference_used_at_inference") is not False
        or source.get("source_sha256") != _source_hashes()
        or source.get("stopped_a11_rollout_sha256")
        != sha256_file(args.a11_rollout)
        or source.get("qualified_a17_rescore_sha256")
        != sha256_file(args.a17_rescore)
    ):
        raise ValueError("A18 slew-limited-tether preflight differs from contract")
    return payload


def run_relaxed_tether_preflight(args: argparse.Namespace) -> dict[str, Any]:
    opened = _open_relaxed_tether_contract(args)
    _, manifest, store, source, *_, a18_rollout = opened
    try:
        provenance = {
            case_id: collection.family_case_provenance(manifest, case_id)
            for case_id in PROSPECTIVE_STRUCTURAL_CASE_IDS
        }
        payload = with_payload_sha256(
            {
                "schema": RELAXED_TETHER_PREFLIGHT_SCHEMA,
                "working_id": RELAXED_TETHER_WORKING_ID,
                "status": "passed",
                "population_status": (
                    "already_open_e12_recurrent_mechanism_calibration"
                ),
                "source_manifest": source,
                "stopped_a18_payload_sha256": a18_rollout["payload_sha256"],
                "case_inventory_present": all(
                    case_id in store.keys
                    for case_id in PROSPECTIVE_STRUCTURAL_CASE_IDS
                ),
                "case_inventory_exact": set(provenance)
                == set(PROSPECTIVE_STRUCTURAL_CASE_IDS),
                "all_cases_in_opened_e12_group": all(
                    row["split_group_id"] == PROSPECTIVE_STRUCTURAL_GROUP_ID
                    for row in provenance.values()
                ),
                "checkpoint_model_built": False,
                "reference_arrays_loaded": False,
                "recurrence_executed": False,
                "new_population_opened": False,
                "warm_start_blocks": FROZEN_OFFSET_WARM_BLOCKS,
                "relaxed_tether": True,
                "relaxation_rate": RELAXED_TETHER_RATE,
                "single_unswept_rate": True,
                "truth_or_reference_used_at_inference": False,
                "animation_contract": RELAXED_TETHER_ANIMATION_CONTRACT,
            }
        )
    finally:
        store.close()
    if not (
        payload["case_inventory_present"]
        and payload["case_inventory_exact"]
        and payload["all_cases_in_opened_e12_group"]
    ):
        raise ValueError("A19 open-e12 calibration inventory is unavailable")
    atomic_write_json(args.output, payload)
    return payload


def _verify_relaxed_tether_preflight(
    path: Path, args: argparse.Namespace
) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    source = payload.get("source_manifest", {})
    if (
        payload.get("schema") != RELAXED_TETHER_PREFLIGHT_SCHEMA
        or payload.get("working_id") != RELAXED_TETHER_WORKING_ID
        or payload.get("status") != "passed"
        or payload.get("reference_arrays_loaded") is not False
        or payload.get("recurrence_executed") is not False
        or payload.get("new_population_opened") is not False
        or payload.get("warm_start_blocks") != FROZEN_OFFSET_WARM_BLOCKS
        or payload.get("relaxed_tether") is not True
        or payload.get("relaxation_rate") != RELAXED_TETHER_RATE
        or payload.get("single_unswept_rate") is not True
        or payload.get("truth_or_reference_used_at_inference") is not False
        or source.get("source_sha256") != _source_hashes()
        or source.get("stopped_a18_rollout_sha256")
        != sha256_file(args.a18_rollout)
    ):
        raise ValueError("A19 relaxed-tether preflight differs from contract")
    return payload


def run_buffered_relaxed_tether_replay_preflight(
    args: argparse.Namespace,
) -> dict[str, Any]:
    opened = _open_buffered_relaxed_tether_replay_contract(args)
    _, manifest, store, source, *_, a22_rescore = opened
    try:
        provenance = {
            case_id: collection.family_case_provenance(manifest, case_id)
            for case_id in PROSPECTIVE_STRUCTURAL_CASE_IDS
        }
        payload = with_payload_sha256(
            {
                "schema": BUFFERED_RELAXED_TETHER_REPLAY_PREFLIGHT_SCHEMA,
                "working_id": BUFFERED_RELAXED_TETHER_REPLAY_WORKING_ID,
                "status": "passed",
                "population_status": (
                    "already_open_e12_buffered_recurrent_replay"
                ),
                "source_manifest": source,
                "qualified_a22_rescore_payload_sha256": a22_rescore[
                    "payload_sha256"
                ],
                "case_inventory_present": all(
                    case_id in store.keys
                    for case_id in PROSPECTIVE_STRUCTURAL_CASE_IDS
                ),
                "case_inventory_exact": set(provenance)
                == set(PROSPECTIVE_STRUCTURAL_CASE_IDS),
                "all_cases_in_opened_e12_group": all(
                    row["split_group_id"] == PROSPECTIVE_STRUCTURAL_GROUP_ID
                    for row in provenance.values()
                ),
                "checkpoint_model_built": False,
                "reference_arrays_loaded": False,
                "recurrence_executed": False,
                "new_population_opened": False,
                "warm_start_blocks": FROZEN_OFFSET_WARM_BLOCKS,
                "buffered_relaxed_tether": True,
                "position_trust_threshold": FROZEN_POSITION_TRUST_THRESHOLD,
                "position_buffer_threshold": FROZEN_POSITION_BUFFER_THRESHOLD,
                "relaxation_rate": RELAXED_TETHER_RATE,
                "truth_or_reference_used_at_inference": False,
                "registered_scored_calls": 580,
                "animation_contract": (
                    BUFFERED_RELAXED_TETHER_REPLAY_ANIMATION_CONTRACT
                ),
            }
        )
    finally:
        store.close()
    if not (
        payload["case_inventory_present"]
        and payload["case_inventory_exact"]
        and payload["all_cases_in_opened_e12_group"]
    ):
        raise ValueError("A22-R1 open-E12 replay inventory is unavailable")
    atomic_write_json(args.output, payload)
    return payload


def _verify_buffered_relaxed_tether_replay_preflight(
    path: Path, args: argparse.Namespace
) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    source = payload.get("source_manifest", {})
    if (
        payload.get("schema") != BUFFERED_RELAXED_TETHER_REPLAY_PREFLIGHT_SCHEMA
        or payload.get("working_id")
        != BUFFERED_RELAXED_TETHER_REPLAY_WORKING_ID
        or payload.get("status") != "passed"
        or payload.get("reference_arrays_loaded") is not False
        or payload.get("recurrence_executed") is not False
        or payload.get("new_population_opened") is not False
        or payload.get("warm_start_blocks") != FROZEN_OFFSET_WARM_BLOCKS
        or payload.get("buffered_relaxed_tether") is not True
        or payload.get("position_trust_threshold")
        != FROZEN_POSITION_TRUST_THRESHOLD
        or payload.get("position_buffer_threshold")
        != FROZEN_POSITION_BUFFER_THRESHOLD
        or payload.get("relaxation_rate") != RELAXED_TETHER_RATE
        or payload.get("truth_or_reference_used_at_inference") is not False
        or payload.get("registered_scored_calls") != 580
        or source.get("source_sha256") != _source_hashes()
        or source.get("qualified_a22_rescore_sha256")
        != sha256_file(args.a22_rescore)
    ):
        raise ValueError("A22-R1 preflight differs from the frozen contract")
    return payload


def run_buffered_relaxed_tether_e14_preflight(
    args: argparse.Namespace,
) -> dict[str, Any]:
    opened = _open_buffered_relaxed_tether_e14_contract(args)
    _, manifest, store, source, *_, a22_r1_rollout, a14_rollout = opened
    try:
        provenance = {
            case_id: collection.family_case_provenance(manifest, case_id)
            for case_id in PROSPECTIVE_BUFFERED_OFFSET_CASE_IDS
        }
        payload = with_payload_sha256(
            {
                "schema": BUFFERED_RELAXED_TETHER_E14_PREFLIGHT_SCHEMA,
                "working_id": BUFFERED_RELAXED_TETHER_E14_WORKING_ID,
                "status": "passed",
                "population_status": "already_open_e14_frozen_protocol_transfer",
                "source_manifest": source,
                "qualified_a22_r1_payload_sha256": a22_r1_rollout[
                    "payload_sha256"
                ],
                "stopped_a14_comparator_payload_sha256": a14_rollout[
                    "payload_sha256"
                ],
                "case_inventory_present": all(
                    case_id in store.keys
                    for case_id in PROSPECTIVE_BUFFERED_OFFSET_CASE_IDS
                ),
                "case_inventory_exact": set(provenance)
                == set(PROSPECTIVE_BUFFERED_OFFSET_CASE_IDS),
                "all_cases_in_opened_e14_group": all(
                    row["case_id"] == case_id
                    and row["split_group_id"] == PROSPECTIVE_BUFFERED_OFFSET_GROUP_ID
                    and row["split"] == "test"
                    and float(row["parameters"]["vortex_epsilon"]) == 0.3875
                    and float(row["parameters"]["vortex_y"])
                    == 0.35 + 0.0375 * index
                    for index, (case_id, row) in enumerate(provenance.items())
                ),
                "checkpoint_model_built": False,
                "reference_arrays_loaded": False,
                "recurrence_executed": False,
                "new_population_opened": False,
                "still_sealed_groups": [],
                "warm_start_blocks": FROZEN_OFFSET_WARM_BLOCKS,
                "buffered_relaxed_tether": True,
                "position_trust_threshold": FROZEN_POSITION_TRUST_THRESHOLD,
                "position_buffer_threshold": FROZEN_POSITION_BUFFER_THRESHOLD,
                "relaxation_rate": RELAXED_TETHER_RATE,
                "truth_or_reference_used_at_inference": False,
                "e14_outcome_used_for_protocol_selection": False,
                "registered_scored_calls": 580,
                "animation_contract": BUFFERED_RELAXED_TETHER_E14_ANIMATION_CONTRACT,
            }
        )
    finally:
        store.close()
    if not (
        payload["case_inventory_present"]
        and payload["case_inventory_exact"]
        and payload["all_cases_in_opened_e14_group"]
    ):
        raise ValueError("A23 opened-E14 inventory is unavailable")
    atomic_write_json(args.output, payload)
    return payload


def _verify_buffered_relaxed_tether_e14_preflight(
    path: Path, args: argparse.Namespace
) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    source = payload.get("source_manifest", {})
    if (
        payload.get("schema") != BUFFERED_RELAXED_TETHER_E14_PREFLIGHT_SCHEMA
        or payload.get("working_id") != BUFFERED_RELAXED_TETHER_E14_WORKING_ID
        or payload.get("status") != "passed"
        or payload.get("reference_arrays_loaded") is not False
        or payload.get("recurrence_executed") is not False
        or payload.get("new_population_opened") is not False
        or payload.get("still_sealed_groups") != []
        or payload.get("all_cases_in_opened_e14_group") is not True
        or payload.get("warm_start_blocks") != FROZEN_OFFSET_WARM_BLOCKS
        or payload.get("buffered_relaxed_tether") is not True
        or payload.get("position_trust_threshold")
        != FROZEN_POSITION_TRUST_THRESHOLD
        or payload.get("position_buffer_threshold")
        != FROZEN_POSITION_BUFFER_THRESHOLD
        or payload.get("relaxation_rate") != RELAXED_TETHER_RATE
        or payload.get("truth_or_reference_used_at_inference") is not False
        or payload.get("e14_outcome_used_for_protocol_selection") is not False
        or payload.get("registered_scored_calls") != 580
        or source.get("source_sha256") != _source_hashes()
        or source.get("qualified_a22_r1_rollout_sha256")
        != sha256_file(args.a22_r1_rollout)
        or source.get("stopped_a14_comparator_sha256")
        != sha256_file(args.a14_rollout)
        or source.get("population", {}).get("case_ids")
        != list(PROSPECTIVE_BUFFERED_OFFSET_CASE_IDS)
    ):
        raise ValueError("A23 preflight differs from the frozen contract")
    return payload


def run_frozen_offset_preflight(args: argparse.Namespace) -> dict[str, Any]:
    _, manifest, store, source, _, _, _, _, _, _, a11_rollout = (
        _open_frozen_offset_contract(args)
    )
    try:
        provenance = {
            case_id: collection.family_case_provenance(manifest, case_id)
            for case_id in PROSPECTIVE_STRUCTURAL_CASE_IDS
        }
        payload = with_payload_sha256(
            {
                "schema": FROZEN_OFFSET_PREFLIGHT_SCHEMA,
                "working_id": FROZEN_OFFSET_WORKING_ID,
                "status": "passed",
                "population_status": "already_open_e12_recurrent_calibration",
                "source_manifest": source,
                "stopped_a11_payload_sha256": a11_rollout["payload_sha256"],
                "case_inventory_present": all(
                    case_id in store.keys for case_id in PROSPECTIVE_STRUCTURAL_CASE_IDS
                ),
                "case_inventory_exact": set(provenance)
                == set(PROSPECTIVE_STRUCTURAL_CASE_IDS),
                "all_cases_in_opened_e12_group": all(
                    row["split_group_id"] == PROSPECTIVE_STRUCTURAL_GROUP_ID
                    for row in provenance.values()
                ),
                "checkpoint_model_built": False,
                "reference_arrays_loaded": False,
                "recurrence_executed": False,
                "new_sealed_population_opened": False,
                "still_sealed_groups": ["strength_ood_e14"],
                "warm_start_blocks": FROZEN_OFFSET_WARM_BLOCKS,
                "frozen_offset": True,
                "coast_offset_feedback": False,
                "coast_projection_active_cells": [
                    list(cell) for cell in FROZEN_ACTIVE_CELLS
                ],
                "coefficient_or_threshold_refit": False,
                "animation_contract": FROZEN_OFFSET_ANIMATION_CONTRACT,
            }
        )
    finally:
        store.close()
    if not (
        payload["case_inventory_present"]
        and payload["case_inventory_exact"]
        and payload["all_cases_in_opened_e12_group"]
    ):
        raise ValueError("A12 open-e12 calibration inventory is unavailable")
    atomic_write_json(args.output, payload)
    return payload


def _verify_frozen_offset_preflight(
    path: Path, args: argparse.Namespace
) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    source = payload.get("source_manifest", {})
    if (
        payload.get("schema") != FROZEN_OFFSET_PREFLIGHT_SCHEMA
        or payload.get("working_id") != FROZEN_OFFSET_WORKING_ID
        or payload.get("status") != "passed"
        or payload.get("reference_arrays_loaded") is not False
        or payload.get("recurrence_executed") is not False
        or payload.get("new_sealed_population_opened") is not False
        or payload.get("still_sealed_groups") != ["strength_ood_e14"]
        or payload.get("all_cases_in_opened_e12_group") is not True
        or payload.get("warm_start_blocks") != FROZEN_OFFSET_WARM_BLOCKS
        or payload.get("frozen_offset") is not True
        or payload.get("coast_offset_feedback") is not False
        or payload.get("coast_projection_active_cells")
        != [list(cell) for cell in FROZEN_ACTIVE_CELLS]
        or payload.get("coefficient_or_threshold_refit") is not False
        or source.get("source_sha256") != _source_hashes()
        or source.get("stopped_a11_rollout_sha256") != sha256_file(args.a11_rollout)
        or source.get("population", {}).get("case_ids")
        != list(PROSPECTIVE_STRUCTURAL_CASE_IDS)
    ):
        raise ValueError("A12 frozen-offset preflight differs from contract")
    return payload


def run_buffered_offset_preflight(args: argparse.Namespace) -> dict[str, Any]:
    opened = _open_buffered_offset_contract(args)
    _, manifest, store, source, *_, a12_rollout, a12_buffer_rescore = opened
    try:
        provenance = {
            case_id: collection.family_case_provenance(manifest, case_id)
            for case_id in PROSPECTIVE_STRUCTURAL_CASE_IDS
        }
        payload = with_payload_sha256(
            {
                "schema": BUFFERED_OFFSET_PREFLIGHT_SCHEMA,
                "working_id": BUFFERED_OFFSET_WORKING_ID,
                "status": "passed",
                "population_status": "already_open_e12_recurrent_calibration",
                "source_manifest": source,
                "stopped_a12_payload_sha256": a12_rollout["payload_sha256"],
                "qualified_a12_buffer_rescore_payload_sha256": a12_buffer_rescore[
                    "payload_sha256"
                ],
                "case_inventory_present": all(
                    case_id in store.keys for case_id in PROSPECTIVE_STRUCTURAL_CASE_IDS
                ),
                "case_inventory_exact": set(provenance)
                == set(PROSPECTIVE_STRUCTURAL_CASE_IDS),
                "all_cases_in_opened_e12_group": all(
                    row["split_group_id"] == PROSPECTIVE_STRUCTURAL_GROUP_ID
                    for row in provenance.values()
                ),
                "checkpoint_model_built": False,
                "reference_arrays_loaded": False,
                "recurrence_executed": False,
                "new_sealed_population_opened": False,
                "still_sealed_groups": ["strength_ood_e14"],
                "warm_start_blocks": FROZEN_OFFSET_WARM_BLOCKS,
                "frozen_offset": True,
                "coast_offset_feedback": False,
                "position_trust_threshold": FROZEN_POSITION_TRUST_THRESHOLD,
                "position_buffer_threshold": FROZEN_POSITION_BUFFER_THRESHOLD,
                "coast_projection_active_cells": [
                    list(cell) for cell in FROZEN_ACTIVE_CELLS
                ],
                "coefficient_refit": False,
                "animation_contract": BUFFERED_OFFSET_ANIMATION_CONTRACT,
            }
        )
    finally:
        store.close()
    if not (
        payload["case_inventory_present"]
        and payload["case_inventory_exact"]
        and payload["all_cases_in_opened_e12_group"]
    ):
        raise ValueError("A13 open-e12 calibration inventory is unavailable")
    atomic_write_json(args.output, payload)
    return payload


def _verify_buffered_offset_preflight(
    path: Path, args: argparse.Namespace
) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    source = payload.get("source_manifest", {})
    if (
        payload.get("schema") != BUFFERED_OFFSET_PREFLIGHT_SCHEMA
        or payload.get("working_id") != BUFFERED_OFFSET_WORKING_ID
        or payload.get("status") != "passed"
        or payload.get("reference_arrays_loaded") is not False
        or payload.get("recurrence_executed") is not False
        or payload.get("new_sealed_population_opened") is not False
        or payload.get("still_sealed_groups") != ["strength_ood_e14"]
        or payload.get("all_cases_in_opened_e12_group") is not True
        or payload.get("warm_start_blocks") != FROZEN_OFFSET_WARM_BLOCKS
        or payload.get("frozen_offset") is not True
        or payload.get("coast_offset_feedback") is not False
        or payload.get("position_trust_threshold") != FROZEN_POSITION_TRUST_THRESHOLD
        or payload.get("position_buffer_threshold") != FROZEN_POSITION_BUFFER_THRESHOLD
        or payload.get("coast_projection_active_cells")
        != [list(cell) for cell in FROZEN_ACTIVE_CELLS]
        or payload.get("coefficient_refit") is not False
        or source.get("source_sha256") != _source_hashes()
        or source.get("stopped_a12_rollout_sha256") != sha256_file(args.a12_rollout)
        or source.get("qualified_a12_buffer_rescore_sha256")
        != sha256_file(args.a12_buffer_rescore)
        or source.get("population", {}).get("case_ids")
        != list(PROSPECTIVE_STRUCTURAL_CASE_IDS)
    ):
        raise ValueError("A13 buffered-offset preflight differs from contract")
    return payload


def run_prospective_buffered_offset_preflight(
    args: argparse.Namespace,
) -> dict[str, Any]:
    opened = _open_prospective_buffered_offset_contract(args)
    _, manifest, store, source, *_, a13_rollout = opened
    try:
        provenance = {
            case_id: collection.family_case_provenance(manifest, case_id)
            for case_id in PROSPECTIVE_BUFFERED_OFFSET_CASE_IDS
        }
        payload = with_payload_sha256(
            {
                "schema": PROSPECTIVE_BUFFERED_OFFSET_PREFLIGHT_SCHEMA,
                "working_id": PROSPECTIVE_BUFFERED_OFFSET_WORKING_ID,
                "status": "passed",
                "population_status": "single_prospectively_named_e14_confirmation",
                "source_manifest": source,
                "qualified_a13_payload_sha256": a13_rollout["payload_sha256"],
                "case_inventory_present": all(
                    case_id in store.keys
                    for case_id in PROSPECTIVE_BUFFERED_OFFSET_CASE_IDS
                ),
                "case_inventory_exact": set(provenance)
                == set(PROSPECTIVE_BUFFERED_OFFSET_CASE_IDS),
                "all_cases_in_named_e14_group": all(
                    row["case_id"] == case_id
                    and row["split_group_id"] == PROSPECTIVE_BUFFERED_OFFSET_GROUP_ID
                    and row["split"] == "test"
                    and float(row["parameters"]["vortex_epsilon"]) == 0.3875
                    and float(row["parameters"]["vortex_y"]) == 0.35 + 0.0375 * index
                    for index, (case_id, row) in enumerate(provenance.items())
                ),
                "checkpoint_model_built": False,
                "reference_arrays_loaded": False,
                "recurrence_executed": False,
                "new_sealed_population_opened": PROSPECTIVE_BUFFERED_OFFSET_GROUP_ID,
                "still_sealed_groups": [],
                "warm_start_blocks": FROZEN_OFFSET_WARM_BLOCKS,
                "frozen_offset": True,
                "buffered_offset": True,
                "coast_offset_feedback": False,
                "position_trust_threshold": FROZEN_POSITION_TRUST_THRESHOLD,
                "position_buffer_threshold": FROZEN_POSITION_BUFFER_THRESHOLD,
                "coast_projection_active_cells": [
                    list(cell) for cell in FROZEN_ACTIVE_CELLS
                ],
                "coefficient_refit": False,
                "additional_e12_selection": False,
                "animation_contract": PROSPECTIVE_BUFFERED_OFFSET_ANIMATION_CONTRACT,
            }
        )
    finally:
        store.close()
    if not (
        payload["case_inventory_present"]
        and payload["case_inventory_exact"]
        and payload["all_cases_in_named_e14_group"]
    ):
        raise ValueError("A14 prospectively named e14 inventory is unavailable")
    atomic_write_json(args.output, payload)
    return payload


def _verify_prospective_buffered_offset_preflight(
    path: Path, args: argparse.Namespace
) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    source = payload.get("source_manifest", {})
    if (
        payload.get("schema") != PROSPECTIVE_BUFFERED_OFFSET_PREFLIGHT_SCHEMA
        or payload.get("working_id") != PROSPECTIVE_BUFFERED_OFFSET_WORKING_ID
        or payload.get("status") != "passed"
        or payload.get("reference_arrays_loaded") is not False
        or payload.get("recurrence_executed") is not False
        or payload.get("new_sealed_population_opened")
        != PROSPECTIVE_BUFFERED_OFFSET_GROUP_ID
        or payload.get("still_sealed_groups") != []
        or payload.get("all_cases_in_named_e14_group") is not True
        or payload.get("warm_start_blocks") != FROZEN_OFFSET_WARM_BLOCKS
        or payload.get("frozen_offset") is not True
        or payload.get("buffered_offset") is not True
        or payload.get("coast_offset_feedback") is not False
        or payload.get("position_trust_threshold") != FROZEN_POSITION_TRUST_THRESHOLD
        or payload.get("position_buffer_threshold") != FROZEN_POSITION_BUFFER_THRESHOLD
        or payload.get("coast_projection_active_cells")
        != [list(cell) for cell in FROZEN_ACTIVE_CELLS]
        or payload.get("coefficient_refit") is not False
        or payload.get("additional_e12_selection") is not False
        or source.get("source_sha256") != _source_hashes()
        or source.get("qualified_a13_rollout_sha256") != sha256_file(args.a13_rollout)
        or source.get("population", {}).get("case_ids")
        != list(PROSPECTIVE_BUFFERED_OFFSET_CASE_IDS)
        or source.get("protocol", {}).get("frozen_from") != BUFFERED_OFFSET_WORKING_ID
    ):
        raise ValueError("A14 prospective buffered-offset preflight differs")
    return payload


def run_persistence_gain_preflight(args: argparse.Namespace) -> dict[str, Any]:
    opened = _open_persistence_gain_contract(args)
    _, manifest, store, source, *_, a13_rollout, a14_rollout = opened
    try:
        provenance = {
            case_id: collection.family_case_provenance(manifest, case_id)
            for case_id in PROSPECTIVE_STRUCTURAL_CASE_IDS
        }
        payload = with_payload_sha256(
            {
                "schema": PERSISTENCE_GAIN_PREFLIGHT_SCHEMA,
                "working_id": PERSISTENCE_GAIN_WORKING_ID,
                "status": "passed",
                "population_status": "already_open_e12_persistence_calibration",
                "source_manifest": source,
                "qualified_a13_payload_sha256": a13_rollout["payload_sha256"],
                "stopped_a14_payload_sha256": a14_rollout["payload_sha256"],
                "case_inventory_present": all(
                    case_id in store.keys for case_id in PROSPECTIVE_STRUCTURAL_CASE_IDS
                ),
                "case_inventory_exact": set(provenance)
                == set(PROSPECTIVE_STRUCTURAL_CASE_IDS),
                "all_cases_in_opened_e12_group": all(
                    row["split_group_id"] == PROSPECTIVE_STRUCTURAL_GROUP_ID
                    for row in provenance.values()
                ),
                "checkpoint_model_built": False,
                "reference_arrays_loaded": False,
                "recurrence_executed": False,
                "new_sealed_population_opened": False,
                "e14_state_or_truth_opened": False,
                "warm_start_blocks": FROZEN_OFFSET_WARM_BLOCKS,
                "persistence_probe": True,
                "additional_fine_probe_calls": 55,
                "coast_offset_feedback": False,
                "coefficient_or_threshold_refit": False,
                "a14_truth_used_for_selection": False,
                "animation_contract": PERSISTENCE_GAIN_ANIMATION_CONTRACT,
            }
        )
    finally:
        store.close()
    if not (
        payload["case_inventory_present"]
        and payload["case_inventory_exact"]
        and payload["all_cases_in_opened_e12_group"]
    ):
        raise ValueError("A15 open-e12 calibration inventory is unavailable")
    atomic_write_json(args.output, payload)
    return payload


def _verify_persistence_gain_preflight(
    path: Path, args: argparse.Namespace
) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    source = payload.get("source_manifest", {})
    if (
        payload.get("schema") != PERSISTENCE_GAIN_PREFLIGHT_SCHEMA
        or payload.get("working_id") != PERSISTENCE_GAIN_WORKING_ID
        or payload.get("status") != "passed"
        or payload.get("reference_arrays_loaded") is not False
        or payload.get("recurrence_executed") is not False
        or payload.get("new_sealed_population_opened") is not False
        or payload.get("e14_state_or_truth_opened") is not False
        or payload.get("warm_start_blocks") != FROZEN_OFFSET_WARM_BLOCKS
        or payload.get("persistence_probe") is not True
        or payload.get("additional_fine_probe_calls") != 55
        or payload.get("coast_offset_feedback") is not False
        or payload.get("coefficient_or_threshold_refit") is not False
        or payload.get("a14_truth_used_for_selection") is not False
        or source.get("source_sha256") != _source_hashes()
        or source.get("qualified_a13_rollout_sha256") != sha256_file(args.a13_rollout)
        or source.get("stopped_a14_rollout_sha256") != sha256_file(args.a14_rollout)
        or source.get("population", {}).get("case_ids")
        != list(PROSPECTIVE_STRUCTURAL_CASE_IDS)
    ):
        raise ValueError("A15 persistence-gain preflight differs from contract")
    return payload


def run_terminal_ramp_preflight(args: argparse.Namespace) -> dict[str, Any]:
    opened = _open_terminal_ramp_contract(args)
    _, manifest, store, source, *_, a15_rollout = opened
    try:
        provenance = {
            case_id: collection.family_case_provenance(manifest, case_id)
            for case_id in PROSPECTIVE_STRUCTURAL_CASE_IDS
        }
        payload = with_payload_sha256(
            {
                "schema": TERMINAL_RAMP_PREFLIGHT_SCHEMA,
                "working_id": TERMINAL_RAMP_WORKING_ID,
                "status": "passed",
                "population_status": "already_open_e12_terminal_ramp_calibration",
                "source_manifest": source,
                "stopped_a15_payload_sha256": a15_rollout["payload_sha256"],
                "case_inventory_present": all(
                    case_id in store.keys for case_id in PROSPECTIVE_STRUCTURAL_CASE_IDS
                ),
                "case_inventory_exact": set(provenance)
                == set(PROSPECTIVE_STRUCTURAL_CASE_IDS),
                "all_cases_in_opened_e12_group": all(
                    row["split_group_id"] == PROSPECTIVE_STRUCTURAL_GROUP_ID
                    for row in provenance.values()
                ),
                "checkpoint_model_built": False,
                "reference_arrays_loaded": False,
                "recurrence_executed": False,
                "new_population_opened": False,
                "e14_state_or_truth_opened": False,
                "phase_trigger": "first_nonpositive_or_unresolved_probe_offset_cosine",
                "terminal_block": 14,
                "coefficient_or_rate_refit": False,
                "a15_truth_used_for_rule": True,
                "animation_contract": TERMINAL_RAMP_ANIMATION_CONTRACT,
            }
        )
    finally:
        store.close()
    if not (
        payload["case_inventory_present"]
        and payload["case_inventory_exact"]
        and payload["all_cases_in_opened_e12_group"]
    ):
        raise ValueError("A16 open-e12 calibration inventory is unavailable")
    atomic_write_json(args.output, payload)
    return payload


def _verify_terminal_ramp_preflight(
    path: Path, args: argparse.Namespace
) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    source = payload.get("source_manifest", {})
    if (
        payload.get("schema") != TERMINAL_RAMP_PREFLIGHT_SCHEMA
        or payload.get("working_id") != TERMINAL_RAMP_WORKING_ID
        or payload.get("status") != "passed"
        or payload.get("reference_arrays_loaded") is not False
        or payload.get("recurrence_executed") is not False
        or payload.get("new_population_opened") is not False
        or payload.get("e14_state_or_truth_opened") is not False
        or payload.get("terminal_block") != 14
        or payload.get("coefficient_or_rate_refit") is not False
        or payload.get("a15_truth_used_for_rule") is not True
        or source.get("source_sha256") != _source_hashes()
        or source.get("stopped_a15_rollout_sha256") != sha256_file(args.a15_rollout)
    ):
        raise ValueError("A16 terminal-ramp preflight differs from contract")
    return payload


def run_fixed_late_ramp_preflight(args: argparse.Namespace) -> dict[str, Any]:
    opened = _open_fixed_late_ramp_contract(args)
    _, manifest, store, source, *_, a13_rollout, a16_rollout = opened
    try:
        provenance = {
            case_id: collection.family_case_provenance(manifest, case_id)
            for case_id in PROSPECTIVE_STRUCTURAL_CASE_IDS
        }
        payload = with_payload_sha256(
            {
                "schema": FIXED_LATE_RAMP_PREFLIGHT_SCHEMA,
                "working_id": FIXED_LATE_RAMP_WORKING_ID,
                "status": "passed",
                "population_status": "already_open_e12_fixed_late_ramp_calibration",
                "source_manifest": source,
                "qualified_a13_payload_sha256": a13_rollout["payload_sha256"],
                "stopped_a16_payload_sha256": a16_rollout["payload_sha256"],
                "case_inventory_present": all(
                    case_id in store.keys for case_id in PROSPECTIVE_STRUCTURAL_CASE_IDS
                ),
                "case_inventory_exact": set(provenance)
                == set(PROSPECTIVE_STRUCTURAL_CASE_IDS),
                "all_cases_in_opened_e12_group": all(
                    row["split_group_id"] == PROSPECTIVE_STRUCTURAL_GROUP_ID
                    for row in provenance.values()
                ),
                "checkpoint_model_built": False,
                "reference_arrays_loaded": False,
                "recurrence_executed": False,
                "new_population_opened": False,
                "schedule": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
                "persistence_probe": False,
                "additional_fine_probe_calls": 0,
                "a13_a15_a16_truth_used_for_schedule_selection": True,
                "truth_or_reference_used_at_inference": False,
                "animation_contract": FIXED_LATE_RAMP_ANIMATION_CONTRACT,
            }
        )
    finally:
        store.close()
    if not (
        payload["case_inventory_present"]
        and payload["case_inventory_exact"]
        and payload["all_cases_in_opened_e12_group"]
    ):
        raise ValueError("A17 open-e12 calibration inventory is unavailable")
    atomic_write_json(args.output, payload)
    return payload


def _verify_fixed_late_ramp_preflight(
    path: Path, args: argparse.Namespace
) -> dict[str, Any]:
    payload = _read_json(path)
    verify_payload_sha256(payload)
    source = payload.get("source_manifest", {})
    if (
        payload.get("schema") != FIXED_LATE_RAMP_PREFLIGHT_SCHEMA
        or payload.get("working_id") != FIXED_LATE_RAMP_WORKING_ID
        or payload.get("status") != "passed"
        or payload.get("reference_arrays_loaded") is not False
        or payload.get("recurrence_executed") is not False
        or payload.get("new_population_opened") is not False
        or payload.get("persistence_probe") is not False
        or payload.get("additional_fine_probe_calls") != 0
        or payload.get("a13_a15_a16_truth_used_for_schedule_selection") is not True
        or payload.get("truth_or_reference_used_at_inference") is not False
        or source.get("source_sha256") != _source_hashes()
        or source.get("qualified_a13_rollout_sha256")
        != sha256_file(args.a13_rollout)
        or source.get("stopped_a16_rollout_sha256")
        != sha256_file(args.a16_rollout)
    ):
        raise ValueError("A17 fixed-late-ramp preflight differs from contract")
    return payload


def _maximum_abs(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.max(np.abs(array))) if array.size else 0.0


def _state_energy(
    error: np.ndarray,
    *,
    volumes: np.ndarray,
    component_scale: np.ndarray,
) -> float:
    value = weighted_scaled_rms(
        error,
        volumes=volumes,
        component_scale=component_scale,
    )
    return float(value * value)


def _account(execution: dict[str, Any], timing: Mapping[str, Any]) -> None:
    execution["logical_model_calls"] += 1
    forwards = timing["forward_seconds"]
    execution["actual_forward_passes_including_repeats"] += len(forwards)
    execution["total_forward_seconds"] += float(sum(forwards))
    repeat = float(timing["repeat_max_abs"])
    if math.isfinite(repeat):
        execution["maximum_repeat_abs_difference"] = max(
            execution["maximum_repeat_abs_difference"], repeat
        )
    peak = timing.get("peak_gpu_memory_bytes")
    if peak is not None:
        execution["maximum_peak_gpu_memory_bytes"] = max(
            execution["maximum_peak_gpu_memory_bytes"], int(peak)
        )


def _case_first_energy(records, key: str, cases: Sequence[str]) -> float:
    return float(
        np.mean(
            [
                np.mean([float(row[key]) for row in records if row["case_id"] == case])
                for case in cases
            ]
        )
    )


def _case_first_rms(records, key: str, cases: Sequence[str]) -> float:
    return float(
        np.sqrt(
            np.mean(
                [
                    np.mean(
                        [
                            float(row[key]) ** 2
                            for row in records
                            if row["case_id"] == case
                        ]
                    )
                    for case in cases
                ]
            )
        )
    )


def _control_ratio(raw: float, candidate: float) -> dict[str, float | str | None]:
    if raw > 1.0e-14:
        return {"ratio": candidate / raw, "status": "ok"}
    if candidate <= 1.0e-14:
        return {"ratio": 1.0, "status": "exact_zero_no_change"}
    return {"ratio": None, "status": "nonzero_from_zero"}


def _diagnostic_ratio(
    numerator: float, denominator: float
) -> dict[str, float | str | None]:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return {"value": None, "status": "unresolved_nonfinite_denominator"}
    if denominator <= A9_DENOMINATOR_FLOOR:
        return {"value": None, "status": "unresolved_small_denominator"}
    return {"value": numerator / denominator, "status": "ok"}


def _weighted_scaled_cosine(
    left: np.ndarray,
    right: np.ndarray,
    *,
    volumes: np.ndarray,
    component_scale: np.ndarray,
) -> dict[str, float | str | None]:
    lhs = np.asarray(left, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    mass = np.asarray(volumes, dtype=np.float64).reshape(-1)
    scale = np.asarray(component_scale, dtype=np.float64).reshape(-1)
    if (
        lhs.ndim != 2
        or rhs.shape != lhs.shape
        or mass.shape != (lhs.shape[0],)
        or scale.shape != (lhs.shape[1],)
        or not np.isfinite(lhs).all()
        or not np.isfinite(rhs).all()
        or not np.isfinite(mass).all()
        or not np.isfinite(scale).all()
        or np.any(mass <= 0.0)
        or np.any(scale <= 0.0)
    ):
        raise ValueError("weighted cosine requires finite, positive, aligned inputs")
    scaled_left = lhs / scale[None, :]
    scaled_right = rhs / scale[None, :]
    inner = float(np.sum(mass[:, None] * scaled_left * scaled_right))
    left_norm = float(np.sqrt(np.sum(mass[:, None] * scaled_left**2)))
    right_norm = float(np.sqrt(np.sum(mass[:, None] * scaled_right**2)))
    denominator = left_norm * right_norm
    if denominator <= A9_DENOMINATOR_FLOOR:
        return {
            "value": None,
            "status": "unresolved_small_denominator",
            "denominator": denominator,
        }
    return {
        "value": inner / denominator,
        "status": "ok",
        "denominator": denominator,
    }


def _weighted_scaled_projection_coefficient(
    direction: np.ndarray,
    target: np.ndarray,
    *,
    volumes: np.ndarray,
    component_scale: np.ndarray,
) -> dict[str, float | str | None]:
    vector = np.asarray(direction, dtype=np.float64)
    value = np.asarray(target, dtype=np.float64)
    mass = np.asarray(volumes, dtype=np.float64).reshape(-1)
    scale = np.asarray(component_scale, dtype=np.float64).reshape(-1)
    if (
        vector.ndim != 2
        or value.shape != vector.shape
        or mass.shape != (vector.shape[0],)
        or scale.shape != (vector.shape[1],)
        or not np.isfinite(vector).all()
        or not np.isfinite(value).all()
        or not np.isfinite(mass).all()
        or not np.isfinite(scale).all()
        or np.any(mass <= 0.0)
        or np.any(scale <= 0.0)
    ):
        raise ValueError("weighted projection requires finite, positive, aligned inputs")
    scaled_direction = vector / scale[None, :]
    scaled_target = value / scale[None, :]
    denominator = float(np.sum(mass[:, None] * scaled_direction**2))
    if denominator <= A9_DENOMINATOR_FLOOR:
        return {
            "value": None,
            "status": "unresolved_small_denominator",
            "denominator": denominator,
        }
    numerator = float(
        np.sum(mass[:, None] * scaled_direction * scaled_target)
    )
    return {
        "value": numerator / denominator,
        "status": "ok",
        "denominator": denominator,
    }


def _persistence_error_structure_row(
    runtime: Any,
    *,
    case_id: str,
    block_index: int,
    raw_prediction: np.ndarray,
    corrected_prediction: np.ndarray,
    target: np.ndarray,
    probe_correction: np.ndarray,
    frozen_offset: np.ndarray,
    applied_gain: float,
) -> dict[str, Any]:
    """Score target-free probe structure offline; targets never affect inference."""

    geometry = runtime.geometry_by_resolution[NATIVE_RESOLUTION]
    volumes = np.asarray(geometry.node_measures, dtype=np.float64).reshape(-1)
    state_scale = np.asarray(runtime.normalization.state_scale, dtype=np.float64)
    needed = np.asarray(target, dtype=np.float64) - np.asarray(
        raw_prediction, dtype=np.float64
    )
    applied_error = np.asarray(corrected_prediction, dtype=np.float64) - np.asarray(
        target, dtype=np.float64
    )
    raw_error = -needed
    rank_parts, _ = runtime.native_projector.split(needed)
    rank8_needed = np.asarray(rank_parts["parallel"], dtype=np.float64)
    orthogonal_needed = np.asarray(rank_parts["orthogonal"], dtype=np.float64)
    masks, _, _ = parent.shock_vortex_regions(
        target,
        geometry.nodes,
        resolution=NATIVE_RESOLUTION,
        gamma=runtime.normalization.gamma,
    )

    def rms(value: np.ndarray, mask: np.ndarray | None = None) -> float:
        return weighted_scaled_rms(
            value,
            volumes=volumes,
            component_scale=state_scale,
            mask=mask,
        )

    needed_rms = rms(needed)
    rank8_rms = rms(rank8_needed)
    orthogonal_rms = rms(orthogonal_needed)
    raw_sse = needed_rms**2
    corrected_sse = rms(applied_error) ** 2
    state_skill = 1.0 - corrected_sse / raw_sse if raw_sse > 1.0e-24 else None
    row: dict[str, Any] = {
        "case_id": case_id,
        "block_index": block_index,
        "output_call": 2 * block_index + 1,
        "applied_gain": float(applied_gain),
        "needed_correction_rms": needed_rms,
        "needed_rank8_rms": rank8_rms,
        "needed_orthogonal_rms": orthogonal_rms,
        "rank8_energy_fraction": (
            rank8_rms**2 / raw_sse if raw_sse > 1.0e-24 else None
        ),
        "applied_state_sse_skill": state_skill,
    }
    for name, direction in (
        ("probe", probe_correction),
        ("handoff_offset", frozen_offset),
    ):
        relation = _weighted_scaled_cosine(
            direction,
            needed,
            volumes=volumes,
            component_scale=state_scale,
        )
        coefficient = _weighted_scaled_projection_coefficient(
            direction,
            needed,
            volumes=volumes,
            component_scale=state_scale,
        )
        row.update(
            {
                f"{name}_needed_cosine": relation["value"],
                f"{name}_needed_cosine_status": relation["status"],
                f"{name}_needed_cosine_denominator": relation["denominator"],
                f"{name}_oracle_coefficient": coefficient["value"],
                f"{name}_oracle_coefficient_status": coefficient["status"],
                f"{name}_oracle_coefficient_denominator": coefficient["denominator"],
            }
        )
    probe_offset = _weighted_scaled_cosine(
        probe_correction,
        frozen_offset,
        volumes=volumes,
        component_scale=state_scale,
    )
    row.update(
        {
            "probe_handoff_offset_cosine": probe_offset["value"],
            "probe_handoff_offset_cosine_status": probe_offset["status"],
            "probe_handoff_offset_cosine_denominator": probe_offset["denominator"],
        }
    )
    for component, name in enumerate(INTEGRAL_COMPONENT_NAMES):
        relation = _weighted_scaled_cosine(
            np.asarray(probe_correction)[:, component : component + 1],
            needed[:, component : component + 1],
            volumes=volumes,
            component_scale=state_scale[component : component + 1],
        )
        row[f"component_{name}_probe_needed_cosine"] = relation["value"]
        row[f"component_{name}_probe_needed_cosine_status"] = relation["status"]
        row[f"component_{name}_probe_needed_cosine_denominator"] = relation[
            "denominator"
        ]
    for label, mask_name in (
        ("boundary", "boundary_le_0.05"),
        ("shock", "partition_shock"),
        ("vortex", "partition_vortex"),
        ("smooth", "partition_smooth"),
    ):
        mask = np.asarray(masks[mask_name], dtype=bool)
        relation = _weighted_scaled_cosine(
            np.asarray(probe_correction)[mask],
            needed[mask],
            volumes=volumes[mask],
            component_scale=state_scale,
        )
        row[f"region_{label}_probe_needed_cosine"] = relation["value"]
        row[f"region_{label}_probe_needed_cosine_status"] = relation["status"]
        row[f"region_{label}_probe_needed_cosine_denominator"] = relation[
            "denominator"
        ]
        row[f"region_{label}_needed_rms"] = rms(needed, mask)
        row[f"region_{label}_raw_error_rms"] = rms(raw_error, mask)
    return row


def _shadow_candidate_feature_row(
    *,
    case_id: str,
    block_index: int,
    raw_input: np.ndarray,
    block: Any,
    volumes: np.ndarray,
    residual_scale: np.ndarray,
    position_descriptor: Any,
    front_audit: Any,
) -> dict[str, Any]:
    raw_first_increment = block.shadow_first_state - raw_input
    raw_second_increment = block.shadow_second_state - block.shadow_first_state
    first_intervention = block.anchored_first_state - block.shadow_first_state
    second_intervention = block.anchored_second_state - block.shadow_second_state

    def rms(value: np.ndarray) -> float:
        return weighted_scaled_rms(
            value,
            volumes=volumes,
            component_scale=residual_scale,
        )

    first_correction_rms = rms(block.first_correction)
    first_intervention_rms = rms(first_intervention)
    second_intervention_rms = rms(second_intervention)
    first_shadow_increment_rms = rms(raw_first_increment)
    second_shadow_increment_rms = rms(raw_second_increment)
    full_response_rms = rms(block.full_response)
    filtered_response_rms = rms(block.filtered_response)

    ratios = {
        "correction_to_native_increment": _diagnostic_ratio(
            first_correction_rms, first_shadow_increment_rms
        ),
        "retained_response_fraction": _diagnostic_ratio(
            filtered_response_rms, full_response_rms
        ),
        "full_response_to_first_intervention": _diagnostic_ratio(
            full_response_rms, first_intervention_rms
        ),
        "filtered_response_to_first_intervention": _diagnostic_ratio(
            filtered_response_rms, first_intervention_rms
        ),
        "second_to_first_intervention": _diagnostic_ratio(
            second_intervention_rms, first_intervention_rms
        ),
        "first_intervention_to_shadow_increment": _diagnostic_ratio(
            first_intervention_rms, first_shadow_increment_rms
        ),
        "second_intervention_to_shadow_increment": _diagnostic_ratio(
            second_intervention_rms, second_shadow_increment_rms
        ),
    }
    cosines = {
        "first_correction_filtered_response_cosine": _weighted_scaled_cosine(
            block.first_correction,
            block.filtered_response,
            volumes=volumes,
            component_scale=residual_scale,
        ),
        "first_second_intervention_cosine": _weighted_scaled_cosine(
            first_intervention,
            second_intervention,
            volumes=volumes,
            component_scale=residual_scale,
        ),
        "full_filtered_response_cosine": _weighted_scaled_cosine(
            block.full_response,
            block.filtered_response,
            volumes=volumes,
            component_scale=residual_scale,
        ),
    }
    row: dict[str, Any] = {
        "case_id": case_id,
        "block_index": block_index,
        "first_output_call": 2 * block_index + 1,
        "second_output_call": 2 * block_index + 2,
        "initial_normalized_wall_distance": (
            position_descriptor.normalized_wall_distance
        ),
        "initial_normalized_wall_distance_status": position_descriptor.status,
        "first_correction_rms": first_correction_rms,
        "first_correction_rms_status": "ok",
        "full_response_rms": full_response_rms,
        "full_response_rms_status": "ok",
        "filtered_response_rms": filtered_response_rms,
        "filtered_response_rms_status": "ok",
        "front_branch_changed": float(front_audit.branch_changed),
        "front_branch_changed_status": front_audit.status,
    }
    for name, relation in (*ratios.items(), *cosines.items()):
        row[name] = relation["value"]
        row[f"{name}_status"] = relation["status"]
        if "denominator" in relation:
            row[f"{name}_denominator"] = relation["denominator"]
    return row


def _block_sse_skill(
    raw_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, float | str | None]:
    raw_sse = float(sum(float(row[key]) ** 2 for row in raw_rows))
    candidate_sse = float(sum(float(row[key]) ** 2 for row in candidate_rows))
    if raw_sse <= A9_DENOMINATOR_FLOOR**2:
        return {
            "value": None,
            "status": "unresolved_small_denominator",
            "raw_sse": raw_sse,
            "candidate_sse": candidate_sse,
        }
    return {
        "value": 1.0 - candidate_sse / raw_sse,
        "status": "ok",
        "raw_sse": raw_sse,
        "candidate_sse": candidate_sse,
    }


def _shadow_candidate_label_row(
    *,
    raw_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(raw_rows) != 2 or len(candidate_rows) != 2:
        raise ValueError("one shadow-candidate label requires exactly two outputs")
    state = _block_sse_skill(raw_rows, candidate_rows, "state_error")
    increment = _block_sse_skill(raw_rows, candidate_rows, "increment_defect")
    rank8 = _block_sse_skill(raw_rows, candidate_rows, "rank8_state_error")
    endpoint = _diagnostic_ratio(
        float(candidate_rows[-1]["state_error"]),
        float(raw_rows[-1]["state_error"]),
    )
    endpoint_log_ratio = None
    endpoint_log_status = endpoint["status"]
    if endpoint["value"] is not None:
        ratio = float(endpoint["value"])
        if ratio > 0.0 and math.isfinite(ratio):
            endpoint_log_ratio = math.log(ratio)
            endpoint_log_status = "ok"
        else:
            endpoint_log_status = "unresolved_nonpositive_ratio"

    control_ratios: list[float] = []
    control_status = "ok"
    for key in A9_CONTROL_KEYS:
        raw_rms = math.sqrt(
            sum(float(row[key]) ** 2 for row in raw_rows) / len(raw_rows)
        )
        candidate_rms = math.sqrt(
            sum(float(row[key]) ** 2 for row in candidate_rows) / len(candidate_rows)
        )
        relation = _control_ratio(raw_rms, candidate_rms)
        if relation["ratio"] is None:
            control_status = str(relation["status"])
        else:
            control_ratios.append(float(relation["ratio"]))
    worst_control_ratio = max(control_ratios) if control_ratios else None
    controls_pass = (
        control_status == "ok"
        and worst_control_ratio is not None
        and worst_control_ratio <= 1.05
    )
    state_value = state["value"]
    return {
        "block_state_sse_skill": state_value,
        "block_state_sse_skill_status": state["status"],
        "raw_block_state_sse": state["raw_sse"],
        "candidate_block_state_sse": state["candidate_sse"],
        "block_increment_sse_skill": increment["value"],
        "block_increment_sse_skill_status": increment["status"],
        "raw_block_increment_sse": increment["raw_sse"],
        "candidate_block_increment_sse": increment["candidate_sse"],
        "block_rank8_sse_skill": rank8["value"],
        "block_rank8_sse_skill_status": rank8["status"],
        "endpoint_state_ratio": endpoint["value"],
        "endpoint_state_ratio_status": endpoint["status"],
        "endpoint_state_log_ratio": endpoint_log_ratio,
        "endpoint_state_log_ratio_status": endpoint_log_status,
        "worst_control_ratio": worst_control_ratio,
        "worst_control_ratio_status": control_status,
        "all_block_controls_at_most_1p05": controls_pass,
        "oracle_joint_accept": bool(
            state_value is not None and float(state_value) > 0.0 and controls_pass
        ),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        stop = start + 1
        while stop < array.size and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> dict[str, float | str | None]:
    lhs = np.asarray(left, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    if lhs.shape != rhs.shape or lhs.ndim != 1 or lhs.size < 2:
        return {"value": None, "status": "unresolved_inventory", "denominator": None}
    lhs_centered = lhs - np.mean(lhs)
    rhs_centered = rhs - np.mean(rhs)
    denominator = float(
        np.sqrt(np.dot(lhs_centered, lhs_centered) * np.dot(rhs_centered, rhs_centered))
    )
    if denominator <= A9_DENOMINATOR_FLOOR:
        return {
            "value": None,
            "status": "unresolved_small_denominator",
            "denominator": denominator,
        }
    return {
        "value": float(np.dot(lhs_centered, rhs_centered) / denominator),
        "status": "ok",
        "denominator": denominator,
    }


def _case_balanced_affine_fit(
    rows: Sequence[Mapping[str, Any]], *, feature: str, target: str
) -> tuple[float, float] | None:
    by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_id"]), []).append(row)
    weighted: list[tuple[float, float, float]] = []
    for case_rows in by_case.values():
        weight = 1.0 / (len(by_case) * len(case_rows))
        weighted.extend(
            (float(row[feature]), float(row[target]), weight) for row in case_rows
        )
    x_mean = sum(x * weight for x, _, weight in weighted)
    y_mean = sum(y * weight for _, y, weight in weighted)
    denominator = sum(weight * (x - x_mean) ** 2 for x, _, weight in weighted)
    if denominator <= A9_DENOMINATOR_FLOOR**2:
        return None
    slope = (
        sum(weight * (x - x_mean) * (y - y_mean) for x, y, weight in weighted)
        / denominator
    )
    return y_mean - slope * x_mean, slope


def _shadow_candidate_associations(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    association_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    case_ids = tuple(sorted({str(row["case_id"]) for row in rows}))
    for feature in A9_FEATURE_NAMES:
        for target in A9_TARGET_NAMES:
            valid = [
                row
                for row in rows
                if row.get(feature) is not None
                and row.get(target) is not None
                and math.isfinite(float(row[feature]))
                and math.isfinite(float(row[target]))
            ]
            x = np.asarray([float(row[feature]) for row in valid], dtype=np.float64)
            y = np.asarray([float(row[target]) for row in valid], dtype=np.float64)
            pearson = _correlation(x, y)
            spearman = _correlation(_average_ranks(x), _average_ranks(y))
            per_case_values: list[float] = []
            for case_id in case_ids:
                selected = [row for row in valid if row["case_id"] == case_id]
                case_x = np.asarray(
                    [float(row[feature]) for row in selected], dtype=np.float64
                )
                case_y = np.asarray(
                    [float(row[target]) for row in selected], dtype=np.float64
                )
                relation = _correlation(_average_ranks(case_x), _average_ranks(case_y))
                case_rows.append(
                    {
                        "feature": feature,
                        "target": target,
                        "case_id": case_id,
                        "valid_blocks": len(selected),
                        "spearman": relation["value"],
                        "spearman_status": relation["status"],
                        "spearman_denominator": relation["denominator"],
                    }
                )
                if relation["value"] is not None:
                    per_case_values.append(float(relation["value"]))

            fold_mse: list[float] = []
            fold_zero: list[float] = []
            sign_hits = 0
            sign_total = 0
            fold_status = "ok"
            for held_out in case_ids:
                train = [row for row in valid if row["case_id"] != held_out]
                test = [row for row in valid if row["case_id"] == held_out]
                fit = _case_balanced_affine_fit(train, feature=feature, target=target)
                if fit is None or not test:
                    fold_status = "unresolved_fold"
                    continue
                intercept, slope = fit
                errors = []
                zero_errors = []
                for row in test:
                    actual = float(row[target])
                    prediction = intercept + slope * float(row[feature])
                    errors.append((prediction - actual) ** 2)
                    zero_errors.append(actual**2)
                    sign_hits += int((prediction > 0.0) == (actual > 0.0))
                    sign_total += 1
                    oof_rows.append(
                        {
                            "feature": feature,
                            "target": target,
                            "case_id": held_out,
                            "block_index": int(row["block_index"]),
                            "feature_value": float(row[feature]),
                            "target_value": actual,
                            "prediction": prediction,
                            "intercept": intercept,
                            "slope": slope,
                        }
                    )
                fold_mse.append(float(np.mean(errors)))
                fold_zero.append(float(np.mean(zero_errors)))
            cv_denominator = float(np.mean(fold_zero)) if fold_zero else 0.0
            if fold_status != "ok" or cv_denominator <= A9_DENOMINATOR_FLOOR**2:
                cv_r2 = None
                cv_status = (
                    fold_status
                    if fold_status != "ok"
                    else "unresolved_small_denominator"
                )
            else:
                cv_r2 = 1.0 - float(np.mean(fold_mse)) / cv_denominator
                cv_status = "ok"

            population_sign = (
                math.copysign(1.0, float(spearman["value"]))
                if spearman["value"] not in (None, 0.0)
                else None
            )
            same_sign = (
                sum(
                    math.copysign(1.0, value) == population_sign
                    for value in per_case_values
                )
                if population_sign is not None
                else 0
            )
            association_rows.append(
                {
                    "feature": feature,
                    "target": target,
                    "total_blocks": len(rows),
                    "valid_blocks": len(valid),
                    "unresolved_blocks": len(rows) - len(valid),
                    "pearson": pearson["value"],
                    "pearson_status": pearson["status"],
                    "pearson_denominator": pearson["denominator"],
                    "spearman": spearman["value"],
                    "spearman_status": spearman["status"],
                    "spearman_denominator": spearman["denominator"],
                    "valid_case_spearman_count": len(per_case_values),
                    "case_spearman_median": (
                        float(np.median(per_case_values)) if per_case_values else None
                    ),
                    "case_spearman_same_population_sign_count": same_sign,
                    "cv_r2_vs_zero": cv_r2,
                    "cv_r2_status": cv_status,
                    "cv_sign_accuracy": (
                        sign_hits / sign_total if sign_total else None
                    ),
                    "cv_prediction_count": sign_total,
                }
            )
    return association_rows, case_rows, oof_rows


def _controls_pass(
    rows: Sequence[Mapping[str, Any]], *, kind: str, limit: float
) -> bool:
    accepted = {"ok", "exact_zero_no_change"}
    return all(
        row["status"] in accepted
        and row["ratio"] is not None
        and float(row["ratio"]) <= limit
        for row in rows
        if row["kind"] == kind
    )


def _control_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    case_ids: Sequence[str],
    raw_prefix: str,
    candidate_prefix: str,
    stage: str,
) -> list[dict[str, Any]]:
    output = []
    scopes = [("population", tuple(case_ids))]
    scopes.extend((case_id, (case_id,)) for case_id in case_ids)
    for scope, cases in scopes:
        for key in FRONT_CONTROL_KEYS:
            raw = _case_first_rms(records, f"{raw_prefix}front::{key}", cases)
            candidate = _case_first_rms(
                records, f"{candidate_prefix}front::{key}", cases
            )
            relation = _control_ratio(raw, candidate)
            output.append(
                {
                    "stage": stage,
                    "kind": "front",
                    "key": key,
                    "scope": scope,
                    "raw_rms": raw,
                    "candidate_rms": candidate,
                    **relation,
                }
            )
        for name in INTEGRAL_COMPONENT_NAMES:
            raw = _case_first_rms(records, f"{raw_prefix}integral::{name}", cases)
            candidate = _case_first_rms(
                records, f"{candidate_prefix}integral::{name}", cases
            )
            relation = _control_ratio(raw, candidate)
            output.append(
                {
                    "stage": stage,
                    "kind": "integral",
                    "key": name,
                    "scope": scope,
                    "raw_rms": raw,
                    "candidate_rms": candidate,
                    **relation,
                }
            )
    return output


def _state_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    case_ids: Sequence[str],
    groups: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    raw_first = _case_first_energy(records, "raw_first_state_energy", case_ids)
    corrected_first = _case_first_energy(
        records, "corrected_first_state_energy", case_ids
    )
    raw_second = _case_first_energy(records, "raw_second_state_energy", case_ids)
    full_second = _case_first_energy(records, "full_second_state_energy", case_ids)
    filtered_second = _case_first_energy(
        records, "filtered_second_state_energy", case_ids
    )
    case_wins = 0
    for case_id in case_ids:
        raw = _case_first_energy(records, "raw_second_state_energy", (case_id,))
        filtered = _case_first_energy(
            records, "filtered_second_state_energy", (case_id,)
        )
        case_wins += filtered < raw
    group_wins = 0
    for group_cases in groups.values():
        raw = _case_first_energy(records, "raw_second_state_energy", group_cases)
        filtered = _case_first_energy(
            records, "filtered_second_state_energy", group_cases
        )
        group_wins += filtered < raw
    oracle_wins = sum(
        float(row["filtered_second_state_energy"])
        < float(row["raw_second_state_energy"])
        for row in records
    )
    return {
        "one_step_sp19_rms_ratio": math.sqrt(corrected_first / raw_first),
        "full_propagated_second_rms_ratio": math.sqrt(full_second / raw_second),
        "response_filtered_second_rms_ratio": math.sqrt(filtered_second / raw_second),
        "case_win_count": case_wins,
        "group_win_count": group_wins,
        "per_sample_oracle_win_count": oracle_wins,
        "per_sample_oracle_win_fraction": oracle_wins / len(records),
    }


def _flatten_metrics(
    prefix: str,
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    runtime: Any,
) -> dict[str, float]:
    values = {
        f"{prefix}state_energy": _state_energy(
            prediction - target,
            volumes=runtime.geometry_by_resolution[NATIVE_RESOLUTION].node_measures,
            component_scale=runtime.normalization.state_scale,
        )
    }
    front = collection._front_errors(prediction, target, runtime)
    values.update(
        {f"{prefix}front::{key}": float(front[key]) for key in FRONT_CONTROL_KEYS}
    )
    integral = collection._physical_integral(
        prediction - target,
        runtime.geometry_by_resolution[NATIVE_RESOLUTION].node_measures,
    )
    values.update(
        {
            f"{prefix}integral::{name}": float(integral[component])
            for component, name in enumerate(INTEGRAL_COMPONENT_NAMES)
        }
    )
    return values


def run_teacher_population(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if args.command == "calibrate":
        preflight = _verify_preflight(args.preflight, args)
        case_ids = CALIBRATION_CASE_IDS
        groups = CALIBRATION_GROUPS
        minimum_case_wins = 14
        minimum_group_wins = 8
        case_gate_name = "minimum_fourteen_case_wins"
        group_gate_name = "minimum_eight_group_wins"
        result_schema = RESULT_SCHEMA
        working_id = WORKING_ID
        result_filename = "calibration.json"
        population_status = "adaptive_open_validation_calibration_only"
        gate_key = "prospective_gate"
        authorization_key = "evaluation_authorized"
        authorization_status = "evaluation_authorized"
        evaluation_executed = False
        rescore = None
        claim_boundary = (
            "Calibration-only teacher-forced two-step dynamic-FV response "
            "filtering. No recurrent, evaluation, bump, cross-family, "
            "conservative-solver, or sealed claim."
        )
    elif args.command == "evaluate":
        preflight = _verify_evaluation_preflight(args.preflight, args)
        case_ids = EVALUATION_CASE_IDS
        groups = EVALUATION_GROUPS
        minimum_case_wins = 4
        minimum_group_wins = 2
        case_gate_name = "minimum_four_case_wins"
        group_gate_name = "minimum_two_group_wins"
        result_schema = EVALUATION_SCHEMA
        working_id = EVALUATION_WORKING_ID
        result_filename = "evaluation.json"
        population_status = "adaptive_open_validation_evaluation"
        gate_key = "teacher_gate"
        authorization_key = "recurrent_pilot_authorized"
        authorization_status = "recurrent_pilot_authorized"
        evaluation_executed = True
        claim_boundary = (
            "Disjoint adaptive-open teacher-forced two-step dynamic-FV result. "
            "Passing authorizes only a separately preregistered recurrent pilot, "
            "not recurrent benefit, bump, cross-family, solver, or sealed claims."
        )
    else:  # pragma: no cover
        raise AssertionError(f"unsupported teacher population: {args.command}")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if args.command == "calibrate":
        runtime, source, p5 = _build_runtime(args)
    else:
        runtime, source, p5, rescore = _build_evaluation_runtime(args)
    try:
        if source != preflight["source_manifest"]:
            raise ValueError("runtime source differs from the frozen preflight")
        started = perf_counter()
        records: list[dict[str, Any]] = []
        reference_rows: list[dict[str, Any]] = []
        execution = {
            "logical_model_calls": 0,
            "actual_forward_passes_including_repeats": 0,
            "total_forward_seconds": 0.0,
            "maximum_repeat_abs_difference": 0.0,
            "maximum_peak_gpu_memory_bytes": 0,
            "logical_calls_per_block": 4,
            "emitted_states_per_block": 2,
        }
        maxima = {
            "common_native_input_abs": 0.0,
            "lookahead_input_abs": 0.0,
            "first_correction_integral_abs": 0.0,
            "filtered_response_integral_abs": 0.0,
            "first_boundary_abs": 0.0,
            "filtered_boundary_abs": 0.0,
            "projection_idempotence_abs": 0.0,
            "call_order_errors": 0,
            "cap_active_count": 0,
        }
        for case_index, case_id in enumerate(case_ids, start=1):
            provenance = collection.family_case_provenance(runtime.manifest, case_id)
            if provenance["split"] != "validation" or case_id not in runtime.store.keys:
                raise ValueError("response-filtered case is outside open validation")
            reference, reference_check = load_resolution_reference(
                runtime.args.family_root,
                runtime.args.multires_reference_root,
                runtime.store,
                runtime.manifest,
                case_id,
                training_resolution=NATIVE_RESOLUTION,
            )
            reference_rows.append(reference_check)
            reference_resolution = tuple(
                int(value) for value in reference["retained_resolution"]
            )
            if reference_resolution != parent.RESOLUTION_CONTRACT.fine:
                raise ValueError("response-filtered block requires retained fine truth")
            print(
                f"response-filtered block: case {case_index}/{len(case_ids)} "
                f"{case_id} calls=0..28",
                flush=True,
            )
            for input_call in INPUT_CALLS:
                states = []
                for offset in (0, 1, 2):
                    state = reference_at_resolution(
                        reference["conservative_states"][(input_call + offset) * 2],
                        reference_resolution=reference_resolution,
                        target_resolution=NATIVE_RESOLUTION,
                    )
                    if state is None:
                        raise ValueError(
                            "two-step response-filtered truth is unavailable"
                        )
                    states.append(np.asarray(state, dtype=np.float64))
                native_input, first_target, second_target = states
                calls: list[tuple[tuple[int, int], np.ndarray]] = []
                repeats = args.repeat_forward if input_call == INPUT_CALLS[0] else 1

                def predictor(
                    resolution,
                    state,
                    call_records=calls,
                    repeat_count=repeats,
                ):
                    call_records.append(
                        (resolution, np.asarray(state, dtype=np.float64).copy())
                    )
                    prediction, timing = predict_resolution_sample(
                        runtime.model,
                        runtime.sample_by_resolution[resolution],
                        state,
                        device=runtime.device,
                        amp="none",
                        repeats=repeat_count,
                    )
                    _account(execution, timing)
                    return prediction

                block = synchronized_response_filtered_block(
                    native_input,
                    contract=parent.RESOLUTION_CONTRACT,
                    projector=runtime.native_projector,
                    predictor=predictor,
                    volumes=runtime.geometry_by_resolution[
                        NATIVE_RESOLUTION
                    ].node_measures,
                    residual_scale=runtime.normalization.residual_scale,
                    state_scale=runtime.normalization.state_scale,
                    active_cells=FROZEN_ACTIVE_CELLS,
                )
                expected_order = [
                    parent.RESOLUTION_CONTRACT.native,
                    parent.RESOLUTION_CONTRACT.fine,
                    parent.RESOLUTION_CONTRACT.native,
                    parent.RESOLUTION_CONTRACT.native,
                ]
                if [resolution for resolution, _ in calls] != expected_order:
                    maxima["call_order_errors"] += 1
                prepared = prepare_common_native_inputs(
                    native_input, contract=parent.RESOLUTION_CONTRACT
                )
                maxima["common_native_input_abs"] = max(
                    maxima["common_native_input_abs"],
                    _maximum_abs(
                        calls[0][1]
                        - prepared.model_inputs[parent.RESOLUTION_CONTRACT.native]
                    ),
                    _maximum_abs(
                        calls[1][1]
                        - prepared.model_inputs[parent.RESOLUTION_CONTRACT.fine]
                    ),
                )
                maxima["lookahead_input_abs"] = max(
                    maxima["lookahead_input_abs"],
                    _maximum_abs(calls[2][1] - block.raw_first_state),
                    _maximum_abs(calls[3][1] - block.corrected_first_state),
                )
                audit = block.projection_audit
                maxima["first_correction_integral_abs"] = max(
                    maxima["first_correction_integral_abs"],
                    audit.maximum_first_correction_integral_abs,
                )
                maxima["filtered_response_integral_abs"] = max(
                    maxima["filtered_response_integral_abs"],
                    audit.maximum_filtered_response_integral_abs,
                )
                maxima["first_boundary_abs"] = max(
                    maxima["first_boundary_abs"], audit.maximum_first_boundary_abs
                )
                maxima["filtered_boundary_abs"] = max(
                    maxima["filtered_boundary_abs"],
                    audit.maximum_filtered_boundary_abs,
                )
                maxima["projection_idempotence_abs"] = max(
                    maxima["projection_idempotence_abs"],
                    audit.maximum_projection_idempotence_abs,
                )
                maxima["cap_active_count"] += int(block.first_audit.cap_active)
                first_admissibility = conservative_admissibility_summary(
                    block.corrected_first_state,
                    gamma=runtime.normalization.gamma,
                )
                second_admissibility = conservative_admissibility_summary(
                    block.filtered_second_state,
                    gamma=runtime.normalization.gamma,
                )
                row = {
                    "case_id": case_id,
                    "group_id": case_id.split("_")[1],
                    "input_call": input_call,
                    **_flatten_metrics(
                        "raw_first_",
                        block.raw_first_state,
                        first_target,
                        runtime=runtime,
                    ),
                    **_flatten_metrics(
                        "corrected_first_",
                        block.corrected_first_state,
                        first_target,
                        runtime=runtime,
                    ),
                    **_flatten_metrics(
                        "raw_second_",
                        block.raw_second_state,
                        second_target,
                        runtime=runtime,
                    ),
                    **_flatten_metrics(
                        "full_second_",
                        block.fully_corrected_second_state,
                        second_target,
                        runtime=runtime,
                    ),
                    **_flatten_metrics(
                        "filtered_second_",
                        block.filtered_second_state,
                        second_target,
                        runtime=runtime,
                    ),
                    "first_finite": bool(
                        np.isfinite(block.corrected_first_state).all()
                    ),
                    "first_admissible": bool(first_admissibility["admissible"]),
                    "first_minimum_density": first_admissibility["minimum_density"],
                    "first_minimum_pressure": first_admissibility["minimum_pressure"],
                    "second_finite": bool(
                        np.isfinite(block.filtered_second_state).all()
                    ),
                    "second_admissible": bool(second_admissibility["admissible"]),
                    "second_minimum_density": second_admissibility["minimum_density"],
                    "second_minimum_pressure": second_admissibility["minimum_pressure"],
                    "first_correction_rms": block.first_audit.correction_rms,
                    "correction_to_native_increment": block.first_audit.correction_to_native_increment,
                    **asdict(audit),
                }
                records.append(row)
            del reference
            if runtime.device.type == "cuda":
                torch.cuda.empty_cache()

        state = _state_summary(records, case_ids=case_ids, groups=groups)
        controls = [
            *_control_rows(
                records,
                case_ids=case_ids,
                raw_prefix="raw_first_",
                candidate_prefix="corrected_first_",
                stage="one_step",
            ),
            *_control_rows(
                records,
                case_ids=case_ids,
                raw_prefix="raw_second_",
                candidate_prefix="filtered_second_",
                stage="two_step",
            ),
        ]
        front_controls_pass = _controls_pass(controls, kind="front", limit=1.05)
        integral_controls_pass = _controls_pass(
            controls, kind="integral", limit=1.0 + 1.0e-8
        )
        all_valid = all(
            row["first_finite"]
            and row["first_admissible"]
            and row["second_finite"]
            and row["second_admissible"]
            for row in records
        )
        execution["wall_seconds"] = perf_counter() - started
        execution["snapshot_count"] = len(records)
        execution["device"] = str(runtime.device)
        execution["amp"] = "none"
        checks = {
            "inventory_and_four_call_order_exact": len(records)
            == len(case_ids) * len(INPUT_CALLS)
            and execution["logical_model_calls"] == 4 * len(records)
            and maxima["call_order_errors"] == 0,
            "live_common_source_and_lookahead_exact": maxima["common_native_input_abs"]
            <= 1.0e-12
            and maxima["lookahead_input_abs"] <= 1.0e-12,
            "two_step_state_rms_ratio_at_most_0p995": state[
                "response_filtered_second_rms_ratio"
            ]
            <= 0.995,
            case_gate_name: state["case_win_count"] >= minimum_case_wins,
            group_gate_name: state["group_win_count"] >= minimum_group_wins,
            "all_front_controls_within_1p05": front_controls_pass,
            "all_integral_controls_numerically_no_harm": integral_controls_pass,
            "all_first_and_second_states_finite_admissible": all_valid,
            "first_and_filtered_integrals_neutral": maxima[
                "first_correction_integral_abs"
            ]
            <= 1.0e-10
            and maxima["filtered_response_integral_abs"] <= 1.0e-10,
            "first_and_filtered_boundaries_zero": maxima["first_boundary_abs"]
            <= 1.0e-12
            and maxima["filtered_boundary_abs"] <= 1.0e-12,
            "projection_idempotence_at_most_1e_10": maxima["projection_idempotence_abs"]
            <= 1.0e-10,
            "correction_cap_inactive": maxima["cap_active_count"] == 0,
            "deterministic_repeat_at_most_1e_6": execution[
                "maximum_repeat_abs_difference"
            ]
            <= 1.0e-6,
            "source_and_reference_inventory_exact": len(reference_rows)
            == len(case_ids),
        }
        qualified = all(checks.values())
        write_csv(output_dir / "sample_records.csv", records)
        write_csv(output_dir / "control_ratios.csv", controls)
        write_csv(output_dir / "reference_checks.csv", reference_rows)
        atomic_write_json(output_dir / "source_manifest.json", source)
        files = (
            "sample_records.csv",
            "control_ratios.csv",
            "reference_checks.csv",
            "source_manifest.json",
        )
        artifact_hashes = sha256_files(files, root=output_dir)
        retained = [
            float(row["retained_response_fraction"])
            for row in records
            if row["retained_response_fraction"] not in (None, "")
        ]
        gate = {
            "status": authorization_status if qualified else "stopped",
            authorization_key: qualified,
            "checks": checks,
        }
        result = {
            "schema": result_schema,
            "working_id": working_id,
            "status": "qualified" if qualified else "stopped",
            "population_status": population_status,
            gate_key: gate,
            "state_summary": state,
            "control_ratios": controls,
            "closure_maxima": maxima,
            "retained_response_fraction": {
                "minimum": min(retained),
                "median": float(np.median(retained)),
                "maximum": max(retained),
            },
            "execution": execution,
            "source_manifest_sha256": sha256_file(output_dir / "source_manifest.json"),
            "source_manifest_payload_sha256": source["payload_sha256"],
            "p5_calibration_sha256": sha256_file(args.p5_calibration),
            "p5_calibration_payload_sha256": p5["payload_sha256"],
            "preflight_sha256": sha256_file(args.preflight),
            "preflight_payload_sha256": preflight["payload_sha256"],
            "artifact_hashes": artifact_hashes,
            "evaluation_executed": evaluation_executed,
            "recurrence_executed": False,
            "sealed_population_opened": False,
            "claim_boundary": claim_boundary,
        }
        if rescore is not None:
            result.update(
                {
                    "qualified_rescore_sha256": sha256_file(args.rescore),
                    "qualified_rescore_payload_sha256": rescore["payload_sha256"],
                }
            )
        payload = with_payload_sha256(result)
        atomic_write_json(output_dir / result_filename, payload)
        return payload, 0 if qualified else 4
    finally:
        collection._close_runtime(runtime)


def _verify_prior_calibration(
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    if sha256_file(path) != EXPECTED_P6_CALIBRATION_SHA256:
        raise ValueError("P6 calibration JSON differs from the registered artifact")
    payload = _read_json(path)
    verify_payload_sha256(payload)
    checks = payload.get("prospective_gate", {}).get("checks", {})
    failed = {key for key, value in checks.items() if not value}
    if (
        payload.get("schema") != RESULT_SCHEMA
        or payload.get("status") != "stopped"
        or failed != {"all_front_controls_within_1p05"}
        or payload.get("evaluation_executed") is not False
        or payload.get("recurrence_executed") is not False
        or payload.get("artifact_hashes") != EXPECTED_P6_ARTIFACT_SHA256
    ):
        raise ValueError("P6 calibration is not the registered scorer-edge result")
    root = path.parent
    for name, expected in EXPECTED_P6_ARTIFACT_SHA256.items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"P6 calibration artifact differs: {name}")
    records = _read_csv(root / "sample_records.csv")
    expected_pairs = {
        (case_id, input_call)
        for case_id in CALIBRATION_CASE_IDS
        for input_call in INPUT_CALLS
    }
    actual_pairs = {(row["case_id"], int(row["input_call"])) for row in records}
    if len(records) != len(expected_pairs) or actual_pairs != expected_pairs:
        raise ValueError("P6 sample inventory is not the registered calibration cell")
    recalculated = _state_summary(
        records, case_ids=CALIBRATION_CASE_IDS, groups=CALIBRATION_GROUPS
    )
    prior_state = payload.get("state_summary", {})
    for key, value in recalculated.items():
        if not math.isclose(float(value), float(prior_state[key]), abs_tol=1.0e-14):
            raise ValueError(f"P6 state summary does not replay from CSV: {key}")
    return payload, records


def run_rescore(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    prior, records = _verify_prior_calibration(args.prior_calibration)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    controls = [
        *_control_rows(
            records,
            case_ids=CALIBRATION_CASE_IDS,
            raw_prefix="raw_first_",
            candidate_prefix="corrected_first_",
            stage="one_step",
        ),
        *_control_rows(
            records,
            case_ids=CALIBRATION_CASE_IDS,
            raw_prefix="raw_second_",
            candidate_prefix="filtered_second_",
            stage="two_step",
        ),
    ]
    old_unresolved = [row for row in prior["control_ratios"] if row["status"] != "ok"]
    zero_edge_exact = len(old_unresolved) == 22 and all(
        float(row["raw_rms"]) <= 1.0e-14 and float(row["candidate_rms"]) <= 1.0e-14
        for row in old_unresolved
    )
    checks = dict(prior["prospective_gate"]["checks"])
    checks["all_front_controls_within_1p05"] = _controls_pass(
        controls, kind="front", limit=1.05
    )
    checks["all_integral_controls_numerically_no_harm"] = _controls_pass(
        controls, kind="integral", limit=1.0 + 1.0e-8
    )
    checks["registered_zero_to_zero_rows_exact"] = zero_edge_exact
    qualified = all(checks.values())
    source = with_payload_sha256(
        {
            "schema": "pcno_response_filtered_block_rescore_source_v1",
            "working_id": RESCORE_WORKING_ID,
            "source_sha256": _source_hashes(),
            "source_status": _git_status_short(SOURCE_PATHS),
            "prior_calibration_sha256": sha256_file(args.prior_calibration),
            "prior_calibration_payload_sha256": prior["payload_sha256"],
            "prior_artifact_sha256": EXPECTED_P6_ARTIFACT_SHA256,
            "semantic_correction": (
                "raw RMS and candidate RMS both at most 1e-14 is exact zero "
                "no-change with ratio 1; nonzero candidate from zero fails"
            ),
            "model_calls": 0,
        }
    )
    write_csv(output_dir / "control_ratios.csv", controls)
    atomic_write_json(output_dir / "source_manifest.json", source)
    artifact_hashes = sha256_files(
        ("control_ratios.csv", "source_manifest.json"), root=output_dir
    )
    payload = with_payload_sha256(
        {
            "schema": RESCORE_SCHEMA,
            "working_id": RESCORE_WORKING_ID,
            "status": "qualified" if qualified else "stopped",
            "population_status": "adaptive_open_validation_calibration_only",
            "prospective_gate": {
                "status": "evaluation_authorized" if qualified else "stopped",
                "evaluation_authorized": qualified,
                "checks": checks,
            },
            "state_summary": _state_summary(
                records, case_ids=CALIBRATION_CASE_IDS, groups=CALIBRATION_GROUPS
            ),
            "control_ratios": controls,
            "prior_calibration_sha256": sha256_file(args.prior_calibration),
            "prior_calibration_payload_sha256": prior["payload_sha256"],
            "prior_artifact_sha256": EXPECTED_P6_ARTIFACT_SHA256,
            "source_manifest_sha256": sha256_file(output_dir / "source_manifest.json"),
            "source_manifest_payload_sha256": source["payload_sha256"],
            "artifact_hashes": artifact_hashes,
            "model_calls": 0,
            "predictions_recomputed": False,
            "thresholds_changed": False,
            "evaluation_executed": False,
            "recurrence_executed": False,
            "sealed_population_opened": False,
            "claim_boundary": (
                "Scorer-only replay of the immutable dynamic-FV calibration "
                "records. No new prediction, evaluation, recurrence, bump, "
                "cross-family, or sealed claim."
            ),
        }
    )
    atomic_write_json(output_dir / "rescore.json", payload)
    return payload, 0 if qualified else 4


@contextmanager
def _sp19_rollout_driver() -> Iterator[None]:
    original = parent.synchronized_fine_discrepancy_step

    def stepper(
        native_state,
        *,
        contract,
        projector,
        predictor,
        policy,
        volumes,
        component_scale,
    ):
        if policy != "rank7_fine_away_half":
            raise ValueError("SP19 adapter received an unexpected driver token")
        return synchronized_sparse_modal_step(
            native_state,
            contract=contract,
            projector=projector,
            predictor=predictor,
            policy="sp19_fine_away_half",
            volumes=volumes,
            component_scale=component_scale,
            active_cells=FROZEN_ACTIVE_CELLS,
        )

    parent.synchronized_fine_discrepancy_step = stepper
    try:
        yield
    finally:
        parent.synchronized_fine_discrepancy_step = original


def _relabel_rollout(rollout: dict[str, Any], policy: str) -> dict[str, Any]:
    rollout["policy"] = policy
    for row in rollout["rows"]:
        row["policy"] = policy
    return rollout


def _block_metric_row(
    runtime: Any,
    *,
    case_id: str,
    policy: str,
    input_call: int,
    previous_state: np.ndarray,
    next_state: np.ndarray,
    current_reference: np.ndarray,
    target: np.ndarray,
    cumulative_defect: np.ndarray,
    intervention: np.ndarray,
    cap_active: bool,
    correction_status: str,
) -> tuple[dict[str, Any], np.ndarray]:
    native_geometry = runtime.geometry_by_resolution[NATIVE_RESOLUTION]
    volumes = np.asarray(native_geometry.node_measures, dtype=np.float64)
    state_scale = np.asarray(runtime.normalization.state_scale, dtype=np.float64)
    residual_scale = np.asarray(runtime.normalization.residual_scale, dtype=np.float64)
    applied_increment = next_state - previous_state
    true_increment = target - current_reference
    defect = applied_increment - true_increment
    cumulative = cumulative_defect + defect
    state_error = next_state - target
    state_parts, _ = runtime.native_projector.split(state_error)
    masks, _, _ = parent.shock_vortex_regions(
        target,
        native_geometry.nodes,
        resolution=NATIVE_RESOLUTION,
        gamma=runtime.normalization.gamma,
    )
    admissibility = conservative_admissibility_summary(
        next_state, gamma=runtime.normalization.gamma
    )
    front = collection._front_errors(next_state, target, runtime)
    integral = collection._physical_integral(state_error, volumes)
    correction_integral = collection._physical_integral(intervention, volumes)
    intervention_rms = weighted_scaled_rms(
        intervention,
        volumes=volumes,
        component_scale=residual_scale,
    )
    increment_rms = weighted_scaled_rms(
        applied_increment,
        volumes=volumes,
        component_scale=residual_scale,
    )
    row = {
        "case_id": case_id,
        "policy": policy,
        "input_call": input_call,
        "output_call": input_call + 1,
        "state_error": weighted_scaled_rms(
            state_error,
            volumes=volumes,
            component_scale=state_scale,
        ),
        "rank8_state_error": weighted_scaled_rms(
            state_parts["parallel"],
            volumes=volumes,
            component_scale=state_scale,
        ),
        "increment_defect": weighted_scaled_rms(
            defect,
            volumes=volumes,
            component_scale=residual_scale,
        ),
        "cumulative_defect": weighted_scaled_rms(
            cumulative,
            volumes=volumes,
            component_scale=residual_scale,
        ),
        "correction_rms": intervention_rms,
        "finite": bool(np.isfinite(next_state).all()),
        "admissible": bool(admissibility["admissible"]),
        "minimum_density": admissibility["minimum_density"],
        "minimum_pressure": admissibility["minimum_pressure"],
        "minimum_internal_energy": admissibility["minimum_internal_energy"],
        "cap_active": cap_active,
        "correction_status": correction_status,
        "correction_to_native_increment": (
            intervention_rms / increment_rms if increment_rms > 1.0e-12 else None
        ),
        "constant_mode_energy_fraction": None,
        "front_position": front["front_position"],
        "front_strength_log_ratio": front["front_strength_log_ratio"],
        "front_thickness_log_ratio": front["front_thickness_log_ratio"],
    }
    for component, name in enumerate(INTEGRAL_COMPONENT_NAMES):
        row[f"component_{name}_state_error"] = weighted_scaled_rms(
            state_error[:, component : component + 1],
            volumes=volumes,
            component_scale=state_scale[component : component + 1],
        )
        row[f"integral_{name}_error"] = float(integral[component])
        row[f"correction_integral_{name}"] = float(correction_integral[component])
    for key, mask_name in (
        ("boundary_state_error", "boundary_le_0.05"),
        ("shock_state_error", "partition_shock"),
        ("vortex_state_error", "partition_vortex"),
        ("smooth_state_error", "partition_smooth"),
    ):
        row[key] = weighted_scaled_rms(
            state_error,
            volumes=volumes,
            component_scale=state_scale,
            mask=masks[mask_name],
        )
    return row, cumulative


def _block_rollout_arm(
    runtime: Any,
    *,
    case_id: str,
    response_mode: str,
    horizon: int,
) -> dict[str, Any]:
    if horizon < 2 or horizon % 2:
        raise ValueError("response-filtered rollout horizon must be positive and even")
    if response_mode not in {"full", "filtered"}:
        raise ValueError("unsupported block response mode")
    policy = f"block_{response_mode}"
    reference, reference_check = load_resolution_reference(
        runtime.args.family_root,
        runtime.args.multires_reference_root,
        runtime.store,
        runtime.manifest,
        case_id,
        training_resolution=NATIVE_RESOLUTION,
    )
    reference_resolution = tuple(
        int(value) for value in reference["retained_resolution"]
    )
    initial = reference_at_resolution(
        reference["conservative_states"][0],
        reference_resolution=reference_resolution,
        target_resolution=NATIVE_RESOLUTION,
    )
    if initial is None:
        raise ValueError("native rollout initial state is unavailable")
    initial = np.asarray(initial, dtype=np.float64)
    volumes = np.asarray(
        runtime.geometry_by_resolution[NATIVE_RESOLUTION].node_measures,
        dtype=np.float64,
    )
    execution = {
        "logical_model_calls": 0,
        "actual_forward_passes": 0,
        "total_forward_seconds": 0.0,
        "maximum_peak_gpu_memory_bytes": 0,
        "native_logical_calls": 0,
        "fine_logical_calls": 0,
        "native_actual_forward_passes": 0,
        "fine_actual_forward_passes": 0,
        "native_forward_seconds": 0.0,
        "fine_forward_seconds": 0.0,
    }
    maxima = {
        "call_order_errors": 0,
        "common_native_input_abs": 0.0,
        "lookahead_input_abs": 0.0,
        "accepted_recurrence_abs": 0.0,
        "first_correction_integral_abs": 0.0,
        "filtered_response_integral_abs": 0.0,
        "first_boundary_abs": 0.0,
        "filtered_boundary_abs": 0.0,
        "projection_idempotence_abs": 0.0,
        "cap_active_count": 0,
    }
    expected_input = np.array(initial, copy=True)
    started = perf_counter()

    def build_block(value: np.ndarray):
        nonlocal expected_input
        maxima["accepted_recurrence_abs"] = max(
            maxima["accepted_recurrence_abs"], _maximum_abs(value - expected_input)
        )
        calls: list[tuple[tuple[int, int], np.ndarray]] = []

        def predictor(resolution, state, call_records=calls):
            call_records.append(
                (resolution, np.asarray(state, dtype=np.float64).copy())
            )
            prediction, timing = predict_resolution_sample(
                runtime.model,
                runtime.sample_by_resolution[resolution],
                state,
                device=runtime.device,
                amp="none",
                repeats=1,
            )
            parent._account_execution(execution, timing, resolution=resolution)
            return prediction

        block = synchronized_response_filtered_block(
            value,
            contract=parent.RESOLUTION_CONTRACT,
            projector=runtime.native_projector,
            predictor=predictor,
            volumes=volumes,
            residual_scale=runtime.normalization.residual_scale,
            state_scale=runtime.normalization.state_scale,
            active_cells=FROZEN_ACTIVE_CELLS,
        )
        expected_order = [
            parent.RESOLUTION_CONTRACT.native,
            parent.RESOLUTION_CONTRACT.fine,
            parent.RESOLUTION_CONTRACT.native,
            parent.RESOLUTION_CONTRACT.native,
        ]
        if [resolution for resolution, _ in calls] != expected_order:
            maxima["call_order_errors"] += 1
        prepared = prepare_common_native_inputs(
            value, contract=parent.RESOLUTION_CONTRACT
        )
        maxima["common_native_input_abs"] = max(
            maxima["common_native_input_abs"],
            _maximum_abs(
                calls[0][1] - prepared.model_inputs[parent.RESOLUTION_CONTRACT.native]
            ),
            _maximum_abs(
                calls[1][1] - prepared.model_inputs[parent.RESOLUTION_CONTRACT.fine]
            ),
        )
        maxima["lookahead_input_abs"] = max(
            maxima["lookahead_input_abs"],
            _maximum_abs(calls[2][1] - block.raw_first_state),
            _maximum_abs(calls[3][1] - block.corrected_first_state),
        )
        audit = block.projection_audit
        maxima["first_correction_integral_abs"] = max(
            maxima["first_correction_integral_abs"],
            audit.maximum_first_correction_integral_abs,
        )
        maxima["filtered_response_integral_abs"] = max(
            maxima["filtered_response_integral_abs"],
            audit.maximum_filtered_response_integral_abs,
        )
        maxima["first_boundary_abs"] = max(
            maxima["first_boundary_abs"], audit.maximum_first_boundary_abs
        )
        maxima["filtered_boundary_abs"] = max(
            maxima["filtered_boundary_abs"], audit.maximum_filtered_boundary_abs
        )
        maxima["projection_idempotence_abs"] = max(
            maxima["projection_idempotence_abs"],
            audit.maximum_projection_idempotence_abs,
        )
        maxima["cap_active_count"] += int(block.first_audit.cap_active)
        expected_input = np.array(
            block.fully_corrected_second_state
            if response_mode == "full"
            else block.filtered_second_state,
            copy=True,
        )
        return block

    trajectory = recurrent_response_filtered_blocks(
        initial,
        block_count=horizon // 2,
        block_builder=build_block,
        response_mode=response_mode,
    )
    rows = []
    cumulative_defect = np.zeros_like(initial)
    first_invalid_call = None
    first_nonfinite_call = None
    for input_call in range(horizon):
        current_reference = reference_at_resolution(
            reference["conservative_states"][input_call * 2],
            reference_resolution=reference_resolution,
            target_resolution=NATIVE_RESOLUTION,
        )
        target = reference_at_resolution(
            reference["conservative_states"][(input_call + 1) * 2],
            reference_resolution=reference_resolution,
            target_resolution=NATIVE_RESOLUTION,
        )
        if current_reference is None or target is None:
            raise ValueError("native rollout target is unavailable")
        block = trajectory.blocks[input_call // 2]
        first_in_block = input_call % 2 == 0
        intervention = (
            block.first_correction
            if first_in_block
            else (
                block.full_response
                if response_mode == "full"
                else block.filtered_response
            )
        )
        row, cumulative_defect = _block_metric_row(
            runtime,
            case_id=case_id,
            policy=policy,
            input_call=input_call,
            previous_state=trajectory.states[input_call],
            next_state=trajectory.states[input_call + 1],
            current_reference=np.asarray(current_reference, dtype=np.float64),
            target=np.asarray(target, dtype=np.float64),
            cumulative_defect=cumulative_defect,
            intervention=np.asarray(intervention, dtype=np.float64),
            cap_active=bool(block.first_audit.cap_active) if first_in_block else False,
            correction_status=(
                block.first_audit.status
                if first_in_block
                else f"{response_mode}_block_response"
            ),
        )
        rows.append(row)
        if not row["finite"] and first_nonfinite_call is None:
            first_nonfinite_call = input_call
        if (not row["finite"] or not row["admissible"]) and first_invalid_call is None:
            first_invalid_call = input_call
    execution["wall_seconds"] = perf_counter() - started
    execution["completed_calls"] = len(rows)
    return {
        "case_id": case_id,
        "policy": policy,
        "rows": rows,
        "states": list(trajectory.states),
        "execution": execution,
        "maxima": maxima,
        "first_invalid_call": first_invalid_call,
        "first_nonfinite_call": first_nonfinite_call,
        "reference_check": reference_check,
    }


def _comparison_with_raw(
    raw_rollouts: Sequence[Mapping[str, Any]],
    candidate_rollouts: Sequence[Mapping[str, Any]],
    *,
    candidate: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = [
        *(row for rollout in raw_rollouts for row in rollout["rows"]),
        *(row for rollout in candidate_rollouts for row in rollout["rows"]),
    ]
    case_rows, controls, population = parent._paired_rollout_controls(rows)
    return (
        [{"candidate": candidate, **row} for row in case_rows],
        [{"candidate": candidate, **row} for row in controls],
        population,
    )


def _policy_cost(
    rollouts: Sequence[Mapping[str, Any]], *, policy: str
) -> dict[str, Any]:
    selected = [row for row in rollouts if row["policy"] == policy]
    rows = [row for rollout in selected for row in rollout["rows"]]
    endpoint = [row for row in rows if int(row["input_call"]) == 29]
    return {
        "logical_model_calls": sum(
            row["execution"]["logical_model_calls"] for row in selected
        ),
        "native_logical_calls": sum(
            row["execution"]["native_logical_calls"] for row in selected
        ),
        "fine_logical_calls": sum(
            row["execution"]["fine_logical_calls"] for row in selected
        ),
        "actual_forward_passes": sum(
            row["execution"]["actual_forward_passes"] for row in selected
        ),
        "total_forward_seconds": sum(
            row["execution"]["total_forward_seconds"] for row in selected
        ),
        "wall_seconds": sum(row["execution"]["wall_seconds"] for row in selected),
        "maximum_peak_gpu_memory_bytes": max(
            row["execution"]["maximum_peak_gpu_memory_bytes"] for row in selected
        ),
        "aggregate_state_rms": math.sqrt(
            np.mean([float(row["state_error"]) ** 2 for row in rows])
        ),
        "endpoint_state_rms": math.sqrt(
            np.mean([float(row["state_error"]) ** 2 for row in endpoint])
        ),
    }


def _pareto_dominated(
    candidate: Mapping[str, Any], comparator: Mapping[str, Any]
) -> bool:
    keys = (
        "aggregate_state_rms",
        "endpoint_state_rms",
        "wall_seconds",
        "maximum_peak_gpu_memory_bytes",
    )
    weak = all(float(comparator[key]) <= float(candidate[key]) for key in keys)
    strict = any(float(comparator[key]) < float(candidate[key]) for key in keys)
    return weak and strict


def run_rollout(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    preflight = _verify_rollout_preflight(args.preflight, args)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    runtime, source, _, _, evaluation = _build_rollout_runtime(args)
    try:
        if source != preflight["source_manifest"]:
            raise ValueError("rollout runtime source differs from frozen preflight")
        prefix_a = _block_rollout_arm(
            runtime,
            case_id=EVALUATION_CASE_IDS[0],
            response_mode="filtered",
            horizon=2,
        )
        prefix_b = _block_rollout_arm(
            runtime,
            case_id=EVALUATION_CASE_IDS[0],
            response_mode="filtered",
            horizon=2,
        )
        prefix_abs = max(
            _maximum_abs(left - right)
            for left, right in zip(prefix_a["states"], prefix_b["states"], strict=True)
        )
        rollouts: list[dict[str, Any]] = []
        for case_id in EVALUATION_CASE_IDS:
            print(f"response-filtered rollout: {case_id} raw", flush=True)
            rollouts.append(
                parent._rollout_arm(runtime, case_id=case_id, policy="zero", horizon=30)
            )
            print(f"response-filtered rollout: {case_id} always-SP19", flush=True)
            with _sp19_rollout_driver():
                always = parent._rollout_arm(
                    runtime,
                    case_id=case_id,
                    policy="rank7_fine_away_half",
                    horizon=30,
                )
            rollouts.append(_relabel_rollout(always, "sp19_always"))
            print(f"response-filtered rollout: {case_id} block-full", flush=True)
            rollouts.append(
                _block_rollout_arm(
                    runtime, case_id=case_id, response_mode="full", horizon=30
                )
            )
            print(f"response-filtered rollout: {case_id} block-filtered", flush=True)
            rollouts.append(
                _block_rollout_arm(
                    runtime, case_id=case_id, response_mode="filtered", horizon=30
                )
            )
            if runtime.device.type == "cuda":
                torch.cuda.empty_cache()

        by_policy = {
            policy: [row for row in rollouts if row["policy"] == policy]
            for policy in ("zero", "sp19_always", "block_full", "block_filtered")
        }
        comparisons = {}
        all_case_rows = []
        all_control_rows = []
        for candidate in ("sp19_always", "block_full", "block_filtered"):
            case_rows, controls, population = _comparison_with_raw(
                by_policy["zero"], by_policy[candidate], candidate=candidate
            )
            comparisons[candidate] = population
            all_case_rows.extend(case_rows)
            all_control_rows.extend(controls)
        costs = {policy: _policy_cost(rollouts, policy=policy) for policy in by_policy}
        filtered = comparisons["block_filtered"]
        filtered_controls = [
            row for row in all_control_rows if row["candidate"] == "block_filtered"
        ]
        complete = all(
            row["execution"]["completed_calls"] == 30
            and row["first_invalid_call"] is None
            and row["first_nonfinite_call"] is None
            for row in rollouts
        )
        block_contract = all(
            row["execution"]["logical_model_calls"] == 60
            and row["execution"]["native_logical_calls"] == 45
            and row["execution"]["fine_logical_calls"] == 15
            and row["maxima"]["call_order_errors"] == 0
            and row["maxima"]["common_native_input_abs"] <= 1.0e-12
            and row["maxima"]["lookahead_input_abs"] <= 1.0e-12
            and row["maxima"]["accepted_recurrence_abs"] <= 1.0e-12
            for policy in ("block_full", "block_filtered")
            for row in by_policy[policy]
        )
        filtered_projection = all(
            row["maxima"]["first_correction_integral_abs"] <= 1.0e-10
            and row["maxima"]["filtered_response_integral_abs"] <= 1.0e-10
            and row["maxima"]["first_boundary_abs"] <= 1.0e-12
            and row["maxima"]["filtered_boundary_abs"] <= 1.0e-12
            and row["maxima"]["projection_idempotence_abs"] <= 1.0e-10
            and row["maxima"]["cap_active_count"] == 0
            for row in by_policy["block_filtered"]
        )
        comparator_contract = all(
            row["execution"]["native_logical_calls"] == 30
            and row["execution"]["fine_logical_calls"] == 0
            for row in by_policy["zero"]
        ) and all(
            row["execution"]["native_logical_calls"] == 30
            and row["execution"]["fine_logical_calls"] == 30
            for row in by_policy["sp19_always"]
        )
        checks = {
            "a3_teacher_gate_passed": evaluation["teacher_gate"][
                "recurrent_pilot_authorized"
            ]
            is True,
            "all_four_arms_complete_finite_admissible": complete,
            "median_endpoint_ratio_at_most_0p98": filtered[
                "median_endpoint_state_ratio"
            ]
            <= 0.98,
            "minimum_four_endpoint_wins": filtered["endpoint_win_count"] >= 4,
            "maximum_endpoint_ratio_at_most_1p02": filtered[
                "maximum_endpoint_state_ratio"
            ]
            <= 1.02,
            "aggregate_state_rms_ratio_at_most_0p99": filtered[
                "aggregate_state_rms_ratio"
            ]
            <= 0.99,
            "increment_and_cumulative_ratios_at_most_one": filtered[
                "aggregate_increment_defect_rms_ratio"
            ]
            <= 1.0
            and filtered["median_endpoint_cumulative_defect_ratio"] <= 1.0,
            "all_filtered_controls_no_harm": bool(filtered_controls)
            and all(parent._control_passed(row) for row in filtered_controls),
            "block_call_common_source_and_recurrence_contract": block_contract,
            "filtered_projection_closure": filtered_projection,
            "raw_and_always_call_contract": comparator_contract,
            "deterministic_two_step_prefix_exact": prefix_abs == 0.0,
            "source_reference_and_artifact_inventory_exact": source
            == preflight["source_manifest"],
        }
        qualified = all(checks.values())
        rows = [row for rollout in rollouts for row in rollout["rows"]]
        write_csv(output_dir / "rollout_call_metrics.csv", rows)
        write_csv(output_dir / "rollout_case_metrics.csv", all_case_rows)
        write_csv(output_dir / "rollout_controls.csv", all_control_rows)
        write_csv(
            output_dir / "rollout_execution.csv",
            [
                {
                    "case_id": row["case_id"],
                    "policy": row["policy"],
                    **row["execution"],
                }
                for row in rollouts
            ],
        )
        write_csv(
            output_dir / "rollout_closure.csv",
            [
                {
                    "case_id": row["case_id"],
                    "policy": row["policy"],
                    **row["maxima"],
                }
                for row in rollouts
            ],
        )
        write_csv(
            output_dir / "reference_checks.csv",
            [
                {
                    "case_id": row["case_id"],
                    "policy": row["policy"],
                    **row["reference_check"],
                }
                for row in rollouts
            ],
        )
        atomic_write_json(output_dir / "source_manifest.json", source)
        files = (
            "rollout_call_metrics.csv",
            "rollout_case_metrics.csv",
            "rollout_controls.csv",
            "rollout_execution.csv",
            "rollout_closure.csv",
            "reference_checks.csv",
            "source_manifest.json",
        )
        artifact_hashes = sha256_files(files, root=output_dir)
        costs["block_filtered"]["pareto_dominated_by_sp19_always"] = _pareto_dominated(
            costs["block_filtered"], costs["sp19_always"]
        )
        costs["block_filtered"]["pareto_dominated_by_block_full"] = _pareto_dominated(
            costs["block_filtered"], costs["block_full"]
        )
        payload = with_payload_sha256(
            {
                "schema": ROLLOUT_SCHEMA,
                "working_id": ROLLOUT_WORKING_ID,
                "status": "qualified" if qualified else "stopped",
                "population_status": "adaptive_open_validation_recurrent_pilot",
                "recurrent_gate": {
                    "status": "qualified" if qualified else "stopped",
                    "claim_authorized": qualified,
                    "checks": checks,
                },
                "comparisons_vs_raw": comparisons,
                "failed_controls": {
                    candidate: [
                        row
                        for row in all_control_rows
                        if row["candidate"] == candidate
                        and not parent._control_passed(row)
                    ]
                    for candidate in (
                        "sp19_always",
                        "block_full",
                        "block_filtered",
                    )
                },
                "cost": costs,
                "deterministic_prefix_max_abs": prefix_abs,
                "qualified_evaluation_sha256": sha256_file(args.evaluation),
                "qualified_evaluation_payload_sha256": evaluation["payload_sha256"],
                "preflight_sha256": sha256_file(args.preflight),
                "preflight_payload_sha256": preflight["payload_sha256"],
                "source_manifest_sha256": sha256_file(
                    output_dir / "source_manifest.json"
                ),
                "source_manifest_payload_sha256": source["payload_sha256"],
                "artifact_hashes": artifact_hashes,
                "recurrence_executed": True,
                "sealed_population_opened": False,
                "claim_boundary": (
                    "Adaptive-open H30 dynamic-FV recurrent pilot only. No "
                    "independent, strength-OOD, test, bump, cross-family, direct "
                    "off-grid, solver-conservation, convergence, or asymptotic claim."
                ),
            }
        )
        atomic_write_json(output_dir / "rollout.json", payload)
        return payload, 0 if qualified else 4
    finally:
        collection._close_runtime(runtime)


def _paired_rollout_controls_for_cases(
    rows: Sequence[Mapping[str, Any]],
    *,
    case_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    case_ids = tuple(case_ids)
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError("paired rollout case_ids must be nonempty and unique")
    raw = [row for row in rows if row["policy"] == "zero"]
    corrected = [row for row in rows if row["policy"] != "zero"]
    raw_lookup = {(row["case_id"], row["input_call"]): row for row in raw}
    corrected_lookup = {(row["case_id"], row["input_call"]): row for row in corrected}
    if set(raw_lookup) != set(corrected_lookup):
        raise ValueError("raw and corrected rollout inventories differ")
    expected_inventory = {
        (case_id, input_call)
        for case_id in case_ids
        for input_call in parent.ALL_INPUT_CALLS
    }
    if set(raw_lookup) != expected_inventory:
        raise ValueError("paired rollout inventory differs from the requested cases")

    def rms_ratio(
        key: str,
        raw_rows: Sequence[Mapping[str, Any]],
        corrected_rows: Sequence[Mapping[str, Any]],
    ) -> float:
        zero = math.sqrt(np.mean([float(row[key]) ** 2 for row in raw_rows]))
        candidate = math.sqrt(np.mean([float(row[key]) ** 2 for row in corrected_rows]))
        if zero <= parent.DENOMINATOR_FLOOR:
            return 1.0 if candidate <= parent.DENOMINATOR_FLOOR else float("inf")
        return candidate / zero

    case_rows = []
    for case_id in case_ids:
        raw_case = [raw_lookup[(case_id, call)] for call in parent.ALL_INPUT_CALLS]
        corrected_case = [
            corrected_lookup[(case_id, call)] for call in parent.ALL_INPUT_CALLS
        ]
        raw_endpoint = raw_case[-1]
        corrected_endpoint = corrected_case[-1]
        case_rows.append(
            {
                "case_id": case_id,
                "endpoint_state_ratio": corrected_endpoint["state_error"]
                / raw_endpoint["state_error"],
                "trajectory_state_rms_ratio": rms_ratio(
                    "state_error", raw_case, corrected_case
                ),
                "increment_defect_rms_ratio": rms_ratio(
                    "increment_defect", raw_case, corrected_case
                ),
                "endpoint_cumulative_defect_ratio": corrected_endpoint[
                    "cumulative_defect"
                ]
                / raw_endpoint["cumulative_defect"],
                "raw_endpoint_state_error": raw_endpoint["state_error"],
                "corrected_endpoint_state_error": corrected_endpoint["state_error"],
            }
        )

    controls: list[dict[str, Any]] = []
    endpoint_keys = (
        "rank8_state_error",
        "boundary_state_error",
        "shock_state_error",
        "vortex_state_error",
        "smooth_state_error",
        "front_position",
        "front_strength_log_ratio",
        "front_thickness_log_ratio",
        *(f"component_{name}_state_error" for name in INTEGRAL_COMPONENT_NAMES),
    )
    for key in endpoint_keys:
        for scope in ("population", *case_ids):
            cases = case_ids if scope == "population" else (scope,)
            trajectory_rows = [
                {
                    "zero_error": raw_lookup[(case_id, input_call)][key],
                    "corrected_error": corrected_lookup[(case_id, input_call)][key],
                }
                for case_id in cases
                for input_call in parent.ALL_INPUT_CALLS
            ]
            endpoint_rows = [
                {
                    "zero_error": raw_lookup[(case_id, parent.ALL_INPUT_CALLS[-1])][
                        key
                    ],
                    "corrected_error": corrected_lookup[
                        (case_id, parent.ALL_INPUT_CALLS[-1])
                    ][key],
                }
                for case_id in cases
            ]
            controls.append(
                parent._rms_control(
                    key=f"trajectory::{key}", scope=scope, rows=trajectory_rows
                )
            )
            controls.append(
                parent._rms_control(
                    key=f"endpoint::{key}", scope=scope, rows=endpoint_rows
                )
            )
    for name in INTEGRAL_COMPONENT_NAMES:
        key = f"integral_{name}_error"
        for scope in ("population", *case_ids):
            cases = case_ids if scope == "population" else (scope,)
            all_rows = [
                {
                    "zero_error": raw_lookup[(case_id, input_call)][key],
                    "corrected_error": corrected_lookup[(case_id, input_call)][key],
                }
                for case_id in cases
                for input_call in parent.ALL_INPUT_CALLS
            ]
            endpoint_rows = [
                {
                    "zero_error": raw_lookup[(case_id, parent.ALL_INPUT_CALLS[-1])][
                        key
                    ],
                    "corrected_error": corrected_lookup[
                        (case_id, parent.ALL_INPUT_CALLS[-1])
                    ][key],
                }
                for case_id in cases
            ]
            controls.append(
                parent._rms_control(
                    key=f"integral_rms::{name}", scope=scope, rows=all_rows
                )
            )
            controls.append(
                parent._rms_control(
                    key=f"integral_endpoint::{name}",
                    scope=scope,
                    rows=endpoint_rows,
                )
            )
    population = {
        "median_endpoint_state_ratio": float(
            np.median([row["endpoint_state_ratio"] for row in case_rows])
        ),
        "maximum_endpoint_state_ratio": max(
            row["endpoint_state_ratio"] for row in case_rows
        ),
        "endpoint_win_count": sum(
            row["endpoint_state_ratio"] <= 1.0 for row in case_rows
        ),
        "aggregate_state_rms_ratio": math.sqrt(
            sum(float(row["state_error"]) ** 2 for row in corrected)
            / sum(float(row["state_error"]) ** 2 for row in raw)
        ),
        "aggregate_increment_defect_rms_ratio": math.sqrt(
            sum(float(row["increment_defect"]) ** 2 for row in corrected)
            / sum(float(row["increment_defect"]) ** 2 for row in raw)
        ),
        "median_endpoint_cumulative_defect_ratio": float(
            np.median([row["endpoint_cumulative_defect_ratio"] for row in case_rows])
        ),
    }
    return case_rows, controls, population


def _explicit_shadow_call_checks(
    execution_rows: Sequence[Mapping[str, Any]],
    closure_rows: Sequence[Mapping[str, Any]],
    *,
    case_ids: Sequence[str] = EVALUATION_CASE_IDS,
) -> dict[str, bool]:
    case_ids = tuple(case_ids)
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError("shadow call-check case_ids must be nonempty and unique")
    expected_cases = set(case_ids)
    execution = {row["case_id"]: row for row in execution_rows}
    closure = {row["case_id"]: row for row in closure_rows}
    inventory_exact = (
        len(execution_rows) == len(case_ids)
        and len(closure_rows) == len(case_ids)
        and len(execution) == len(case_ids)
        and len(closure) == len(case_ids)
        and set(execution) == expected_cases
        and set(closure) == expected_cases
    )
    if not inventory_exact:
        return {
            "shadow_call_inventory_exact": False,
            "shadow_call_counts_exact": False,
            "shadow_call_order_and_recurrence_exact": False,
            "accepted_post_fp32_floor_at_most_1e_6": False,
            "candidate_common_source_exact": False,
            "candidate_lookahead_inputs_exact": False,
        }
    return {
        "shadow_call_inventory_exact": True,
        "shadow_call_counts_exact": all(
            int(execution[case_id]["logical_model_calls"]) == 90
            and int(execution[case_id]["native_logical_calls"]) == 75
            and int(execution[case_id]["fine_logical_calls"]) == 15
            and int(execution[case_id]["shadow_native_logical_calls"]) == 30
            and int(execution[case_id]["candidate_native_logical_calls"]) == 45
            and int(execution[case_id]["candidate_fine_logical_calls"]) == 15
            and int(execution[case_id]["completed_calls"]) == 30
            for case_id in case_ids
        ),
        "shadow_call_order_and_recurrence_exact": all(
            int(float(closure[case_id]["call_order_errors"])) == 0
            and float(closure[case_id]["shadow_recurrence_abs"]) <= 1.0e-12
            for case_id in case_ids
        ),
        "accepted_post_fp32_floor_at_most_1e_6": all(
            float(closure[case_id]["accepted_recurrence_abs"]) <= 1.0e-6
            for case_id in case_ids
        ),
        "candidate_common_source_exact": all(
            float(closure[case_id]["candidate_common_native_input_abs"]) <= 1.0e-12
            for case_id in case_ids
        ),
        "candidate_lookahead_inputs_exact": all(
            float(closure[case_id]["candidate_lookahead_input_abs"]) <= 1.0e-12
            for case_id in case_ids
        ),
    }


def _rescore_shadow_candidate_call_checks(
    prior_checks: Mapping[str, Any],
    execution_rows: Sequence[Mapping[str, Any]],
    closure_rows: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    checks = {str(key): bool(value) for key, value in prior_checks.items()}
    failed = {key for key, value in checks.items() if not value}
    if failed != A9_MISWIRED_CALL_CHECKS:
        raise ValueError(
            "A9 rescore input does not have the exact call-inventory failure"
        )
    checks.update(
        _explicit_shadow_call_checks(
            execution_rows,
            closure_rows,
            case_ids=PROSPECTIVE_STRUCTURAL_CASE_IDS,
        )
    )
    return checks


def _structural_gate_call_checks(
    rollouts: Sequence[Mapping[str, Any]],
    *,
    case_ids: Sequence[str] = STRENGTH_OOD_CASE_IDS,
) -> dict[str, bool]:
    expected_cases = tuple(case_ids)
    by_case = {str(row["case_id"]): row for row in rollouts}
    inventory_exact = (
        len(expected_cases) == len(set(expected_cases))
        and len(rollouts) == len(expected_cases)
        and len(by_case) == len(expected_cases)
        and set(by_case) == set(expected_cases)
    )
    if not inventory_exact:
        return {
            "structural_gate_inventory_exact": False,
            "structural_gate_decisions_target_free_and_consistent": False,
            "structural_gate_call_counts_exact": False,
            "structural_gate_fallback_recurrence_exact": False,
            "structural_gate_common_source_and_lookahead_exact": False,
        }

    decisions_ok = True
    counts_ok = True
    fallback_ok = True
    closure_ok = True
    for case_id in expected_cases:
        rollout = by_case[case_id]
        descriptor = rollout.get("position_descriptor")
        trusted = bool(rollout.get("position_trusted"))
        front_rows = list(rollout.get("front_audits", ()))
        veto_call = rollout.get("front_branch_veto_call")
        expected_trusted = bool(
            descriptor
            and descriptor.get("status") == "ok"
            and descriptor.get("normalized_wall_distance") is not None
            and float(descriptor["normalized_wall_distance"])
            <= FROZEN_POSITION_TRUST_THRESHOLD
        )
        changed = [row for row in front_rows if bool(row["branch_changed"])]
        decisions_ok = decisions_ok and trusted == expected_trusted
        decisions_ok = decisions_ok and (not front_rows if not trusted else True)
        decisions_ok = decisions_ok and len(changed) <= 1
        decisions_ok = decisions_ok and (
            (veto_call is None and not changed)
            or (
                len(changed) == 1
                and int(veto_call) == int(changed[0]["first_output_call"])
                and changed[0] is front_rows[-1]
            )
        )

        candidate_blocks = len(front_rows)
        execution = rollout["execution"]
        counts_ok = counts_ok and all(
            (
                int(execution["logical_model_calls"]) == 30 + 4 * candidate_blocks,
                int(execution["native_logical_calls"]) == 30 + 3 * candidate_blocks,
                int(execution["fine_logical_calls"]) == candidate_blocks,
                int(execution["shadow_native_logical_calls"]) == 30,
                int(execution["candidate_native_logical_calls"])
                == 3 * candidate_blocks,
                int(execution["candidate_fine_logical_calls"]) == candidate_blocks,
                int(execution["completed_calls"]) == 30,
            )
        )

        fallback_start = 0 if not trusted else veto_call
        if fallback_start is not None:
            fallback_ok = fallback_ok and all(
                np.array_equal(left, right)
                for left, right in zip(
                    rollout["accepted_states"][int(fallback_start) :],
                    rollout["shadow_states"][int(fallback_start) :],
                    strict=True,
                )
            )
        maxima = rollout["maxima"]
        closure_ok = closure_ok and (
            int(maxima["call_order_errors"]) == 0
            and float(maxima["shadow_recurrence_abs"]) <= 1.0e-12
            and float(maxima["accepted_recurrence_abs"]) <= 1.0e-6
            and float(maxima["candidate_common_native_input_abs"]) <= 1.0e-12
            and float(maxima["candidate_lookahead_input_abs"]) <= 1.0e-12
        )
    return {
        "structural_gate_inventory_exact": True,
        "structural_gate_decisions_target_free_and_consistent": decisions_ok,
        "structural_gate_call_counts_exact": counts_ok,
        "structural_gate_fallback_recurrence_exact": fallback_ok,
        "structural_gate_common_source_and_lookahead_exact": closure_ok,
    }


def _warm_start_call_checks(
    rollouts: Sequence[Mapping[str, Any]],
    *,
    case_ids: Sequence[str] = PROSPECTIVE_STRUCTURAL_CASE_IDS,
) -> dict[str, bool]:
    expected_cases = tuple(case_ids)
    by_case = {str(row["case_id"]): row for row in rollouts}
    inventory_exact = (
        len(expected_cases) == len(set(expected_cases))
        and len(rollouts) == len(expected_cases)
        and len(by_case) == len(expected_cases)
        and set(by_case) == set(expected_cases)
    )
    if not inventory_exact:
        return {
            "warm_start_inventory_exact": False,
            "warm_start_decisions_target_free_and_consistent": False,
            "warm_start_call_counts_exact": False,
            "warm_start_coast_and_fallback_recurrence_exact": False,
            "warm_start_common_source_and_lookahead_exact": False,
        }

    decisions_ok = True
    counts_ok = True
    recurrence_ok = True
    closure_ok = True
    for case_id in expected_cases:
        rollout = by_case[case_id]
        descriptor = rollout.get("position_descriptor")
        trusted = bool(rollout.get("position_trusted"))
        expected_trusted = bool(
            descriptor
            and descriptor.get("status") == "ok"
            and descriptor.get("normalized_wall_distance") is not None
            and float(descriptor["normalized_wall_distance"])
            <= FROZEN_POSITION_TRUST_THRESHOLD
        )
        front_rows = list(rollout.get("front_audits", ()))
        changed = [row for row in front_rows if bool(row["branch_changed"])]
        veto_call = rollout.get("front_branch_veto_call")
        decisions_ok = decisions_ok and all(
            (
                rollout.get("warm_start_blocks") == WARM_START_BLOCKS,
                trusted == expected_trusted,
                len(changed) <= 1,
                (veto_call is None and not changed)
                or (
                    len(changed) == 1
                    and int(veto_call) == int(changed[0]["first_output_call"])
                    and changed[0] is front_rows[-1]
                ),
            )
        )
        candidate_blocks = len(front_rows)
        if veto_call is None:
            expected_candidate_blocks = 15 if trusted else WARM_START_BLOCKS
            expected_coast_blocks = 0 if trusted else 15 - WARM_START_BLOCKS
        else:
            expected_candidate_blocks = (int(veto_call) + 1) // 2
            expected_coast_blocks = 0
        execution = rollout["execution"]
        counts_ok = counts_ok and all(
            (
                candidate_blocks == expected_candidate_blocks,
                int(execution["logical_model_calls"])
                == 30 + 4 * candidate_blocks + 2 * expected_coast_blocks,
                int(execution["native_logical_calls"])
                == 30 + 3 * candidate_blocks + 2 * expected_coast_blocks,
                int(execution["fine_logical_calls"]) == candidate_blocks,
                int(execution["shadow_native_logical_calls"]) == 30,
                int(execution["candidate_native_logical_calls"])
                == 3 * candidate_blocks,
                int(execution["candidate_fine_logical_calls"]) == candidate_blocks,
                int(execution["accepted_coast_native_logical_calls"])
                == 2 * expected_coast_blocks,
                int(execution["completed_calls"]) == 30,
            )
        )
        accepted_statuses = [
            row["correction_status"] for row in rollout["accepted_rows"]
        ]
        if veto_call is not None:
            recurrence_ok = recurrence_ok and all(
                np.array_equal(left, right)
                for left, right in zip(
                    rollout["accepted_states"][int(veto_call) :],
                    rollout["shadow_states"][int(veto_call) :],
                    strict=True,
                )
            )
        elif not trusted:
            recurrence_ok = recurrence_ok and all(
                status == "accepted_native_coast"
                for status in accepted_statuses[2 * WARM_START_BLOCKS :]
            )
            recurrence_ok = recurrence_ok and any(
                not np.array_equal(left, right)
                for left, right in zip(
                    rollout["accepted_states"][2 * WARM_START_BLOCKS :],
                    rollout["shadow_states"][2 * WARM_START_BLOCKS :],
                    strict=True,
                )
            )
        maxima = rollout["maxima"]
        closure_ok = closure_ok and all(
            (
                int(maxima["call_order_errors"]) == 0,
                float(maxima["shadow_recurrence_abs"]) <= 1.0e-12,
                float(maxima["accepted_recurrence_abs"]) <= 1.0e-6,
                float(maxima["accepted_coast_recurrence_abs"]) <= 1.0e-6,
                float(maxima["candidate_common_native_input_abs"]) <= 1.0e-12,
                float(maxima["candidate_lookahead_input_abs"]) <= 1.0e-12,
            )
        )
    return {
        "warm_start_inventory_exact": True,
        "warm_start_decisions_target_free_and_consistent": decisions_ok,
        "warm_start_call_counts_exact": counts_ok,
        "warm_start_coast_and_fallback_recurrence_exact": recurrence_ok,
        "warm_start_common_source_and_lookahead_exact": closure_ok,
    }


def _projected_coast_call_checks(
    rollouts: Sequence[Mapping[str, Any]],
    *,
    case_ids: Sequence[str] = PROSPECTIVE_STRUCTURAL_CASE_IDS,
) -> dict[str, bool]:
    expected_cases = tuple(case_ids)
    by_case = {str(row["case_id"]): row for row in rollouts}
    inventory_exact = (
        len(expected_cases) == len(set(expected_cases))
        and len(rollouts) == len(expected_cases)
        and len(by_case) == len(expected_cases)
        and set(by_case) == set(expected_cases)
    )
    keys = (
        "projected_coast_inventory_exact",
        "projected_coast_decisions_target_free_and_consistent",
        "projected_coast_call_counts_exact",
        "projected_coast_recurrence_and_fallback_exact",
        "projected_coast_common_source_lookahead_and_projection_exact",
    )
    if not inventory_exact:
        return {key: False for key in keys}

    decisions_ok = True
    counts_ok = True
    recurrence_ok = True
    closure_ok = True
    for case_id in expected_cases:
        rollout = by_case[case_id]
        descriptor = rollout.get("position_descriptor")
        trusted = bool(rollout.get("position_trusted"))
        expected_trusted = bool(
            descriptor
            and descriptor.get("status") == "ok"
            and descriptor.get("normalized_wall_distance") is not None
            and float(descriptor["normalized_wall_distance"])
            <= FROZEN_POSITION_TRUST_THRESHOLD
        )
        front_rows = list(rollout.get("front_audits", ()))
        warm_rows = [
            row
            for row in front_rows
            if row.get("phase") in {"warm_candidate", "trusted_candidate"}
        ]
        coast_rows = [
            row for row in front_rows if row.get("phase") == "projected_coast"
        ]
        changed = [row for row in front_rows if bool(row["branch_changed"])]
        veto_call = rollout.get("front_branch_veto_call")
        decisions_ok = decisions_ok and all(
            (
                rollout.get("warm_start_blocks") == WARM_START_BLOCKS,
                rollout.get("projected_coast") is True,
                trusted == expected_trusted,
                len(changed) <= 1,
                (veto_call is None and not changed)
                or (
                    len(changed) == 1
                    and int(veto_call) == int(changed[0]["first_output_call"])
                    and changed[0] is front_rows[-1]
                ),
            )
        )
        if veto_call is None:
            expected_warm_blocks = 15 if trusted else WARM_START_BLOCKS
            expected_coast_blocks = 0 if trusted else 15 - WARM_START_BLOCKS
        else:
            veto_block = (int(veto_call) - 1) // 2
            if trusted or veto_block < WARM_START_BLOCKS:
                expected_warm_blocks = veto_block + 1
                expected_coast_blocks = 0
            else:
                expected_warm_blocks = WARM_START_BLOCKS
                expected_coast_blocks = veto_block - WARM_START_BLOCKS + 1
        execution = rollout["execution"]
        counts_ok = counts_ok and all(
            (
                len(warm_rows) == expected_warm_blocks,
                len(coast_rows) == expected_coast_blocks,
                len(front_rows) == expected_warm_blocks + expected_coast_blocks,
                len(rollout.get("projected_coast_audits", ()))
                == 2 * expected_coast_blocks,
                int(execution["logical_model_calls"])
                == 30 + 4 * expected_warm_blocks + 2 * expected_coast_blocks,
                int(execution["native_logical_calls"])
                == 30 + 3 * expected_warm_blocks + 2 * expected_coast_blocks,
                int(execution["fine_logical_calls"]) == expected_warm_blocks,
                int(execution["shadow_native_logical_calls"]) == 30,
                int(execution["candidate_native_logical_calls"])
                == 3 * expected_warm_blocks,
                int(execution["candidate_fine_logical_calls"]) == expected_warm_blocks,
                int(execution["accepted_coast_native_logical_calls"])
                == 2 * expected_coast_blocks,
                int(execution["completed_calls"]) == 30,
            )
        )
        accepted_statuses = [
            row["correction_status"] for row in rollout["accepted_rows"]
        ]
        if veto_call is not None:
            recurrence_ok = recurrence_ok and all(
                np.array_equal(left, right)
                for left, right in zip(
                    rollout["accepted_states"][int(veto_call) :],
                    rollout["shadow_states"][int(veto_call) :],
                    strict=True,
                )
            )
        elif not trusted:
            recurrence_ok = recurrence_ok and all(
                status == "projected_coast_sp19_tether"
                for status in accepted_statuses[2 * WARM_START_BLOCKS :]
            )
            recurrence_ok = recurrence_ok and any(
                not np.array_equal(left, right)
                for left, right in zip(
                    rollout["accepted_states"][2 * WARM_START_BLOCKS :],
                    rollout["shadow_states"][2 * WARM_START_BLOCKS :],
                    strict=True,
                )
            )
        maxima = rollout["maxima"]
        closure_ok = closure_ok and all(
            (
                int(maxima["call_order_errors"]) == 0,
                float(maxima["shadow_recurrence_abs"]) <= 1.0e-12,
                float(maxima["accepted_recurrence_abs"]) <= 1.0e-6,
                float(maxima["accepted_coast_recurrence_abs"]) <= 1.0e-6,
                float(maxima["candidate_common_native_input_abs"]) <= 1.0e-12,
                float(maxima["candidate_lookahead_input_abs"]) <= 1.0e-12,
                float(maxima["accepted_coast_shadow_integral_abs"]) <= 1.0e-10,
                float(maxima["projected_coast_boundary_abs"]) <= 1.0e-12,
                float(maxima["projected_coast_idempotence_abs"]) <= 1.0e-10,
                float(maxima["projected_coast_tether_abs"]) <= 1.0e-12,
            )
        )
    return dict(
        zip(
            keys,
            (True, decisions_ok, counts_ok, recurrence_ok, closure_ok),
            strict=True,
        )
    )


def _slew_limited_tether_call_checks(
    rollouts: Sequence[Mapping[str, Any]],
    *,
    case_ids: Sequence[str] = PROSPECTIVE_STRUCTURAL_CASE_IDS,
    relaxed: bool = False,
) -> dict[str, bool]:
    expected_cases = tuple(case_ids)
    by_case = {str(row["case_id"]): row for row in rollouts}
    inventory_exact = (
        len(expected_cases) == len(set(expected_cases))
        and len(rollouts) == len(expected_cases)
        and len(by_case) == len(expected_cases)
        and set(by_case) == set(expected_cases)
    )
    prefix = "relaxed_tether" if relaxed else "slew_limited_tether"
    phase = "relaxed_tether_coast" if relaxed else "slew_limited_tether_coast"
    status = "relaxed_sp19_tether" if relaxed else "slew_limited_sp19_tether"
    audit_key = "relaxed_tether_audits" if relaxed else "slew_limited_tether_audits"
    keys = (
        f"{prefix}_inventory_exact",
        f"{prefix}_decisions_target_free_and_consistent",
        f"{prefix}_call_counts_exact",
        f"{prefix}_recurrence_and_fallback_exact",
        f"{prefix}_bound_and_closure_exact",
    )
    if not inventory_exact:
        return {key: False for key in keys}

    decisions_ok = True
    counts_ok = True
    recurrence_ok = True
    closure_ok = True
    for case_id in expected_cases:
        rollout = by_case[case_id]
        descriptor = rollout.get("position_descriptor")
        trusted = bool(rollout.get("position_trusted"))
        expected_trusted = bool(
            descriptor
            and descriptor.get("status") == "ok"
            and descriptor.get("normalized_wall_distance") is not None
            and float(descriptor["normalized_wall_distance"])
            <= FROZEN_POSITION_TRUST_THRESHOLD
        )
        front_rows = list(rollout.get("front_audits", ()))
        warm_rows = [
            row
            for row in front_rows
            if row.get("phase") in {"warm_candidate", "trusted_candidate"}
        ]
        coast_rows = [
            row
            for row in front_rows
            if row.get("phase") == phase
        ]
        changed = [row for row in front_rows if bool(row["branch_changed"])]
        veto_call = rollout.get("front_branch_veto_call")
        decisions_ok = decisions_ok and all(
            (
                rollout.get("warm_start_blocks") == FROZEN_OFFSET_WARM_BLOCKS,
                rollout.get(prefix) is True,
                trusted == expected_trusted,
                len(changed) <= 1,
                (veto_call is None and not changed)
                or (
                    len(changed) == 1
                    and int(veto_call) == int(changed[0]["first_output_call"])
                    and changed[0] is front_rows[-1]
                ),
            )
        )
        if veto_call is None:
            expected_warm_blocks = 15 if trusted else FROZEN_OFFSET_WARM_BLOCKS
            expected_coast_blocks = (
                0 if trusted else 15 - FROZEN_OFFSET_WARM_BLOCKS
            )
        else:
            veto_block = (int(veto_call) - 1) // 2
            if trusted or veto_block < FROZEN_OFFSET_WARM_BLOCKS:
                expected_warm_blocks = veto_block + 1
                expected_coast_blocks = 0
            else:
                expected_warm_blocks = FROZEN_OFFSET_WARM_BLOCKS
                expected_coast_blocks = (
                    veto_block - FROZEN_OFFSET_WARM_BLOCKS + 1
                )
        execution = rollout["execution"]
        counts_ok = counts_ok and all(
            (
                len(warm_rows) == expected_warm_blocks,
                len(coast_rows) == expected_coast_blocks,
                len(front_rows) == expected_warm_blocks + expected_coast_blocks,
                len(rollout.get(audit_key, ()))
                == 2 * expected_coast_blocks,
                int(execution["logical_model_calls"])
                == 30 + 4 * expected_warm_blocks + 2 * expected_coast_blocks,
                int(execution["native_logical_calls"])
                == 30 + 3 * expected_warm_blocks + 2 * expected_coast_blocks,
                int(execution["fine_logical_calls"]) == expected_warm_blocks,
                int(execution["shadow_native_logical_calls"]) == 30,
                int(execution["candidate_native_logical_calls"])
                == 3 * expected_warm_blocks,
                int(execution["candidate_fine_logical_calls"])
                == expected_warm_blocks,
                int(execution["accepted_coast_native_logical_calls"])
                == 2 * expected_coast_blocks,
                int(execution["completed_calls"]) == 30,
            )
        )
        accepted_statuses = [
            row["correction_status"] for row in rollout["accepted_rows"]
        ]
        if veto_call is not None:
            recurrence_ok = recurrence_ok and all(
                np.array_equal(left, right)
                for left, right in zip(
                    rollout["accepted_states"][int(veto_call) :],
                    rollout["shadow_states"][int(veto_call) :],
                    strict=True,
                )
            )
        elif not trusted:
            recurrence_ok = recurrence_ok and all(
                row_status == status
                for row_status in accepted_statuses[
                    2 * FROZEN_OFFSET_WARM_BLOCKS :
                ]
            )
            recurrence_ok = recurrence_ok and any(
                not np.array_equal(left, right)
                for left, right in zip(
                    rollout["accepted_states"][
                        2 * FROZEN_OFFSET_WARM_BLOCKS :
                    ],
                    rollout["shadow_states"][
                        2 * FROZEN_OFFSET_WARM_BLOCKS :
                    ],
                    strict=True,
                )
            )
        maxima = rollout["maxima"]
        tether_audits = list(rollout.get(audit_key, ()))
        relaxation_rows_exact = (
            all(
                row.get("status") in {"ok", "zero_target_gap"}
                and abs(float(row["applied_scale"]) - RELAXED_TETHER_RATE)
                <= 1.0e-12
                and abs(
                    float(row["applied_change_rms"])
                    - RELAXED_TETHER_RATE * float(row["requested_change_rms"])
                )
                <= 1.0e-12
                and bool(row.get("cap_active"))
                == (float(row["requested_change_rms"]) > 0.0)
                for row in tether_audits
            )
            if relaxed
            else True
        )
        closure_ok = closure_ok and all(
            (
                int(maxima["call_order_errors"]) == 0,
                float(maxima["shadow_recurrence_abs"]) <= 1.0e-12,
                float(maxima["accepted_recurrence_abs"]) <= 1.0e-6,
                float(maxima["accepted_coast_recurrence_abs"]) <= 1.0e-6,
                float(maxima["candidate_common_native_input_abs"]) <= 1.0e-12,
                float(maxima["candidate_lookahead_input_abs"]) <= 1.0e-12,
                (
                    int(maxima["slew_cap_active_count"])
                    == sum(bool(row.get("cap_active")) for row in tether_audits)
                    if relaxed
                    else float(maxima["slew_applied_to_shadow_increment_ratio"])
                    <= SLEW_LIMITED_TETHER_RELATIVE_CHANGE_LIMIT + 1.0e-12
                ),
                float(maxima["slew_change_limit_violation_rms"]) <= 1.0e-12,
                float(maxima["slew_boundary_contraction_violation_abs"])
                <= 1.0e-12,
                float(maxima["slew_projection_residual_increase_rms"])
                <= 1.0e-12,
                float(maxima["slew_update_identity_abs"]) <= 1.0e-12,
                float(maxima["slew_integral_abs"])
                <= (1.0e-12 if relaxed else 1.0e-10),
                float(maxima["accepted_coast_shadow_integral_abs"])
                <= (1.0e-12 if relaxed else 1.0e-10),
                relaxation_rows_exact,
            )
        )
    return dict(
        zip(
            keys,
            (True, decisions_ok, counts_ok, recurrence_ok, closure_ok),
            strict=True,
        )
    )


def _buffered_relaxed_tether_call_checks(
    rollouts: Sequence[Mapping[str, Any]],
    *,
    case_ids: Sequence[str] = PROSPECTIVE_STRUCTURAL_CASE_IDS,
) -> dict[str, bool]:
    expected_cases = tuple(case_ids)
    by_case = {str(row["case_id"]): row for row in rollouts}
    keys = (
        "buffered_relaxed_tether_inventory_exact",
        "buffered_relaxed_tether_routes_target_free_and_exact_2_2_5",
        "buffered_relaxed_tether_call_counts_exact_580",
        "buffered_relaxed_tether_recurrence_and_raw_buffer_exact",
        "buffered_relaxed_tether_bound_and_closure_exact",
    )
    inventory_exact = (
        len(expected_cases) == len(set(expected_cases))
        and len(rollouts) == len(expected_cases)
        and len(by_case) == len(expected_cases)
        and set(by_case) == set(expected_cases)
    )
    if not inventory_exact:
        return {key: False for key in keys}

    routes = {}
    for case_id, rollout in by_case.items():
        descriptor = rollout.get("position_descriptor")
        if not isinstance(descriptor, Mapping):
            return {key: False for key in keys}
        parsed = TransverseVelocityPositionDescriptor(
            status=str(descriptor.get("status")),
            vertical_centroid=descriptor.get("vertical_centroid"),
            normalized_wall_distance=descriptor.get("normalized_wall_distance"),
            transverse_velocity_l1=float(descriptor["transverse_velocity_l1"]),
        )
        routes[case_id] = buffered_frozen_offset_position_route(parsed)
    route_counts = {
        route: sum(value == route for value in routes.values())
        for route in (
            "edge_candidate",
            "raw_uncertainty_buffer",
            "interior_frozen_offset",
            "raw_unresolved",
        )
    }
    routes_exact = route_counts == {
        "edge_candidate": 2,
        "raw_uncertainty_buffer": 2,
        "interior_frozen_offset": 5,
        "raw_unresolved": 0,
    } and all(by_case[case_id].get("position_route") == route for case_id, route in routes.items())

    buffer_routes = {"raw_uncertainty_buffer", "raw_unresolved"}
    buffer_cases = [case_id for case_id, route in routes.items() if route in buffer_routes]
    active_cases = [case_id for case_id, route in routes.items() if route not in buffer_routes]
    active_checks = _slew_limited_tether_call_checks(
        [by_case[case_id] for case_id in active_cases],
        case_ids=active_cases,
        relaxed=True,
    )
    buffer_exact = True
    for case_id in buffer_cases:
        rollout = by_case[case_id]
        execution = rollout["execution"]
        maxima = rollout["maxima"]
        buffer_exact = buffer_exact and all(
            (
                rollout.get("buffered_relaxed_tether") is True,
                rollout.get("position_buffered") is True,
                rollout.get("policy") == "buffered_relaxed_tether",
                int(execution["logical_model_calls"]) == 30,
                int(execution["native_logical_calls"]) == 30,
                int(execution["fine_logical_calls"]) == 0,
                int(execution["shadow_native_logical_calls"]) == 30,
                int(execution["candidate_native_logical_calls"]) == 0,
                int(execution["candidate_fine_logical_calls"]) == 0,
                int(execution["accepted_coast_native_logical_calls"]) == 0,
                int(execution["completed_calls"]) == 30,
                not rollout.get("front_audits"),
                not rollout.get("anchor_rows"),
                not rollout.get("relaxed_tether_audits"),
                int(maxima["call_order_errors"]) == 0,
                float(maxima["shadow_recurrence_abs"]) <= 1.0e-12,
                all(
                    np.array_equal(left, right)
                    for left, right in zip(
                        rollout["accepted_states"],
                        rollout["shadow_states"],
                        strict=True,
                    )
                ),
                all(
                    row["correction_status"] == "position_uncertainty_buffer_raw"
                    for row in rollout["accepted_rows"]
                ),
            )
        )
    total = {
        key: sum(int(row["execution"][key]) for row in rollouts)
        for key in ("logical_model_calls", "native_logical_calls", "fine_logical_calls")
    }
    counts_exact = total == {
        "logical_model_calls": 580,
        "native_logical_calls": 530,
        "fine_logical_calls": 50,
    }
    return dict(
        zip(
            keys,
            (
                True,
                routes_exact,
                counts_exact,
                buffer_exact
                and active_checks["relaxed_tether_recurrence_and_fallback_exact"],
                buffer_exact
                and all(
                    passed
                    for name, passed in active_checks.items()
                    if name != "relaxed_tether_recurrence_and_fallback_exact"
                ),
            ),
            strict=True,
        )
    )


def _a22_rows_close(
    actual_rows: Sequence[Mapping[str, Any]],
    expected_rows: Sequence[Mapping[str, Any]],
    *,
    key_fields: Sequence[str],
    numeric_fields: Sequence[str],
    boolean_fields: Sequence[str] = (),
    text_fields: Sequence[str] = (),
    atol: float = 1.0e-12,
) -> bool:
    def key(row: Mapping[str, Any]) -> tuple[str, ...]:
        return tuple(str(row.get(field, "")) for field in key_fields)

    actual = {key(row): row for row in actual_rows}
    expected = {key(row): row for row in expected_rows}
    if (
        len(actual) != len(actual_rows)
        or len(expected) != len(expected_rows)
        or set(actual) != set(expected)
    ):
        return False
    for row_key, expected_row in expected.items():
        actual_row = actual[row_key]
        for field in numeric_fields:
            if field not in actual_row or field not in expected_row:
                return False
            expected_value = expected_row[field]
            actual_value = actual_row[field]
            if expected_value in (None, ""):
                if actual_value not in (None, ""):
                    return False
                continue
            try:
                expected_number = float(expected_value)
                actual_number = float(actual_value)
            except (TypeError, ValueError):
                return False
            if not (
                math.isfinite(expected_number)
                and math.isfinite(actual_number)
                and abs(actual_number - expected_number) <= atol
            ):
                return False
        for field in boolean_fields:
            expected_value = expected_row.get(field)
            expected_bool = (
                expected_value
                if isinstance(expected_value, bool)
                else str(expected_value) == "True"
            )
            if isinstance(actual_row.get(field), bool):
                actual_bool = actual_row[field]
            elif str(actual_row.get(field)) in {"True", "False"}:
                actual_bool = str(actual_row[field]) == "True"
            else:
                return False
            if actual_bool is not expected_bool:
                return False
        if any(
            str(actual_row.get(field, "")) != str(expected_row.get(field, ""))
            for field in text_fields
        ):
            return False
    return True


def _a22_replay_scoring_checks(
    *,
    a22_rescore_path: Path,
    a22_rescore: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    case_rows: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    population: Mapping[str, Any],
    descriptor_rows: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    root = a22_rescore_path.parent
    expected_calls = _read_csv(root / "selected_call_metrics.csv")
    call_numeric_fields = tuple(
        field
        for field in expected_calls[0]
        if field
        not in {
            "case_id",
            "policy",
            "correction_status",
            "finite",
            "admissible",
            "cap_active",
            "source_policy",
            "position_route",
        }
    )
    call_rows_close = _a22_rows_close(
        rows,
        expected_calls,
        key_fields=("case_id", "policy", "input_call"),
        numeric_fields=call_numeric_fields,
        boolean_fields=("finite", "admissible", "cap_active"),
    )

    expected_cases = _read_csv(root / "rollout_case_metrics.csv")
    case_rows_close = _a22_rows_close(
        case_rows,
        expected_cases,
        key_fields=("case_id",),
        numeric_fields=tuple(
            field for field in expected_cases[0] if field != "case_id"
        ),
    )
    expected_controls = _read_csv(root / "rollout_controls.csv")
    controls_close = _a22_rows_close(
        controls,
        expected_controls,
        key_fields=("key", "scope"),
        numeric_fields=("zero_rms", "corrected_rms", "ratio"),
        text_fields=("status",),
    )
    actual_population = dict(population)
    deep_cases = {
        str(row["case_id"])
        for row in descriptor_rows
        if row["position_route"] == "interior_frozen_offset"
    }
    actual_population["strict_deep_interior_trajectory_win_count"] = sum(
        str(row["case_id"]) in deep_cases
        and float(row["trajectory_state_rms_ratio"]) < 1.0
        for row in case_rows
    )
    expected_population = a22_rescore.get("population", {})
    population_close = bool(expected_population) and all(
        key in actual_population
        and math.isfinite(float(actual_population[key]))
        and abs(float(actual_population[key]) - float(expected_value)) <= 1.0e-12
        for key, expected_value in expected_population.items()
    )
    expected_routes = {
        str(row["case_id"]): str(row["route"])
        for row in a22_rescore.get("position_routes", ())
    }
    actual_routes = {
        str(row["case_id"]): str(row["position_route"])
        for row in descriptor_rows
    }
    return {
        "a22_selected_call_metrics_reproduced_1e_12": call_rows_close,
        "a22_case_metrics_reproduced_1e_12": case_rows_close,
        "a22_controls_reproduced_1e_12": controls_close,
        "a22_population_reproduced_1e_12": population_close,
        "a22_routes_reproduced_exact": actual_routes == expected_routes,
    }


def _frozen_offset_call_checks(
    rollouts: Sequence[Mapping[str, Any]],
    *,
    case_ids: Sequence[str] = PROSPECTIVE_STRUCTURAL_CASE_IDS,
) -> dict[str, bool]:
    expected_cases = tuple(case_ids)
    by_case = {str(row["case_id"]): row for row in rollouts}
    keys = (
        "frozen_offset_inventory_exact",
        "frozen_offset_decisions_target_free_and_consistent",
        "frozen_offset_call_counts_exact",
        "frozen_offset_raw_coast_and_fallback_exact",
        "frozen_offset_increment_and_projection_closure_exact",
    )
    inventory_exact = (
        len(expected_cases) == len(set(expected_cases))
        and len(rollouts) == len(expected_cases)
        and len(by_case) == len(expected_cases)
        and set(by_case) == set(expected_cases)
    )
    if not inventory_exact:
        return {key: False for key in keys}

    decisions_ok = True
    counts_ok = True
    recurrence_ok = True
    closure_ok = True
    for case_id in expected_cases:
        rollout = by_case[case_id]
        descriptor = rollout.get("position_descriptor")
        trusted = bool(rollout.get("position_trusted"))
        expected_trusted = bool(
            descriptor
            and descriptor.get("status") == "ok"
            and descriptor.get("normalized_wall_distance") is not None
            and float(descriptor["normalized_wall_distance"])
            <= FROZEN_POSITION_TRUST_THRESHOLD
        )
        front_rows = list(rollout.get("front_audits", ()))
        warm_rows = [
            row
            for row in front_rows
            if row.get("phase") in {"warm_candidate", "trusted_candidate"}
        ]
        coast_rows = [
            row for row in front_rows if row.get("phase") == "frozen_offset_coast"
        ]
        changed = [row for row in front_rows if bool(row["branch_changed"])]
        veto_call = rollout.get("front_branch_veto_call")
        decisions_ok = decisions_ok and all(
            (
                rollout.get("warm_start_blocks") == FROZEN_OFFSET_WARM_BLOCKS,
                rollout.get("frozen_offset") is True,
                trusted == expected_trusted,
                len(changed) <= 1,
                (veto_call is None and not changed)
                or (
                    len(changed) == 1
                    and int(veto_call) == int(changed[0]["first_output_call"])
                    and changed[0] is front_rows[-1]
                ),
            )
        )
        if veto_call is None:
            expected_warm_blocks = 15 if trusted else FROZEN_OFFSET_WARM_BLOCKS
            expected_coast_blocks = 0 if trusted else 15 - FROZEN_OFFSET_WARM_BLOCKS
        else:
            veto_block = (int(veto_call) - 1) // 2
            if trusted or veto_block < FROZEN_OFFSET_WARM_BLOCKS:
                expected_warm_blocks = veto_block + 1
                expected_coast_blocks = 0
            else:
                expected_warm_blocks = FROZEN_OFFSET_WARM_BLOCKS
                expected_coast_blocks = veto_block - FROZEN_OFFSET_WARM_BLOCKS + 1
        expected_handoff_audits = int(
            not trusted and expected_warm_blocks == FROZEN_OFFSET_WARM_BLOCKS
        )
        execution = rollout["execution"]
        counts_ok = counts_ok and all(
            (
                len(warm_rows) == expected_warm_blocks,
                len(coast_rows) == expected_coast_blocks,
                len(front_rows) == expected_warm_blocks + expected_coast_blocks,
                len(rollout.get("frozen_offset_audits", ())) == expected_handoff_audits,
                int(execution["logical_model_calls"]) == 30 + 4 * expected_warm_blocks,
                int(execution["native_logical_calls"]) == 30 + 3 * expected_warm_blocks,
                int(execution["fine_logical_calls"]) == expected_warm_blocks,
                int(execution["shadow_native_logical_calls"]) == 30,
                int(execution["candidate_native_logical_calls"])
                == 3 * expected_warm_blocks,
                int(execution["candidate_fine_logical_calls"]) == expected_warm_blocks,
                int(execution["accepted_coast_native_logical_calls"]) == 0,
                int(execution["completed_calls"]) == 30,
            )
        )
        accepted_states = rollout["accepted_states"]
        shadow_states = rollout["shadow_states"]
        accepted_statuses = [
            row["correction_status"] for row in rollout["accepted_rows"]
        ]
        if veto_call is not None:
            recurrence_ok = recurrence_ok and all(
                np.array_equal(left, right)
                for left, right in zip(
                    accepted_states[int(veto_call) :],
                    shadow_states[int(veto_call) :],
                    strict=True,
                )
            )
        elif not trusted:
            handoff_call = 2 * FROZEN_OFFSET_WARM_BLOCKS
            offset = accepted_states[handoff_call] - shadow_states[handoff_call]
            recurrence_ok = recurrence_ok and all(
                status == "frozen_sp19_output_offset"
                for status in accepted_statuses[handoff_call:]
            )
            recurrence_ok = recurrence_ok and all(
                _maximum_abs(accepted_states[call] - shadow_states[call] - offset)
                <= 1.0e-12
                for call in range(handoff_call, 31)
            )
            recurrence_ok = recurrence_ok and all(
                _maximum_abs(
                    (accepted_states[call] - accepted_states[call - 1])
                    - (shadow_states[call] - shadow_states[call - 1])
                )
                <= 1.0e-12
                for call in range(handoff_call + 1, 31)
            )
        maxima = rollout["maxima"]
        closure_ok = closure_ok and all(
            (
                int(maxima["call_order_errors"]) == 0,
                float(maxima["shadow_recurrence_abs"]) <= 1.0e-12,
                float(maxima["accepted_recurrence_abs"]) <= 1.0e-6,
                float(maxima["candidate_common_native_input_abs"]) <= 1.0e-12,
                float(maxima["candidate_lookahead_input_abs"]) <= 1.0e-12,
                float(maxima["accepted_coast_shadow_integral_abs"]) <= 1.0e-10,
                float(maxima["frozen_offset_integral_abs"]) <= 1.0e-10,
                float(maxima["frozen_offset_boundary_abs"]) <= 1.0e-12,
                float(maxima["frozen_offset_idempotence_abs"]) <= 1.0e-10,
                float(maxima["frozen_offset_constancy_abs"]) <= 1.0e-12,
                float(maxima["frozen_offset_increment_identity_abs"]) <= 1.0e-12,
            )
        )
    return dict(
        zip(
            keys,
            (True, decisions_ok, counts_ok, recurrence_ok, closure_ok),
            strict=True,
        )
    )


def _buffered_offset_call_checks(
    rollouts: Sequence[Mapping[str, Any]],
    *,
    case_ids: Sequence[str] = PROSPECTIVE_STRUCTURAL_CASE_IDS,
) -> dict[str, bool]:
    expected_cases = tuple(case_ids)
    by_case = {str(row["case_id"]): row for row in rollouts}
    keys = (
        "buffered_offset_inventory_exact",
        "buffered_offset_route_target_free_and_consistent",
        "buffered_offset_raw_buffer_recurrence_exact",
        "buffered_offset_edge_and_interior_contract_exact",
    )
    if (
        len(expected_cases) != len(set(expected_cases))
        or len(rollouts) != len(expected_cases)
        or len(by_case) != len(expected_cases)
        or set(by_case) != set(expected_cases)
    ):
        return {key: False for key in keys}

    routes_ok = True
    raw_ok = True
    other_ok = True
    for case_id in expected_cases:
        rollout = by_case[case_id]
        descriptor = rollout.get("position_descriptor")
        if not descriptor:
            expected_route = "raw_unresolved"
        else:
            expected_route = buffered_frozen_offset_position_route(
                TransverseVelocityPositionDescriptor(**descriptor)
            )
        route = rollout.get("position_route")
        buffered = bool(rollout.get("position_buffered"))
        routes_ok = routes_ok and all(
            (
                rollout.get("buffered_offset") is True,
                route == expected_route,
                buffered == (route in {"raw_uncertainty_buffer", "raw_unresolved"}),
            )
        )
        if buffered:
            execution = rollout["execution"]
            raw_ok = raw_ok and all(
                (
                    not rollout.get("front_audits"),
                    not rollout.get("frozen_offset_audits"),
                    int(execution["logical_model_calls"]) == 30,
                    int(execution["native_logical_calls"]) == 30,
                    int(execution["fine_logical_calls"]) == 0,
                    int(execution["candidate_native_logical_calls"]) == 0,
                    int(execution["candidate_fine_logical_calls"]) == 0,
                    all(
                        np.array_equal(left, right)
                        for left, right in zip(
                            rollout["accepted_states"],
                            rollout["shadow_states"],
                            strict=True,
                        )
                    ),
                    all(
                        row["correction_status"] == "position_uncertainty_buffer_raw"
                        for row in rollout["accepted_rows"]
                    ),
                )
            )
        else:
            single = _frozen_offset_call_checks([rollout], case_ids=(case_id,))
            other_ok = other_ok and all(single.values())
    return dict(zip(keys, (True, routes_ok, raw_ok, other_ok), strict=True))


def _persistence_gain_call_checks(
    rollouts: Sequence[Mapping[str, Any]],
    *,
    case_ids: Sequence[str] = PROSPECTIVE_STRUCTURAL_CASE_IDS,
) -> dict[str, bool]:
    expected_cases = tuple(case_ids)
    by_case = {str(row["case_id"]): row for row in rollouts}
    keys = (
        "persistence_inventory_exact",
        "persistence_route_target_free_and_consistent",
        "persistence_raw_buffer_and_edge_contract_exact",
        "persistence_probe_count_and_common_source_exact",
        "persistence_gain_and_offset_bookkeeping_exact",
    )
    if (
        len(expected_cases) != len(set(expected_cases))
        or len(rollouts) != len(expected_cases)
        or len(by_case) != len(expected_cases)
        or set(by_case) != set(expected_cases)
    ):
        return {key: False for key in keys}

    routes_ok = True
    unchanged_ok = True
    probes_ok = True
    gains_ok = True
    for case_id in expected_cases:
        rollout = by_case[case_id]
        descriptor = rollout.get("position_descriptor")
        expected_route = (
            buffered_frozen_offset_position_route(
                TransverseVelocityPositionDescriptor(**descriptor)
            )
            if descriptor
            else "raw_unresolved"
        )
        route = rollout.get("position_route")
        routes_ok = routes_ok and all(
            (
                rollout.get("buffered_offset") is True,
                rollout.get("persistence_probe") is True,
                route == expected_route,
                bool(rollout.get("position_buffered"))
                == (route in {"raw_uncertainty_buffer", "raw_unresolved"}),
            )
        )
        audits = list(rollout.get("persistence_probe_audits", ()))
        execution = rollout["execution"]
        if route != "interior_frozen_offset":
            unchanged_ok = unchanged_ok and not audits
            if route in {"raw_uncertainty_buffer", "raw_unresolved"}:
                unchanged_ok = unchanged_ok and all(
                    (
                        int(execution["logical_model_calls"]) == 30,
                        int(execution["native_logical_calls"]) == 30,
                        int(execution["fine_logical_calls"]) == 0,
                        int(execution["persistence_probe_fine_logical_calls"]) == 0,
                        all(
                            np.array_equal(left, right)
                            for left, right in zip(
                                rollout["accepted_states"],
                                rollout["shadow_states"],
                                strict=True,
                            )
                        ),
                    )
                )
            else:
                single = _frozen_offset_call_checks(
                    [rollout], case_ids=(case_id,)
                )
                unchanged_ok = unchanged_ok and all(single.values())
        else:
            maxima = rollout["maxima"]
            probes_ok = probes_ok and all(
                (
                    len(audits) == 11,
                    len(rollout.get("front_audits", ())) == 15,
                    len(rollout.get("frozen_offset_audits", ())) == 1,
                    int(execution["logical_model_calls"]) == 57,
                    int(execution["native_logical_calls"]) == 42,
                    int(execution["fine_logical_calls"]) == 15,
                    int(execution["candidate_native_logical_calls"]) == 12,
                    int(execution["candidate_fine_logical_calls"]) == 4,
                    int(execution["persistence_probe_fine_logical_calls"]) == 11,
                    int(execution["accepted_coast_native_logical_calls"]) == 0,
                    all(
                        int(row["output_call"]) == 2 * int(row["block_index"]) + 1
                        for row in audits
                    ),
                    float(maxima["persistence_common_native_input_abs"]) <= 1.0e-6,
                    float(maxima["persistence_fine_input_abs"]) <= 1.0e-12,
                    float(maxima["persistence_offset_boundary_abs"]) <= 1.0e-12,
                    float(maxima["persistence_offset_integral_abs"]) <= 1.0e-10,
                )
            )
            applied = [float(row["applied_gain"]) for row in audits]
            gains_ok = gains_ok and all(
                (
                    all(0.0 <= gain <= 1.0 for gain in applied),
                    all(
                        right <= left + 1.0e-15
                        for left, right in pairwise(applied)
                    ),
                    float(maxima["persistence_gain_increase_abs"]) <= 1.0e-15,
                    float(maxima["persistence_offset_bookkeeping_abs"])
                    <= 1.0e-12,
                    float(maxima["persistence_within_block_identity_abs"])
                    <= 1.0e-12,
                    int(maxima["call_order_errors"]) == 0,
                )
            )
    return dict(zip(keys, (True, routes_ok, unchanged_ok, probes_ok, gains_ok), strict=True))


def _terminal_ramp_call_checks(
    rollouts: Sequence[Mapping[str, Any]],
    *,
    case_ids: Sequence[str] = PROSPECTIVE_STRUCTURAL_CASE_IDS,
) -> dict[str, bool]:
    checks = _persistence_gain_call_checks(rollouts, case_ids=case_ids)
    checks = {
        key.replace("persistence_", "terminal_ramp_", 1): value
        for key, value in checks.items()
    }
    paths_ok = True
    for rollout in rollouts:
        if rollout.get("position_route") != "interior_frozen_offset":
            continue
        audits = list(rollout.get("persistence_probe_audits", ()))
        trigger = rollout.get("terminal_ramp_trigger_block")
        gains = [float(row["applied_gain"]) for row in audits]
        phase_trigger = next(
            (
                int(row["block_index"])
                for row in audits
                if row["alignment_cosine"] in (None, "")
                or float(row["alignment_cosine"]) <= 0.0
            ),
            None,
        )
        effective_trigger = 14 if phase_trigger is None else phase_trigger
        expected = []
        for row in audits:
            block = int(row["block_index"])
            if phase_trigger is None or phase_trigger >= 14:
                expected.append(0.0 if block == 14 else 1.0)
            elif block <= phase_trigger:
                expected.append(1.0)
            else:
                expected.append((14 - block) / (14 - phase_trigger))
        paths_ok = paths_ok and all(
            (
                rollout.get("terminal_ramp") is True,
                trigger == effective_trigger,
                len(gains) == 11,
                all(
                    abs(left - right) <= 1.0e-15
                    for left, right in zip(gains, expected, strict=True)
                ),
                gains[-1] == 0.0,
                all(
                    row["gain_status"]
                    in {
                        "pretrigger_unit_gain",
                        "phase_triggered_terminal_ramp",
                        "forced_terminal_raw",
                    }
                    for row in audits
                ),
            )
        )
    checks["terminal_ramp_trigger_and_linear_path_exact"] = paths_ok
    return checks


def _fixed_late_ramp_call_checks(
    rollouts: Sequence[Mapping[str, Any]],
    *,
    case_ids: Sequence[str] = PROSPECTIVE_STRUCTURAL_CASE_IDS,
) -> dict[str, bool]:
    expected_cases = tuple(case_ids)
    by_case = {str(row["case_id"]): row for row in rollouts}
    keys = (
        "fixed_late_ramp_inventory_exact",
        "fixed_late_ramp_route_target_free_and_consistent",
        "fixed_late_ramp_raw_buffer_recurrence_exact",
        "fixed_late_ramp_edge_and_interior_contract_exact",
        "fixed_late_ramp_schedule_cost_and_bookkeeping_exact",
    )
    if (
        len(expected_cases) != len(set(expected_cases))
        or len(rollouts) != len(expected_cases)
        or len(by_case) != len(expected_cases)
        or set(by_case) != set(expected_cases)
    ):
        return {key: False for key in keys}

    routes_ok = True
    raw_ok = True
    edge_and_interior_ok = True
    schedule_ok = True
    for case_id in expected_cases:
        rollout = by_case[case_id]
        descriptor = rollout.get("position_descriptor")
        expected_route = (
            buffered_frozen_offset_position_route(
                TransverseVelocityPositionDescriptor(**descriptor)
            )
            if descriptor
            else "raw_unresolved"
        )
        route = rollout.get("position_route")
        routes_ok = routes_ok and all(
            (
                rollout.get("buffered_offset") is True,
                rollout.get("fixed_late_ramp") is True,
                rollout.get("persistence_probe") is False,
                route == expected_route,
                bool(rollout.get("position_buffered"))
                == (route in {"raw_uncertainty_buffer", "raw_unresolved"}),
            )
        )
        audits = list(rollout.get("persistence_probe_audits", ()))
        execution = rollout["execution"]
        if route in {"raw_uncertainty_buffer", "raw_unresolved"}:
            raw_ok = raw_ok and all(
                (
                    not audits,
                    not rollout.get("front_audits"),
                    not rollout.get("frozen_offset_audits"),
                    int(execution["logical_model_calls"]) == 30,
                    int(execution["native_logical_calls"]) == 30,
                    int(execution["fine_logical_calls"]) == 0,
                    int(execution["persistence_probe_fine_logical_calls"]) == 0,
                    all(
                        np.array_equal(left, right)
                        for left, right in zip(
                            rollout["accepted_states"],
                            rollout["shadow_states"],
                            strict=True,
                        )
                    ),
                    all(
                        row["correction_status"]
                        == "position_uncertainty_buffer_raw"
                        for row in rollout["accepted_rows"]
                    ),
                )
            )
            continue
        if route == "edge_candidate":
            edge_and_interior_ok = edge_and_interior_ok and not audits and all(
                _frozen_offset_call_checks([rollout], case_ids=(case_id,)).values()
            )
            continue

        expected = [
            fixed_late_terminal_ramp_gain(
                block_index=block,
                ramp_start_block=9,
                terminal_block=14,
            )
            for block in range(4, 15)
        ]
        handoff_call = 2 * FROZEN_OFFSET_WARM_BLOCKS
        accepted_states = rollout["accepted_states"]
        shadow_states = rollout["shadow_states"]
        offset = accepted_states[handoff_call] - shadow_states[handoff_call]
        front_rows = list(rollout.get("front_audits", ()))
        edge_and_interior_ok = edge_and_interior_ok and all(
            (
                route == "interior_frozen_offset",
                len(front_rows) == 15,
                [row["phase"] for row in front_rows[:4]]
                == ["warm_candidate"] * 4,
                [row["phase"] for row in front_rows[4:]]
                == ["fixed_late_ramp_offset_coast"] * 11,
                len(rollout.get("frozen_offset_audits", ())) == 1,
                all(
                    _maximum_abs(
                        accepted_states[call]
                        - shadow_states[call]
                        - fixed_late_terminal_ramp_gain(
                            block_index=(call - 1) // 2,
                            ramp_start_block=9,
                            terminal_block=14,
                        )
                        * offset
                    )
                    <= 1.0e-12
                    for call in range(handoff_call + 1, 31)
                ),
                all(
                    row["correction_status"]
                    == "fixed_late_terminal_ramp_sp19_output_offset"
                    for row in rollout["accepted_rows"][handoff_call:]
                ),
            )
        )
        schedule_ok = schedule_ok and all(
            (
                int(execution["logical_model_calls"]) == 46,
                int(execution["native_logical_calls"]) == 42,
                int(execution["fine_logical_calls"]) == 4,
                int(execution["candidate_native_logical_calls"]) == 12,
                int(execution["candidate_fine_logical_calls"]) == 4,
                int(execution["persistence_probe_fine_logical_calls"]) == 0,
                int(execution["accepted_coast_native_logical_calls"]) == 0,
                int(execution["completed_calls"]) == 30,
                len(audits) == 11,
                [int(row["block_index"]) for row in audits] == list(range(4, 15)),
                all(
                    int(row["output_call"]) == 2 * int(row["block_index"]) + 1
                    for row in audits
                ),
                all(
                    abs(float(row["applied_gain"]) - gain) <= 1.0e-15
                    for row, gain in zip(audits, expected, strict=True)
                ),
                all(
                    row["gain_status"] == "fixed_late_terminal_ramp"
                    for row in audits
                ),
                float(rollout["maxima"]["persistence_gain_increase_abs"])
                <= 1.0e-15,
                float(rollout["maxima"]["persistence_offset_bookkeeping_abs"])
                <= 1.0e-12,
                float(rollout["maxima"]["persistence_within_block_identity_abs"])
                <= 1.0e-12,
                int(rollout["maxima"]["call_order_errors"]) == 0,
            )
        )
    return dict(
        zip(
            keys,
            (True, routes_ok, raw_ok, edge_and_interior_ok, schedule_ok),
            strict=True,
        )
    )


def _shadow_rollout_arm(
    runtime: Any,
    *,
    case_id: str,
    horizon: int,
    retain_reference_states: bool = False,
    native_truth_only: bool = False,
    structural_gate: bool = False,
    diagnostic_shadow_candidates: bool = False,
    warm_start_blocks: int | None = None,
    projected_coast: bool = False,
    frozen_offset: bool = False,
    buffered_offset: bool = False,
    persistence_probe: bool = False,
    terminal_ramp: bool = False,
    fixed_late_ramp: bool = False,
    slew_limited_tether: bool = False,
    relaxed_tether: bool = False,
    buffered_relaxed_tether: bool = False,
    fixed_late_ramp_start_block: int = 9,
    shard_native_reference: bool = False,
) -> dict[str, Any]:
    if horizon < 2 or horizon % 2:
        raise ValueError("raw-shadow rollout horizon must be positive and even")
    if diagnostic_shadow_candidates and not structural_gate:
        raise ValueError("shadow-candidate diagnostics require structural audits")
    if warm_start_blocks is not None:
        if not structural_gate or diagnostic_shadow_candidates:
            raise ValueError(
                "warm-start coast requires a non-diagnostic structural gate"
            )
        if warm_start_blocks < 1 or warm_start_blocks >= horizon // 2:
            raise ValueError("warm-start blocks must leave at least one coast block")
    if projected_coast and warm_start_blocks is None:
        raise ValueError("projected coast requires a warm-start window")
    if slew_limited_tether and warm_start_blocks is None:
        raise ValueError("slew-limited tether requires a warm-start window")
    if relaxed_tether and warm_start_blocks is None:
        raise ValueError("relaxed tether requires a warm-start window")
    if buffered_relaxed_tether and not relaxed_tether:
        raise ValueError("buffered relaxed tether requires relaxed_tether=True")
    if frozen_offset and warm_start_blocks is None:
        raise ValueError("frozen offset requires a warm-start window")
    if buffered_offset and not frozen_offset:
        raise ValueError("buffered offset requires frozen_offset=True")
    if persistence_probe and not buffered_offset:
        raise ValueError("persistence probes require buffered_offset=True")
    if terminal_ramp and not persistence_probe:
        raise ValueError("terminal ramp requires persistence_probe=True")
    if fixed_late_ramp and (persistence_probe or not buffered_offset):
        raise ValueError(
            "fixed late ramp requires buffered offset without persistence probes"
        )
    if sum((projected_coast, frozen_offset, slew_limited_tether, relaxed_tether)) > 1:
        raise ValueError(
            "projected coast, frozen offset, slew-limited tether, and relaxed "
            "tether are mutually exclusive"
        )
    if native_truth_only:
        reference, reference_check = _load_native_scoring_reference(
            runtime,
            case_id,
            shard_native_reference=shard_native_reference,
        )
    else:
        reference, reference_check = load_resolution_reference(
            runtime.args.family_root,
            runtime.args.multires_reference_root,
            runtime.store,
            runtime.manifest,
            case_id,
            training_resolution=NATIVE_RESOLUTION,
        )
    reference_resolution = tuple(
        int(value) for value in reference["retained_resolution"]
    )
    initial = reference_at_resolution(
        reference["conservative_states"][0],
        reference_resolution=reference_resolution,
        target_resolution=NATIVE_RESOLUTION,
    )
    if initial is None:
        raise ValueError("raw-shadow initial reference is unavailable")
    accepted = np.asarray(initial, dtype=np.float64)
    shadow = np.asarray(initial, dtype=np.float64)
    volumes = np.asarray(
        runtime.geometry_by_resolution[NATIVE_RESOLUTION].node_measures,
        dtype=np.float64,
    )
    position_descriptor = (
        transverse_velocity_position_descriptor(
            accepted,
            nodes=runtime.geometry_by_resolution[NATIVE_RESOLUTION].nodes,
            volumes=volumes,
            y_min=runtime.first_config.y_min,
            y_max=runtime.first_config.y_max,
        )
        if structural_gate
        else None
    )
    position_trusted = (
        position_trusts_cross_resolution(position_descriptor)
        if position_descriptor is not None
        else True
    )
    position_route = (
        buffered_frozen_offset_position_route(position_descriptor)
        if (buffered_offset or buffered_relaxed_tether)
        and position_descriptor is not None
        else ("edge_candidate" if position_trusted else "interior_frozen_offset")
    )
    position_buffered = position_route in {
        "raw_uncertainty_buffer",
        "raw_unresolved",
    }
    execution = {
        "logical_model_calls": 0,
        "actual_forward_passes": 0,
        "total_forward_seconds": 0.0,
        "maximum_peak_gpu_memory_bytes": 0,
        "native_logical_calls": 0,
        "fine_logical_calls": 0,
        "native_actual_forward_passes": 0,
        "fine_actual_forward_passes": 0,
        "native_forward_seconds": 0.0,
        "fine_forward_seconds": 0.0,
        "shadow_native_logical_calls": 0,
        "candidate_native_logical_calls": 0,
        "candidate_fine_logical_calls": 0,
        "accepted_coast_native_logical_calls": 0,
        "persistence_probe_fine_logical_calls": 0,
    }
    maxima = {
        "call_order_errors": 0,
        "shadow_recurrence_abs": 0.0,
        "accepted_recurrence_abs": 0.0,
        "candidate_common_native_input_abs": 0.0,
        "candidate_lookahead_input_abs": 0.0,
        "candidate_shadow_integral_abs": 0.0,
        "anchor_post_integral_abs": 0.0,
        "anchor_boundary_abs": 0.0,
        "anchor_idempotence_abs": 0.0,
        "anchor_correction_rms": 0.0,
        "first_correction_integral_abs": 0.0,
        "filtered_response_integral_abs": 0.0,
        "first_boundary_abs": 0.0,
        "filtered_boundary_abs": 0.0,
        "projection_idempotence_abs": 0.0,
        "cap_active_count": 0,
        "front_branch_veto_count": 0,
        "discard_to_raw_recurrence_abs": 0.0,
        "accepted_coast_recurrence_abs": 0.0,
        "accepted_coast_shadow_integral_abs": 0.0,
        "projected_coast_boundary_abs": 0.0,
        "projected_coast_idempotence_abs": 0.0,
        "projected_coast_tether_abs": 0.0,
        "slew_applied_to_shadow_increment_ratio": 0.0,
        "slew_change_limit_violation_rms": 0.0,
        "slew_integral_abs": 0.0,
        "slew_boundary_contraction_violation_abs": 0.0,
        "slew_projection_residual_increase_rms": 0.0,
        "slew_update_identity_abs": 0.0,
        "slew_cap_active_count": 0,
        "frozen_offset_integral_abs": 0.0,
        "frozen_offset_boundary_abs": 0.0,
        "frozen_offset_idempotence_abs": 0.0,
        "frozen_offset_constancy_abs": 0.0,
        "frozen_offset_increment_identity_abs": 0.0,
        "persistence_common_native_input_abs": 0.0,
        "persistence_fine_input_abs": 0.0,
        "persistence_offset_integral_abs": 0.0,
        "persistence_offset_boundary_abs": 0.0,
        "persistence_gain_increase_abs": 0.0,
        "persistence_offset_bookkeeping_abs": 0.0,
        "persistence_within_block_identity_abs": 0.0,
        "persistence_unresolved_count": 0,
    }
    accepted_states = [np.array(accepted, copy=True)]
    shadow_states = [np.array(shadow, copy=True)]
    reference_states = (
        [np.array(accepted, copy=True)] if retain_reference_states else []
    )
    accepted_rows = []
    shadow_rows = []
    anchor_rows = []
    accepted_cumulative = np.zeros_like(accepted)
    shadow_cumulative = np.zeros_like(shadow)
    first_invalid_call = None
    first_nonfinite_call = None
    front_branch_veto_call = None
    front_branch_veto_latched = False
    front_audits = []
    diagnostic_blocks = []
    projected_coast_audits = []
    slew_limited_tether_audits = []
    relaxed_tether_audits = []
    frozen_offset_audits = []
    persistence_probe_audits = []
    persistence_error_structure = []
    frozen_coast_offset = None
    persistence_baseline_alignment = None
    persistence_gain = 1.0
    terminal_ramp_trigger_block = None
    started = perf_counter()

    for block_index in range(horizon // 2):
        accepted_input = np.array(accepted, copy=True)
        shadow_input = np.array(shadow, copy=True)
        if diagnostic_shadow_candidates:
            maxima["discard_to_raw_recurrence_abs"] = max(
                maxima["discard_to_raw_recurrence_abs"],
                _maximum_abs(accepted_input - shadow_input),
            )
            accepted_cumulative = np.array(shadow_cumulative, copy=True)
        calls: list[tuple[tuple[int, int], np.ndarray]] = []
        block_vetoed = False
        front_audit = None
        coast_block = bool(
            warm_start_blocks is not None
            and not position_trusted
            and not position_buffered
            and block_index >= warm_start_blocks
            and not front_branch_veto_latched
        )

        def predictor(resolution, state, call_records=calls):
            call_records.append(
                (resolution, np.asarray(state, dtype=np.float64).copy())
            )
            prediction, timing = predict_resolution_sample(
                runtime.model,
                runtime.sample_by_resolution[resolution],
                state,
                device=runtime.device,
                amp="none",
                repeats=1,
            )
            parent._account_execution(execution, timing, resolution=resolution)
            return prediction

        candidate_block = (
            diagnostic_shadow_candidates
            or not structural_gate
            or (
                not position_buffered
                and not front_branch_veto_latched
                and (
                    position_trusted
                    or (
                        warm_start_blocks is not None
                        and block_index < warm_start_blocks
                    )
                )
            )
        )
        if candidate_block:
            block = synchronized_shadow_anchored_block(
                accepted_input,
                shadow_input,
                contract=parent.RESOLUTION_CONTRACT,
                projector=runtime.native_projector,
                predictor=predictor,
                volumes=volumes,
                residual_scale=runtime.normalization.residual_scale,
                state_scale=runtime.normalization.state_scale,
                active_cells=FROZEN_ACTIVE_CELLS,
            )
            execution["shadow_native_logical_calls"] += 2
            execution["candidate_native_logical_calls"] += 3
            execution["candidate_fine_logical_calls"] += 1
            expected_order = [
                parent.RESOLUTION_CONTRACT.native,
                parent.RESOLUTION_CONTRACT.native,
                parent.RESOLUTION_CONTRACT.native,
                parent.RESOLUTION_CONTRACT.fine,
                parent.RESOLUTION_CONTRACT.native,
                parent.RESOLUTION_CONTRACT.native,
            ]
            if [resolution for resolution, _ in calls] != expected_order:
                maxima["call_order_errors"] += 1
            prepared = prepare_common_native_inputs(
                accepted_input, contract=parent.RESOLUTION_CONTRACT
            )
            maxima["shadow_recurrence_abs"] = max(
                maxima["shadow_recurrence_abs"],
                _maximum_abs(calls[0][1] - shadow_input),
                _maximum_abs(calls[1][1] - block.shadow_first_state),
            )
            maxima["accepted_recurrence_abs"] = max(
                maxima["accepted_recurrence_abs"],
                _maximum_abs(calls[2][1] - accepted_input),
            )
            maxima["candidate_common_native_input_abs"] = max(
                maxima["candidate_common_native_input_abs"],
                _maximum_abs(
                    calls[2][1]
                    - prepared.model_inputs[parent.RESOLUTION_CONTRACT.native]
                ),
                _maximum_abs(
                    calls[3][1] - prepared.model_inputs[parent.RESOLUTION_CONTRACT.fine]
                ),
            )
            maxima["candidate_lookahead_input_abs"] = max(
                maxima["candidate_lookahead_input_abs"],
                _maximum_abs(calls[4][1] - block.candidate_raw_first_state),
                _maximum_abs(calls[5][1] - block.anchored_first_state),
            )
            response_audit = block.response_audit
            for output_call, audit in (
                (2 * block_index + 1, block.first_anchor_audit),
                (2 * block_index + 2, block.second_anchor_audit),
            ):
                maxima["anchor_post_integral_abs"] = max(
                    maxima["anchor_post_integral_abs"],
                    audit.maximum_integral_mismatch_after_abs,
                )
                maxima["anchor_boundary_abs"] = max(
                    maxima["anchor_boundary_abs"],
                    audit.maximum_boundary_correction_abs,
                )
                maxima["anchor_idempotence_abs"] = max(
                    maxima["anchor_idempotence_abs"],
                    audit.maximum_idempotence_abs,
                )
                maxima["anchor_correction_rms"] = max(
                    maxima["anchor_correction_rms"], audit.correction_rms
                )
                anchor_rows.append(
                    {
                        "case_id": case_id,
                        "block_index": block_index,
                        "output_call": output_call,
                        **asdict(audit),
                    }
                )
            maxima["first_correction_integral_abs"] = max(
                maxima["first_correction_integral_abs"],
                response_audit.maximum_first_correction_integral_abs,
            )
            maxima["filtered_response_integral_abs"] = max(
                maxima["filtered_response_integral_abs"],
                response_audit.maximum_filtered_response_integral_abs,
            )
            maxima["first_boundary_abs"] = max(
                maxima["first_boundary_abs"],
                response_audit.maximum_first_boundary_abs,
            )
            maxima["filtered_boundary_abs"] = max(
                maxima["filtered_boundary_abs"],
                response_audit.maximum_filtered_boundary_abs,
            )
            maxima["projection_idempotence_abs"] = max(
                maxima["projection_idempotence_abs"],
                response_audit.maximum_projection_idempotence_abs,
            )
            maxima["cap_active_count"] += int(block.first_audit.cap_active)

            next_accepted = (
                block.anchored_first_state,
                block.anchored_second_state,
            )
            next_shadow = (block.shadow_first_state, block.shadow_second_state)
            raw_candidate = (
                block.candidate_raw_first_state,
                block.candidate_raw_second_state,
            )
            if (
                frozen_offset
                and not position_trusted
                and not position_buffered
                and warm_start_blocks is not None
                and block_index == warm_start_blocks - 1
            ):
                handoff_state, handoff_offset, handoff_audit = projected_shadow_tether(
                    next_accepted[1],
                    next_shadow[1],
                    projector=runtime.native_projector,
                    volumes=volumes,
                    component_scale=runtime.normalization.state_scale,
                    active_cells=FROZEN_ACTIVE_CELLS,
                )
                next_accepted = (next_accepted[0], handoff_state)
                frozen_coast_offset = handoff_offset
                maxima["frozen_offset_integral_abs"] = max(
                    maxima["frozen_offset_integral_abs"],
                    handoff_audit.maximum_integral_difference_abs,
                )
                maxima["frozen_offset_boundary_abs"] = max(
                    maxima["frozen_offset_boundary_abs"],
                    handoff_audit.maximum_boundary_difference_abs,
                )
                maxima["frozen_offset_idempotence_abs"] = max(
                    maxima["frozen_offset_idempotence_abs"],
                    handoff_audit.maximum_projection_idempotence_abs,
                )
                frozen_offset_audits.append(
                    {
                        "case_id": case_id,
                        "block_index": block_index,
                        "output_call": 2 * block_index + 2,
                        "phase": "handoff",
                        **asdict(handoff_audit),
                    }
                )
            if structural_gate:
                front_audit = target_free_front_branch_audit(
                    *next_accepted,
                    *next_shadow,
                    resolution=NATIVE_RESOLUTION,
                    x_min=runtime.first_config.x_min,
                    x_max=runtime.first_config.x_max,
                    gamma=runtime.normalization.gamma,
                    shock_center_x=runtime.first_config.shock_x,
                )
                front_audits.append(
                    {
                        "case_id": case_id,
                        "block_index": block_index,
                        "phase": (
                            "warm_candidate"
                            if warm_start_blocks is not None and not position_trusted
                            else "trusted_candidate"
                        ),
                        "first_output_call": 2 * block_index + 1,
                        "second_output_call": 2 * block_index + 2,
                        **asdict(front_audit),
                    }
                )
                if front_audit.branch_changed and not diagnostic_shadow_candidates:
                    next_accepted = next_shadow
                    frozen_coast_offset = None
                    front_branch_veto_latched = True
                    front_branch_veto_call = 2 * block_index + 1
                    maxima["front_branch_veto_count"] += 1
                    block_vetoed = True
        elif coast_block:
            shadow_first = np.asarray(
                predictor(parent.RESOLUTION_CONTRACT.native, shadow_input),
                dtype=np.float64,
            )
            shadow_second = np.asarray(
                predictor(parent.RESOLUTION_CONTRACT.native, shadow_first),
                dtype=np.float64,
            )
            accepted_raw_first = None
            if frozen_offset:
                if frozen_coast_offset is None:
                    raise AssertionError("frozen-offset coast lacks its handoff offset")
                previous_persistence_gain = persistence_gain
                if fixed_late_ramp:
                    persistence_gain = fixed_late_terminal_ramp_gain(
                        block_index=block_index,
                        ramp_start_block=fixed_late_ramp_start_block,
                        terminal_block=horizon // 2 - 1,
                    )
                    applied_offset = persistence_gain * frozen_coast_offset
                if persistence_probe:
                    probe = synchronized_persistence_probe(
                        shadow_input,
                        shadow_first,
                        contract=parent.RESOLUTION_CONTRACT,
                        projector=runtime.native_projector,
                        fine_predictor=predictor,
                        volumes=volumes,
                        component_scale=runtime.normalization.residual_scale,
                        active_cells=FROZEN_ACTIVE_CELLS,
                    )
                    execution["persistence_probe_fine_logical_calls"] += 1
                    gain_audit = monotone_persistence_gain(
                        frozen_coast_offset,
                        probe.correction,
                        baseline_alignment=persistence_baseline_alignment,
                        previous_gain=persistence_gain,
                        volumes=volumes,
                        component_scale=runtime.normalization.state_scale,
                    )
                    if persistence_baseline_alignment is None:
                        persistence_baseline_alignment = gain_audit.baseline_alignment
                    if terminal_ramp:
                        ramp_audit = phase_triggered_terminal_ramp_gain(
                            block_index=block_index,
                            terminal_block=horizon // 2 - 1,
                            alignment_cosine=gain_audit.alignment_cosine,
                            trigger_block=terminal_ramp_trigger_block,
                        )
                        terminal_ramp_trigger_block = ramp_audit.trigger_block
                        persistence_gain = ramp_audit.applied_gain
                    else:
                        ramp_audit = None
                        persistence_gain = gain_audit.applied_gain
                    maxima["persistence_unresolved_count"] += int(
                        gain_audit.alignment_cosine is None
                        if terminal_ramp
                        else gain_audit.status != "ok"
                    )
                    maxima["persistence_gain_increase_abs"] = max(
                        maxima["persistence_gain_increase_abs"],
                        persistence_gain - previous_persistence_gain,
                    )
                    maxima["persistence_common_native_input_abs"] = max(
                        maxima["persistence_common_native_input_abs"],
                        _maximum_abs(calls[0][1] - probe.native_model_input),
                    )
                    maxima["persistence_fine_input_abs"] = max(
                        maxima["persistence_fine_input_abs"],
                        _maximum_abs(calls[2][1] - probe.fine_model_input),
                    )
                    applied_offset = persistence_gain * frozen_coast_offset
                    boundary = applied_offset[
                        ~np.asarray(runtime.native_projector.interior_mask, dtype=bool)
                    ]
                    maxima["persistence_offset_boundary_abs"] = max(
                        maxima["persistence_offset_boundary_abs"],
                        _maximum_abs(boundary),
                    )
                    maxima["persistence_offset_integral_abs"] = max(
                        maxima["persistence_offset_integral_abs"],
                        _maximum_abs(
                            collection._physical_integral(applied_offset, volumes)
                        ),
                    )
                    persistence_probe_audits.append(
                        {
                            "case_id": case_id,
                            "block_index": block_index,
                            "output_call": 2 * block_index + 1,
                            "previous_gain": previous_persistence_gain,
                            **asdict(gain_audit),
                            "gain_status": (
                                ramp_audit.status
                                if ramp_audit is not None
                                else gain_audit.status
                            ),
                            "ramp_trigger_block": (
                                ramp_audit.trigger_block
                                if ramp_audit is not None
                                else None
                            ),
                            "applied_gain": persistence_gain,
                            **{
                                f"probe_{key}": value
                                for key, value in asdict(probe.audit).items()
                            },
                        }
                    )
                elif not fixed_late_ramp:
                    applied_offset = frozen_coast_offset
                accepted_first = shadow_first + applied_offset
                retained_first = applied_offset
                first_tether_audit = None
            else:
                accepted_raw_first = np.asarray(
                    predictor(parent.RESOLUTION_CONTRACT.native, accepted_input),
                    dtype=np.float64,
                )
            if projected_coast:
                accepted_first, retained_first, first_tether_audit = (
                    projected_shadow_tether(
                        accepted_raw_first,
                        shadow_first,
                        projector=runtime.native_projector,
                        volumes=volumes,
                        component_scale=runtime.normalization.state_scale,
                        active_cells=FROZEN_ACTIVE_CELLS,
                    )
                )
            elif slew_limited_tether:
                accepted_first, retained_first, first_tether_audit = (
                    slew_limited_projected_shadow_tether(
                        accepted_raw_first,
                        shadow_first,
                        accepted_input,
                        shadow_input,
                        projector=runtime.native_projector,
                        volumes=volumes,
                        component_scale=runtime.normalization.state_scale,
                        relative_change_limit=(
                            SLEW_LIMITED_TETHER_RELATIVE_CHANGE_LIMIT
                        ),
                        active_cells=FROZEN_ACTIVE_CELLS,
                    )
                )
            elif relaxed_tether:
                accepted_first, retained_first, first_tether_audit = (
                    relaxed_projected_shadow_tether(
                        accepted_raw_first,
                        shadow_first,
                        accepted_input,
                        shadow_input,
                        projector=runtime.native_projector,
                        volumes=volumes,
                        component_scale=runtime.normalization.state_scale,
                        relaxation=RELAXED_TETHER_RATE,
                        active_cells=FROZEN_ACTIVE_CELLS,
                    )
                )
            elif not frozen_offset:
                accepted_first = accepted_raw_first
                retained_first = accepted_first - shadow_first
                first_tether_audit = None
            accepted_raw_second = None
            if frozen_offset:
                accepted_second = shadow_second + applied_offset
                retained_second = applied_offset
                second_tether_audit = None
            else:
                accepted_raw_second = np.asarray(
                    predictor(parent.RESOLUTION_CONTRACT.native, accepted_first),
                    dtype=np.float64,
                )
            if projected_coast:
                accepted_second, retained_second, second_tether_audit = (
                    projected_shadow_tether(
                        accepted_raw_second,
                        shadow_second,
                        projector=runtime.native_projector,
                        volumes=volumes,
                        component_scale=runtime.normalization.state_scale,
                        active_cells=FROZEN_ACTIVE_CELLS,
                    )
                )
            elif slew_limited_tether:
                accepted_second, retained_second, second_tether_audit = (
                    slew_limited_projected_shadow_tether(
                        accepted_raw_second,
                        shadow_second,
                        accepted_first,
                        shadow_first,
                        projector=runtime.native_projector,
                        volumes=volumes,
                        component_scale=runtime.normalization.state_scale,
                        relative_change_limit=(
                            SLEW_LIMITED_TETHER_RELATIVE_CHANGE_LIMIT
                        ),
                        active_cells=FROZEN_ACTIVE_CELLS,
                    )
                )
            elif relaxed_tether:
                accepted_second, retained_second, second_tether_audit = (
                    relaxed_projected_shadow_tether(
                        accepted_raw_second,
                        shadow_second,
                        accepted_first,
                        shadow_first,
                        projector=runtime.native_projector,
                        volumes=volumes,
                        component_scale=runtime.normalization.state_scale,
                        relaxation=RELAXED_TETHER_RATE,
                        active_cells=FROZEN_ACTIVE_CELLS,
                    )
                )
            elif not frozen_offset:
                accepted_second = accepted_raw_second
                retained_second = accepted_second - shadow_second
                second_tether_audit = None
            execution["shadow_native_logical_calls"] += 2
            if not frozen_offset:
                execution["accepted_coast_native_logical_calls"] += 2
            expected_coast_order = [
                parent.RESOLUTION_CONTRACT.native,
                parent.RESOLUTION_CONTRACT.native,
            ]
            if persistence_probe:
                expected_coast_order.append(parent.RESOLUTION_CONTRACT.fine)
            if not frozen_offset:
                expected_coast_order.extend(
                    [
                        parent.RESOLUTION_CONTRACT.native,
                        parent.RESOLUTION_CONTRACT.native,
                    ]
                )
            if [resolution for resolution, _ in calls] != expected_coast_order:
                maxima["call_order_errors"] += 1
            maxima["shadow_recurrence_abs"] = max(
                maxima["shadow_recurrence_abs"],
                _maximum_abs(calls[0][1] - shadow_input),
                _maximum_abs(calls[1][1] - shadow_first),
            )
            if not frozen_offset:
                maxima["accepted_coast_recurrence_abs"] = max(
                    maxima["accepted_coast_recurrence_abs"],
                    _maximum_abs(calls[2][1] - accepted_input),
                    _maximum_abs(calls[3][1] - accepted_first),
                )
            next_shadow = (shadow_first, shadow_second)
            next_accepted = (accepted_first, accepted_second)
            raw_candidate = (
                next_shadow
                if frozen_offset
                else (accepted_raw_first, accepted_raw_second)
            )
            if frozen_offset:
                if persistence_probe or fixed_late_ramp:
                    expected_gain_change = (
                        persistence_gain - previous_persistence_gain
                    ) * frozen_coast_offset
                    maxima["persistence_offset_bookkeeping_abs"] = max(
                        maxima["persistence_offset_bookkeeping_abs"],
                        _maximum_abs(
                            (accepted_first - accepted_input)
                            - (shadow_first - shadow_input)
                            - expected_gain_change
                        ),
                    )
                    maxima["persistence_within_block_identity_abs"] = max(
                        maxima["persistence_within_block_identity_abs"],
                        _maximum_abs(
                            (accepted_second - accepted_first)
                            - (shadow_second - shadow_first)
                        ),
                    )
                    if fixed_late_ramp:
                        persistence_probe_audits.append(
                            {
                                "case_id": case_id,
                                "block_index": block_index,
                                "output_call": 2 * block_index + 1,
                                "previous_gain": previous_persistence_gain,
                                "applied_gain": persistence_gain,
                                "gain_status": "fixed_late_terminal_ramp",
                            }
                        )
                else:
                    maxima["frozen_offset_constancy_abs"] = max(
                        maxima["frozen_offset_constancy_abs"],
                        _maximum_abs(retained_first - frozen_coast_offset),
                        _maximum_abs(retained_second - frozen_coast_offset),
                    )
                    maxima["frozen_offset_increment_identity_abs"] = max(
                        maxima["frozen_offset_increment_identity_abs"],
                        _maximum_abs(
                            (accepted_first - accepted_input)
                            - (shadow_first - shadow_input)
                        ),
                        _maximum_abs(
                            (accepted_second - accepted_first)
                            - (shadow_second - shadow_first)
                        ),
                    )
            if projected_coast:
                for offset, (retained, audit) in enumerate(
                    (
                        (retained_first, first_tether_audit),
                        (retained_second, second_tether_audit),
                    )
                ):
                    if audit is None:  # pragma: no cover - guarded by construction
                        raise AssertionError("projected coast requires a tether audit")
                    maxima["projected_coast_boundary_abs"] = max(
                        maxima["projected_coast_boundary_abs"],
                        audit.maximum_boundary_difference_abs,
                    )
                    maxima["projected_coast_idempotence_abs"] = max(
                        maxima["projected_coast_idempotence_abs"],
                        audit.maximum_projection_idempotence_abs,
                    )
                    maxima["projected_coast_tether_abs"] = max(
                        maxima["projected_coast_tether_abs"],
                        _maximum_abs(
                            next_accepted[offset] - next_shadow[offset] - retained
                        ),
                    )
                    projected_coast_audits.append(
                        {
                            "case_id": case_id,
                            "block_index": block_index,
                            "output_call": 2 * block_index + offset + 1,
                            **asdict(audit),
                        }
                    )
                front_audit = target_free_front_branch_audit(
                    *next_accepted,
                    *next_shadow,
                    resolution=NATIVE_RESOLUTION,
                    x_min=runtime.first_config.x_min,
                    x_max=runtime.first_config.x_max,
                    gamma=runtime.normalization.gamma,
                    shock_center_x=runtime.first_config.shock_x,
                )
                front_audits.append(
                    {
                        "case_id": case_id,
                        "block_index": block_index,
                        "phase": "projected_coast",
                        "first_output_call": 2 * block_index + 1,
                        "second_output_call": 2 * block_index + 2,
                        **asdict(front_audit),
                    }
                )
                if front_audit.branch_changed:
                    next_accepted = next_shadow
                    front_branch_veto_latched = True
                    front_branch_veto_call = 2 * block_index + 1
                    maxima["front_branch_veto_count"] += 1
                    block_vetoed = True
            elif slew_limited_tether:
                for offset, (retained, audit) in enumerate(
                    (
                        (retained_first, first_tether_audit),
                        (retained_second, second_tether_audit),
                    )
                ):
                    if audit is None:  # pragma: no cover - guarded by construction
                        raise AssertionError("slew-limited coast requires an audit")
                    maxima["slew_applied_to_shadow_increment_ratio"] = max(
                        maxima["slew_applied_to_shadow_increment_ratio"],
                        audit.applied_to_shadow_increment_ratio or 0.0,
                    )
                    maxima["slew_change_limit_violation_rms"] = max(
                        maxima["slew_change_limit_violation_rms"],
                        audit.applied_change_rms - audit.change_limit_rms,
                    )
                    maxima["slew_integral_abs"] = max(
                        maxima["slew_integral_abs"],
                        audit.maximum_integral_difference_abs,
                    )
                    maxima["slew_boundary_contraction_violation_abs"] = max(
                        maxima["slew_boundary_contraction_violation_abs"],
                        audit.maximum_boundary_contraction_violation_abs,
                    )
                    maxima["slew_projection_residual_increase_rms"] = max(
                        maxima["slew_projection_residual_increase_rms"],
                        audit.retained_projection_residual_rms
                        - audit.previous_projection_residual_rms,
                    )
                    maxima["slew_update_identity_abs"] = max(
                        maxima["slew_update_identity_abs"],
                        audit.maximum_update_identity_abs,
                        _maximum_abs(
                            next_accepted[offset] - next_shadow[offset] - retained
                        ),
                    )
                    maxima["slew_cap_active_count"] += int(audit.cap_active)
                    slew_limited_tether_audits.append(
                        {
                            "case_id": case_id,
                            "block_index": block_index,
                            "output_call": 2 * block_index + offset + 1,
                            **asdict(audit),
                        }
                    )
                front_audit = target_free_front_branch_audit(
                    *next_accepted,
                    *next_shadow,
                    resolution=NATIVE_RESOLUTION,
                    x_min=runtime.first_config.x_min,
                    x_max=runtime.first_config.x_max,
                    gamma=runtime.normalization.gamma,
                    shock_center_x=runtime.first_config.shock_x,
                )
                front_audits.append(
                    {
                        "case_id": case_id,
                        "block_index": block_index,
                        "phase": "slew_limited_tether_coast",
                        "first_output_call": 2 * block_index + 1,
                        "second_output_call": 2 * block_index + 2,
                        **asdict(front_audit),
                    }
                )
                if front_audit.branch_changed:
                    next_accepted = next_shadow
                    front_branch_veto_latched = True
                    front_branch_veto_call = 2 * block_index + 1
                    maxima["front_branch_veto_count"] += 1
                    block_vetoed = True
            elif relaxed_tether:
                for offset, (retained, audit) in enumerate(
                    (
                        (retained_first, first_tether_audit),
                        (retained_second, second_tether_audit),
                    )
                ):
                    if audit is None:  # pragma: no cover - guarded by construction
                        raise AssertionError("relaxed coast requires an audit")
                    maxima["slew_applied_to_shadow_increment_ratio"] = max(
                        maxima["slew_applied_to_shadow_increment_ratio"],
                        audit.applied_to_shadow_increment_ratio or 0.0,
                    )
                    maxima["slew_change_limit_violation_rms"] = max(
                        maxima["slew_change_limit_violation_rms"],
                        abs(
                            audit.applied_change_rms
                            - RELAXED_TETHER_RATE * audit.requested_change_rms
                        ),
                    )
                    maxima["slew_integral_abs"] = max(
                        maxima["slew_integral_abs"],
                        audit.maximum_integral_difference_abs,
                    )
                    maxima["slew_boundary_contraction_violation_abs"] = max(
                        maxima["slew_boundary_contraction_violation_abs"],
                        audit.maximum_boundary_contraction_violation_abs,
                    )
                    maxima["slew_projection_residual_increase_rms"] = max(
                        maxima["slew_projection_residual_increase_rms"],
                        audit.retained_projection_residual_rms
                        - audit.previous_projection_residual_rms,
                    )
                    maxima["slew_update_identity_abs"] = max(
                        maxima["slew_update_identity_abs"],
                        audit.maximum_update_identity_abs,
                        _maximum_abs(
                            next_accepted[offset] - next_shadow[offset] - retained
                        ),
                    )
                    maxima["slew_cap_active_count"] += int(audit.cap_active)
                    relaxed_tether_audits.append(
                        {
                            "case_id": case_id,
                            "block_index": block_index,
                            "output_call": 2 * block_index + offset + 1,
                            **asdict(audit),
                        }
                    )
                front_audit = target_free_front_branch_audit(
                    *next_accepted,
                    *next_shadow,
                    resolution=NATIVE_RESOLUTION,
                    x_min=runtime.first_config.x_min,
                    x_max=runtime.first_config.x_max,
                    gamma=runtime.normalization.gamma,
                    shock_center_x=runtime.first_config.shock_x,
                )
                front_audits.append(
                    {
                        "case_id": case_id,
                        "block_index": block_index,
                        "phase": "relaxed_tether_coast",
                        "first_output_call": 2 * block_index + 1,
                        "second_output_call": 2 * block_index + 2,
                        **asdict(front_audit),
                    }
                )
                if front_audit.branch_changed:
                    next_accepted = next_shadow
                    front_branch_veto_latched = True
                    front_branch_veto_call = 2 * block_index + 1
                    maxima["front_branch_veto_count"] += 1
                    block_vetoed = True
            elif frozen_offset:
                front_audit = target_free_front_branch_audit(
                    *next_accepted,
                    *next_shadow,
                    resolution=NATIVE_RESOLUTION,
                    x_min=runtime.first_config.x_min,
                    x_max=runtime.first_config.x_max,
                    gamma=runtime.normalization.gamma,
                    shock_center_x=runtime.first_config.shock_x,
                )
                front_audits.append(
                    {
                        "case_id": case_id,
                        "block_index": block_index,
                        "phase": (
                            "fixed_late_ramp_offset_coast"
                            if fixed_late_ramp
                            else (
                                "terminal_ramp_offset_coast"
                                if terminal_ramp
                                else (
                                    "persistence_offset_coast"
                                    if persistence_probe
                                    else "frozen_offset_coast"
                                )
                            )
                        ),
                        "first_output_call": 2 * block_index + 1,
                        "second_output_call": 2 * block_index + 2,
                        **asdict(front_audit),
                    }
                )
                if front_audit.branch_changed:
                    next_accepted = next_shadow
                    frozen_coast_offset = None
                    front_branch_veto_latched = True
                    front_branch_veto_call = 2 * block_index + 1
                    maxima["front_branch_veto_count"] += 1
                    block_vetoed = True
        else:
            shadow_first = np.asarray(
                predictor(parent.RESOLUTION_CONTRACT.native, shadow_input),
                dtype=np.float64,
            )
            shadow_second = np.asarray(
                predictor(parent.RESOLUTION_CONTRACT.native, shadow_first),
                dtype=np.float64,
            )
            execution["shadow_native_logical_calls"] += 2
            if [resolution for resolution, _ in calls] != [
                parent.RESOLUTION_CONTRACT.native,
                parent.RESOLUTION_CONTRACT.native,
            ]:
                maxima["call_order_errors"] += 1
            maxima["shadow_recurrence_abs"] = max(
                maxima["shadow_recurrence_abs"],
                _maximum_abs(calls[0][1] - shadow_input),
                _maximum_abs(calls[1][1] - shadow_first),
            )
            next_shadow = (shadow_first, shadow_second)
            next_accepted = next_shadow
            raw_candidate = next_shadow
        previous_accepted = accepted_input
        previous_shadow = shadow_input
        block_shadow_rows = []
        block_candidate_rows = []
        for offset in range(2):
            input_call = 2 * block_index + offset
            current_reference = reference_at_resolution(
                reference["conservative_states"][input_call * 2],
                reference_resolution=reference_resolution,
                target_resolution=NATIVE_RESOLUTION,
            )
            target = reference_at_resolution(
                reference["conservative_states"][(input_call + 1) * 2],
                reference_resolution=reference_resolution,
                target_resolution=NATIVE_RESOLUTION,
            )
            if current_reference is None or target is None:
                raise ValueError("raw-shadow rollout target is unavailable")
            if retain_reference_states:
                reference_states.append(np.asarray(target, dtype=np.float64).copy())
            if persistence_probe and coast_block and offset == 0:
                if frozen_coast_offset is None or not persistence_probe_audits:
                    raise AssertionError("persistence diagnostic lacks probe state")
                persistence_error_structure.append(
                    _persistence_error_structure_row(
                        runtime,
                        case_id=case_id,
                        block_index=block_index,
                        raw_prediction=next_shadow[offset],
                        corrected_prediction=next_accepted[offset],
                        target=np.asarray(target, dtype=np.float64),
                        probe_correction=probe.correction,
                        frozen_offset=frozen_coast_offset,
                        applied_gain=persistence_gain,
                    )
                )
            shadow_row, shadow_cumulative = _block_metric_row(
                runtime,
                case_id=case_id,
                policy="zero",
                input_call=input_call,
                previous_state=previous_shadow,
                next_state=next_shadow[offset],
                current_reference=np.asarray(current_reference, dtype=np.float64),
                target=np.asarray(target, dtype=np.float64),
                cumulative_defect=shadow_cumulative,
                intervention=np.zeros_like(shadow_input),
                cap_active=False,
                correction_status="raw_shadow",
            )
            intervention = next_accepted[offset] - raw_candidate[offset]
            if diagnostic_shadow_candidates:
                accepted_policy = "shadow_candidate_diagnostic"
                correction_status = (
                    "diagnostic_sp19_plus_integral_anchor"
                    if offset == 0
                    else "diagnostic_filtered_response_plus_integral_anchor"
                )
            elif block_vetoed:
                accepted_policy = (
                    "buffered_relaxed_tether"
                    if buffered_relaxed_tether
                    else (
                        "buffered_offset"
                        if buffered_offset
                        else (
                            "relaxed_tether"
                            if relaxed_tether
                            else (
                                "slew_limited_tether"
                                if slew_limited_tether
                                else (
                                    "projected_coast"
                                    if projected_coast
                                    else "structural_gated"
                                )
                            )
                        )
                    )
                )
                correction_status = "front_branch_veto_raw_shadow"
            elif coast_block:
                accepted_policy = (
                    "buffered_relaxed_tether"
                    if buffered_relaxed_tether
                    else (
                        "buffered_offset"
                        if buffered_offset
                        else (
                            "frozen_offset"
                            if frozen_offset
                            else (
                                "relaxed_tether"
                                if relaxed_tether
                                else (
                                    "slew_limited_tether"
                                    if slew_limited_tether
                                    else (
                                        "projected_coast"
                                        if projected_coast
                                        else "warm_start_coast"
                                    )
                                )
                            )
                        )
                    )
                )
                correction_status = (
                    "fixed_late_terminal_ramp_sp19_output_offset"
                    if fixed_late_ramp
                    else (
                        "phase_triggered_terminal_ramp_sp19_output_offset"
                        if terminal_ramp
                        else (
                            "persistence_gated_sp19_output_offset"
                            if persistence_probe
                            else (
                                "frozen_sp19_output_offset"
                                if frozen_offset
                                else (
                                    "slew_limited_sp19_tether"
                                    if slew_limited_tether
                                    else (
                                        "relaxed_sp19_tether"
                                        if relaxed_tether
                                        else (
                                            "projected_coast_sp19_tether"
                                            if projected_coast
                                            else "accepted_native_coast"
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            elif not structural_gate:
                accepted_policy = "shadow_anchored"
                correction_status = (
                    "sp19_plus_integral_anchor"
                    if offset == 0
                    else "filtered_response_plus_integral_anchor"
                )
            elif not candidate_block:
                accepted_policy = (
                    "buffered_relaxed_tether"
                    if buffered_relaxed_tether
                    else (
                        "buffered_offset"
                        if buffered_offset
                        else (
                            "frozen_offset"
                            if frozen_offset
                            else (
                                "relaxed_tether"
                                if relaxed_tether
                                else (
                                    "slew_limited_tether"
                                    if slew_limited_tether
                                    else (
                                        "projected_coast"
                                        if projected_coast
                                        else "structural_gated"
                                    )
                                )
                            )
                        )
                    )
                )
                correction_status = (
                    "position_uncertainty_buffer_raw"
                    if position_buffered
                    else (
                        "position_rejected_raw"
                        if not position_trusted
                        else "latched_raw_fallback"
                    )
                )
            else:
                accepted_policy = (
                    "buffered_relaxed_tether"
                    if buffered_relaxed_tether
                    else (
                        "buffered_offset"
                        if buffered_offset
                        else (
                            "frozen_offset"
                            if frozen_offset
                            else (
                                "relaxed_tether"
                                if relaxed_tether
                                else (
                                    "slew_limited_tether"
                                    if slew_limited_tether
                                    else (
                                        "projected_coast"
                                        if projected_coast
                                        else (
                                            "warm_start_coast"
                                            if warm_start_blocks is not None
                                            else "structural_gated"
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
                if warm_start_blocks is not None and not position_trusted:
                    correction_status = (
                        "warm_sp19_plus_integral_anchor"
                        if offset == 0
                        else "warm_filtered_response_plus_integral_anchor"
                    )
                else:
                    correction_status = (
                        "trusted_sp19_plus_integral_anchor"
                        if offset == 0
                        else "trusted_filtered_response_plus_integral_anchor"
                    )
            accepted_row, accepted_cumulative = _block_metric_row(
                runtime,
                case_id=case_id,
                policy=accepted_policy,
                input_call=input_call,
                previous_state=previous_accepted,
                next_state=next_accepted[offset],
                current_reference=np.asarray(current_reference, dtype=np.float64),
                target=np.asarray(target, dtype=np.float64),
                cumulative_defect=accepted_cumulative,
                intervention=intervention,
                cap_active=(
                    bool(block.first_audit.cap_active)
                    if candidate_block and not block_vetoed and offset == 0
                    else False
                ),
                correction_status=correction_status,
            )
            integral_difference = collection._physical_integral(
                next_accepted[offset] - next_shadow[offset], volumes
            )
            if coast_block:
                maxima["accepted_coast_shadow_integral_abs"] = max(
                    maxima["accepted_coast_shadow_integral_abs"],
                    _maximum_abs(integral_difference),
                )
            else:
                maxima["candidate_shadow_integral_abs"] = max(
                    maxima["candidate_shadow_integral_abs"],
                    _maximum_abs(integral_difference),
                )
            shadow_rows.append(shadow_row)
            accepted_rows.append(accepted_row)
            block_shadow_rows.append(shadow_row)
            block_candidate_rows.append(accepted_row)
            previous_shadow = next_shadow[offset]
            previous_accepted = next_accepted[offset]
            shadow_states.append(np.array(next_shadow[offset], copy=True))
            accepted_states.append(np.array(next_accepted[offset], copy=True))
            for row in (shadow_row, accepted_row):
                if not row["finite"] and first_nonfinite_call is None:
                    first_nonfinite_call = input_call
                if (
                    not row["finite"] or not row["admissible"]
                ) and first_invalid_call is None:
                    first_invalid_call = input_call
        if diagnostic_shadow_candidates:
            if block is None or front_audit is None:  # pragma: no cover
                raise AssertionError("diagnostic block and front audit are required")
            diagnostic_blocks.append(
                {
                    **_shadow_candidate_feature_row(
                        case_id=case_id,
                        block_index=block_index,
                        raw_input=shadow_input,
                        block=block,
                        volumes=volumes,
                        residual_scale=np.asarray(
                            runtime.normalization.residual_scale, dtype=np.float64
                        ),
                        position_descriptor=position_descriptor,
                        front_audit=front_audit,
                    ),
                    **_shadow_candidate_label_row(
                        raw_rows=block_shadow_rows,
                        candidate_rows=block_candidate_rows,
                    ),
                }
            )
        shadow = np.array(next_shadow[1], copy=True)
        accepted = np.array(
            next_shadow[1] if diagnostic_shadow_candidates else next_accepted[1],
            copy=True,
        )

    execution["wall_seconds"] = perf_counter() - started
    execution["completed_calls"] = horizon
    return {
        "case_id": case_id,
        "policy": (
            "shadow_candidate_diagnostic"
            if diagnostic_shadow_candidates
            else (
                "fixed_late_ramp"
                if fixed_late_ramp
                else (
                    "terminal_ramp"
                    if terminal_ramp
                    else (
                        "persistence_gain"
                        if persistence_probe
                        else (
                            "frozen_offset"
                            if frozen_offset and not buffered_offset
                            else (
                                "buffered_relaxed_tether"
                                if buffered_relaxed_tether
                                else (
                                    "buffered_offset"
                                    if buffered_offset
                                    else (
                                        "relaxed_tether"
                                        if relaxed_tether
                                        else (
                                            "slew_limited_tether"
                                            if slew_limited_tether
                                            else (
                                                "projected_coast"
                                                if projected_coast
                                                else (
                                                    "warm_start_coast"
                                                    if warm_start_blocks is not None
                                                    else (
                                                        "structural_gated"
                                                        if structural_gate
                                                        else "shadow_anchored"
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        ),
        "rows": [*shadow_rows, *accepted_rows],
        "shadow_rows": shadow_rows,
        "accepted_rows": accepted_rows,
        "shadow_states": shadow_states,
        "accepted_states": accepted_states,
        "reference_states": reference_states,
        "anchor_rows": anchor_rows,
        "front_audits": front_audits,
        "diagnostic_blocks": diagnostic_blocks,
        "projected_coast_audits": projected_coast_audits,
        "slew_limited_tether_audits": slew_limited_tether_audits,
        "relaxed_tether_audits": relaxed_tether_audits,
        "frozen_offset_audits": frozen_offset_audits,
        "persistence_probe_audits": persistence_probe_audits,
        "persistence_error_structure": persistence_error_structure,
        "position_descriptor": (
            asdict(position_descriptor) if position_descriptor is not None else None
        ),
        "position_trusted": position_trusted,
        "position_route": position_route,
        "position_buffered": position_buffered,
        "front_branch_veto_call": front_branch_veto_call,
        "warm_start_blocks": warm_start_blocks,
        "projected_coast": projected_coast,
        "slew_limited_tether": slew_limited_tether,
        "relaxed_tether": relaxed_tether,
        "buffered_relaxed_tether": buffered_relaxed_tether,
        "frozen_offset": frozen_offset,
        "buffered_offset": buffered_offset,
        "persistence_probe": persistence_probe,
        "terminal_ramp": terminal_ramp,
        "fixed_late_ramp": fixed_late_ramp,
        "terminal_ramp_trigger_block": terminal_ramp_trigger_block,
        "execution": execution,
        "maxima": maxima,
        "first_invalid_call": first_invalid_call,
        "first_nonfinite_call": first_nonfinite_call,
        "reference_check": reference_check,
    }


def _write_strength_ood_animation_bundles(
    output_dir: Path,
    *,
    rollouts: Sequence[Mapping[str, Any]],
    runtime: Any,
    working_id: str = STRENGTH_OOD_WORKING_ID,
    animation_contract: Mapping[str, Any] | None = None,
    animation_case_ids: Sequence[str] | None = None,
    split_group_id: str | None = None,
    frame_calls: Sequence[int] | None = None,
) -> dict[str, Any]:
    if animation_contract is None:
        animation_contract = STRENGTH_OOD_ANIMATION_CONTRACT
    if animation_case_ids is None:
        animation_case_ids = STRENGTH_OOD_ANIMATION_CASE_IDS
    if split_group_id is None:
        split_group_id = STRENGTH_OOD_GROUP_ID
    if frame_calls is None:
        frame_calls = STRENGTH_OOD_ANIMATION_CALLS
    animation_case_ids = tuple(animation_case_ids)
    frame_calls = tuple(int(value) for value in frame_calls)
    if list(animation_case_ids) != animation_contract.get("case_ids"):
        raise ValueError("animation cases differ from the registered contract")
    if list(frame_calls) != animation_contract.get("output_calls"):
        raise ValueError("animation calls differ from the registered contract")
    lookup = {str(row["case_id"]): row for row in rollouts}
    if set(animation_case_ids) - set(lookup):
        raise ValueError("strength-OOD animation case inventory is incomplete")
    output_dir.mkdir(parents=True, exist_ok=False)
    geometry = runtime.geometry_by_resolution[NATIVE_RESOLUTION]
    nodes = np.asarray(geometry.nodes, dtype=np.float32)
    volumes = np.asarray(geometry.node_measures, dtype=np.float32).reshape(-1)
    state_scale = np.asarray(runtime.normalization.state_scale, dtype=np.float32)
    residual_scale = np.asarray(runtime.normalization.residual_scale, dtype=np.float32)
    frame_indices = np.asarray(frame_calls, dtype=np.int64)
    bundle_rows = []
    for case_id in animation_case_ids:
        rollout = lookup[case_id]
        reference_states = rollout["reference_states"]
        if (
            len(reference_states) != 31
            or len(rollout["shadow_states"]) != 31
            or len(rollout["accepted_states"]) != 31
        ):
            raise ValueError(f"animation trajectory is incomplete: {case_id}")
        truth = np.asarray(reference_states, dtype=np.float64)[frame_indices]
        raw = np.asarray(rollout["shadow_states"], dtype=np.float64)[frame_indices]
        corrected = np.asarray(rollout["accepted_states"], dtype=np.float64)[
            frame_indices
        ]
        if not all(np.isfinite(value).all() for value in (truth, raw, corrected)):
            raise ValueError(f"animation trajectory is non-finite: {case_id}")
        bundle_path = output_dir / f"{case_id}_shadow_anchored_h30.npz"
        np.savez_compressed(
            bundle_path,
            case_id=np.asarray(case_id),
            split_group_id=np.asarray(split_group_id),
            output_calls=frame_indices,
            physical_times=np.asarray(
                animation_contract["physical_times"],
                dtype=np.float64,
            ),
            native_resolution=np.asarray(NATIVE_RESOLUTION, dtype=np.int64),
            nodes=nodes,
            volumes=volumes,
            state_scale=state_scale,
            residual_scale=residual_scale,
            gamma=np.asarray(1.4, dtype=np.float64),
            truth_conservative=truth.astype(np.float32),
            raw_shadow_conservative=raw.astype(np.float32),
            corrected_conservative=corrected.astype(np.float32),
        )
        bundle_rows.append(
            {
                "case_id": case_id,
                "path": bundle_path.name,
                "sha256": sha256_file(bundle_path),
                "frames": len(frame_indices),
                "storage_dtype": "float32_visualization_only",
            }
        )
    manifest = with_payload_sha256(
        {
            "schema": "pcno_response_filtered_block_animation_bundles_v1",
            "working_id": working_id,
            "contract": dict(animation_contract),
            "bundles": bundle_rows,
            "inference_or_gate_input": False,
        }
    )
    atomic_write_json(output_dir / "bundle_manifest.json", manifest)
    return manifest


def run_shadow_rollout(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    strength_ood = args.command == "strength-ood-rollout"
    if strength_ood:
        preflight = _verify_strength_ood_preflight(args.preflight, args)
        case_ids = STRENGTH_OOD_CASE_IDS
        minimum_endpoint_wins = 6
    else:
        preflight = _verify_shadow_preflight(args.preflight, args)
        case_ids = EVALUATION_CASE_IDS
        minimum_endpoint_wins = 4
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if strength_ood:
        runtime, source, a5_rescore = _build_strength_ood_runtime(args)
        evaluation = None
        a4_rollout = None
    else:
        runtime, source, _, _, evaluation, a4_rollout = _build_shadow_runtime(args)
        a5_rescore = None
    try:
        if source != preflight["source_manifest"]:
            raise ValueError("raw-shadow runtime source differs from frozen preflight")
        prefix_a = _shadow_rollout_arm(
            runtime,
            case_id=case_ids[0],
            horizon=2,
            native_truth_only=strength_ood,
        )
        prefix_b = _shadow_rollout_arm(
            runtime,
            case_id=case_ids[0],
            horizon=2,
            native_truth_only=strength_ood,
        )
        prefix_abs = max(
            *(
                _maximum_abs(left - right)
                for left, right in zip(
                    prefix_a["shadow_states"],
                    prefix_b["shadow_states"],
                    strict=True,
                )
            ),
            *(
                _maximum_abs(left - right)
                for left, right in zip(
                    prefix_a["accepted_states"],
                    prefix_b["accepted_states"],
                    strict=True,
                )
            ),
        )
        rollouts = []
        for case_id in case_ids:
            print(f"raw-shadow anchored rollout: {case_id}", flush=True)
            arm = _shadow_rollout_arm(
                runtime,
                case_id=case_id,
                horizon=30,
                retain_reference_states=(
                    strength_ood and case_id in STRENGTH_OOD_ANIMATION_CASE_IDS
                ),
                native_truth_only=strength_ood,
            )
            rollouts.append(arm)
            if runtime.device.type == "cuda":
                torch.cuda.empty_cache()
        rows = [row for rollout in rollouts for row in rollout["rows"]]
        case_rows, controls, population = _paired_rollout_controls_for_cases(
            rows, case_ids=case_ids
        )
        completed = all(
            row["execution"]["completed_calls"] == 30
            and row["first_invalid_call"] is None
            and row["first_nonfinite_call"] is None
            for row in rollouts
        )
        execution_rows = [
            {
                "case_id": row["case_id"],
                "policy": row["policy"],
                **row["execution"],
            }
            for row in rollouts
        ]
        closure_rows = [
            {
                "case_id": row["case_id"],
                "policy": row["policy"],
                **row["maxima"],
            }
            for row in rollouts
        ]
        call_checks = _explicit_shadow_call_checks(
            execution_rows, closure_rows, case_ids=case_ids
        )
        anchor_closure = all(
            row["maxima"]["candidate_shadow_integral_abs"] <= 1.0e-10
            and row["maxima"]["anchor_post_integral_abs"] <= 1.0e-10
            and row["maxima"]["anchor_boundary_abs"] <= 1.0e-12
            and row["maxima"]["anchor_idempotence_abs"] <= 1.0e-10
            for row in rollouts
        )
        response_closure = all(
            row["maxima"]["first_correction_integral_abs"] <= 1.0e-10
            and row["maxima"]["filtered_response_integral_abs"] <= 1.0e-10
            and row["maxima"]["first_boundary_abs"] <= 1.0e-12
            and row["maxima"]["filtered_boundary_abs"] <= 1.0e-12
            and row["maxima"]["projection_idempotence_abs"] <= 1.0e-10
            and row["maxima"]["cap_active_count"] == 0
            for row in rollouts
        )
        native_reference_contract = all(
            row["reference_check"].get("retained_resolution") == "250x100"
            and row["reference_check"].get("active_reference_artifact_sha256")
            == row["reference_check"].get("frozen_training_reference_sha256")
            and row["reference_check"].get("restriction_crosscheck_max_abs") == 0.0
            for row in rollouts
        )
        prerequisite_checks = (
            {
                "qualified_a5_r1_prerequisite_exact": (
                    a5_rescore is not None and a5_rescore["status"] == "qualified"
                ),
                "named_strength_ood_group_exact": (
                    source["population"]["split_group_id"] == STRENGTH_OOD_GROUP_ID
                    and source["population"]["case_ids"] == list(case_ids)
                ),
                "neighboring_strength_ood_groups_not_opened": source["population"][
                    "still_sealed_groups"
                ]
                == ["strength_ood_e12", "strength_ood_e14"],
                "native_truth_reference_contract_exact": native_reference_contract,
                "fine_reference_truth_not_used": source["population"][
                    "fine_reference_truth"
                ]
                == "not_required_or_loaded",
            }
            if strength_ood
            else {
                "a3_teacher_gate_passed": evaluation is not None
                and evaluation["teacher_gate"]["recurrent_pilot_authorized"] is True,
                "a4_stopped_only_on_filtered_controls": a4_rollout is not None
                and a4_rollout["status"] == "stopped",
            }
        )
        checks = {
            **prerequisite_checks,
            "all_shadow_and_accepted_rollouts_complete_finite_admissible": completed,
            "median_endpoint_ratio_at_most_0p98": population[
                "median_endpoint_state_ratio"
            ]
            <= 0.98,
            (
                "minimum_six_endpoint_wins"
                if strength_ood
                else "minimum_four_endpoint_wins"
            ): population["endpoint_win_count"] >= minimum_endpoint_wins,
            "maximum_endpoint_ratio_at_most_1p02": population[
                "maximum_endpoint_state_ratio"
            ]
            <= 1.02,
            "aggregate_state_rms_ratio_at_most_0p99": population[
                "aggregate_state_rms_ratio"
            ]
            <= 0.99,
            "increment_and_cumulative_ratios_at_most_one": population[
                "aggregate_increment_defect_rms_ratio"
            ]
            <= 1.0
            and population["median_endpoint_cumulative_defect_ratio"] <= 1.0,
            "all_controls_no_harm": bool(controls)
            and all(parent._control_passed(row) for row in controls),
            **call_checks,
            "integral_anchor_closure": anchor_closure,
            "response_projection_closure": response_closure,
            "deterministic_two_output_prefix_exact": prefix_abs == 0.0,
            "source_reference_and_artifact_inventory_exact": source
            == preflight["source_manifest"],
        }
        qualified = all(checks.values())
        accepted_rows = [
            row for rollout in rollouts for row in rollout["accepted_rows"]
        ]
        shadow_rows = [row for rollout in rollouts for row in rollout["shadow_rows"]]
        anchor_rows = [row for rollout in rollouts for row in rollout["anchor_rows"]]
        write_csv(output_dir / "rollout_call_metrics.csv", rows)
        write_csv(output_dir / "rollout_case_metrics.csv", case_rows)
        write_csv(output_dir / "rollout_controls.csv", controls)
        write_csv(output_dir / "anchor_audits.csv", anchor_rows)
        write_csv(
            output_dir / "rollout_execution.csv",
            execution_rows,
        )
        write_csv(
            output_dir / "rollout_closure.csv",
            closure_rows,
        )
        write_csv(
            output_dir / "reference_checks.csv",
            [{"case_id": row["case_id"], **row["reference_check"]} for row in rollouts],
        )
        atomic_write_json(output_dir / "source_manifest.json", source)
        animation_manifest = (
            _write_strength_ood_animation_bundles(
                output_dir / "animation_bundles",
                rollouts=rollouts,
                runtime=runtime,
            )
            if strength_ood
            else None
        )
        files = (
            "rollout_call_metrics.csv",
            "rollout_case_metrics.csv",
            "rollout_controls.csv",
            "anchor_audits.csv",
            "rollout_execution.csv",
            "rollout_closure.csv",
            "reference_checks.csv",
            "source_manifest.json",
        )
        artifact_hashes = sha256_files(files, root=output_dir)
        cost = {
            "logical_model_calls": sum(
                row["execution"]["logical_model_calls"] for row in rollouts
            ),
            "native_logical_calls": sum(
                row["execution"]["native_logical_calls"] for row in rollouts
            ),
            "fine_logical_calls": sum(
                row["execution"]["fine_logical_calls"] for row in rollouts
            ),
            "actual_forward_passes": sum(
                row["execution"]["actual_forward_passes"] for row in rollouts
            ),
            "total_forward_seconds": sum(
                row["execution"]["total_forward_seconds"] for row in rollouts
            ),
            "wall_seconds": sum(row["execution"]["wall_seconds"] for row in rollouts),
            "maximum_peak_gpu_memory_bytes": max(
                row["execution"]["maximum_peak_gpu_memory_bytes"] for row in rollouts
            ),
            "aggregate_state_rms": math.sqrt(
                np.mean([float(row["state_error"]) ** 2 for row in accepted_rows])
            ),
            "shadow_aggregate_state_rms": math.sqrt(
                np.mean([float(row["state_error"]) ** 2 for row in shadow_rows])
            ),
            "prior_a4_cost": (
                a5_rescore["cost"]["prior_a4_cost"]
                if strength_ood
                else a4_rollout["cost"]
            ),
        }
        payload_fields = {
            "schema": STRENGTH_OOD_SCHEMA if strength_ood else SHADOW_SCHEMA,
            "working_id": (
                STRENGTH_OOD_WORKING_ID if strength_ood else SHADOW_WORKING_ID
            ),
            "status": "qualified" if qualified else "stopped",
            "population_status": (
                "sealed_strength_ood_e13_named_confirmation"
                if strength_ood
                else "adaptive_open_validation_shadow_pilot"
            ),
            "recurrent_gate": {
                "status": "qualified" if qualified else "stopped",
                "claim_authorized": qualified,
                "checks": checks,
            },
            "population": population,
            "failed_controls": [
                row for row in controls if not parent._control_passed(row)
            ],
            "cost": cost,
            "maximum_anchor_correction_rms": max(
                row["maxima"]["anchor_correction_rms"] for row in rollouts
            ),
            "deterministic_prefix_max_abs": prefix_abs,
            "preflight_sha256": sha256_file(args.preflight),
            "preflight_payload_sha256": preflight["payload_sha256"],
            "source_manifest_sha256": sha256_file(output_dir / "source_manifest.json"),
            "source_manifest_payload_sha256": source["payload_sha256"],
            "artifact_hashes": artifact_hashes,
            "recurrence_executed": True,
            "sealed_population_opened": strength_ood,
        }
        if strength_ood:
            payload_fields.update(
                {
                    "opened_group": STRENGTH_OOD_GROUP_ID,
                    "opened_case_ids": list(STRENGTH_OOD_CASE_IDS),
                    "still_sealed_groups": [
                        "strength_ood_e12",
                        "strength_ood_e14",
                    ],
                    "qualified_a5_rescore_sha256": sha256_file(args.a5_rescore),
                    "qualified_a5_rescore_payload_sha256": a5_rescore["payload_sha256"],
                    "animation_bundle_manifest": animation_manifest,
                    "animation_bundle_manifest_sha256": sha256_file(
                        output_dir / "animation_bundles" / "bundle_manifest.json"
                    ),
                    "claim_boundary": (
                        "One frozen H30 confirmation on the single named dynamic-FV "
                        "strength-OOD e13 group. It does not establish neighboring-"
                        "strength, full-test, cross-family, physical-conservation, "
                        "resolution-convergence, or direct off-grid superiority claims."
                    ),
                }
            )
            output_name = "strength_ood_rollout.json"
        else:
            payload_fields.update(
                {
                    "qualified_evaluation_sha256": sha256_file(args.evaluation),
                    "qualified_evaluation_payload_sha256": evaluation["payload_sha256"],
                    "stopped_a4_rollout_sha256": sha256_file(args.a4_rollout),
                    "stopped_a4_rollout_payload_sha256": a4_rollout["payload_sha256"],
                    "claim_boundary": (
                        "Adaptive-open H30 dynamic-FV raw-shadow pilot only. The "
                        "anchor matches raw model integrals; it does not establish "
                        "physical conservation, independent validation, cross-family "
                        "transfer, sealed performance, or direct off-grid superiority."
                    ),
                }
            )
            output_name = "shadow_rollout.json"
        payload = with_payload_sha256(payload_fields)
        atomic_write_json(output_dir / output_name, payload)
        return payload, 0 if qualified else 4
    finally:
        collection._close_runtime(runtime)


def _run_structural_gate_population(
    args: argparse.Namespace,
    *,
    prospective: bool,
    warm_start: bool = False,
    projected_coast: bool = False,
    frozen_offset: bool = False,
    buffered_offset: bool = False,
    prospective_buffered_offset: bool = False,
    persistence_gain: bool = False,
    terminal_ramp: bool = False,
    fixed_late_ramp: bool = False,
    slew_limited_tether: bool = False,
    relaxed_tether: bool = False,
    buffered_relaxed_tether: bool = False,
    buffered_relaxed_tether_e14: bool = False,
) -> tuple[dict[str, Any], int]:
    shard_native_reference = bool(
        getattr(args, "shard_native_reference_audit", None) is not None
    )
    if prospective and warm_start:
        raise ValueError("warm-start calibration cannot open a prospective population")
    if projected_coast and not warm_start:
        raise ValueError("projected-coast calibration requires warm_start=True")
    if slew_limited_tether and not warm_start:
        raise ValueError("slew-limited tether calibration requires warm_start=True")
    if relaxed_tether and not warm_start:
        raise ValueError("relaxed tether calibration requires warm_start=True")
    if buffered_relaxed_tether and not relaxed_tether:
        raise ValueError("buffered relaxed tether requires relaxed_tether=True")
    if buffered_relaxed_tether_e14 and not buffered_relaxed_tether:
        raise ValueError("E14 buffered relaxed transfer requires buffered mode")
    if frozen_offset and not warm_start:
        raise ValueError("frozen-offset calibration requires warm_start=True")
    if buffered_offset and not frozen_offset:
        raise ValueError("buffered-offset calibration requires frozen_offset=True")
    if prospective_buffered_offset and not buffered_offset:
        raise ValueError("prospective buffered offset requires buffered_offset=True")
    if persistence_gain and (prospective_buffered_offset or not buffered_offset):
        raise ValueError("persistence gain requires non-prospective buffered offset")
    if terminal_ramp and not persistence_gain:
        raise ValueError("terminal ramp requires persistence_gain=True")
    if fixed_late_ramp and persistence_gain:
        raise ValueError("fixed late ramp does not use persistence gain")
    if fixed_late_ramp and (prospective_buffered_offset or not buffered_offset):
        raise ValueError("fixed late ramp requires non-prospective buffered offset")
    if sum((projected_coast, frozen_offset, slew_limited_tether, relaxed_tether)) > 1:
        raise ValueError(
            "projected coast, frozen offset, and slew-limited tether are "
            "and relaxed tether are mutually exclusive"
        )
    a22_r1_rollout = None
    a14_lineage_rollout = None
    if buffered_relaxed_tether_e14:
        preflight = _verify_buffered_relaxed_tether_e14_preflight(
            args.preflight, args
        )
        case_ids = PROSPECTIVE_BUFFERED_OFFSET_CASE_IDS
        animation_case_ids = PROSPECTIVE_BUFFERED_OFFSET_ANIMATION_CASE_IDS
        working_id = BUFFERED_RELAXED_TETHER_E14_WORKING_ID
        schema = BUFFERED_RELAXED_TETHER_E14_SCHEMA
        animation_contract = BUFFERED_RELAXED_TETHER_E14_ANIMATION_CONTRACT
        population_status = "already_open_e14_frozen_protocol_transfer"
        qualified_status = "qualified_retrospective_transfer"
        stopped_status = "stopped_retrospective_transfer"
        gate_key = "retrospective_transfer_gate"
        inventory_check_name = "already_open_e14_inventory_exact"
        sealed_check_name = "no_new_population_opened"
        result_name = "buffered_relaxed_tether_e14_rollout.json"
        rollout_label = "frozen E14 buffered relaxed-tether transfer"
    elif buffered_relaxed_tether:
        preflight = _verify_buffered_relaxed_tether_replay_preflight(
            args.preflight, args
        )
        case_ids = PROSPECTIVE_STRUCTURAL_CASE_IDS
        animation_case_ids = BUFFERED_OFFSET_ANIMATION_CASE_IDS
        working_id = BUFFERED_RELAXED_TETHER_REPLAY_WORKING_ID
        schema = BUFFERED_RELAXED_TETHER_REPLAY_SCHEMA
        animation_contract = BUFFERED_RELAXED_TETHER_REPLAY_ANIMATION_CONTRACT
        population_status = "already_open_e12_buffered_recurrent_replay"
        qualified_status = "qualified_calibration_replay"
        stopped_status = "stopped_calibration_replay"
        gate_key = "calibration_gate"
        inventory_check_name = "already_open_e12_inventory_exact"
        sealed_check_name = "no_new_population_opened"
        result_name = "buffered_relaxed_tether_replay.json"
        rollout_label = "buffered relaxed-tether replay"
    elif relaxed_tether:
        preflight = _verify_relaxed_tether_preflight(args.preflight, args)
        case_ids = PROSPECTIVE_STRUCTURAL_CASE_IDS
        animation_case_ids = BUFFERED_OFFSET_ANIMATION_CASE_IDS
        working_id = RELAXED_TETHER_WORKING_ID
        schema = RELAXED_TETHER_SCHEMA
        animation_contract = RELAXED_TETHER_ANIMATION_CONTRACT
        population_status = "already_open_e12_recurrent_mechanism_calibration"
        qualified_status = "qualified_calibration"
        stopped_status = "stopped_calibration"
        gate_key = "calibration_gate"
        inventory_check_name = "already_open_e12_inventory_exact"
        sealed_check_name = "no_new_population_opened"
        result_name = "relaxed_tether_rollout.json"
        rollout_label = "relaxed projected-tether calibration"
    elif slew_limited_tether:
        preflight = _verify_slew_limited_tether_preflight(args.preflight, args)
        case_ids = PROSPECTIVE_STRUCTURAL_CASE_IDS
        animation_case_ids = BUFFERED_OFFSET_ANIMATION_CASE_IDS
        working_id = SLEW_LIMITED_TETHER_WORKING_ID
        schema = SLEW_LIMITED_TETHER_SCHEMA
        animation_contract = SLEW_LIMITED_TETHER_ANIMATION_CONTRACT
        population_status = "already_open_e12_recurrent_mechanism_calibration"
        qualified_status = "qualified_calibration"
        stopped_status = "stopped_calibration"
        gate_key = "calibration_gate"
        inventory_check_name = "already_open_e12_inventory_exact"
        sealed_check_name = "no_new_population_opened"
        result_name = "slew_limited_tether_rollout.json"
        rollout_label = "slew-limited projected-tether calibration"
    elif fixed_late_ramp:
        preflight = _verify_fixed_late_ramp_preflight(args.preflight, args)
        case_ids = PROSPECTIVE_STRUCTURAL_CASE_IDS
        animation_case_ids = BUFFERED_OFFSET_ANIMATION_CASE_IDS
        working_id = FIXED_LATE_RAMP_WORKING_ID
        schema = FIXED_LATE_RAMP_SCHEMA
        animation_contract = FIXED_LATE_RAMP_ANIMATION_CONTRACT
        population_status = "already_open_e12_fixed_late_ramp_calibration"
        qualified_status = "qualified_calibration"
        stopped_status = "stopped_calibration"
        gate_key = "calibration_gate"
        inventory_check_name = "already_open_e12_inventory_exact"
        sealed_check_name = "no_new_population_opened"
        result_name = "fixed_late_ramp_rollout.json"
        rollout_label = "truth-selected fixed late-ramp calibration"
    elif terminal_ramp:
        preflight = _verify_terminal_ramp_preflight(args.preflight, args)
        case_ids = PROSPECTIVE_STRUCTURAL_CASE_IDS
        animation_case_ids = BUFFERED_OFFSET_ANIMATION_CASE_IDS
        working_id = TERMINAL_RAMP_WORKING_ID
        schema = TERMINAL_RAMP_SCHEMA
        animation_contract = TERMINAL_RAMP_ANIMATION_CONTRACT
        population_status = "already_open_e12_terminal_ramp_calibration"
        qualified_status = "qualified_calibration"
        stopped_status = "stopped_calibration"
        gate_key = "calibration_gate"
        inventory_check_name = "already_open_e12_inventory_exact"
        sealed_check_name = "no_new_population_opened"
        result_name = "terminal_ramp_rollout.json"
        rollout_label = "phase-triggered terminal-ramp calibration"
    elif persistence_gain:
        preflight = _verify_persistence_gain_preflight(args.preflight, args)
        case_ids = PROSPECTIVE_STRUCTURAL_CASE_IDS
        animation_case_ids = BUFFERED_OFFSET_ANIMATION_CASE_IDS
        working_id = PERSISTENCE_GAIN_WORKING_ID
        schema = PERSISTENCE_GAIN_SCHEMA
        animation_contract = PERSISTENCE_GAIN_ANIMATION_CONTRACT
        population_status = "already_open_e12_persistence_calibration"
        qualified_status = "qualified_calibration"
        stopped_status = "stopped_calibration"
        gate_key = "calibration_gate"
        inventory_check_name = "already_open_e12_inventory_exact"
        sealed_check_name = "no_new_population_opened"
        result_name = "persistence_gain_rollout.json"
        rollout_label = "target-free persistence-gain calibration"
    elif prospective_buffered_offset:
        preflight = _verify_prospective_buffered_offset_preflight(args.preflight, args)
        case_ids = PROSPECTIVE_BUFFERED_OFFSET_CASE_IDS
        animation_case_ids = PROSPECTIVE_BUFFERED_OFFSET_ANIMATION_CASE_IDS
        working_id = PROSPECTIVE_BUFFERED_OFFSET_WORKING_ID
        schema = PROSPECTIVE_BUFFERED_OFFSET_SCHEMA
        animation_contract = PROSPECTIVE_BUFFERED_OFFSET_ANIMATION_CONTRACT
        population_status = "single_prospectively_named_e14_confirmation"
        qualified_status = "qualified_prospective"
        stopped_status = "stopped_prospective"
        gate_key = "prospective_gate"
        inventory_check_name = "prospectively_named_e14_inventory_exact"
        sealed_check_name = "no_additional_sealed_group_opened"
        result_name = "prospective_buffered_offset_rollout.json"
        rollout_label = "prospective e14 buffered-offset rollout"
    elif buffered_offset:
        preflight = _verify_buffered_offset_preflight(args.preflight, args)
        case_ids = PROSPECTIVE_STRUCTURAL_CASE_IDS
        animation_case_ids = BUFFERED_OFFSET_ANIMATION_CASE_IDS
        working_id = BUFFERED_OFFSET_WORKING_ID
        schema = BUFFERED_OFFSET_SCHEMA
        animation_contract = BUFFERED_OFFSET_ANIMATION_CONTRACT
        population_status = "already_open_e12_recurrent_calibration"
        qualified_status = "qualified_calibration"
        stopped_status = "stopped_calibration"
        gate_key = "calibration_gate"
        inventory_check_name = "already_open_e12_inventory_exact"
        sealed_check_name = "e14_remained_sealed"
        result_name = "buffered_offset_rollout.json"
        rollout_label = "buffered-offset calibration"
    elif frozen_offset:
        preflight = _verify_frozen_offset_preflight(args.preflight, args)
        case_ids = PROSPECTIVE_STRUCTURAL_CASE_IDS
        animation_case_ids = PROSPECTIVE_STRUCTURAL_ANIMATION_CASE_IDS
        working_id = FROZEN_OFFSET_WORKING_ID
        schema = FROZEN_OFFSET_SCHEMA
        animation_contract = FROZEN_OFFSET_ANIMATION_CONTRACT
        population_status = "already_open_e12_recurrent_calibration"
        qualified_status = "qualified_calibration"
        stopped_status = "stopped_calibration"
        gate_key = "calibration_gate"
        inventory_check_name = "already_open_e12_inventory_exact"
        sealed_check_name = "e14_remained_sealed"
        result_name = "frozen_offset_rollout.json"
        rollout_label = "frozen-offset calibration"
    elif projected_coast:
        preflight = _verify_projected_coast_preflight(args.preflight, args)
        case_ids = PROSPECTIVE_STRUCTURAL_CASE_IDS
        animation_case_ids = PROSPECTIVE_STRUCTURAL_ANIMATION_CASE_IDS
        working_id = PROJECTED_COAST_WORKING_ID
        schema = PROJECTED_COAST_SCHEMA
        animation_contract = PROJECTED_COAST_ANIMATION_CONTRACT
        population_status = "already_open_e12_recurrent_calibration"
        qualified_status = "qualified_calibration"
        stopped_status = "stopped_calibration"
        gate_key = "calibration_gate"
        inventory_check_name = "already_open_e12_inventory_exact"
        sealed_check_name = "e14_remained_sealed"
        result_name = "projected_coast_rollout.json"
        rollout_label = "projected-coast calibration"
    elif warm_start:
        preflight = _verify_warm_start_preflight(args.preflight, args)
        case_ids = PROSPECTIVE_STRUCTURAL_CASE_IDS
        animation_case_ids = PROSPECTIVE_STRUCTURAL_ANIMATION_CASE_IDS
        working_id = WARM_START_WORKING_ID
        schema = WARM_START_SCHEMA
        animation_contract = WARM_START_ANIMATION_CONTRACT
        population_status = "already_open_e12_recurrent_calibration"
        qualified_status = "qualified_calibration"
        stopped_status = "stopped_calibration"
        gate_key = "calibration_gate"
        inventory_check_name = "already_open_e12_inventory_exact"
        sealed_check_name = "e14_remained_sealed"
        result_name = "warm_start_rollout.json"
        rollout_label = "warm-start/coast calibration"
    elif prospective:
        preflight = _verify_prospective_structural_preflight(args.preflight, args)
        case_ids = PROSPECTIVE_STRUCTURAL_CASE_IDS
        animation_case_ids = PROSPECTIVE_STRUCTURAL_ANIMATION_CASE_IDS
        working_id = PROSPECTIVE_STRUCTURAL_WORKING_ID
        schema = PROSPECTIVE_STRUCTURAL_SCHEMA
        animation_contract = PROSPECTIVE_STRUCTURAL_ANIMATION_CONTRACT
        population_status = "prospectively_named_e12_confirmation"
        qualified_status = "qualified_prospective"
        stopped_status = "stopped_prospective"
        gate_key = "prospective_gate"
        inventory_check_name = "prospectively_named_e12_inventory_exact"
        sealed_check_name = "e14_remained_sealed"
        result_name = "prospective_structural_rollout.json"
        rollout_label = "prospective structural-gated rollout"
    else:
        preflight = _verify_structural_gate_preflight(args.preflight, args)
        case_ids = STRENGTH_OOD_CASE_IDS
        animation_case_ids = STRENGTH_OOD_ANIMATION_CASE_IDS
        working_id = STRUCTURAL_GATE_WORKING_ID
        schema = STRUCTURAL_GATE_SCHEMA
        animation_contract = STRUCTURAL_GATE_ANIMATION_CONTRACT
        population_status = "already_open_e13_retrospective_calibration"
        qualified_status = "qualified_retrospective"
        stopped_status = "stopped_retrospective"
        gate_key = "retrospective_gate"
        inventory_check_name = "already_open_e13_inventory_exact"
        sealed_check_name = "neighboring_strength_groups_not_opened"
        result_name = "structural_gate_rollout.json"
        rollout_label = "structural-gated rollout"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if buffered_relaxed_tether_e14:
        (
            runtime,
            source,
            _,
            a6_rollout,
            a7_rollout,
            a8_rollout,
            a9_rescore,
            a10_rollout,
            a11_rollout,
            a17_rescore,
            a18_rollout,
            a22_rescore,
            a22_r1_rollout,
            a14_rollout,
        ) = _build_buffered_relaxed_tether_e14_runtime(args)
        a12_rollout = None
        a12_buffer_rescore = None
        a13_rollout = None
        a14_lineage_rollout = a14_rollout
        a15_rollout = None
        a16_rollout = None
    elif buffered_relaxed_tether:
        (
            runtime,
            source,
            _,
            a6_rollout,
            a7_rollout,
            a8_rollout,
            a9_rescore,
            a10_rollout,
            a11_rollout,
            a17_rescore,
            a18_rollout,
            a22_rescore,
        ) = _build_buffered_relaxed_tether_replay_runtime(args)
        a12_rollout = None
        a12_buffer_rescore = None
        a13_rollout = None
        a14_rollout = None
        a15_rollout = None
        a16_rollout = None
    elif relaxed_tether:
        (
            runtime,
            source,
            _,
            a6_rollout,
            a7_rollout,
            a8_rollout,
            a9_rescore,
            a10_rollout,
            a11_rollout,
            a17_rescore,
            a18_rollout,
        ) = _build_relaxed_tether_runtime(args)
        a22_rescore = None
        a12_rollout = None
        a12_buffer_rescore = None
        a13_rollout = None
        a14_rollout = None
        a15_rollout = None
        a16_rollout = None
    elif slew_limited_tether:
        (
            runtime,
            source,
            _,
            a6_rollout,
            a7_rollout,
            a8_rollout,
            a9_rescore,
            a10_rollout,
            a11_rollout,
            a17_rescore,
        ) = _build_slew_limited_tether_runtime(args)
        a22_rescore = None
        a12_rollout = None
        a12_buffer_rescore = None
        a13_rollout = None
        a14_rollout = None
        a15_rollout = None
        a16_rollout = None
    elif fixed_late_ramp:
        built = _build_fixed_late_ramp_runtime(args)
        (
            runtime,
            source,
            _,
            a6_rollout,
            a7_rollout,
            a8_rollout,
            a9_rescore,
            a10_rollout,
            a11_rollout,
            a12_rollout,
            a12_buffer_rescore,
            a13_rollout,
            a16_rollout,
        ) = built
        a14_rollout = None
        a15_rollout = None
        a17_rescore = None
        a22_rescore = None
    elif terminal_ramp:
        built = _build_terminal_ramp_runtime(args)
        (
            runtime,
            source,
            _,
            a6_rollout,
            a7_rollout,
            a8_rollout,
            a9_rescore,
            a10_rollout,
            a11_rollout,
            a12_rollout,
            a12_buffer_rescore,
            a13_rollout,
            a14_rollout,
            a15_rollout,
        ) = built
        a16_rollout = None
        a22_rescore = None
    elif persistence_gain:
        built = _build_persistence_gain_runtime(args)
        (
            runtime,
            source,
            _,
            a6_rollout,
            a7_rollout,
            a8_rollout,
            a9_rescore,
            a10_rollout,
            a11_rollout,
            a12_rollout,
            a12_buffer_rescore,
            a13_rollout,
            a14_rollout,
        ) = built
        a15_rollout = None
        a16_rollout = None
        a22_rescore = None
    elif prospective_buffered_offset:
        built = _build_prospective_buffered_offset_runtime(args)
        (
            runtime,
            source,
            _,
            a6_rollout,
            a7_rollout,
            a8_rollout,
            a9_rescore,
            a10_rollout,
            a11_rollout,
            a12_rollout,
            a12_buffer_rescore,
            a13_rollout,
        ) = built
        a14_rollout = None
        a15_rollout = None
        a22_rescore = None
    elif buffered_offset:
        built = _build_buffered_offset_runtime(args)
        (
            runtime,
            source,
            _,
            a6_rollout,
            a7_rollout,
            a8_rollout,
            a9_rescore,
            a10_rollout,
            a11_rollout,
            a12_rollout,
            a12_buffer_rescore,
        ) = built
        a13_rollout = None
        a14_rollout = None
        a15_rollout = None
        a16_rollout = None
        a22_rescore = None
    elif frozen_offset:
        (
            runtime,
            source,
            _,
            a6_rollout,
            a7_rollout,
            a8_rollout,
            a9_rescore,
            a10_rollout,
            a11_rollout,
        ) = _build_frozen_offset_runtime(args)
        a12_rollout = None
        a12_buffer_rescore = None
        a13_rollout = None
        a14_rollout = None
        a15_rollout = None
        a16_rollout = None
        a22_rescore = None
    elif projected_coast:
        (
            runtime,
            source,
            _,
            a6_rollout,
            a7_rollout,
            a8_rollout,
            a9_rescore,
            a10_rollout,
        ) = _build_projected_coast_runtime(args)
        a11_rollout = None
        a12_rollout = None
        a12_buffer_rescore = None
        a13_rollout = None
        a14_rollout = None
        a15_rollout = None
        a16_rollout = None
        a22_rescore = None
    elif warm_start:
        runtime, source, _, a6_rollout, a7_rollout, a8_rollout, a9_rescore = (
            _build_warm_start_runtime(args)
        )
        a10_rollout = None
        a11_rollout = None
        a12_rollout = None
        a12_buffer_rescore = None
        a13_rollout = None
        a14_rollout = None
        a15_rollout = None
        a16_rollout = None
        a22_rescore = None
    elif prospective:
        runtime, source, _, a6_rollout, a7_rollout = (
            _build_prospective_structural_runtime(args)
        )
        a8_rollout = None
        a9_rescore = None
        a10_rollout = None
        a11_rollout = None
        a12_rollout = None
        a12_buffer_rescore = None
        a13_rollout = None
        a14_rollout = None
        a15_rollout = None
        a16_rollout = None
        a22_rescore = None
    else:
        runtime, source, _, a6_rollout = _build_structural_gate_runtime(args)
        a7_rollout = None
        a8_rollout = None
        a9_rescore = None
        a10_rollout = None
        a11_rollout = None
        a12_rollout = None
        a12_buffer_rescore = None
        a13_rollout = None
        a14_rollout = None
        a15_rollout = None
        a16_rollout = None
        a22_rescore = None
    try:
        if source != preflight["source_manifest"]:
            raise ValueError("structural-gate runtime source differs from preflight")
        prefix_a = _shadow_rollout_arm(
            runtime,
            case_id=case_ids[0],
            horizon=2,
            native_truth_only=True,
            structural_gate=True,
            shard_native_reference=shard_native_reference,
        )
        prefix_b = _shadow_rollout_arm(
            runtime,
            case_id=case_ids[0],
            horizon=2,
            native_truth_only=True,
            structural_gate=True,
            shard_native_reference=shard_native_reference,
        )
        prefix_abs = max(
            _maximum_abs(left - right)
            for left, right in zip(
                prefix_a["accepted_states"],
                prefix_b["accepted_states"],
                strict=True,
            )
        )
        prefix_decisions_exact = (
            prefix_a["position_descriptor"] == prefix_b["position_descriptor"]
            and prefix_a["position_trusted"] == prefix_b["position_trusted"]
            and prefix_a["front_audits"] == prefix_b["front_audits"]
        )

        rollouts = []
        for case_id in case_ids:
            print(f"{rollout_label}: {case_id}", flush=True)
            rollouts.append(
                _shadow_rollout_arm(
                    runtime,
                    case_id=case_id,
                    horizon=30,
                    retain_reference_states=(case_id in animation_case_ids),
                    native_truth_only=True,
                    structural_gate=True,
                    warm_start_blocks=(
                        FROZEN_OFFSET_WARM_BLOCKS
                        if frozen_offset or slew_limited_tether or relaxed_tether
                        else (WARM_START_BLOCKS if warm_start else None)
                    ),
                    projected_coast=projected_coast,
                    slew_limited_tether=slew_limited_tether,
                    relaxed_tether=relaxed_tether,
                    buffered_relaxed_tether=buffered_relaxed_tether,
                    frozen_offset=frozen_offset,
                    buffered_offset=buffered_offset,
                    persistence_probe=persistence_gain,
                    terminal_ramp=terminal_ramp,
                    fixed_late_ramp=fixed_late_ramp,
                    shard_native_reference=shard_native_reference,
                )
            )
            if runtime.device.type == "cuda":
                torch.cuda.empty_cache()

        rows = [row for rollout in rollouts for row in rollout["rows"]]
        case_rows, controls, population = _paired_rollout_controls_for_cases(
            rows, case_ids=case_ids
        )
        execution_rows = [
            {
                "case_id": row["case_id"],
                "policy": row["policy"],
                **row["execution"],
            }
            for row in rollouts
        ]
        closure_rows = [
            {
                "case_id": row["case_id"],
                "policy": row["policy"],
                **row["maxima"],
            }
            for row in rollouts
        ]
        descriptor_rows = [
            {
                "case_id": row["case_id"],
                "position_trusted": row["position_trusted"],
                "position_trust_threshold": FROZEN_POSITION_TRUST_THRESHOLD,
                "position_buffer_threshold": (
                    FROZEN_POSITION_BUFFER_THRESHOLD
                    if buffered_offset or buffered_relaxed_tether
                    else None
                ),
                "position_route": row["position_route"],
                "front_branch_veto_call": row["front_branch_veto_call"],
                **row["position_descriptor"],
            }
            for row in rollouts
        ]
        front_rows = [row for rollout in rollouts for row in rollout["front_audits"]]
        anchor_rows = [row for rollout in rollouts for row in rollout["anchor_rows"]]
        call_checks = (
            _buffered_relaxed_tether_call_checks(rollouts, case_ids=case_ids)
            if buffered_relaxed_tether
            else (
            _slew_limited_tether_call_checks(
                rollouts, case_ids=case_ids, relaxed=True
            )
            if relaxed_tether
            else (
                _slew_limited_tether_call_checks(rollouts, case_ids=case_ids)
                if slew_limited_tether
                else (
                    _fixed_late_ramp_call_checks(rollouts, case_ids=case_ids)
                    if fixed_late_ramp
                    else (
                        _terminal_ramp_call_checks(rollouts, case_ids=case_ids)
                        if terminal_ramp
                        else (
                            _persistence_gain_call_checks(
                                rollouts, case_ids=case_ids
                            )
                            if persistence_gain
                            else (
                                _buffered_offset_call_checks(
                                    rollouts, case_ids=case_ids
                                )
                                if buffered_offset
                                else (
                                    _frozen_offset_call_checks(
                                        rollouts, case_ids=case_ids
                                    )
                                    if frozen_offset
                                    else (
                                        _projected_coast_call_checks(
                                            rollouts, case_ids=case_ids
                                        )
                                        if projected_coast
                                        else (
                                            _warm_start_call_checks(
                                                rollouts, case_ids=case_ids
                                            )
                                            if warm_start
                                            else _structural_gate_call_checks(
                                                rollouts, case_ids=case_ids
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
            )
        )
        completed = all(
            row["execution"]["completed_calls"] == 30
            and row["first_invalid_call"] is None
            and row["first_nonfinite_call"] is None
            for row in rollouts
        )
        anchor_closure = all(
            row["maxima"]["candidate_shadow_integral_abs"] <= 1.0e-10
            and row["maxima"]["anchor_post_integral_abs"] <= 1.0e-10
            and row["maxima"]["anchor_boundary_abs"] <= 1.0e-12
            and row["maxima"]["anchor_idempotence_abs"] <= 1.0e-10
            for row in rollouts
        )
        response_closure = all(
            row["maxima"]["first_correction_integral_abs"] <= 1.0e-10
            and row["maxima"]["filtered_response_integral_abs"] <= 1.0e-10
            and row["maxima"]["first_boundary_abs"] <= 1.0e-12
            and row["maxima"]["filtered_boundary_abs"] <= 1.0e-12
            and row["maxima"]["projection_idempotence_abs"] <= 1.0e-10
            and row["maxima"]["cap_active_count"] == 0
            for row in rollouts
        )
        truth_contract = source["population"].get(
            "native_scoring_truth_contract",
            {"mode": "frozen_reference_npz_float64", "recovery_audit": None},
        )
        native_reference_contract = all(
            _native_reference_check_exact(
                row["reference_check"], truth_contract=truth_contract
            )
            for row in rollouts
        )
        if warm_start:
            if a7_rollout is None or a8_rollout is None or a9_rescore is None:
                raise AssertionError("warm-start run requires frozen A7/A8/A9 lineage")
            lineage_checks = {
                "exact_stopped_a6_lineage": a6_rollout["status"] == "stopped",
                "exact_qualified_a7_calibration_prerequisite": (
                    a7_rollout["status"] == "qualified_retrospective"
                    and a7_rollout["retrospective_gate"]["prospective_claim_authorized"]
                    is False
                ),
                "exact_qualified_a8_prerequisite": (
                    a8_rollout["status"] == "qualified_prospective"
                    and a8_rollout["prospective_gate"]["prospective_claim_authorized"]
                    is True
                ),
                "exact_completed_a9_rescore_prerequisite": (
                    a9_rescore["status"] == "completed_diagnostic_rescore"
                    and a9_rescore["model_calls"] == 0
                    and a9_rescore["recurrence_reexecuted"] is False
                ),
                "coefficient_trust_threshold_and_window_not_refit": (
                    source["protocol"]["coefficient"] == -0.5
                    and source["protocol"]["position_trust_threshold"]
                    == FROZEN_POSITION_TRUST_THRESHOLD
                    and source["protocol"]["interior_warm_blocks"]
                    == (
                        FROZEN_OFFSET_WARM_BLOCKS
                        if frozen_offset or slew_limited_tether or relaxed_tether
                        else WARM_START_BLOCKS
                    )
                    and (
                        source["protocol"].get(
                            "coefficient_or_trust_threshold_refit",
                            source["protocol"].get("coefficient_or_threshold_refit"),
                        )
                        is False
                    )
                ),
            }
            if projected_coast:
                if a10_rollout is None:  # pragma: no cover - guarded by construction
                    raise AssertionError(
                        "projected-coast run requires the frozen A10 artifact"
                    )
                lineage_checks.update(
                    {
                        "exact_stopped_a10_prerequisite": (
                            a10_rollout["status"] == "stopped_calibration"
                            and a10_rollout["population"][
                                "strict_interior_trajectory_win_count"
                            ]
                            == 6
                        ),
                        "projected_coast_cells_and_rank_not_refit": (
                            source["protocol"]["coast_projection_rank"] == 8
                            and source["protocol"]["coast_projection_active_cells"]
                            == [list(cell) for cell in FROZEN_ACTIVE_CELLS]
                            and source["protocol"]["coast_projection_nonconstant_only"]
                            is True
                        ),
                    }
                )
            if slew_limited_tether:
                if a11_rollout is None or a17_rescore is None:  # pragma: no cover
                    raise AssertionError(
                        "slew-limited tether requires frozen A11 and A17-R1"
                    )
                lineage_checks.update(
                    {
                        "exact_stopped_a11_prerequisite": (
                            a11_rollout["status"] == "stopped_calibration"
                            and a11_rollout["population"][
                                "strict_interior_trajectory_win_count"
                            ]
                            == 7
                        ),
                        "exact_qualified_a17_rescore_prerequisite": (
                            a17_rescore["status"]
                            == "qualified_calibration_rescore"
                            and a17_rescore["model_calls"] == 0
                            and a17_rescore["recurrence_reexecuted"] is False
                        ),
                        "single_unswept_slew_rule_and_cells_exact": (
                            source["protocol"][
                                "relative_displacement_change_limit"
                            ]
                            == SLEW_LIMITED_TETHER_RELATIVE_CHANGE_LIMIT
                            and source["protocol"]["single_unswept_relative_limit"]
                            is True
                            and source["protocol"]["coast_projection_rank"] == 8
                            and source["protocol"][
                                "coast_projection_active_cells"
                            ]
                            == [list(cell) for cell in FROZEN_ACTIVE_CELLS]
                            and source["protocol"][
                                "true_error_or_reference_at_inference"
                            ]
                            is False
                        ),
                    }
                )
            if relaxed_tether:
                lineage_checks.update(
                    {
                        "exact_stopped_a18_prerequisite": (
                            a18_rollout["status"] == "stopped_calibration"
                            and a18_rollout["population"][
                                "aggregate_increment_defect_rms_ratio"
                            ]
                            == 0.9970256010493702
                            and a18_rollout["population"][
                                "strict_interior_trajectory_win_count"
                            ]
                            == 7
                        ),
                        "single_unswept_relaxation_rule_and_cells_exact": (
                            source["protocol"]["relaxation_rate"]
                            == RELAXED_TETHER_RATE
                            and source["protocol"][
                                "single_unswept_relaxation_rate"
                            ]
                            is True
                            and source["protocol"]["coast_projection_rank"] == 8
                            and source["protocol"][
                                "coast_projection_active_cells"
                            ]
                            == [list(cell) for cell in FROZEN_ACTIVE_CELLS]
                            and source["protocol"][
                                "true_error_or_reference_at_inference"
                            ]
                            is False
                        ),
                    }
                )
            if buffered_relaxed_tether:
                if a22_rescore is None:  # pragma: no cover - guarded above
                    raise AssertionError(
                        "buffered relaxed-tether replay requires qualified A22"
                    )
                lineage_checks.update(
                    {
                        "exact_qualified_a22_rescore_prerequisite": (
                            a22_rescore["status"]
                            == "qualified_calibration_rescore"
                            and a22_rescore["model_calls"] == 0
                            and a22_rescore["recurrence_reexecuted"] is False
                            and a22_rescore["calibration_gate"][
                                "one_exact_replay_authorized"
                            ]
                            is True
                        ),
                        "a22_route_and_parameters_not_refit": (
                            source["protocol"]["position_trust_threshold"]
                            == FROZEN_POSITION_TRUST_THRESHOLD
                            and source["protocol"]["position_buffer_threshold"]
                            == FROZEN_POSITION_BUFFER_THRESHOLD
                            and source["protocol"]["relaxation_rate"]
                            == RELAXED_TETHER_RATE
                            and source["protocol"][
                                "route_or_numeric_parameter_refit_after_a22"
                            ]
                            is False
                            and source["protocol"][
                                "true_error_or_reference_at_inference"
                            ]
                            is False
                        ),
                    }
                )
            if buffered_relaxed_tether_e14:
                if (
                    a22_r1_rollout is None or a14_lineage_rollout is None
                ):  # pragma: no cover - guarded above
                    raise AssertionError("A23 requires frozen A22-R1 and A14")
                lineage_checks.update(
                    {
                        "exact_qualified_a22_r1_prerequisite": (
                            a22_r1_rollout["status"]
                            == "qualified_calibration_replay"
                            and all(
                                a22_r1_rollout["calibration_gate"]["checks"].values()
                            )
                        ),
                        "exact_stopped_a14_comparator_prerequisite": (
                            a14_lineage_rollout["status"] == "stopped_prospective"
                            and a14_lineage_rollout["population"][
                                "strict_interior_trajectory_win_count"
                            ]
                            == 5
                        ),
                        "e14_protocol_frozen_without_outcome_selection": (
                            source["protocol"]["frozen_from"]
                            == BUFFERED_RELAXED_TETHER_REPLAY_WORKING_ID
                            and source["protocol"][
                                "e14_outcome_used_for_protocol_selection"
                            ]
                            is False
                            and source["protocol"][
                                "coefficient_threshold_rate_cells_or_window_refit"
                            ]
                            is False
                        ),
                    }
                )
            if buffered_offset:
                if a12_rollout is None:  # pragma: no cover - guarded by construction
                    raise AssertionError(
                        "buffered-offset run requires the frozen A12 artifact"
                    )
                if a12_buffer_rescore is None:  # pragma: no cover
                    raise AssertionError(
                        "buffered-offset run requires the frozen A12-R1 selector"
                    )
                lineage_checks.update(
                    {
                        "exact_stopped_a12_prerequisite": (
                            a12_rollout["status"] == "stopped_calibration"
                            and a12_rollout["population"][
                                "maximum_endpoint_state_ratio"
                            ]
                            == 1.0013624254751654
                        ),
                        "buffered_offset_route_and_cells_exact": (
                            source["protocol"]["position_trust_threshold"]
                            == FROZEN_POSITION_TRUST_THRESHOLD
                            and source["protocol"]["position_buffer_threshold"]
                            == FROZEN_POSITION_BUFFER_THRESHOLD
                            and source["protocol"]["coast_projection_rank"] == 8
                            and source["protocol"]["coast_projection_active_cells"]
                            == [list(cell) for cell in FROZEN_ACTIVE_CELLS]
                            and source["protocol"]["coast_offset_feedback"] is False
                            and source["protocol"][
                                "buffer_threshold_selected_from_a12_truth"
                            ]
                            is True
                        ),
                        "exact_qualified_a12_buffer_selector": (
                            a12_buffer_rescore["status"]
                            == "qualified_calibration_rescore"
                            and a12_buffer_rescore["model_calls"] == 0
                            and a12_buffer_rescore["truth_used_for_selector"] is True
                        ),
                    }
                )
            if prospective_buffered_offset:
                if a13_rollout is None:  # pragma: no cover - guarded by construction
                    raise AssertionError(
                        "prospective buffered-offset run requires frozen A13"
                    )
                lineage_checks.update(
                    {
                        "exact_qualified_a13_prerequisite": (
                            a13_rollout["status"] == "qualified_calibration"
                            and all(a13_rollout["calibration_gate"]["checks"].values())
                            and a13_rollout["calibration_gate"][
                                "prospective_claim_authorized"
                            ]
                            is False
                        ),
                        "a13_source_route_coefficient_cells_and_window_frozen": (
                            source["qualified_a13_frozen_source_sha256"]
                            == EXPECTED_P6_A13_SOURCE_SHA256
                            and source["protocol"]["frozen_from"]
                            == BUFFERED_OFFSET_WORKING_ID
                            and source["protocol"]["additional_e12_selection"] is False
                        ),
                    }
                )
            if persistence_gain:
                if a13_rollout is None or a14_rollout is None:  # pragma: no cover
                    raise AssertionError("persistence run requires frozen A13 and A14")
                lineage_checks.update(
                    {
                        "exact_qualified_a13_prerequisite": (
                            a13_rollout["status"] == "qualified_calibration"
                            and all(
                                a13_rollout["calibration_gate"]["checks"].values()
                            )
                        ),
                        "exact_stopped_a14_mechanism_prerequisite": (
                            a14_rollout["status"] == "stopped_prospective"
                            and a14_rollout["population"][
                                "strict_interior_trajectory_win_count"
                            ]
                            == 5
                        ),
                        "persistence_rule_target_free_and_not_refit": (
                            source["protocol"]["persistence_probe_coefficient"] == -0.5
                            and source["protocol"][
                                "coefficient_cell_route_or_window_refit"
                            ]
                            is False
                            and source["protocol"]["a14_truth_used_for_selection"]
                            is False
                        ),
                    }
                )
            if terminal_ramp:
                if a15_rollout is None:  # pragma: no cover - guarded above
                    raise AssertionError("terminal-ramp run requires frozen A15")
                lineage_checks.update(
                    {
                        "exact_stopped_a15_prerequisite": (
                            a15_rollout["status"] == "stopped_calibration"
                            and {
                                name
                                for name, passed in a15_rollout["calibration_gate"][
                                    "checks"
                                ].items()
                                if not bool(passed)
                            }
                            == {"aggregate_increment_defect_rms_ratio_at_most_a8"}
                        ),
                        "terminal_ramp_truth_informed_but_runtime_target_free": (
                            source["protocol"]["a15_truth_used_for_rule"] is True
                            and source["protocol"][
                                "true_error_or_reference_at_inference"
                            ]
                            is False
                            and source["protocol"]["ramp_terminal_block"] == 14
                            and source["protocol"]["ramp_terminal_gain"] == 0.0
                            and source["protocol"][
                                "coefficient_cell_route_or_numeric_rate_fit"
                            ]
                            is False
                        ),
                    }
                )
            if fixed_late_ramp:
                if a13_rollout is None or a16_rollout is None:  # pragma: no cover
                    raise AssertionError("fixed late ramp requires frozen A13 and A16")
                lineage_checks.update(
                    {
                        "exact_qualified_a13_prerequisite": (
                            a13_rollout["status"] == "qualified_calibration"
                            and all(a13_rollout["calibration_gate"]["checks"].values())
                        ),
                        "exact_stopped_a16_prerequisite": (
                            a16_rollout["status"] == "stopped_calibration"
                            and {
                                name
                                for name, passed in a16_rollout["calibration_gate"][
                                    "checks"
                                ].items()
                                if not bool(passed)
                            }
                            == {"aggregate_increment_defect_rms_ratio_at_most_a8"}
                        ),
                        "fixed_schedule_truth_selected_and_runtime_target_free": (
                            source["protocol"][
                                "a13_a15_a16_truth_used_for_schedule_selection"
                            ]
                            is True
                            and source["protocol"][
                                "true_error_or_reference_at_inference"
                            ]
                            is False
                            and source["protocol"]["persistence_probe"] is False
                            and source["protocol"]["ramp_start_block"] == 9
                            and source["protocol"]["ramp_terminal_block"] == 14
                        ),
                    }
                )
            if frozen_offset:
                if a11_rollout is None:  # pragma: no cover - guarded by construction
                    raise AssertionError(
                        "frozen-offset run requires the frozen A11 artifact"
                    )
                lineage_checks.update(
                    {
                        "exact_stopped_a11_prerequisite": (
                            a11_rollout["status"] == "stopped_calibration"
                            and a11_rollout["population"][
                                "strict_interior_trajectory_win_count"
                            ]
                            == 7
                        ),
                        "frozen_offset_cells_rank_and_window_exact": (
                            source["protocol"]["interior_warm_blocks"]
                            == FROZEN_OFFSET_WARM_BLOCKS
                            and source["protocol"]["coast_projection_rank"] == 8
                            and source["protocol"]["coast_projection_active_cells"]
                            == [list(cell) for cell in FROZEN_ACTIVE_CELLS]
                            and source["protocol"]["coast_offset_feedback"] is False
                        ),
                    }
                )
            sealed_check = source["population"]["still_sealed_groups"] == (
                []
                if prospective_buffered_offset or slew_limited_tether or relaxed_tether
                else ["strength_ood_e14"]
            )
            if persistence_gain or fixed_late_ramp or slew_limited_tether or relaxed_tether:
                sealed_check = source["population"]["still_sealed_groups"] == []
        elif prospective:
            if a7_rollout is None:  # pragma: no cover - guarded by construction
                raise AssertionError("prospective run requires the frozen A7 artifact")
            lineage_checks = {
                "exact_stopped_a6_lineage": a6_rollout["status"] == "stopped",
                "exact_qualified_a7_calibration_prerequisite": (
                    a7_rollout["status"] == "qualified_retrospective"
                    and a7_rollout["retrospective_gate"]["prospective_claim_authorized"]
                    is False
                ),
                "coefficient_and_threshold_not_refit": (
                    source["protocol"]["coefficient"] == -0.5
                    and source["protocol"]["position_trust_threshold"]
                    == FROZEN_POSITION_TRUST_THRESHOLD
                    and source["protocol"]["coefficient_or_threshold_refit"] is False
                ),
            }
            sealed_check = source["population"]["still_sealed_groups"] == [
                "strength_ood_e14"
            ]
        else:
            lineage_checks = {
                "exact_stopped_a6_calibration_prerequisite": (
                    a6_rollout["status"] == "stopped"
                )
            }
            sealed_check = source["population"]["still_sealed_groups"] == [
                "strength_ood_e12",
                "strength_ood_e14",
            ]
        if warm_start:
            if a8_rollout is None:  # pragma: no cover - guarded above
                raise AssertionError("warm-start gate requires the frozen A8 result")
            interior_case_ids = (
                set(PROSPECTIVE_BUFFERED_OFFSET_CASE_IDS)
                if prospective_buffered_offset or buffered_relaxed_tether_e14
                else set(PROSPECTIVE_STRUCTURAL_INTERIOR_CASE_IDS)
            )
            interior_wins = sum(
                row["case_id"] in interior_case_ids
                and next(
                    rollout["position_route"]
                    for rollout in rollouts
                    if rollout["case_id"] == row["case_id"]
                )
                == "interior_frozen_offset"
                and float(row["trajectory_state_rms_ratio"]) < 1.0
                for row in case_rows
            )
            population["strict_interior_trajectory_win_count"] = interior_wins
            performance_checks = {
                (
                    "aggregate_state_rms_ratio_strictly_below_one"
                    if persistence_gain
                    else "aggregate_state_rms_ratio_strictly_below_a8"
                ): population["aggregate_state_rms_ratio"]
                < (
                    1.0
                    if persistence_gain
                    else float(a8_rollout["population"]["aggregate_state_rms_ratio"])
                ),
                "aggregate_increment_defect_rms_ratio_at_most_a8": population[
                    "aggregate_increment_defect_rms_ratio"
                ]
                <= float(
                    a8_rollout["population"]["aggregate_increment_defect_rms_ratio"]
                ),
                (
                    "strict_interior_trajectory_win_count_at_least_five"
                        if projected_coast or frozen_offset or slew_limited_tether or relaxed_tether
                    else "strict_interior_trajectory_win_count_at_least_one"
                ): interior_wins
                >= (
                    5
                    if projected_coast
                    or frozen_offset
                    or slew_limited_tether
                    or relaxed_tether
                    else 1
                ),
                "median_endpoint_cumulative_defect_ratio_at_most_one": population[
                    "median_endpoint_cumulative_defect_ratio"
                ]
                <= 1.0,
                "maximum_endpoint_state_ratio_at_most_one": population[
                    "maximum_endpoint_state_ratio"
                ]
                <= 1.0,
            }
        else:
            performance_checks = {
                "aggregate_state_rms_ratio_at_most_0p995": population[
                    "aggregate_state_rms_ratio"
                ]
                <= 0.995,
                "aggregate_increment_defect_ratio_at_most_one": population[
                    "aggregate_increment_defect_rms_ratio"
                ]
                <= 1.0,
                "median_endpoint_cumulative_defect_ratio_at_most_one": population[
                    "median_endpoint_cumulative_defect_ratio"
                ]
                <= 1.0,
                "maximum_endpoint_state_ratio_at_most_one": population[
                    "maximum_endpoint_state_ratio"
                ]
                <= 1.0,
            }
        a22_reproduction_checks = (
            _a22_replay_scoring_checks(
                a22_rescore_path=args.a22_rescore,
                a22_rescore=a22_rescore,
                rows=rows,
                case_rows=case_rows,
                controls=controls,
                population=population,
                descriptor_rows=descriptor_rows,
            )
            if buffered_relaxed_tether
            and not buffered_relaxed_tether_e14
            and a22_rescore is not None
            else {}
        )
        checks = {
            **lineage_checks,
            inventory_check_name: (
                {row["case_id"] for row in rollouts} == set(case_ids)
            ),
            sealed_check_name: sealed_check,
            "all_rollouts_complete_finite_admissible": completed,
            **performance_checks,
            "all_registered_controls_no_harm": bool(controls)
            and all(parent._control_passed(row) for row in controls),
            **call_checks,
            "candidate_integral_anchor_closure": anchor_closure,
            "candidate_response_projection_closure": response_closure,
            "native_truth_reference_contract_exact": native_reference_contract,
            "fine_reference_truth_not_used": source["population"][
                "fine_reference_truth"
            ]
            == "not_required_or_loaded",
            "deterministic_two_output_prefix_and_decisions_exact": (
                prefix_abs == 0.0 and prefix_decisions_exact
            ),
            "source_reference_and_artifact_inventory_exact": source
            == preflight["source_manifest"],
            **a22_reproduction_checks,
        }
        if buffered_offset or buffered_relaxed_tether:
            route_counts = {
                route: sum(rollout["position_route"] == route for rollout in rollouts)
                for route in (
                    "edge_candidate",
                    "raw_uncertainty_buffer",
                    "interior_frozen_offset",
                    "raw_unresolved",
                )
            }
            route_check_name = (
                "buffered_relaxed_tether_route_counts_exact_2_2_5"
                if buffered_relaxed_tether
                else "buffered_offset_route_counts_exact_2_2_5"
            )
            checks[route_check_name] = route_counts == {
                "edge_candidate": 2,
                "raw_uncertainty_buffer": 2,
                "interior_frozen_offset": 5,
                "raw_unresolved": 0,
            }
        qualified = all(checks.values())

        write_csv(output_dir / "rollout_call_metrics.csv", rows)
        write_csv(output_dir / "rollout_case_metrics.csv", case_rows)
        write_csv(output_dir / "rollout_controls.csv", controls)
        write_csv(output_dir / "rollout_execution.csv", execution_rows)
        write_csv(output_dir / "rollout_closure.csv", closure_rows)
        write_csv(output_dir / "position_decisions.csv", descriptor_rows)
        write_csv(output_dir / "front_branch_audits.csv", front_rows)
        write_csv(output_dir / "anchor_audits.csv", anchor_rows)
        projected_coast_rows = [
            row for rollout in rollouts for row in rollout["projected_coast_audits"]
        ]
        if projected_coast:
            write_csv(output_dir / "projected_coast_audits.csv", projected_coast_rows)
        slew_rows = [
            row
            for rollout in rollouts
            for row in rollout["slew_limited_tether_audits"]
        ]
        if slew_limited_tether:
            write_csv(output_dir / "slew_limited_tether_audits.csv", slew_rows)
        relaxed_rows = [
            row for rollout in rollouts for row in rollout["relaxed_tether_audits"]
        ]
        if relaxed_tether:
            write_csv(output_dir / "relaxed_tether_audits.csv", relaxed_rows)
        frozen_offset_rows = [
            row for rollout in rollouts for row in rollout["frozen_offset_audits"]
        ]
        if frozen_offset:
            write_csv(output_dir / "frozen_offset_audits.csv", frozen_offset_rows)
        persistence_rows = [
            row for rollout in rollouts for row in rollout["persistence_probe_audits"]
        ]
        persistence_error_rows = [
            row for rollout in rollouts for row in rollout["persistence_error_structure"]
        ]
        if persistence_gain or fixed_late_ramp:
            write_csv(output_dir / "persistence_probe_audits.csv", persistence_rows)
        if persistence_gain:
            write_csv(
                output_dir / "persistence_error_structure.csv",
                persistence_error_rows,
            )
        write_csv(
            output_dir / "reference_checks.csv",
            [{"case_id": row["case_id"], **row["reference_check"]} for row in rollouts],
        )
        atomic_write_json(output_dir / "source_manifest.json", source)
        animation_manifest = _write_strength_ood_animation_bundles(
            output_dir / "animation_bundles",
            rollouts=rollouts,
            runtime=runtime,
            working_id=working_id,
            animation_contract=animation_contract,
            animation_case_ids=animation_case_ids,
            split_group_id=source["population"]["split_group_id"],
            frame_calls=STRENGTH_OOD_ANIMATION_CALLS,
        )
        files = [
            "rollout_call_metrics.csv",
            "rollout_case_metrics.csv",
            "rollout_controls.csv",
            "rollout_execution.csv",
            "rollout_closure.csv",
            "position_decisions.csv",
            "front_branch_audits.csv",
            "anchor_audits.csv",
            "reference_checks.csv",
            "source_manifest.json",
        ]
        if projected_coast:
            files.append("projected_coast_audits.csv")
        if slew_limited_tether:
            files.append("slew_limited_tether_audits.csv")
        if relaxed_tether:
            files.append("relaxed_tether_audits.csv")
        if frozen_offset:
            files.append("frozen_offset_audits.csv")
        if persistence_gain or fixed_late_ramp:
            files.append("persistence_probe_audits.csv")
        if persistence_gain:
            files.append("persistence_error_structure.csv")
        artifact_hashes = sha256_files(files, root=output_dir)
        cost = {
            "logical_model_calls": sum(
                row["execution"]["logical_model_calls"] for row in rollouts
            ),
            "native_logical_calls": sum(
                row["execution"]["native_logical_calls"] for row in rollouts
            ),
            "fine_logical_calls": sum(
                row["execution"]["fine_logical_calls"] for row in rollouts
            ),
            "actual_forward_passes": sum(
                row["execution"]["actual_forward_passes"] for row in rollouts
            ),
            "total_forward_seconds": sum(
                row["execution"]["total_forward_seconds"] for row in rollouts
            ),
            "wall_seconds": sum(row["execution"]["wall_seconds"] for row in rollouts),
            "maximum_peak_gpu_memory_bytes": max(
                row["execution"]["maximum_peak_gpu_memory_bytes"] for row in rollouts
            ),
            "raw_native_logical_call_comparator": 30 * len(case_ids),
            "a6_always_candidate_logical_calls": a6_rollout["cost"][
                "logical_model_calls"
            ],
        }
        if warm_start:
            if a7_rollout is None or a8_rollout is None or a9_rescore is None:
                raise AssertionError("warm-start payload requires frozen lineage")
            prefix_calls = (
                prefix_a["execution"]["logical_model_calls"]
                + prefix_b["execution"]["logical_model_calls"]
            )
            cost.update(
                {
                    "a8_position_gate_logical_call_comparator": a8_rollout["cost"][
                        "logical_model_calls"
                    ],
                    "deterministic_prefix_logical_model_calls": prefix_calls,
                    "logical_model_calls_including_deterministic_prefix": (
                        cost["logical_model_calls"] + prefix_calls
                    ),
                }
            )
            if frozen_offset:
                cost.update(
                    {
                        "a11_projected_coast_logical_call_comparator": (
                            a11_rollout["cost"]["logical_model_calls"]
                            if a11_rollout is not None
                            else None
                        ),
                        "registered_no_veto_deployment_raw_coast_plus_offset": 502,
                    }
                )
                if buffered_offset:
                    cost.update(
                        {
                            "a12_frozen_offset_logical_call_comparator": (
                                a12_rollout["cost"]["logical_model_calls"]
                                if a12_rollout is not None
                                else None
                            ),
                            "registered_no_veto_buffered_deployment_calls": 470,
                        }
                    )
                    if prospective_buffered_offset:
                        cost["a13_buffered_offset_logical_call_comparator"] = (
                            a13_rollout["cost"]["logical_model_calls"]
                            if a13_rollout is not None
                            else None
                        )
                    if persistence_gain:
                        cost.update(
                            {
                                "a13_buffered_offset_logical_call_comparator": (
                                    a13_rollout["cost"]["logical_model_calls"]
                                    if a13_rollout is not None
                                    else None
                                ),
                                "registered_persistence_probe_calls": 55,
                                "registered_persistence_deployment_calls": 525,
                            }
                        )
                        if terminal_ramp:
                            if a15_rollout is None:  # pragma: no cover
                                raise AssertionError(
                                    "terminal-ramp cost requires frozen A15"
                                )
                            cost.update(
                                {
                                    "a15_persistence_gain_logical_call_comparator": (
                                        a15_rollout["cost"]["logical_model_calls"]
                                    ),
                                    "registered_terminal_ramp_calls": 525,
                                }
                            )
                    if fixed_late_ramp:
                        if a13_rollout is None or a16_rollout is None:  # pragma: no cover
                            raise AssertionError(
                                "fixed late-ramp cost requires frozen A13 and A16"
                            )
                        cost.update(
                            {
                                "a13_buffered_offset_logical_call_comparator": (
                                    a13_rollout["cost"]["logical_model_calls"]
                                ),
                                "a16_terminal_ramp_logical_call_comparator": (
                                    a16_rollout["cost"]["logical_model_calls"]
                                ),
                                "registered_fixed_late_ramp_calls": 470,
                                "registered_additional_persistence_probe_calls": 0,
                            }
                        )
            elif projected_coast or slew_limited_tether or relaxed_tether:
                cost.update(
                    {
                        "a10_unprojected_coast_logical_call_comparator": (
                            a10_rollout["cost"]["logical_model_calls"]
                            if a10_rollout is not None
                            else None
                        ),
                        "registered_no_veto_deployment_with_required_shadow_calls": (
                            580
                            if buffered_relaxed_tether
                            else (656 if slew_limited_tether or relaxed_tether else 670)
                        ),
                    }
                )
                if slew_limited_tether:
                    cost.update(
                        {
                            "a11_projected_coast_logical_call_comparator": (
                                a11_rollout["cost"]["logical_model_calls"]
                            ),
                            "a17_output_offset_logical_call_comparator": (
                                a17_rescore["cost"]["logical_model_calls"]
                            ),
                            "registered_slew_limited_tether_calls": 656,
                        }
                    )
                if relaxed_tether:
                    cost.update(
                        {
                            "a18_slew_limited_tether_logical_call_comparator": (
                                a18_rollout["cost"]["logical_model_calls"]
                            ),
                            "registered_relaxed_tether_calls": 656,
                        }
                    )
                if buffered_relaxed_tether:
                    if a22_rescore is None:  # pragma: no cover - guarded above
                        raise AssertionError(
                            "buffered replay cost requires qualified A22"
                        )
                    cost.update(
                        {
                            "a19_relaxed_tether_logical_call_comparator": (
                                a22_rescore["projected_replay_cost"][
                                    "a19_relaxed_tether_logical_call_comparator"
                                ]
                            ),
                            "registered_buffered_relaxed_tether_calls": 580,
                            "registered_buffered_relaxed_native_calls": 530,
                            "registered_buffered_relaxed_fine_calls": 50,
                        }
                    )
                    if buffered_relaxed_tether_e14:
                        if (
                            a22_r1_rollout is None
                            or a14_lineage_rollout is None
                        ):  # pragma: no cover - guarded above
                            raise AssertionError(
                                "A23 cost requires A22-R1 and A14"
                            )
                        cost.update(
                            {
                                "a22_r1_buffered_relaxed_logical_call_comparator": (
                                    a22_r1_rollout["cost"]["logical_model_calls"]
                                ),
                                "a14_buffered_offset_logical_call_comparator": (
                                    a14_lineage_rollout["cost"]["logical_model_calls"]
                                ),
                            }
                        )
            else:
                cost["registered_no_veto_deployment_without_coast_shadow_calls"] = 530
            lineage_payload = {
                "qualified_a7_rollout_sha256": sha256_file(args.a7_rollout),
                "qualified_a7_rollout_payload_sha256": a7_rollout["payload_sha256"],
                "qualified_a8_rollout_sha256": sha256_file(args.a8_rollout),
                "qualified_a8_rollout_payload_sha256": a8_rollout["payload_sha256"],
                "completed_a9_rescore_sha256": sha256_file(args.a9_rescore),
                "completed_a9_rescore_payload_sha256": a9_rescore["payload_sha256"],
            }
            if projected_coast:
                if a10_rollout is None:  # pragma: no cover - guarded above
                    raise AssertionError(
                        "projected-coast payload requires the frozen A10 artifact"
                    )
                lineage_payload.update(
                    {
                        "stopped_a10_rollout_sha256": sha256_file(args.a10_rollout),
                        "stopped_a10_rollout_payload_sha256": a10_rollout[
                            "payload_sha256"
                        ],
                    }
                )
            if buffered_relaxed_tether_e14:
                if (
                    a22_r1_rollout is None or a14_lineage_rollout is None
                ):  # pragma: no cover - guarded above
                    raise AssertionError("A23 payload requires A22-R1 and A14")
                lineage_payload.update(
                    {
                        "qualified_a22_r1_rollout_sha256": sha256_file(
                            args.a22_r1_rollout
                        ),
                        "qualified_a22_r1_rollout_payload_sha256": a22_r1_rollout[
                            "payload_sha256"
                        ],
                        "stopped_a14_comparator_sha256": sha256_file(
                            args.a14_rollout
                        ),
                        "stopped_a14_comparator_payload_sha256": (
                            a14_lineage_rollout["payload_sha256"]
                        ),
                    }
                )
            if slew_limited_tether:
                if a11_rollout is None or a17_rescore is None:  # pragma: no cover
                    raise AssertionError(
                        "slew-limited payload requires frozen A11 and A17-R1"
                    )
                lineage_payload.update(
                    {
                        "stopped_a11_rollout_sha256": sha256_file(args.a11_rollout),
                        "stopped_a11_rollout_payload_sha256": a11_rollout[
                            "payload_sha256"
                        ],
                        "qualified_a17_rescore_sha256": sha256_file(
                            args.a17_rescore
                        ),
                        "qualified_a17_rescore_payload_sha256": a17_rescore[
                            "payload_sha256"
                        ],
                    }
                )
            if relaxed_tether:
                lineage_payload.update(
                    {
                        "stopped_a18_rollout_sha256": sha256_file(
                            args.a18_rollout
                        ),
                        "stopped_a18_rollout_payload_sha256": a18_rollout[
                            "payload_sha256"
                        ],
                    }
                )
            if buffered_relaxed_tether:
                if a22_rescore is None:  # pragma: no cover - guarded above
                    raise AssertionError(
                        "buffered replay payload requires qualified A22"
                    )
                lineage_payload.update(
                    {
                        "qualified_a22_rescore_sha256": sha256_file(
                            args.a22_rescore
                        ),
                        "qualified_a22_rescore_payload_sha256": a22_rescore[
                            "payload_sha256"
                        ],
                    }
                )
            if frozen_offset:
                if a11_rollout is None:  # pragma: no cover - guarded above
                    raise AssertionError(
                        "frozen-offset payload requires the frozen A11 artifact"
                    )
                lineage_payload.update(
                    {
                        "stopped_a11_rollout_sha256": sha256_file(args.a11_rollout),
                        "stopped_a11_rollout_payload_sha256": a11_rollout[
                            "payload_sha256"
                        ],
                    }
                )
            if buffered_offset:
                if a12_rollout is None:  # pragma: no cover - guarded above
                    raise AssertionError(
                        "buffered-offset payload requires the frozen A12 artifact"
                    )
                lineage_payload.update(
                    {
                        "stopped_a12_rollout_sha256": sha256_file(args.a12_rollout),
                        "stopped_a12_rollout_payload_sha256": a12_rollout[
                            "payload_sha256"
                        ],
                        "qualified_a12_buffer_rescore_sha256": sha256_file(
                            args.a12_buffer_rescore
                        ),
                        "qualified_a12_buffer_rescore_payload_sha256": (
                            a12_buffer_rescore["payload_sha256"]
                            if a12_buffer_rescore is not None
                            else None
                        ),
                    }
                )
                if prospective_buffered_offset:
                    if a13_rollout is None:  # pragma: no cover - guarded above
                        raise AssertionError(
                            "prospective payload requires the frozen A13 artifact"
                        )
                    lineage_payload.update(
                        {
                            "qualified_a13_rollout_sha256": sha256_file(
                                args.a13_rollout
                            ),
                            "qualified_a13_rollout_payload_sha256": a13_rollout[
                                "payload_sha256"
                            ],
                        }
                    )
                if persistence_gain and (
                    a13_rollout is None or a14_rollout is None
                ):  # pragma: no cover
                    raise AssertionError(
                        "persistence payload requires frozen A13 and A14"
                    )
                if fixed_late_ramp:
                    if a13_rollout is None or a16_rollout is None:  # pragma: no cover
                        raise AssertionError(
                            "fixed late-ramp payload requires frozen A13 and A16"
                        )
                    lineage_payload.update(
                        {
                            "qualified_a13_rollout_sha256": sha256_file(
                                args.a13_rollout
                            ),
                            "qualified_a13_rollout_payload_sha256": a13_rollout[
                                "payload_sha256"
                            ],
                            "stopped_a16_rollout_sha256": sha256_file(
                                args.a16_rollout
                            ),
                            "stopped_a16_rollout_payload_sha256": a16_rollout[
                                "payload_sha256"
                            ],
                        }
                    )
                if persistence_gain:
                    lineage_payload.update(
                        {
                            "qualified_a13_rollout_sha256": sha256_file(
                                args.a13_rollout
                            ),
                            "qualified_a13_rollout_payload_sha256": a13_rollout[
                                "payload_sha256"
                            ],
                            "stopped_a14_rollout_sha256": sha256_file(
                                args.a14_rollout
                            ),
                            "stopped_a14_rollout_payload_sha256": a14_rollout[
                                "payload_sha256"
                            ],
                        }
                    )
                    if terminal_ramp:
                        if a15_rollout is None:  # pragma: no cover
                            raise AssertionError(
                                "terminal-ramp payload requires frozen A15"
                            )
                        lineage_payload.update(
                            {
                                "stopped_a15_rollout_sha256": sha256_file(
                                    args.a15_rollout
                                ),
                                "stopped_a15_rollout_payload_sha256": a15_rollout[
                                    "payload_sha256"
                                ],
                            }
                        )
            if prospective_buffered_offset:
                new_sealed_population_opened = PROSPECTIVE_BUFFERED_OFFSET_GROUP_ID
                still_sealed_groups = []
                claim_boundary = (
                    "Single prospective confirmation of the exact A13 target-free "
                    "buffered frozen-output-offset protocol on the pre-named e14 "
                    "dynamic-FV group. It authorizes no further population access, "
                    "cross-family coefficient transfer, physical-conservation, "
                    "resolution-convergence, Richardson-extrapolation, or direct "
                    "off-grid claim."
                )
            else:
                new_sealed_population_opened = False
                still_sealed_groups = (
                    []
                    if persistence_gain
                    or fixed_late_ramp
                    or slew_limited_tether
                    or relaxed_tether
                    else ["strength_ood_e14"]
                )
                claim_boundary = (
                    "Single non-retuned transfer of the exact A22-R1 buffered "
                    "relaxed recurrent protocol to the already-open e14 group; "
                    "truth is used only for offline scoring. "
                    if buffered_relaxed_tether_e14
                    else (
                        "Single A22-authorized integrated replay of the fixed buffered "
                        "relaxed-tether composition on already-open e12; routing uses "
                        "only the initial physical-state position descriptor, and truth "
                        "is used only for offline scoring. "
                        if buffered_relaxed_tether
                        else (
                        "Single unswept relaxed corrected-state recurrence calibration "
                        "on already-open e12; A18 localized the transition defect and "
                        "truth is used only for offline scoring. "
                        if relaxed_tether
                        else (
                            "Single unswept slew-limited corrected-state recurrence "
                            "calibration on already-open e12; A11/A17 motivated the "
                            "mechanism and truth is used only for offline scoring. "
                            if slew_limited_tether
                            else (
                                "Truth-selected fixed late-ramp calibration on already-open "
                                "e12; A13/A15/A16 truth selected block 9, but inference uses "
                                "only the frozen schedule and raw-shadow state. "
                                if fixed_late_ramp
                                else (
                                    "Truth-informed terminal-ramp rule calibration on already-"
                                    "open e12; A15 truth selected the rule class, but inference "
                                    "uses only the synchronized probe/offset cosine and known "
                                    "horizon. "
                                    if terminal_ramp
                                    else (
                                        "Target-free persistence-gain calibration on already-open "
                                        "e12; A14 is hash-bound mechanism evidence only and no "
                                        "e14 state or truth is opened or used. "
                                        if persistence_gain
                                        else (
                                            "Buffered frozen-output-offset protocol calibration "
                                            "on the already-open e12 dynamic-FV group; buffer "
                                            "trajectories and coast model inputs are raw native. "
                                            if buffered_offset
                                            else (
                                                "Frozen-output-offset protocol calibration on "
                                                "the already-open e12 dynamic-FV group; the coast "
                                                "model consumes only the raw shadow. "
                                                if frozen_offset
                                                else (
                                                    "Projected-coast recurrent protocol "
                                                    "calibration on the already-open e12 "
                                                    "dynamic-FV group. "
                                                    if projected_coast
                                                    else "Recurrent protocol calibration on the "
                                                    "already-open e12 dynamic-FV group. "
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                        )
                    )
                )
                claim_boundary += (
                    "It authorizes no new prospective, cross-family, coefficient-"
                    "transfer, "
                    "physical-conservation, resolution-convergence, or direct "
                    "off-grid claim."
                )
        elif prospective:
            if a7_rollout is None:  # pragma: no cover - guarded by construction
                raise AssertionError("prospective run requires the frozen A7 artifact")
            lineage_payload = {
                "qualified_a7_rollout_sha256": sha256_file(args.a7_rollout),
                "qualified_a7_rollout_payload_sha256": a7_rollout["payload_sha256"],
            }
            new_sealed_population_opened: str | bool = PROSPECTIVE_STRUCTURAL_GROUP_ID
            still_sealed_groups = ["strength_ood_e14"]
            claim_boundary = (
                "Prospective confirmation on one pre-named adjacent dynamic-FV "
                "strength group. It authorizes no e14, cross-family, coefficient-"
                "transfer, physical-conservation, resolution-convergence, or direct "
                "off-grid claim."
            )
        else:
            lineage_payload = {}
            new_sealed_population_opened = False
            still_sealed_groups = ["strength_ood_e12", "strength_ood_e14"]
            claim_boundary = (
                "Retrospective target-free protocol calibration on the already-open "
                "e13 dynamic-FV group. It authorizes no prospective, neighboring-"
                "strength, cross-family, physical-conservation, resolution-"
                "convergence, or direct off-grid claim."
            )
        payload = with_payload_sha256(
            {
                "schema": schema,
                "working_id": working_id,
                "status": qualified_status if qualified else stopped_status,
                "population_status": population_status,
                gate_key: {
                    "status": qualified_status if qualified else stopped_status,
                    "checks": checks,
                    "prospective_claim_authorized": (
                        (
                            prospective_buffered_offset
                            or (prospective and not warm_start)
                        )
                        and qualified
                    ),
                },
                "population": population,
                "failed_controls": [
                    row for row in controls if not parent._control_passed(row)
                ],
                "position_decisions": descriptor_rows,
                "front_branch_vetoes": [
                    row for row in front_rows if bool(row["branch_changed"])
                ],
                "cost": cost,
                "deterministic_prefix_max_abs": prefix_abs,
                "preflight_sha256": sha256_file(args.preflight),
                "preflight_payload_sha256": preflight["payload_sha256"],
                "stopped_a6_rollout_sha256": sha256_file(args.a6_rollout),
                "stopped_a6_rollout_payload_sha256": a6_rollout["payload_sha256"],
                **lineage_payload,
                "source_manifest_sha256": sha256_file(
                    output_dir / "source_manifest.json"
                ),
                "source_manifest_payload_sha256": source["payload_sha256"],
                "artifact_hashes": artifact_hashes,
                "animation_bundle_manifest": animation_manifest,
                "animation_bundle_manifest_sha256": sha256_file(
                    output_dir / "animation_bundles" / "bundle_manifest.json"
                ),
                "recurrence_executed": True,
                "new_sealed_population_opened": new_sealed_population_opened,
                "opened_case_ids": (
                    []
                    if buffered_relaxed_tether_e14
                    else (
                        list(case_ids)
                        if prospective_buffered_offset
                        or (prospective and not warm_start)
                        else []
                    )
                ),
                "still_sealed_groups": still_sealed_groups,
                "claim_boundary": claim_boundary,
            }
        )
        atomic_write_json(output_dir / result_name, payload)
        return payload, 0 if qualified else 4
    finally:
        collection._close_runtime(runtime)


def run_structural_gate_rollout(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], int]:
    return _run_structural_gate_population(args, prospective=False)


def run_prospective_structural_rollout(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], int]:
    return _run_structural_gate_population(args, prospective=True)


def run_warm_start_rollout(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], int]:
    return _run_structural_gate_population(args, prospective=False, warm_start=True)


def run_projected_coast_rollout(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], int]:
    return _run_structural_gate_population(
        args,
        prospective=False,
        warm_start=True,
        projected_coast=True,
    )


def run_slew_limited_tether_rollout(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], int]:
    return _run_structural_gate_population(
        args,
        prospective=False,
        warm_start=True,
        slew_limited_tether=True,
    )


def run_relaxed_tether_rollout(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], int]:
    return _run_structural_gate_population(
        args,
        prospective=False,
        warm_start=True,
        relaxed_tether=True,
    )


def run_buffered_relaxed_tether_replay(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], int]:
    return _run_structural_gate_population(
        args,
        prospective=False,
        warm_start=True,
        relaxed_tether=True,
        buffered_relaxed_tether=True,
    )


def run_buffered_relaxed_tether_e14_rollout(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], int]:
    return _run_structural_gate_population(
        args,
        prospective=False,
        warm_start=True,
        relaxed_tether=True,
        buffered_relaxed_tether=True,
        buffered_relaxed_tether_e14=True,
    )


def run_frozen_offset_rollout(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], int]:
    return _run_structural_gate_population(
        args,
        prospective=False,
        warm_start=True,
        frozen_offset=True,
    )


def run_buffered_offset_rollout(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], int]:
    return _run_structural_gate_population(
        args,
        prospective=False,
        warm_start=True,
        frozen_offset=True,
        buffered_offset=True,
    )


def run_prospective_buffered_offset_rollout(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], int]:
    return _run_structural_gate_population(
        args,
        prospective=False,
        warm_start=True,
        frozen_offset=True,
        buffered_offset=True,
        prospective_buffered_offset=True,
    )


def run_persistence_gain_rollout(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], int]:
    return _run_structural_gate_population(
        args,
        prospective=False,
        warm_start=True,
        frozen_offset=True,
        buffered_offset=True,
        persistence_gain=True,
    )


def run_terminal_ramp_rollout(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], int]:
    return _run_structural_gate_population(
        args,
        prospective=False,
        warm_start=True,
        frozen_offset=True,
        buffered_offset=True,
        persistence_gain=True,
        terminal_ramp=True,
    )


def run_fixed_late_ramp_rollout(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], int]:
    return _run_structural_gate_population(
        args,
        prospective=False,
        warm_start=True,
        frozen_offset=True,
        buffered_offset=True,
        fixed_late_ramp=True,
    )


def _fixed_late_ramp_saved_evidence_checks(
    *,
    root: Path,
    prior: Mapping[str, Any],
    preflight: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, bool]:
    expected_routes = {
        "sv_e12_y00": "edge_candidate",
        "sv_e12_y01": "raw_uncertainty_buffer",
        "sv_e12_y02": "interior_frozen_offset",
        "sv_e12_y03": "interior_frozen_offset",
        "sv_e12_y04": "interior_frozen_offset",
        "sv_e12_y05": "interior_frozen_offset",
        "sv_e12_y06": "interior_frozen_offset",
        "sv_e12_y07": "raw_uncertainty_buffer",
        "sv_e12_y08": "edge_candidate",
    }
    decisions = _read_csv(root / "position_decisions.csv")
    decision_by_case = {row["case_id"]: row for row in decisions}
    execution = _read_csv(root / "rollout_execution.csv")
    execution_by_case = {row["case_id"]: row for row in execution}
    closure = _read_csv(root / "rollout_closure.csv")
    closure_by_case = {row["case_id"]: row for row in closure}
    audits = _read_csv(root / "persistence_probe_audits.csv")
    call_rows = _read_csv(root / "rollout_call_metrics.csv")
    expected_gains = [1.0] * 6 + [0.8, 0.6, 0.4, 0.2, 0.0]

    inventory_exact = all(
        (
            len(decisions) == 9,
            len(decision_by_case) == 9,
            set(decision_by_case) == set(expected_routes),
            len(execution) == 9,
            len(execution_by_case) == 9,
            set(execution_by_case) == set(expected_routes),
            len(closure) == 9,
            len(closure_by_case) == 9,
            set(closure_by_case) == set(expected_routes),
            len(audits) == 55,
            len(call_rows) == 540,
        )
    )
    route_exact = inventory_exact and all(
        decision_by_case[case_id]["position_route"] == route
        for case_id, route in expected_routes.items()
    )

    execution_exact = inventory_exact
    for case_id, route in expected_routes.items():
        row = execution_by_case[case_id]
        expected = {
            "edge_candidate": (90, 75, 15),
            "raw_uncertainty_buffer": (30, 30, 0),
            "interior_frozen_offset": (46, 42, 4),
        }[route]
        execution_exact = execution_exact and all(
            (
                row["policy"] == "fixed_late_ramp",
                int(row["logical_model_calls"]) == expected[0],
                int(row["actual_forward_passes"]) == expected[0],
                int(row["native_logical_calls"]) == expected[1],
                int(row["fine_logical_calls"]) == expected[2],
                int(row["persistence_probe_fine_logical_calls"]) == 0,
                int(row["completed_calls"]) == 30,
            )
        )
    execution_exact = execution_exact and all(
        (
            sum(int(row["logical_model_calls"]) for row in execution) == 470,
            sum(int(row["native_logical_calls"]) for row in execution) == 420,
            sum(int(row["fine_logical_calls"]) for row in execution) == 50,
            prior["cost"]["logical_model_calls"] == 470,
            prior["cost"]["native_logical_calls"] == 420,
            prior["cost"]["fine_logical_calls"] == 50,
            prior["cost"]["registered_additional_persistence_probe_calls"] == 0,
        )
    )

    gain_exact = inventory_exact
    for case_id in tuple(f"sv_e12_y{index:02d}" for index in range(2, 7)):
        rows = [row for row in audits if row["case_id"] == case_id]
        gain_exact = gain_exact and all(
            (
                [int(row["block_index"]) for row in rows] == list(range(4, 15)),
                [int(row["output_call"]) for row in rows]
                == list(range(9, 31, 2)),
                all(
                    abs(float(row["applied_gain"]) - expected) <= 1.0e-15
                    for row, expected in zip(rows, expected_gains, strict=True)
                ),
                all(
                    row["gain_status"] == "fixed_late_terminal_ramp"
                    for row in rows
                ),
            )
        )

    closure_exact = inventory_exact and all(
        all(
            (
                int(row["call_order_errors"]) == 0,
                float(row["shadow_recurrence_abs"]) <= 1.0e-12,
                float(row["accepted_recurrence_abs"]) <= 1.0e-6,
                float(row["candidate_common_native_input_abs"]) <= 1.0e-12,
                float(row["candidate_lookahead_input_abs"]) <= 1.0e-12,
                float(row["persistence_offset_integral_abs"]) <= 1.0e-10,
                float(row["persistence_offset_boundary_abs"]) <= 1.0e-12,
                float(row["persistence_gain_increase_abs"]) <= 1.0e-15,
                float(row["persistence_offset_bookkeeping_abs"]) <= 1.0e-12,
                float(row["persistence_within_block_identity_abs"]) <= 1.0e-12,
            )
        )
        for row in closure
    )

    raw_exact = inventory_exact
    for case_id in ("sv_e12_y01", "sv_e12_y07"):
        raw = sorted(
            (row for row in call_rows if row["case_id"] == case_id and row["policy"] == "zero"),
            key=lambda row: int(row["output_call"]),
        )
        candidate = sorted(
            (
                row
                for row in call_rows
                if row["case_id"] == case_id and row["policy"] == "buffered_offset"
            ),
            key=lambda row: int(row["output_call"]),
        )
        raw_exact = raw_exact and len(raw) == len(candidate) == 30
        for left, right in zip(raw, candidate, strict=True):
            raw_exact = raw_exact and all(
                left[key] == right[key]
                for key in left
                if key not in {"policy", "correction_status"}
            )
            raw_exact = raw_exact and right["correction_status"] == (
                "position_uncertainty_buffer_raw"
            )

    protocol = source["protocol"]
    schedule_exact = all(
        (
            protocol["gain_by_coast_block_4_to_14"] == expected_gains,
            preflight["schedule"] == expected_gains,
            protocol["ramp_start_block"] == 9,
            protocol["ramp_terminal_block"] == 14,
            protocol["persistence_probe"] is False,
            protocol["additional_fine_probe_calls"] == 0,
            protocol["a13_a15_a16_truth_used_for_schedule_selection"] is True,
            protocol["true_error_or_reference_at_inference"] is False,
            protocol["coefficient_cell_route_or_schedule_refit_at_inference"]
            is False,
        )
    )
    population_exact = all(
        (
            preflight["status"] == "passed",
            preflight["population_status"]
            == "already_open_e12_fixed_late_ramp_calibration",
            preflight["case_inventory_exact"] is True,
            preflight["case_inventory_present"] is True,
            preflight["all_cases_in_opened_e12_group"] is True,
            preflight["new_population_opened"] is False,
            preflight["checkpoint_model_built"] is False,
            preflight["reference_arrays_loaded"] is False,
            preflight["recurrence_executed"] is False,
            preflight["truth_or_reference_used_at_inference"] is False,
            prior["new_sealed_population_opened"] is False,
            prior["opened_case_ids"] == [],
            prior["still_sealed_groups"] == [],
            source["population"]["split_group_id"] == "strength_ood_e12",
            source["population"]["case_ids"]
            == list(PROSPECTIVE_STRUCTURAL_CASE_IDS),
            source["population"]["new_sealed_population_opened"] is False,
        )
    )
    inherited_label_exact = all(
        (
            source["population"]["population_role"]
            == "already_open_e12_recurrent_calibration",
            source["population"]["still_sealed_groups"]
            == ["strength_ood_e14"],
            preflight["source_manifest"]["population"]["still_sealed_groups"]
            == ["strength_ood_e14"],
        )
    )
    prior_checks = prior["calibration_gate"]["checks"]
    original_gate_exact = all(
        passed
        for name, passed in prior_checks.items()
        if name
        not in {
            "fixed_late_ramp_edge_and_interior_contract_exact",
            "no_new_population_opened",
        }
    ) and {
        name for name, passed in prior_checks.items() if not passed
    } == {
        "fixed_late_ramp_edge_and_interior_contract_exact",
        "no_new_population_opened",
    }
    return {
        "exact_frozen_a17_and_artifact_inventory": True,
        "original_gate_failed_only_two_bookkeeping_checks": original_gate_exact,
        "direct_route_inventory_exact_2_2_5": route_exact,
        "direct_fixed_late_ramp_execution_inventory_exact": execution_exact,
        "direct_fixed_late_ramp_gain_inventory_exact": gain_exact,
        "direct_recurrence_and_offset_closure_exact": closure_exact,
        "direct_raw_buffer_identity_exact": raw_exact,
        "registered_schedule_and_thresholds_unchanged": schedule_exact,
        "inherited_a13_sealed_label_exact_and_nonoperational": (
            inherited_label_exact
        ),
        "direct_no_new_population_evidence_exact": population_exact,
        "model_calls_zero": True,
        "predictions_and_recurrence_not_recomputed": True,
    }


def run_fixed_late_ramp_rescore(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], int]:
    prior, preflight, source = _verify_a17_rollout_for_rescore(
        args.prior_fixed_late_ramp_rollout
    )
    root = args.prior_fixed_late_ramp_rollout.parent
    checks = _fixed_late_ramp_saved_evidence_checks(
        root=root,
        prior=prior,
        preflight=preflight,
        source=source,
    )
    qualified = all(checks.values())
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(
        output_dir / "gate_checks.csv",
        [{"check": key, "passed": value} for key, value in checks.items()],
    )
    rescore_source = with_payload_sha256(
        {
            "schema": (
                "pcno_response_filtered_block_fixed_late_ramp_rescore_source_v1"
            ),
            "working_id": FIXED_LATE_RAMP_RESCORE_WORKING_ID,
            "source_sha256": _source_hashes(),
            "source_status": _git_status_short(SOURCE_PATHS),
            "prior_a17_sha256": sha256_file(args.prior_fixed_late_ramp_rollout),
            "prior_a17_payload_sha256": prior["payload_sha256"],
            "prior_a17_source_manifest_sha256": prior[
                "source_manifest_sha256"
            ],
            "prior_a17_artifact_sha256": EXPECTED_P6_A17_ARTIFACT_SHA256,
            "scope": (
                "zero_model_source_bound_bookkeeping_rescore_of_saved_a17_only"
            ),
            "schedule_threshold_or_population_change": False,
        }
    )
    atomic_write_json(output_dir / "source_manifest.json", rescore_source)
    artifact_hashes = sha256_files(
        ("gate_checks.csv", "source_manifest.json"), root=output_dir
    )
    payload = with_payload_sha256(
        {
            "schema": FIXED_LATE_RAMP_RESCORE_SCHEMA,
            "working_id": FIXED_LATE_RAMP_RESCORE_WORKING_ID,
            "status": (
                "qualified_calibration_rescore" if qualified else "stopped_rescore"
            ),
            "calibration_gate": {
                "checks": checks,
                "prospective_claim_authorized": False,
            },
            "population": prior["population"],
            "cost": prior["cost"],
            "prior_a17_sha256": sha256_file(args.prior_fixed_late_ramp_rollout),
            "prior_a17_payload_sha256": prior["payload_sha256"],
            "prior_a17_source_manifest_sha256": prior[
                "source_manifest_sha256"
            ],
            "prior_a17_animation_bundle_manifest_sha256": prior[
                "animation_bundle_manifest_sha256"
            ],
            "source_manifest_sha256": sha256_file(
                output_dir / "source_manifest.json"
            ),
            "source_manifest_payload_sha256": rescore_source["payload_sha256"],
            "artifact_hashes": artifact_hashes,
            "model_calls": 0,
            "predictions_recomputed": False,
            "recurrence_reexecuted": False,
            "schedule_or_threshold_refit": False,
            "truth_used_for_schedule_selection": True,
            "new_sealed_population_opened": False,
            "opened_case_ids": [],
            "still_sealed_groups": [],
            "claim_boundary": (
                "Zero-model bookkeeping repair of the already-scored, truth-selected "
                "E12 A17 calibration. It authorizes no prospective, cross-family, "
                "corrected-state stabilization, conservation, convergence, off-grid, "
                "or Richardson-extrapolation claim."
            ),
        }
    )
    atomic_write_json(output_dir / "fixed_late_ramp_rescore.json", payload)
    return payload, 0 if qualified else 4


def run_buffered_offset_rescore(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], int]:
    prior = _verify_a12_rollout(args.prior_frozen_offset_rollout)
    root = args.prior_frozen_offset_rollout.parent
    rows = _typed_rollout_rows(_read_csv(root / "rollout_call_metrics.csv"))
    decisions = _read_csv(root / "position_decisions.csv")
    routes = {}
    for row in decisions:
        descriptor = TransverseVelocityPositionDescriptor(
            status=str(row["status"]),
            vertical_centroid=(
                None
                if row["vertical_centroid"] == ""
                else float(row["vertical_centroid"])
            ),
            normalized_wall_distance=(
                None
                if row["normalized_wall_distance"] == ""
                else float(row["normalized_wall_distance"])
            ),
            transverse_velocity_l1=float(row["transverse_velocity_l1"]),
        )
        routes[str(row["case_id"])] = buffered_frozen_offset_position_route(descriptor)
    if set(routes) != set(PROSPECTIVE_STRUCTURAL_CASE_IDS):
        raise ValueError("A12-R1 position inventory is incomplete")

    selected_rows = [row for row in rows if row["policy"] == "zero"]
    for row in rows:
        if row["policy"] == "zero":
            continue
        if routes[str(row["case_id"])] == "raw_uncertainty_buffer":
            continue
        selected_rows.append({**row, "policy": "buffered_offset"})
    for row in rows:
        if row["policy"] != "zero" or routes[str(row["case_id"])] != (
            "raw_uncertainty_buffer"
        ):
            continue
        selected_rows.append({**row, "policy": "buffered_offset"})
    case_rows, controls, population = _paired_rollout_controls_for_cases(
        selected_rows,
        case_ids=PROSPECTIVE_STRUCTURAL_CASE_IDS,
    )
    corrected_interior_wins = sum(
        routes[str(row["case_id"])] == "interior_frozen_offset"
        and float(row["trajectory_state_rms_ratio"]) < 1.0
        for row in case_rows
    )
    population["strict_corrected_interior_trajectory_win_count"] = (
        corrected_interior_wins
    )
    checks = {
        "exact_stopped_a12_lineage": prior["status"] == "stopped_calibration",
        "a12_artifact_inventory_exact": all(
            sha256_file(root / name) == expected
            for name, expected in EXPECTED_P6_A12_ARTIFACT_SHA256.items()
        ),
        "route_inventory_exact": set(routes) == set(PROSPECTIVE_STRUCTURAL_CASE_IDS),
        "route_counts_exact": {
            route: sum(value == route for value in routes.values())
            for route in (
                "edge_candidate",
                "raw_uncertainty_buffer",
                "interior_frozen_offset",
                "raw_unresolved",
            )
        }
        == {
            "edge_candidate": 2,
            "raw_uncertainty_buffer": 2,
            "interior_frozen_offset": 5,
            "raw_unresolved": 0,
        },
        "aggregate_state_rms_ratio_strictly_below_a8": population[
            "aggregate_state_rms_ratio"
        ]
        < 0.9884779121207531,
        "aggregate_increment_defect_rms_ratio_at_most_a8": population[
            "aggregate_increment_defect_rms_ratio"
        ]
        <= 0.9970100807663629,
        "strict_corrected_interior_trajectory_win_count_at_least_five": (
            corrected_interior_wins >= 5
        ),
        "median_endpoint_cumulative_defect_ratio_at_most_one": population[
            "median_endpoint_cumulative_defect_ratio"
        ]
        <= 1.0,
        "maximum_endpoint_state_ratio_at_most_one": population[
            "maximum_endpoint_state_ratio"
        ]
        <= 1.0,
        "all_registered_controls_no_harm": bool(controls)
        and all(parent._control_passed(row) for row in controls),
        "model_calls_zero": True,
        "predictions_and_recurrence_not_recomputed": True,
        "e14_remained_sealed": prior["still_sealed_groups"] == ["strength_ood_e14"],
    }
    qualified = all(checks.values())
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    route_rows = [
        {"case_id": case_id, "route": routes[case_id]}
        for case_id in PROSPECTIVE_STRUCTURAL_CASE_IDS
    ]
    write_csv(output_dir / "position_routes.csv", route_rows)
    write_csv(output_dir / "rollout_case_metrics.csv", case_rows)
    write_csv(output_dir / "rollout_controls.csv", controls)
    write_csv(
        output_dir / "gate_checks.csv",
        [{"check": key, "passed": value} for key, value in checks.items()],
    )
    source = with_payload_sha256(
        {
            "schema": "pcno_response_filtered_block_buffered_offset_rescore_source_v1",
            "working_id": BUFFERED_OFFSET_RESCORE_WORKING_ID,
            "source_sha256": _source_hashes(),
            "source_status": _git_status_short(SOURCE_PATHS),
            "prior_a12_sha256": sha256_file(args.prior_frozen_offset_rollout),
            "prior_a12_payload_sha256": prior["payload_sha256"],
            "position_trust_threshold": FROZEN_POSITION_TRUST_THRESHOLD,
            "position_buffer_threshold": FROZEN_POSITION_BUFFER_THRESHOLD,
            "selector": "symmetric_open_e12_truth_informed_no_forward_rescore",
            "still_sealed_groups": ["strength_ood_e14"],
        }
    )
    atomic_write_json(output_dir / "source_manifest.json", source)
    artifact_hashes = sha256_files(
        (
            "position_routes.csv",
            "rollout_case_metrics.csv",
            "rollout_controls.csv",
            "gate_checks.csv",
            "source_manifest.json",
        ),
        root=output_dir,
    )
    payload = with_payload_sha256(
        {
            "schema": BUFFERED_OFFSET_RESCORE_SCHEMA,
            "working_id": BUFFERED_OFFSET_RESCORE_WORKING_ID,
            "status": "qualified_calibration_rescore"
            if qualified
            else "stopped_rescore",
            "calibration_gate": {"checks": checks},
            "population": population,
            "position_routes": route_rows,
            "failed_controls": [
                row for row in controls if not parent._control_passed(row)
            ],
            "prior_a12_sha256": sha256_file(args.prior_frozen_offset_rollout),
            "prior_a12_payload_sha256": prior["payload_sha256"],
            "source_manifest_sha256": sha256_file(output_dir / "source_manifest.json"),
            "source_manifest_payload_sha256": source["payload_sha256"],
            "artifact_hashes": artifact_hashes,
            "model_calls": 0,
            "predictions_recomputed": False,
            "recurrence_reexecuted": False,
            "truth_used_for_selector": True,
            "inference_route_inputs": ["initial_normalized_wall_distance"],
            "new_sealed_population_opened": False,
            "still_sealed_groups": ["strength_ood_e14"],
            "claim_boundary": (
                "Truth-informed selector calibration on the already-open e12 dynamic-"
                "FV population. This is no-forward evidence and authorizes no e14, "
                "cross-family, conservation, convergence, or off-grid claim."
            ),
        }
    )
    atomic_write_json(output_dir / "buffered_offset_rescore.json", payload)
    return payload, 0 if qualified else 4


def _verify_a19_rollout_for_buffered_rescore(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_P6_A19_ROLLOUT_SHA256:
        raise ValueError("A22 requires the exact immutable A19 result")
    prior = _read_json(path)
    verify_payload_sha256(prior)
    if (
        prior.get("schema") != RELAXED_TETHER_SCHEMA
        or prior.get("working_id") != RELAXED_TETHER_WORKING_ID
        or prior.get("status") != "stopped_calibration"
        or prior.get("payload_sha256") != EXPECTED_P6_A19_PAYLOAD_SHA256
        or prior.get("source_manifest_sha256")
        != EXPECTED_P6_A19_SOURCE_MANIFEST_SHA256
        or prior.get("recurrence_executed") is not True
        or prior.get("artifact_hashes") != EXPECTED_P6_A19_ARTIFACT_SHA256
    ):
        raise ValueError("A19 result identity or artifact inventory mismatch")
    root = path.parent
    actual = {
        name: sha256_file(root / name) for name in EXPECTED_P6_A19_ARTIFACT_SHA256
    }
    if actual != EXPECTED_P6_A19_ARTIFACT_SHA256:
        raise ValueError("A19 side-artifact hash mismatch")
    source = _read_json(root / "source_manifest.json")
    verify_payload_sha256(source)
    if sha256_file(root / "source_manifest.json") != (
        EXPECTED_P6_A19_SOURCE_MANIFEST_SHA256
    ):
        raise ValueError("A19 source-manifest identity mismatch")
    checks = prior.get("calibration_gate", {}).get("checks")
    if not isinstance(checks, Mapping):
        raise TypeError("A19 calibration checks are missing")
    failed = {str(name) for name, passed in checks.items() if passed is not True}
    if failed != {"maximum_endpoint_state_ratio_at_most_one"}:
        raise ValueError("A19 did not stop solely on its registered endpoint gate")
    return prior


def _buffered_relaxed_tether_routes(
    decisions: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    if len(decisions) != len(PROSPECTIVE_STRUCTURAL_CASE_IDS):
        raise ValueError("A22 position-decision row count mismatch")
    routes: dict[str, str] = {}
    for row in decisions:
        case_id = str(row.get("case_id", ""))
        if not case_id or case_id in routes:
            raise ValueError("A22 position decisions require unique case identifiers")
        descriptor = TransverseVelocityPositionDescriptor(
            status=str(row["status"]),
            vertical_centroid=(
                None
                if row["vertical_centroid"] == ""
                else float(row["vertical_centroid"])
            ),
            normalized_wall_distance=(
                None
                if row["normalized_wall_distance"] == ""
                else float(row["normalized_wall_distance"])
            ),
            transverse_velocity_l1=float(row["transverse_velocity_l1"]),
        )
        routes[case_id] = buffered_frozen_offset_position_route(descriptor)
    if set(routes) != set(PROSPECTIVE_STRUCTURAL_CASE_IDS):
        raise ValueError("A22 position-decision case inventory mismatch")
    expected_counts = {
        "edge_candidate": 2,
        "raw_uncertainty_buffer": 2,
        "interior_frozen_offset": 5,
        "raw_unresolved": 0,
    }
    actual_counts = {
        route: sum(value == route for value in routes.values())
        for route in expected_counts
    }
    if actual_counts != expected_counts:
        raise ValueError("A22 frozen route inventory mismatch")
    return routes


def _compose_buffered_relaxed_tether_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    routes: Mapping[str, str],
) -> list[dict[str, Any]]:
    expected_pairs = {
        (case_id, input_call)
        for case_id in PROSPECTIVE_STRUCTURAL_CASE_IDS
        for input_call in parent.ALL_INPUT_CALLS
    }
    lookup: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id", ""))
        input_call = int(row.get("input_call", -1))
        policy = str(row.get("policy", ""))
        key = (case_id, input_call, policy)
        if key in lookup:
            raise ValueError("A22 source rows contain duplicate case/call/policy keys")
        lookup[key] = row
    expected_keys = {
        (case_id, input_call, policy)
        for case_id, input_call in expected_pairs
        for policy in ("zero", "relaxed_tether")
    }
    if set(lookup) != expected_keys or set(routes) != set(
        PROSPECTIVE_STRUCTURAL_CASE_IDS
    ):
        raise ValueError("A22 source row or route inventory mismatch")

    selected: list[dict[str, Any]] = []
    for case_id in PROSPECTIVE_STRUCTURAL_CASE_IDS:
        route = routes[case_id]
        source_policy = (
            "zero" if route in {"raw_uncertainty_buffer", "raw_unresolved"}
            else "relaxed_tether"
        )
        for input_call in parent.ALL_INPUT_CALLS:
            raw = lookup[(case_id, input_call, "zero")]
            selected.append(dict(raw))
            chosen = dict(lookup[(case_id, input_call, source_policy)])
            chosen["policy"] = "buffered_relaxed_tether"
            chosen["source_policy"] = source_policy
            chosen["position_route"] = route
            selected.append(chosen)
    return selected


def _buffered_relaxed_tether_call_budget(
    routes: Mapping[str, str],
) -> dict[str, int]:
    counts = {
        route: sum(value == route for value in routes.values())
        for route in (
            "edge_candidate",
            "raw_uncertainty_buffer",
            "interior_frozen_offset",
            "raw_unresolved",
        )
    }
    native = (
        75 * counts["edge_candidate"]
        + 30 * (counts["raw_uncertainty_buffer"] + counts["raw_unresolved"])
        + 64 * counts["interior_frozen_offset"]
    )
    fine = 15 * counts["edge_candidate"] + 4 * counts["interior_frozen_offset"]
    return {
        "logical_model_calls": native + fine,
        "native_logical_calls": native,
        "fine_logical_calls": fine,
        "raw_native_logical_call_comparator": 30
        * len(PROSPECTIVE_STRUCTURAL_CASE_IDS),
        "a19_relaxed_tether_logical_call_comparator": 656,
        "deterministic_prefix_logical_model_calls": 12,
        "logical_model_calls_including_deterministic_prefix": native + fine + 12,
    }


def _buffered_relaxed_tether_saved_branch_checks(
    *,
    root: Path,
    routes: Mapping[str, str],
    selected_rows: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    execution_rows = _read_csv(root / "rollout_execution.csv")
    closure_rows = _read_csv(root / "rollout_closure.csv")
    audit_rows = _read_csv(root / "relaxed_tether_audits.csv")
    execution = {str(row["case_id"]): row for row in execution_rows}
    closure = {str(row["case_id"]): row for row in closure_rows}
    candidate = [row for row in selected_rows if row["policy"] != "zero"]
    buffer_rows = [
        row
        for row in candidate
        if row["position_route"] in {"raw_uncertainty_buffer", "raw_unresolved"}
    ]
    nonbuffer_rows = [row for row in candidate if row not in buffer_rows]
    inventory = set(PROSPECTIVE_STRUCTURAL_CASE_IDS)
    return {
        "saved_execution_and_closure_case_inventory_exact": (
            len(execution_rows) == len(inventory)
            and len(closure_rows) == len(inventory)
            and set(execution) == inventory
            and set(closure) == inventory
        ),
        "selected_row_inventory_exact": (
            len(candidate) == len(inventory) * len(parent.ALL_INPUT_CALLS)
            and len(buffer_rows) == 2 * len(parent.ALL_INPUT_CALLS)
            and len(nonbuffer_rows) == 7 * len(parent.ALL_INPUT_CALLS)
        ),
        "buffer_rows_are_exact_raw_metrics": all(
            row["source_policy"] == "zero" for row in buffer_rows
        ),
        "edge_and_deep_rows_are_exact_a19_metrics": all(
            row["source_policy"] == "relaxed_tether" for row in nonbuffer_rows
        ),
        "selected_branch_call_inventories_exact": all(
            (
                int(execution[case_id]["logical_model_calls"]),
                int(execution[case_id]["native_logical_calls"]),
                int(execution[case_id]["fine_logical_calls"]),
            )
            == (
                (90, 75, 15)
                if route == "edge_candidate"
                else (68, 64, 4)
            )
            and int(execution[case_id]["shadow_native_logical_calls"]) == 30
            and int(execution[case_id]["completed_calls"]) == 30
            for case_id, route in routes.items()
        ),
        "selected_branch_recurrence_and_closure_exact": all(
            int(closure[case_id]["call_order_errors"]) == 0
            and float(closure[case_id]["shadow_recurrence_abs"]) <= 1.0e-12
            and float(closure[case_id]["accepted_recurrence_abs"]) <= 1.0e-6
            and float(closure[case_id]["candidate_common_native_input_abs"])
            <= 1.0e-12
            and float(closure[case_id]["candidate_lookahead_input_abs"])
            <= 1.0e-12
            and int(closure[case_id]["front_branch_veto_count"]) == 0
            for case_id, route in routes.items()
            if route not in {"raw_uncertainty_buffer", "raw_unresolved"}
        ),
        "deep_relaxed_tether_audit_inventory_exact": {
            (str(row["case_id"]), int(row["output_call"]))
            for row in audit_rows
            if routes[str(row["case_id"])] == "interior_frozen_offset"
        }
        == {
            (case_id, output_call)
            for case_id, route in routes.items()
            if route == "interior_frozen_offset"
            for output_call in range(9, 31)
        },
    }


def run_buffered_relaxed_tether_rescore(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], int]:
    prior = _verify_a19_rollout_for_buffered_rescore(
        args.prior_relaxed_tether_rollout
    )
    root = args.prior_relaxed_tether_rollout.parent
    routes = _buffered_relaxed_tether_routes(
        _read_csv(root / "position_decisions.csv")
    )
    source_rows = _typed_rollout_rows(_read_csv(root / "rollout_call_metrics.csv"))
    selected_rows = _compose_buffered_relaxed_tether_rows(
        source_rows, routes=routes
    )
    case_rows, controls, population = _paired_rollout_controls_for_cases(
        selected_rows,
        case_ids=PROSPECTIVE_STRUCTURAL_CASE_IDS,
    )
    deep_wins = sum(
        routes[str(row["case_id"])] == "interior_frozen_offset"
        and float(row["trajectory_state_rms_ratio"]) < 1.0
        for row in case_rows
    )
    population["strict_deep_interior_trajectory_win_count"] = deep_wins
    branch_checks = _buffered_relaxed_tether_saved_branch_checks(
        root=root,
        routes=routes,
        selected_rows=selected_rows,
    )
    checks = {
        "exact_stopped_a19_lineage_and_artifacts": True,
        "route_inventory_exact_2_edge_2_buffer_5_deep": {
            route: sum(value == route for value in routes.values())
            for route in (
                "edge_candidate",
                "raw_uncertainty_buffer",
                "interior_frozen_offset",
                "raw_unresolved",
            )
        }
        == {
            "edge_candidate": 2,
            "raw_uncertainty_buffer": 2,
            "interior_frozen_offset": 5,
            "raw_unresolved": 0,
        },
        **branch_checks,
        "aggregate_state_rms_ratio_strictly_below_a8": population[
            "aggregate_state_rms_ratio"
        ]
        < 0.9884779121207531,
        "aggregate_increment_defect_rms_ratio_at_most_a8": population[
            "aggregate_increment_defect_rms_ratio"
        ]
        <= 0.9970100807663629,
        "all_five_deep_interiors_strictly_improve": deep_wins == 5,
        "maximum_endpoint_state_ratio_at_most_one": population[
            "maximum_endpoint_state_ratio"
        ]
        <= 1.0,
        "median_endpoint_cumulative_defect_ratio_at_most_one": population[
            "median_endpoint_cumulative_defect_ratio"
        ]
        <= 1.0,
        "all_registered_controls_no_harm": bool(controls)
        and all(parent._control_passed(row) for row in controls),
        "model_calls_zero": True,
        "predictions_and_recurrence_not_recomputed": True,
        "no_new_population_opened": prior.get("new_sealed_population_opened")
        is False,
    }
    qualified = all(checks.values())
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    route_rows = [
        {"case_id": case_id, "route": routes[case_id]}
        for case_id in PROSPECTIVE_STRUCTURAL_CASE_IDS
    ]
    write_csv(output_dir / "position_routes.csv", route_rows)
    write_csv(output_dir / "selected_call_metrics.csv", selected_rows)
    write_csv(output_dir / "rollout_case_metrics.csv", case_rows)
    write_csv(output_dir / "rollout_controls.csv", controls)
    write_csv(
        output_dir / "gate_checks.csv",
        [{"check": key, "passed": value} for key, value in checks.items()],
    )
    source = with_payload_sha256(
        {
            "schema": (
                "pcno_response_filtered_block_buffered_relaxed_tether_"
                "rescore_source_v1"
            ),
            "working_id": BUFFERED_RELAXED_TETHER_RESCORE_WORKING_ID,
            "source_sha256": _source_hashes(),
            "source_status": _git_status_short(SOURCE_PATHS),
            "prior_a19_sha256": sha256_file(args.prior_relaxed_tether_rollout),
            "prior_a19_payload_sha256": prior["payload_sha256"],
            "prior_a19_source_manifest_sha256": prior["source_manifest_sha256"],
            "prior_a19_artifact_sha256": EXPECTED_P6_A19_ARTIFACT_SHA256,
            "route_contract": {
                "descriptor": "call_zero_normalized_wall_distance",
                "edge_candidate_at_most": FROZEN_POSITION_TRUST_THRESHOLD,
                "raw_uncertainty_buffer_at_most": FROZEN_POSITION_BUFFER_THRESHOLD,
                "deeper_interior": "a19_relaxed_tether",
                "unresolved": "raw",
            },
            "model_calls": 0,
            "predictions_recomputed": False,
            "recurrence_reexecuted": False,
        }
    )
    atomic_write_json(output_dir / "source_manifest.json", source)
    files = (
        "position_routes.csv",
        "selected_call_metrics.csv",
        "rollout_case_metrics.csv",
        "rollout_controls.csv",
        "gate_checks.csv",
        "source_manifest.json",
    )
    artifact_hashes = sha256_files(files, root=output_dir)
    payload = with_payload_sha256(
        {
            "schema": BUFFERED_RELAXED_TETHER_RESCORE_SCHEMA,
            "working_id": BUFFERED_RELAXED_TETHER_RESCORE_WORKING_ID,
            "status": (
                "qualified_calibration_rescore" if qualified else "stopped_rescore"
            ),
            "calibration_gate": {
                "status": "qualified" if qualified else "stopped",
                "checks": checks,
                "prospective_claim_authorized": False,
                "one_exact_replay_authorized": qualified,
            },
            "population": population,
            "position_routes": route_rows,
            "failed_controls": [
                row for row in controls if not parent._control_passed(row)
            ],
            "projected_replay_cost": _buffered_relaxed_tether_call_budget(routes),
            "prior_a19_sha256": sha256_file(args.prior_relaxed_tether_rollout),
            "prior_a19_payload_sha256": prior["payload_sha256"],
            "source_manifest_sha256": sha256_file(
                output_dir / "source_manifest.json"
            ),
            "source_manifest_payload_sha256": source["payload_sha256"],
            "artifact_hashes": artifact_hashes,
            "model_calls": 0,
            "predictions_recomputed": False,
            "recurrence_reexecuted": False,
            "truth_used_for_runtime_route": False,
            "thresholds_truth_informed_on_open_e12": True,
            "coefficient_or_relaxation_refit": False,
            "new_population_opened": False,
            "claim_boundary": (
                "Zero-model composition of exact saved A19 and raw-shadow rows "
                "using the prior E12 position buffer. This is same-population "
                "calibration evidence, not prospective, cross-family, transferable-"
                "coefficient, conservation, convergence, off-grid, or Richardson-"
                "extrapolation evidence."
            ),
        }
    )
    atomic_write_json(output_dir / "buffered_relaxed_tether_rescore.json", payload)
    return payload, 0 if qualified else 4


def run_shadow_candidate_diagnostic(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], int]:
    preflight = _verify_shadow_candidate_preflight(args.preflight, args)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    runtime, source, _, _, _, a8_rollout = _build_shadow_candidate_runtime(args)
    try:
        if source != preflight["source_manifest"]:
            raise ValueError("A9 runtime source differs from the frozen preflight")
        prefix_a = _shadow_rollout_arm(
            runtime,
            case_id=PROSPECTIVE_STRUCTURAL_CASE_IDS[0],
            horizon=2,
            native_truth_only=True,
            structural_gate=True,
            diagnostic_shadow_candidates=True,
        )
        prefix_b = _shadow_rollout_arm(
            runtime,
            case_id=PROSPECTIVE_STRUCTURAL_CASE_IDS[0],
            horizon=2,
            native_truth_only=True,
            structural_gate=True,
            diagnostic_shadow_candidates=True,
        )
        prefix_abs = max(
            _maximum_abs(left - right)
            for left, right in zip(
                prefix_a["accepted_states"],
                prefix_b["accepted_states"],
                strict=True,
            )
        )
        prefix_features_exact = (
            prefix_a["diagnostic_blocks"] == prefix_b["diagnostic_blocks"]
        )

        rollouts = []
        for case_id in PROSPECTIVE_STRUCTURAL_CASE_IDS:
            print(f"shadow-candidate diagnostic: {case_id}", flush=True)
            rollouts.append(
                _shadow_rollout_arm(
                    runtime,
                    case_id=case_id,
                    horizon=30,
                    native_truth_only=True,
                    structural_gate=True,
                    diagnostic_shadow_candidates=True,
                )
            )
            if runtime.device.type == "cuda":
                torch.cuda.empty_cache()

        rows = [row for rollout in rollouts for row in rollout["rows"]]
        block_rows = [
            row for rollout in rollouts for row in rollout["diagnostic_blocks"]
        ]
        front_rows = [row for rollout in rollouts for row in rollout["front_audits"]]
        anchor_rows = [row for rollout in rollouts for row in rollout["anchor_rows"]]
        execution_rows = [
            {
                "case_id": rollout["case_id"],
                "policy": rollout["policy"],
                **rollout["execution"],
            }
            for rollout in rollouts
        ]
        closure_rows = [
            {
                "case_id": rollout["case_id"],
                "policy": rollout["policy"],
                **rollout["maxima"],
            }
            for rollout in rollouts
        ]
        reference_rows = [
            {"case_id": rollout["case_id"], **rollout["reference_check"]}
            for rollout in rollouts
        ]
        association_rows, case_association_rows, oof_rows = (
            _shadow_candidate_associations(block_rows)
        )

        case_summary = []
        for case_id in PROSPECTIVE_STRUCTURAL_CASE_IDS:
            selected = [row for row in block_rows if row["case_id"] == case_id]
            raw_state_sse = sum(float(row["raw_block_state_sse"]) for row in selected)
            candidate_state_sse = sum(
                float(row["candidate_block_state_sse"]) for row in selected
            )
            raw_increment_sse = sum(
                float(row["raw_block_increment_sse"]) for row in selected
            )
            candidate_increment_sse = sum(
                float(row["candidate_block_increment_sse"]) for row in selected
            )
            case_summary.append(
                {
                    "case_id": case_id,
                    "block_count": len(selected),
                    "state_sse_skill": 1.0 - candidate_state_sse / raw_state_sse,
                    "increment_sse_skill": (
                        1.0 - candidate_increment_sse / raw_increment_sse
                    ),
                    "state_benefit_block_count": sum(
                        float(row["block_state_sse_skill"]) > 0.0 for row in selected
                    ),
                    "joint_oracle_accept_block_count": sum(
                        bool(row["oracle_joint_accept"]) for row in selected
                    ),
                    "front_branch_change_count": sum(
                        bool(row["front_branch_changed"]) for row in selected
                    ),
                    "initial_normalized_wall_distance": selected[0][
                        "initial_normalized_wall_distance"
                    ],
                }
            )

        raw_state_sse = sum(float(row["raw_block_state_sse"]) for row in block_rows)
        candidate_state_sse = sum(
            float(row["candidate_block_state_sse"]) for row in block_rows
        )
        raw_increment_sse = sum(
            float(row["raw_block_increment_sse"]) for row in block_rows
        )
        candidate_increment_sse = sum(
            float(row["candidate_block_increment_sse"]) for row in block_rows
        )
        population = {
            "block_count": len(block_rows),
            "counterfactual_state_sse_skill": 1.0 - candidate_state_sse / raw_state_sse,
            "counterfactual_increment_sse_skill": 1.0
            - candidate_increment_sse / raw_increment_sse,
            "state_benefit_block_count": sum(
                float(row["block_state_sse_skill"]) > 0.0 for row in block_rows
            ),
            "state_benefit_block_fraction": sum(
                float(row["block_state_sse_skill"]) > 0.0 for row in block_rows
            )
            / len(block_rows),
            "joint_oracle_accept_block_count": sum(
                bool(row["oracle_joint_accept"]) for row in block_rows
            ),
            "joint_oracle_accept_block_fraction": sum(
                bool(row["oracle_joint_accept"]) for row in block_rows
            )
            / len(block_rows),
            "front_branch_change_count": sum(
                bool(row["front_branch_changed"]) for row in block_rows
            ),
        }

        call_checks = _explicit_shadow_call_checks(
            execution_rows,
            closure_rows,
            case_ids=PROSPECTIVE_STRUCTURAL_CASE_IDS,
        )
        block_inventory = {
            (str(row["case_id"]), int(row["block_index"])) for row in block_rows
        }
        expected_block_inventory = {
            (case_id, block_index)
            for case_id in PROSPECTIVE_STRUCTURAL_CASE_IDS
            for block_index in range(15)
        }
        completed = all(
            rollout["execution"]["completed_calls"] == 30
            and rollout["first_invalid_call"] is None
            and rollout["first_nonfinite_call"] is None
            for rollout in rollouts
        )
        anchor_closure = all(
            rollout["maxima"]["candidate_shadow_integral_abs"] <= 1.0e-10
            and rollout["maxima"]["anchor_post_integral_abs"] <= 1.0e-10
            and rollout["maxima"]["anchor_boundary_abs"] <= 1.0e-12
            and rollout["maxima"]["anchor_idempotence_abs"] <= 1.0e-10
            for rollout in rollouts
        )
        response_closure = all(
            rollout["maxima"]["first_correction_integral_abs"] <= 1.0e-10
            and rollout["maxima"]["filtered_response_integral_abs"] <= 1.0e-10
            and rollout["maxima"]["first_boundary_abs"] <= 1.0e-12
            and rollout["maxima"]["filtered_boundary_abs"] <= 1.0e-12
            and rollout["maxima"]["projection_idempotence_abs"] <= 1.0e-10
            and rollout["maxima"]["cap_active_count"] == 0
            for rollout in rollouts
        )
        native_reference_contract = all(
            row["reference_check"].get("retained_resolution") == "250x100"
            and row["reference_check"].get("active_reference_artifact_sha256")
            == row["reference_check"].get("frozen_training_reference_sha256")
            and row["reference_check"].get("restriction_crosscheck_max_abs") == 0.0
            for row in rollouts
        )
        checks = {
            "exact_qualified_a8_lineage": (
                a8_rollout["status"] == "qualified_prospective"
                and a8_rollout["prospective_gate"]["prospective_claim_authorized"]
                is True
            ),
            "already_open_e12_inventory_exact": (
                len(block_rows) == len(expected_block_inventory)
                and block_inventory == expected_block_inventory
            ),
            "e14_remained_sealed": source["population"]["still_sealed_groups"]
            == ["strength_ood_e14"],
            "all_raw_and_counterfactual_states_complete_finite_admissible": completed,
            "candidate_discard_to_raw_recurrence_exact": all(
                rollout["maxima"]["discard_to_raw_recurrence_abs"] <= 1.0e-12
                for rollout in rollouts
            ),
            "all_blocks_have_target_free_feature_statuses": all(
                all(f"{feature}_status" in row for feature in A9_FEATURE_NAMES)
                for row in block_rows
            ),
            "all_frozen_feature_target_associations_reported": (
                {(row["feature"], row["target"]) for row in association_rows}
                == {
                    (feature, target)
                    for feature in A9_FEATURE_NAMES
                    for target in A9_TARGET_NAMES
                }
            ),
            "candidate_integral_anchor_closure": anchor_closure,
            "candidate_response_projection_closure": response_closure,
            "native_truth_reference_contract_exact": native_reference_contract,
            "fine_reference_truth_not_used": source["population"][
                "fine_reference_truth"
            ]
            == "not_required_or_loaded",
            "deterministic_two_output_prefix_and_features_exact": (
                prefix_abs == 0.0 and prefix_features_exact
            ),
            "source_reference_and_artifact_inventory_exact": source
            == preflight["source_manifest"],
            **call_checks,
        }
        valid = all(checks.values())

        write_csv(output_dir / "shadow_candidate_call_metrics.csv", rows)
        write_csv(output_dir / "shadow_candidate_blocks.csv", block_rows)
        write_csv(output_dir / "feature_associations.csv", association_rows)
        write_csv(output_dir / "case_feature_associations.csv", case_association_rows)
        write_csv(output_dir / "oof_feature_predictions.csv", oof_rows)
        write_csv(output_dir / "case_summary.csv", case_summary)
        write_csv(output_dir / "front_branch_audits.csv", front_rows)
        write_csv(output_dir / "anchor_audits.csv", anchor_rows)
        write_csv(output_dir / "execution.csv", execution_rows)
        write_csv(output_dir / "closure.csv", closure_rows)
        write_csv(output_dir / "reference_checks.csv", reference_rows)
        atomic_write_json(output_dir / "source_manifest.json", source)
        files = (
            "shadow_candidate_call_metrics.csv",
            "shadow_candidate_blocks.csv",
            "feature_associations.csv",
            "case_feature_associations.csv",
            "oof_feature_predictions.csv",
            "case_summary.csv",
            "front_branch_audits.csv",
            "anchor_audits.csv",
            "execution.csv",
            "closure.csv",
            "reference_checks.csv",
            "source_manifest.json",
        )
        artifact_hashes = sha256_files(files, root=output_dir)
        cost = {
            "logical_model_calls": sum(
                row["execution"]["logical_model_calls"] for row in rollouts
            ),
            "native_logical_calls": sum(
                row["execution"]["native_logical_calls"] for row in rollouts
            ),
            "fine_logical_calls": sum(
                row["execution"]["fine_logical_calls"] for row in rollouts
            ),
            "actual_forward_passes": sum(
                row["execution"]["actual_forward_passes"] for row in rollouts
            ),
            "total_forward_seconds": sum(
                row["execution"]["total_forward_seconds"] for row in rollouts
            ),
            "wall_seconds": sum(row["execution"]["wall_seconds"] for row in rollouts),
            "maximum_peak_gpu_memory_bytes": max(
                row["execution"]["maximum_peak_gpu_memory_bytes"] for row in rollouts
            ),
            "raw_native_logical_call_comparator": 30
            * len(PROSPECTIVE_STRUCTURAL_CASE_IDS),
            "deterministic_prefix_logical_model_calls": (
                prefix_a["execution"]["logical_model_calls"]
                + prefix_b["execution"]["logical_model_calls"]
            ),
            "logical_model_calls_including_deterministic_prefix": (
                sum(row["execution"]["logical_model_calls"] for row in rollouts)
                + prefix_a["execution"]["logical_model_calls"]
                + prefix_b["execution"]["logical_model_calls"]
            ),
        }
        payload = with_payload_sha256(
            {
                "schema": SHADOW_CANDIDATE_SCHEMA,
                "working_id": SHADOW_CANDIDATE_WORKING_ID,
                "status": "completed_diagnostic" if valid else "invalid_diagnostic",
                "contract_checks": checks,
                "population": population,
                "case_summary": case_summary,
                "feature_associations": association_rows,
                "cost": cost,
                "deterministic_prefix_max_abs": prefix_abs,
                "preflight_sha256": sha256_file(args.preflight),
                "preflight_payload_sha256": preflight["payload_sha256"],
                "qualified_a8_rollout_sha256": sha256_file(args.a8_rollout),
                "qualified_a8_rollout_payload_sha256": a8_rollout["payload_sha256"],
                "source_manifest_sha256": sha256_file(
                    output_dir / "source_manifest.json"
                ),
                "source_manifest_payload_sha256": source["payload_sha256"],
                "artifact_hashes": artifact_hashes,
                "recurrence_executed": True,
                "candidate_accepted_into_recurrence": False,
                "feature_or_threshold_selection_executed": False,
                "coefficient_refit": False,
                "new_sealed_population_opened": False,
                "still_sealed_groups": ["strength_ood_e14"],
                "claim_boundary": (
                    "Retrospective synchronized block-level mechanism diagnostic on "
                    "the already-open e12 dynamic-FV group. It selects no feature, "
                    "threshold, gain, or deployment policy and supports no e14, "
                    "cross-family, conservation, convergence, or off-grid claim."
                ),
            }
        )
        atomic_write_json(output_dir / "shadow_candidate_diagnostic.json", payload)
        return payload, 0 if valid else 4
    finally:
        collection._close_runtime(runtime)


def run_shadow_candidate_rescore(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], int]:
    prior = _verify_a9_diagnostic(args.prior_shadow_candidate_diagnostic)
    root = args.prior_shadow_candidate_diagnostic.parent
    execution_rows = _read_csv(root / "execution.csv")
    closure_rows = _read_csv(root / "closure.csv")
    checks = _rescore_shadow_candidate_call_checks(
        prior["contract_checks"], execution_rows, closure_rows
    )
    qualified = all(checks.values())
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    source = with_payload_sha256(
        {
            "schema": "pcno_response_filtered_block_shadow_candidate_rescore_source_v1",
            "working_id": SHADOW_CANDIDATE_RESCORE_WORKING_ID,
            "source_sha256": _source_hashes(),
            "source_status": _git_status_short(SOURCE_PATHS),
            "prior_diagnostic_sha256": sha256_file(
                args.prior_shadow_candidate_diagnostic
            ),
            "prior_diagnostic_payload_sha256": prior["payload_sha256"],
            "prior_artifact_hashes": prior["artifact_hashes"],
            "repair": {
                "scope": "call_inventory_rescore_only",
                "original_case_ids_default": list(EVALUATION_CASE_IDS),
                "correct_case_ids": list(PROSPECTIVE_STRUCTURAL_CASE_IDS),
                "model_calls": 0,
                "predictions_recomputed": False,
                "recurrence_reexecuted": False,
                "features_or_labels_recomputed": False,
                "associations_recomputed": False,
                "threshold_or_coefficient_changed": False,
            },
            "still_sealed_groups": ["strength_ood_e14"],
        }
    )
    check_rows = [{"check": key, "passed": value} for key, value in checks.items()]
    write_csv(output_dir / "contract_checks.csv", check_rows)
    atomic_write_json(output_dir / "source_manifest.json", source)
    artifact_hashes = sha256_files(
        ("contract_checks.csv", "source_manifest.json"), root=output_dir
    )
    amended_cost = {
        **prior["cost"],
        "deterministic_prefix_logical_model_calls": 12,
        "logical_model_calls_including_deterministic_prefix": 822,
    }
    payload = with_payload_sha256(
        {
            "schema": SHADOW_CANDIDATE_RESCORE_SCHEMA,
            "working_id": SHADOW_CANDIDATE_RESCORE_WORKING_ID,
            "status": "completed_diagnostic_rescore"
            if qualified
            else "invalid_rescore",
            "contract_checks": checks,
            "original_failed_checks": sorted(A9_MISWIRED_CALL_CHECKS),
            "corrected_checks": sorted(A9_MISWIRED_CALL_CHECKS),
            "population": prior["population"],
            "case_summary": prior["case_summary"],
            "feature_associations": prior["feature_associations"],
            "cost": amended_cost,
            "prior_diagnostic_sha256": sha256_file(
                args.prior_shadow_candidate_diagnostic
            ),
            "prior_diagnostic_payload_sha256": prior["payload_sha256"],
            "prior_artifact_hashes": prior["artifact_hashes"],
            "source_manifest_sha256": sha256_file(output_dir / "source_manifest.json"),
            "source_manifest_payload_sha256": source["payload_sha256"],
            "artifact_hashes": artifact_hashes,
            "model_calls": 0,
            "predictions_recomputed": False,
            "recurrence_reexecuted": False,
            "features_or_labels_recomputed": False,
            "associations_recomputed": False,
            "threshold_or_coefficient_changed": False,
            "feature_or_threshold_selection_executed": False,
            "candidate_accepted_into_recurrence": False,
            "new_sealed_population_opened": False,
            "still_sealed_groups": ["strength_ood_e14"],
            "claim_boundary": prior["claim_boundary"],
        }
    )
    atomic_write_json(output_dir / "shadow_candidate_rescore.json", payload)
    return payload, 0 if qualified else 4


def run_shadow_rescore(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    prior, _, case_rows, controls, population = _verify_a5_rollout(
        args.prior_shadow_rollout
    )
    root = args.prior_shadow_rollout.parent
    execution_rows = _read_csv(root / "rollout_execution.csv")
    closure_rows = _read_csv(root / "rollout_closure.csv")
    call_checks = _explicit_shadow_call_checks(execution_rows, closure_rows)
    checks = dict(prior["recurrent_gate"]["checks"])
    removed = checks.pop("six_call_shadow_and_candidate_contract", None)
    if removed is not False:
        raise ValueError("P6 A5 compound call check is not the registered false value")
    checks.update(call_checks)
    qualified = all(checks.values())
    observed = {
        "maximum_call_order_errors": max(
            int(float(row["call_order_errors"])) for row in closure_rows
        ),
        "maximum_shadow_recurrence_abs": max(
            float(row["shadow_recurrence_abs"]) for row in closure_rows
        ),
        "maximum_accepted_post_fp32_abs": max(
            float(row["accepted_recurrence_abs"]) for row in closure_rows
        ),
        "maximum_candidate_common_native_input_abs": max(
            float(row["candidate_common_native_input_abs"]) for row in closure_rows
        ),
        "maximum_candidate_lookahead_input_abs": max(
            float(row["candidate_lookahead_input_abs"]) for row in closure_rows
        ),
    }
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    source = with_payload_sha256(
        {
            "schema": "pcno_response_filtered_block_shadow_rescore_source_v1",
            "working_id": SHADOW_RESCORE_WORKING_ID,
            "source_sha256": _source_hashes(),
            "source_status": _git_status_short(SOURCE_PATHS),
            "prior_shadow_rollout_sha256": sha256_file(args.prior_shadow_rollout),
            "prior_shadow_rollout_payload_sha256": prior["payload_sha256"],
            "prior_artifact_sha256": EXPECTED_P6_A5_ARTIFACT_SHA256,
            "semantic_correction": (
                "replace the ambiguous compound call gate with exact per-case "
                "call inventory, order, shadow recurrence, prepared common-source, "
                "and lookahead checks plus the registered 1e-6 post-FP32 transfer floor"
            ),
            "model_calls": 0,
        }
    )
    write_csv(
        output_dir / "explicit_call_checks.csv",
        [{"check": key, "passed": value} for key, value in sorted(call_checks.items())],
    )
    write_csv(output_dir / "rollout_case_metrics.csv", case_rows)
    write_csv(output_dir / "rollout_controls.csv", controls)
    atomic_write_json(output_dir / "source_manifest.json", source)
    artifact_hashes = sha256_files(
        (
            "explicit_call_checks.csv",
            "rollout_case_metrics.csv",
            "rollout_controls.csv",
            "source_manifest.json",
        ),
        root=output_dir,
    )
    payload = with_payload_sha256(
        {
            "schema": SHADOW_RESCORE_SCHEMA,
            "working_id": SHADOW_RESCORE_WORKING_ID,
            "status": "qualified" if qualified else "stopped",
            "population_status": "adaptive_open_validation_shadow_pilot",
            "recurrent_gate": {
                "status": "qualified" if qualified else "stopped",
                "claim_authorized": qualified,
                "checks": checks,
            },
            "population": population,
            "failed_controls": [
                row for row in controls if not parent._control_passed(row)
            ],
            "explicit_call_checks": call_checks,
            "call_contract_observed": observed,
            "cost": prior["cost"],
            "maximum_anchor_correction_rms": prior["maximum_anchor_correction_rms"],
            "prior_shadow_rollout_sha256": sha256_file(args.prior_shadow_rollout),
            "prior_shadow_rollout_payload_sha256": prior["payload_sha256"],
            "prior_artifact_sha256": EXPECTED_P6_A5_ARTIFACT_SHA256,
            "source_manifest_sha256": sha256_file(output_dir / "source_manifest.json"),
            "source_manifest_payload_sha256": source["payload_sha256"],
            "artifact_hashes": artifact_hashes,
            "model_calls": 0,
            "predictions_recomputed": False,
            "thresholds_changed": False,
            "recurrence_reexecuted": False,
            "sealed_population_opened": False,
            "claim_boundary": (
                "Scorer-only replay of immutable adaptive-open H30 dynamic-FV "
                "raw-shadow artifacts. No new prediction, recurrence, independent "
                "validation, cross-family, physical-conservation, or sealed claim."
            ),
        }
    )
    atomic_write_json(output_dir / "shadow_rescore.json", payload)
    return payload, 0 if qualified else 4


def synthetic_summary() -> dict[str, Any]:
    from utility.time_dependent_no.pcno_cross_resolution_correction import (
        ResolutionContract,
        prolong_nested_state,
    )
    from utility.time_dependent_no.pcno_cross_resolution_teacher_forced import (
        build_fixed_cosine_projector,
    )

    nx, ny = 8, 4
    x = (np.arange(nx, dtype=np.float64) + 0.5) * (2.0 / nx)
    y = (np.arange(ny, dtype=np.float64) + 0.5) * (1.0 / ny)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    nodes = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    volumes = np.full(nx * ny, 2.0 / (nx * ny), dtype=np.float64)
    node_type = np.zeros(nx * ny, dtype=np.int64)
    node_type[:nx] = 1
    projector = build_fixed_cosine_projector(
        nodes,
        volumes,
        node_type,
        rank=8,
        domain_bounds=(0.0, 2.0, 0.0, 1.0),
    )
    state = np.ones((nx * ny, 4), dtype=np.float64)
    pattern = np.zeros_like(state)
    pattern[:, 0] = projector.basis[:, 1]
    fine_pattern = prolong_nested_state(
        pattern,
        coarse_resolution=(nx, ny),
        fine_resolution=(2 * nx, 2 * ny),
    )
    calls = []

    def predictor(resolution, value):
        calls.append((resolution, np.array(value, copy=True)))
        if resolution == (2 * nx, 2 * ny):
            return value + 0.1 + 0.04 * fine_pattern
        return 1.1 * value + 0.1

    block = synchronized_response_filtered_block(
        state,
        contract=ResolutionContract(
            coarse=(nx // 2, ny // 2),
            native=(nx, ny),
            fine=(2 * nx, 2 * ny),
        ),
        projector=projector,
        predictor=predictor,
        volumes=volumes[:, None],
        residual_scale=np.ones(4),
        state_scale=np.ones(4),
    )
    checks = {
        "exact_four_call_order": [resolution for resolution, _ in calls]
        == [(nx, ny), (2 * nx, 2 * ny), (nx, ny), (nx, ny)],
        "one_native_trajectory_block": block.corrected_first_state.shape
        == block.filtered_second_state.shape
        == state.shape,
        "first_integral_neutral": block.projection_audit.maximum_first_correction_integral_abs
        <= 1.0e-12,
        "filtered_integral_neutral": block.projection_audit.maximum_filtered_response_integral_abs
        <= 1.0e-12,
        "boundary_zero": block.projection_audit.maximum_first_boundary_abs <= 1.0e-12
        and block.projection_audit.maximum_filtered_boundary_abs <= 1.0e-12,
        "projection_idempotent": block.projection_audit.maximum_projection_idempotence_abs
        <= 1.0e-12,
        "evaluation_closed": True,
        "recurrence_closed": True,
    }
    return with_payload_sha256(
        {
            "schema": "pcno_response_filtered_block_synthetic_v1",
            "working_id": WORKING_ID,
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
        }
    )


def _add_external_arguments(parser: argparse.ArgumentParser) -> None:
    parent._add_external_arguments(parser)
    parser.add_argument("--p5-calibration", type=Path, required=True)


def _add_evaluation_arguments(parser: argparse.ArgumentParser) -> None:
    _add_external_arguments(parser)
    parser.add_argument("--rescore", type=Path, required=True)


def _add_rollout_arguments(parser: argparse.ArgumentParser) -> None:
    _add_evaluation_arguments(parser)
    parser.add_argument("--evaluation", type=Path, required=True)


def _add_shadow_arguments(parser: argparse.ArgumentParser) -> None:
    _add_rollout_arguments(parser)
    parser.add_argument("--a4-rollout", type=Path, required=True)


def _add_strength_ood_arguments(parser: argparse.ArgumentParser) -> None:
    _add_shadow_arguments(parser)
    parser.add_argument("--a5-rescore", type=Path, required=True)


def _add_structural_gate_arguments(parser: argparse.ArgumentParser) -> None:
    _add_strength_ood_arguments(parser)
    parser.add_argument("--a6-rollout", type=Path, required=True)


def _add_prospective_structural_arguments(parser: argparse.ArgumentParser) -> None:
    _add_structural_gate_arguments(parser)
    parser.add_argument("--a7-rollout", type=Path, required=True)


def _add_shadow_candidate_arguments(parser: argparse.ArgumentParser) -> None:
    _add_prospective_structural_arguments(parser)
    parser.add_argument("--a8-rollout", type=Path, required=True)


def _add_warm_start_arguments(parser: argparse.ArgumentParser) -> None:
    _add_shadow_candidate_arguments(parser)
    parser.add_argument("--a9-rescore", type=Path, required=True)


def _add_projected_coast_arguments(parser: argparse.ArgumentParser) -> None:
    _add_warm_start_arguments(parser)
    parser.add_argument("--a10-rollout", type=Path, required=True)


def _add_slew_limited_tether_arguments(parser: argparse.ArgumentParser) -> None:
    _add_projected_coast_arguments(parser)
    parser.add_argument("--a11-rollout", type=Path, required=True)
    parser.add_argument("--a17-rescore", type=Path, required=True)


def _add_relaxed_tether_arguments(parser: argparse.ArgumentParser) -> None:
    _add_slew_limited_tether_arguments(parser)
    parser.add_argument("--a18-rollout", type=Path, required=True)


def _add_buffered_relaxed_tether_replay_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    _add_relaxed_tether_arguments(parser)
    parser.add_argument("--a22-rescore", type=Path, required=True)


def _add_buffered_relaxed_tether_e14_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    _add_buffered_relaxed_tether_replay_arguments(parser)
    parser.add_argument("--a22-r1-rollout", type=Path, required=True)
    parser.add_argument("--a14-rollout", type=Path, required=True)
    parser.add_argument("--shard-native-reference-audit", type=Path, required=True)


def _add_frozen_offset_arguments(parser: argparse.ArgumentParser) -> None:
    _add_projected_coast_arguments(parser)
    parser.add_argument("--a11-rollout", type=Path, required=True)


def _add_buffered_offset_arguments(parser: argparse.ArgumentParser) -> None:
    _add_frozen_offset_arguments(parser)
    parser.add_argument("--a12-rollout", type=Path, required=True)
    parser.add_argument("--a12-buffer-rescore", type=Path, required=True)
    parser.add_argument("--shard-native-reference-audit", type=Path)


def _add_prospective_buffered_offset_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    _add_buffered_offset_arguments(parser)
    parser.add_argument("--a13-rollout", type=Path, required=True)
    parser.add_argument("--shard-native-a13-replay", type=Path)


def _add_persistence_gain_arguments(parser: argparse.ArgumentParser) -> None:
    _add_buffered_offset_arguments(parser)
    parser.add_argument("--a13-rollout", type=Path, required=True)
    parser.add_argument("--a14-rollout", type=Path, required=True)


def _add_terminal_ramp_arguments(parser: argparse.ArgumentParser) -> None:
    _add_persistence_gain_arguments(parser)
    parser.add_argument("--a15-rollout", type=Path, required=True)


def _add_fixed_late_ramp_arguments(parser: argparse.ArgumentParser) -> None:
    _add_buffered_offset_arguments(parser)
    parser.add_argument("--a13-rollout", type=Path, required=True)
    parser.add_argument("--a16-rollout", type=Path, required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("synthetic")
    shard_native_audit = commands.add_parser("shard-native-reference-audit")
    shard_native_audit.add_argument("--family-root", type=Path, required=True)
    shard_native_audit.add_argument("--data-dir", type=Path, required=True)
    shard_native_audit.add_argument("--output", type=Path, required=True)
    preflight = commands.add_parser("preflight")
    _add_external_arguments(preflight)
    preflight.add_argument("--output", type=Path, required=True)
    calibration = commands.add_parser("calibrate")
    _add_external_arguments(calibration)
    parent._add_runtime_arguments(calibration)
    calibration.add_argument("--preflight", type=Path, required=True)
    calibration.add_argument("--output-dir", type=Path, required=True)
    rescore = commands.add_parser("rescore")
    rescore.add_argument("--prior-calibration", type=Path, required=True)
    rescore.add_argument("--output-dir", type=Path, required=True)
    evaluation_preflight = commands.add_parser("evaluation-preflight")
    _add_evaluation_arguments(evaluation_preflight)
    evaluation_preflight.add_argument("--output", type=Path, required=True)
    evaluation = commands.add_parser("evaluate")
    _add_evaluation_arguments(evaluation)
    parent._add_runtime_arguments(evaluation)
    evaluation.add_argument("--preflight", type=Path, required=True)
    evaluation.add_argument("--output-dir", type=Path, required=True)
    rollout_preflight = commands.add_parser("rollout-preflight")
    _add_rollout_arguments(rollout_preflight)
    rollout_preflight.add_argument("--output", type=Path, required=True)
    rollout = commands.add_parser("rollout")
    _add_rollout_arguments(rollout)
    parent._add_runtime_arguments(rollout)
    rollout.add_argument("--preflight", type=Path, required=True)
    rollout.add_argument("--output-dir", type=Path, required=True)
    shadow_preflight = commands.add_parser("shadow-preflight")
    _add_shadow_arguments(shadow_preflight)
    shadow_preflight.add_argument("--output", type=Path, required=True)
    shadow = commands.add_parser("shadow-rollout")
    _add_shadow_arguments(shadow)
    parent._add_runtime_arguments(shadow)
    shadow.add_argument("--preflight", type=Path, required=True)
    shadow.add_argument("--output-dir", type=Path, required=True)
    strength_ood_preflight = commands.add_parser("strength-ood-preflight")
    _add_strength_ood_arguments(strength_ood_preflight)
    strength_ood_preflight.add_argument("--output", type=Path, required=True)
    strength_ood = commands.add_parser("strength-ood-rollout")
    _add_strength_ood_arguments(strength_ood)
    parent._add_runtime_arguments(strength_ood)
    strength_ood.add_argument("--preflight", type=Path, required=True)
    strength_ood.add_argument("--output-dir", type=Path, required=True)
    structural_preflight = commands.add_parser("structural-gate-preflight")
    _add_structural_gate_arguments(structural_preflight)
    structural_preflight.add_argument("--output", type=Path, required=True)
    structural = commands.add_parser("structural-gate-rollout")
    _add_structural_gate_arguments(structural)
    parent._add_runtime_arguments(structural)
    structural.add_argument("--preflight", type=Path, required=True)
    structural.add_argument("--output-dir", type=Path, required=True)
    prospective_preflight = commands.add_parser("prospective-structural-preflight")
    _add_prospective_structural_arguments(prospective_preflight)
    prospective_preflight.add_argument("--output", type=Path, required=True)
    prospective = commands.add_parser("prospective-structural-rollout")
    _add_prospective_structural_arguments(prospective)
    parent._add_runtime_arguments(prospective)
    prospective.add_argument("--preflight", type=Path, required=True)
    prospective.add_argument("--output-dir", type=Path, required=True)
    shadow_candidate_preflight = commands.add_parser("shadow-candidate-preflight")
    _add_shadow_candidate_arguments(shadow_candidate_preflight)
    shadow_candidate_preflight.add_argument("--output", type=Path, required=True)
    shadow_candidate = commands.add_parser("shadow-candidate-diagnostic")
    _add_shadow_candidate_arguments(shadow_candidate)
    parent._add_runtime_arguments(shadow_candidate)
    shadow_candidate.add_argument("--preflight", type=Path, required=True)
    shadow_candidate.add_argument("--output-dir", type=Path, required=True)
    shadow_candidate_rescore = commands.add_parser("shadow-candidate-rescore")
    shadow_candidate_rescore.add_argument(
        "--prior-shadow-candidate-diagnostic", type=Path, required=True
    )
    shadow_candidate_rescore.add_argument("--output-dir", type=Path, required=True)
    warm_start_preflight = commands.add_parser("warm-start-preflight")
    _add_warm_start_arguments(warm_start_preflight)
    warm_start_preflight.add_argument("--output", type=Path, required=True)
    warm_start = commands.add_parser("warm-start-rollout")
    _add_warm_start_arguments(warm_start)
    parent._add_runtime_arguments(warm_start)
    warm_start.add_argument("--preflight", type=Path, required=True)
    warm_start.add_argument("--output-dir", type=Path, required=True)
    projected_coast_preflight = commands.add_parser("projected-coast-preflight")
    _add_projected_coast_arguments(projected_coast_preflight)
    projected_coast_preflight.add_argument("--output", type=Path, required=True)
    projected_coast = commands.add_parser("projected-coast-rollout")
    _add_projected_coast_arguments(projected_coast)
    parent._add_runtime_arguments(projected_coast)
    projected_coast.add_argument("--preflight", type=Path, required=True)
    projected_coast.add_argument("--output-dir", type=Path, required=True)
    slew_preflight = commands.add_parser("slew-limited-tether-preflight")
    _add_slew_limited_tether_arguments(slew_preflight)
    slew_preflight.add_argument("--output", type=Path, required=True)
    slew = commands.add_parser("slew-limited-tether-rollout")
    _add_slew_limited_tether_arguments(slew)
    parent._add_runtime_arguments(slew)
    slew.add_argument("--preflight", type=Path, required=True)
    slew.add_argument("--output-dir", type=Path, required=True)
    relaxed_preflight = commands.add_parser("relaxed-tether-preflight")
    _add_relaxed_tether_arguments(relaxed_preflight)
    relaxed_preflight.add_argument("--output", type=Path, required=True)
    relaxed = commands.add_parser("relaxed-tether-rollout")
    _add_relaxed_tether_arguments(relaxed)
    parent._add_runtime_arguments(relaxed)
    relaxed.add_argument("--preflight", type=Path, required=True)
    relaxed.add_argument("--output-dir", type=Path, required=True)
    buffered_relaxed_preflight = commands.add_parser(
        "buffered-relaxed-tether-replay-preflight"
    )
    _add_buffered_relaxed_tether_replay_arguments(buffered_relaxed_preflight)
    buffered_relaxed_preflight.add_argument("--output", type=Path, required=True)
    buffered_relaxed = commands.add_parser("buffered-relaxed-tether-replay")
    _add_buffered_relaxed_tether_replay_arguments(buffered_relaxed)
    parent._add_runtime_arguments(buffered_relaxed)
    buffered_relaxed.add_argument("--preflight", type=Path, required=True)
    buffered_relaxed.add_argument("--output-dir", type=Path, required=True)
    buffered_relaxed_e14_preflight = commands.add_parser(
        "buffered-relaxed-tether-e14-preflight"
    )
    _add_buffered_relaxed_tether_e14_arguments(buffered_relaxed_e14_preflight)
    buffered_relaxed_e14_preflight.add_argument("--output", type=Path, required=True)
    buffered_relaxed_e14 = commands.add_parser(
        "buffered-relaxed-tether-e14-rollout"
    )
    _add_buffered_relaxed_tether_e14_arguments(buffered_relaxed_e14)
    parent._add_runtime_arguments(buffered_relaxed_e14)
    buffered_relaxed_e14.add_argument("--preflight", type=Path, required=True)
    buffered_relaxed_e14.add_argument("--output-dir", type=Path, required=True)
    frozen_offset_preflight = commands.add_parser("frozen-offset-preflight")
    _add_frozen_offset_arguments(frozen_offset_preflight)
    frozen_offset_preflight.add_argument("--output", type=Path, required=True)
    frozen_offset = commands.add_parser("frozen-offset-rollout")
    _add_frozen_offset_arguments(frozen_offset)
    parent._add_runtime_arguments(frozen_offset)
    frozen_offset.add_argument("--preflight", type=Path, required=True)
    frozen_offset.add_argument("--output-dir", type=Path, required=True)
    buffered_offset_preflight = commands.add_parser("buffered-offset-preflight")
    _add_buffered_offset_arguments(buffered_offset_preflight)
    buffered_offset_preflight.add_argument("--output", type=Path, required=True)
    buffered_offset = commands.add_parser("buffered-offset-rollout")
    _add_buffered_offset_arguments(buffered_offset)
    parent._add_runtime_arguments(buffered_offset)
    buffered_offset.add_argument("--preflight", type=Path, required=True)
    buffered_offset.add_argument("--output-dir", type=Path, required=True)
    prospective_buffered_offset_preflight = commands.add_parser(
        "prospective-buffered-offset-preflight"
    )
    _add_prospective_buffered_offset_arguments(prospective_buffered_offset_preflight)
    prospective_buffered_offset_preflight.add_argument(
        "--output", type=Path, required=True
    )
    prospective_buffered_offset = commands.add_parser(
        "prospective-buffered-offset-rollout"
    )
    _add_prospective_buffered_offset_arguments(prospective_buffered_offset)
    parent._add_runtime_arguments(prospective_buffered_offset)
    prospective_buffered_offset.add_argument("--preflight", type=Path, required=True)
    prospective_buffered_offset.add_argument("--output-dir", type=Path, required=True)
    persistence_gain_preflight = commands.add_parser("persistence-gain-preflight")
    _add_persistence_gain_arguments(persistence_gain_preflight)
    persistence_gain_preflight.add_argument("--output", type=Path, required=True)
    persistence_gain = commands.add_parser("persistence-gain-rollout")
    _add_persistence_gain_arguments(persistence_gain)
    parent._add_runtime_arguments(persistence_gain)
    persistence_gain.add_argument("--preflight", type=Path, required=True)
    persistence_gain.add_argument("--output-dir", type=Path, required=True)
    terminal_ramp_preflight = commands.add_parser("terminal-ramp-preflight")
    _add_terminal_ramp_arguments(terminal_ramp_preflight)
    terminal_ramp_preflight.add_argument("--output", type=Path, required=True)
    terminal_ramp = commands.add_parser("terminal-ramp-rollout")
    _add_terminal_ramp_arguments(terminal_ramp)
    parent._add_runtime_arguments(terminal_ramp)
    terminal_ramp.add_argument("--preflight", type=Path, required=True)
    terminal_ramp.add_argument("--output-dir", type=Path, required=True)
    fixed_late_ramp_preflight = commands.add_parser("fixed-late-ramp-preflight")
    _add_fixed_late_ramp_arguments(fixed_late_ramp_preflight)
    fixed_late_ramp_preflight.add_argument("--output", type=Path, required=True)
    fixed_late_ramp = commands.add_parser("fixed-late-ramp-rollout")
    _add_fixed_late_ramp_arguments(fixed_late_ramp)
    parent._add_runtime_arguments(fixed_late_ramp)
    fixed_late_ramp.add_argument("--preflight", type=Path, required=True)
    fixed_late_ramp.add_argument("--output-dir", type=Path, required=True)
    fixed_late_ramp_rescore = commands.add_parser("fixed-late-ramp-rescore")
    fixed_late_ramp_rescore.add_argument(
        "--prior-fixed-late-ramp-rollout", type=Path, required=True
    )
    fixed_late_ramp_rescore.add_argument("--output-dir", type=Path, required=True)
    shadow_rescore = commands.add_parser("shadow-rescore")
    shadow_rescore.add_argument("--prior-shadow-rollout", type=Path, required=True)
    shadow_rescore.add_argument("--output-dir", type=Path, required=True)
    buffered_offset_rescore = commands.add_parser("buffered-offset-rescore")
    buffered_offset_rescore.add_argument(
        "--prior-frozen-offset-rollout", type=Path, required=True
    )
    buffered_offset_rescore.add_argument("--output-dir", type=Path, required=True)
    buffered_relaxed_rescore = commands.add_parser(
        "buffered-relaxed-tether-rescore"
    )
    buffered_relaxed_rescore.add_argument(
        "--prior-relaxed-tether-rollout", type=Path, required=True
    )
    buffered_relaxed_rescore.add_argument(
        "--output-dir", type=Path, required=True
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "synthetic":
        payload = synthetic_summary()
        exit_code = 0 if payload["status"] == "passed" else 2
    elif args.command == "shard-native-reference-audit":
        payload = run_shard_native_reference_audit(args)
        exit_code = 0 if payload["status"] == "passed" else 4
    elif args.command == "preflight":
        payload = run_preflight(args)
        exit_code = 0
    elif args.command == "evaluation-preflight":
        payload = run_evaluation_preflight(args)
        exit_code = 0
    elif args.command == "rollout-preflight":
        payload = run_rollout_preflight(args)
        exit_code = 0
    elif args.command == "shadow-preflight":
        payload = run_shadow_preflight(args)
        exit_code = 0
    elif args.command == "strength-ood-preflight":
        payload = run_strength_ood_preflight(args)
        exit_code = 0
    elif args.command == "structural-gate-preflight":
        payload = run_structural_gate_preflight(args)
        exit_code = 0
    elif args.command == "prospective-structural-preflight":
        payload = run_prospective_structural_preflight(args)
        exit_code = 0
    elif args.command in {"calibrate", "evaluate"}:
        if args.repeat_forward < 1:
            raise ValueError("repeat-forward must be positive")
        payload, exit_code = run_teacher_population(args)
    elif args.command == "rollout":
        payload, exit_code = run_rollout(args)
    elif args.command in {"shadow-rollout", "strength-ood-rollout"}:
        payload, exit_code = run_shadow_rollout(args)
    elif args.command == "structural-gate-rollout":
        payload, exit_code = run_structural_gate_rollout(args)
    elif args.command == "prospective-structural-rollout":
        payload, exit_code = run_prospective_structural_rollout(args)
    elif args.command == "shadow-candidate-preflight":
        payload = run_shadow_candidate_preflight(args)
        exit_code = 0
    elif args.command == "shadow-candidate-diagnostic":
        payload, exit_code = run_shadow_candidate_diagnostic(args)
    elif args.command == "shadow-candidate-rescore":
        payload, exit_code = run_shadow_candidate_rescore(args)
    elif args.command == "warm-start-preflight":
        payload = run_warm_start_preflight(args)
        exit_code = 0
    elif args.command == "warm-start-rollout":
        payload, exit_code = run_warm_start_rollout(args)
    elif args.command == "projected-coast-preflight":
        payload = run_projected_coast_preflight(args)
        exit_code = 0
    elif args.command == "projected-coast-rollout":
        payload, exit_code = run_projected_coast_rollout(args)
    elif args.command == "slew-limited-tether-preflight":
        payload = run_slew_limited_tether_preflight(args)
        exit_code = 0
    elif args.command == "slew-limited-tether-rollout":
        payload, exit_code = run_slew_limited_tether_rollout(args)
    elif args.command == "relaxed-tether-preflight":
        payload = run_relaxed_tether_preflight(args)
        exit_code = 0
    elif args.command == "relaxed-tether-rollout":
        payload, exit_code = run_relaxed_tether_rollout(args)
    elif args.command == "buffered-relaxed-tether-replay-preflight":
        payload = run_buffered_relaxed_tether_replay_preflight(args)
        exit_code = 0
    elif args.command == "buffered-relaxed-tether-replay":
        payload, exit_code = run_buffered_relaxed_tether_replay(args)
    elif args.command == "buffered-relaxed-tether-e14-preflight":
        payload = run_buffered_relaxed_tether_e14_preflight(args)
        exit_code = 0
    elif args.command == "buffered-relaxed-tether-e14-rollout":
        payload, exit_code = run_buffered_relaxed_tether_e14_rollout(args)
    elif args.command == "frozen-offset-preflight":
        payload = run_frozen_offset_preflight(args)
        exit_code = 0
    elif args.command == "frozen-offset-rollout":
        payload, exit_code = run_frozen_offset_rollout(args)
    elif args.command == "buffered-offset-preflight":
        payload = run_buffered_offset_preflight(args)
        exit_code = 0
    elif args.command == "buffered-offset-rollout":
        payload, exit_code = run_buffered_offset_rollout(args)
    elif args.command == "prospective-buffered-offset-preflight":
        payload = run_prospective_buffered_offset_preflight(args)
        exit_code = 0
    elif args.command == "prospective-buffered-offset-rollout":
        payload, exit_code = run_prospective_buffered_offset_rollout(args)
    elif args.command == "persistence-gain-preflight":
        payload = run_persistence_gain_preflight(args)
        exit_code = 0
    elif args.command == "persistence-gain-rollout":
        payload, exit_code = run_persistence_gain_rollout(args)
    elif args.command == "terminal-ramp-preflight":
        payload = run_terminal_ramp_preflight(args)
        exit_code = 0
    elif args.command == "terminal-ramp-rollout":
        payload, exit_code = run_terminal_ramp_rollout(args)
    elif args.command == "fixed-late-ramp-preflight":
        payload = run_fixed_late_ramp_preflight(args)
        exit_code = 0
    elif args.command == "fixed-late-ramp-rollout":
        payload, exit_code = run_fixed_late_ramp_rollout(args)
    elif args.command == "fixed-late-ramp-rescore":
        payload, exit_code = run_fixed_late_ramp_rescore(args)
    elif args.command == "buffered-offset-rescore":
        payload, exit_code = run_buffered_offset_rescore(args)
    elif args.command == "buffered-relaxed-tether-rescore":
        payload, exit_code = run_buffered_relaxed_tether_rescore(args)
    elif args.command == "shadow-rescore":
        payload, exit_code = run_shadow_rescore(args)
    elif args.command == "rescore":
        payload, exit_code = run_rescore(args)
    else:  # pragma: no cover
        raise AssertionError(f"unsupported command: {args.command}")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
