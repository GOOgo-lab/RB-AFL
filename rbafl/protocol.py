from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple


CHANNEL_ORDER: Tuple[str, ...] = ("occ", "dist", "orient", "density")
CHANNEL_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(CHANNEL_ORDER)}


@dataclass(frozen=True)
class AblationSpec:
    exp_id: str
    exp_name: str
    description: str
    channels: Tuple[str, ...]
    lambda_consistency: float
    lambda_triplet: float

    @property
    def channel_indices(self) -> Tuple[int, ...]:
        return tuple(CHANNEL_INDEX[name] for name in self.channels)

    def to_dict(self) -> Dict[str, object]:
        result = asdict(self)
        result["channels"] = list(self.channels)
        result["channel_indices"] = list(self.channel_indices)
        return result


# E1-E4 share the identity-classification objective and change only the input
# channels. E5 adds the two proposed losses. E6/E7 remove one loss at a time.
ABLATIONS: Tuple[AblationSpec, ...] = (
    AblationSpec("E1", "occ_only", "仅占据场", ("occ",), 0.0, 0.0),
    AblationSpec("E2", "occ_dist", "占据场 + 距离场", ("occ", "dist"), 0.0, 0.0),
    AblationSpec(
        "E3",
        "occ_dist_orient",
        "占据场 + 距离场 + 方向场",
        ("occ", "dist", "orient"),
        0.0,
        0.0,
    ),
    AblationSpec(
        "E4",
        "full_channels",
        "完整四通道（基础身份分类损失）",
        CHANNEL_ORDER,
        0.0,
        0.0,
    ),
    AblationSpec(
        "E5",
        "proposed",
        "完整通道 + 一致性损失 + 三元组唯一性损失",
        CHANNEL_ORDER,
        1.0,
        1.0,
    ),
    AblationSpec(
        "E6",
        "no_consistency",
        "去除一致性损失",
        CHANNEL_ORDER,
        0.0,
        1.0,
    ),
    AblationSpec(
        "E7",
        "no_triplet",
        "去除三元组唯一性损失",
        CHANNEL_ORDER,
        1.0,
        0.0,
    ),
)


ROTATION_DEGREES: Tuple[float, ...] = (0.0, 30.0, 75.0, 135.0)
# A scale factor is a multiplier, not an attack ratio.  Factor 1.0 is therefore
# the explicit clean baseline; factor 0 is invalid because it collapses geometry.
SCALE_FACTORS: Tuple[float, ...] = (
    0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 1.9, 2.0,
)
# Translation is expressed as a fraction of max(width, height); zero is clean.
TRANSLATION_FACTORS: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0)
OBJECT_ADD_RATIOS: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25)
# Deletion levels visible in manuscript Figure 6(c).
OBJECT_DELETE_RATIOS: Tuple[float, ...] = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50)
# Each non-zero deletion level is evaluated with independent deterministic masks.
# The clean (0%) case is stored once because there is no random mask to repeat.
OBJECT_DELETE_MASK_REPEATS: int = 10
CLIP_MERGE_RATIOS: Tuple[float, ...] = (0.0, 0.15, 0.30, 0.45, 0.60, 0.75)

# Additional single-attack protocol derived from vector-map watermarking literature.
# All strengths are dataset-independent ratios so maps with different units can be compared.
NON_UNIFORM_SCALE_FACTORS: Tuple[float, ...] = (0.4, 0.6, 0.8, 1.2, 1.5, 2.0, 3.0, 4.0)
VERTEX_CHANGE_RATIOS: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
COORDINATE_NOISE_RATIOS: Tuple[float, ...] = (
    0.00010,
    0.00025,
    0.00050,
    0.00100,
    0.00200,
    0.00500,
)
REORDER_RATIOS: Tuple[float, ...] = (0.20, 0.40, 0.60, 0.80, 1.00)
# No public default is permitted: the final decision threshold must be read from
# calibration/threshold_summary.json.  The alias remains for API compatibility.
NC_THRESHOLD: Optional[float] = None


@dataclass(frozen=True)
class AttackCase:
    attack: str
    strength: float
    repeat_index: int = 0

    @property
    def case_id(self) -> str:
        base = f"{self.attack}_{self.strength:g}"
        if self.attack == "object_delete" and self.strength > 0:
            return f"{base}_r{self.repeat_index:02d}"
        return base


def attack_plan() -> List[AttackCase]:
    plan: List[AttackCase] = []
    plan.extend(AttackCase("rotation", x) for x in ROTATION_DEGREES)
    plan.extend(AttackCase("scale", x) for x in SCALE_FACTORS)
    plan.extend(AttackCase("translation", x) for x in TRANSLATION_FACTORS)
    for strength in OBJECT_DELETE_RATIOS:
        repeats = OBJECT_DELETE_MASK_REPEATS if strength > 0 else 1
        plan.extend(
            AttackCase("object_delete", strength, repeat_index)
            for repeat_index in range(repeats)
        )
    return plan


def get_ablation(exp_id_or_name: str) -> AblationSpec:
    key = exp_id_or_name.strip().lower()
    for spec in ABLATIONS:
        if key in (spec.exp_id.lower(), spec.exp_name.lower()):
            return spec
    choices = ", ".join(f"{x.exp_id}/{x.exp_name}" for x in ABLATIONS)
    raise KeyError(f"Unknown experiment '{exp_id_or_name}'. Available: {choices}")


def protocol_dict() -> Dict[str, object]:
    return {
        "version": "1.1.0",
        "implementation_version": "RB-AFL-MANUSCRIPT-ALIGNED-1.1",
        "ablation_design": [x.to_dict() for x in ABLATIONS],
        "attacks": {
            "rotation_degrees_clockwise": list(ROTATION_DEGREES),
            "uniform_scale_factors": list(SCALE_FACTORS),
            "translation_factors": list(TRANSLATION_FACTORS),
            "object_delete_ratios": list(OBJECT_DELETE_RATIOS),
            "object_delete_random_masks_per_nonzero_strength": OBJECT_DELETE_MASK_REPEATS,
        },
        "translation_definition": (
            "xoff=yoff=factor*max(width,height) in the source coordinate system"
        ),
        "clip_definition": (
            "remove the rightmost ratio of the source bounding-box width; retain 1-ratio"
        ),
        "merge_definition": (
            "append a deterministic subset of features from an external vector map; never dissolve"
        ),
        "non_uniform_scale_definition": "xfact=strength, yfact=1.0 around dataset center",
        "interpolation_definition": (
            "linearly add approximately strength*original_vertex_count vertices without changing shape"
        ),
        "simplification_definition": (
            "Douglas-Peucker simplification; binary-search tolerance to approach the target deleted-vertex ratio"
        ),
        "vertex_delete_definition": (
            "randomly delete the requested fraction of vertices while preserving line endpoints and valid rings"
        ),
        "coordinate_noise_definition": (
            "independent uniform coordinate jitter in [-strength*span,+strength*span]"
        ),
        "reorder_definition": (
            "reorder the requested fraction of features and reverse their coordinate storage direction without changing shape"
        ),
        "decision_rule": {
            "metric": "NC",
            "operator": ">=",
            "threshold": 0.75,
            "threshold_source": "predeclared 0.75; diagnostic calibration artifact required",
        },
    }
