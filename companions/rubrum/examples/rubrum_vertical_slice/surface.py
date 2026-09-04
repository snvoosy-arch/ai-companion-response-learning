from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .contracts import CandidateVerdict, ContentPlan, SemanticFeature, SurfaceCandidate
from .semantic_features import feature_map, replace_features


@dataclass(frozen=True, slots=True)
class _SurfaceSpec:
    candidate_id: str
    family: str
    response_act: str
    topic: str
    experiencer: str
    semantic_features: tuple[SemanticFeature, ...]
    register: str
    form_id: str
    naturalness: float
    clarity: float


@dataclass(frozen=True, slots=True)
class _SurfaceProjection:
    text: str
    atoms: tuple[str, ...]
    atom_roles: tuple[str, ...]
    register: str


def _render_weather(spec: _SurfaceSpec, values: dict[str, str]) -> _SurfaceProjection:
    time, topic_particle = {
        "yesterday": ("어제", "는"),
        "tomorrow": ("내일", "은"),
    }[values["target_time"]]
    comparison = {"less": "덜", "more": "더"}[values["comparison"]]
    predicate = {"cold": "추울", "hot": "더울"}[values["predicate"]]
    form_register = {
        "weather.standard": "casual_banmal",
        "weather.concise": "casual_banmal",
        "weather.polite": "polite_haeyo",
    }[spec.form_id]
    if spec.form_id == "weather.concise":
        if values["degree"] != "slight" or values["evidentiality"] != "seem":
            raise ValueError("concise weather form does not support these semantics")
        degree, evidential_noun, ending, attach = "좀", "듯", "해", True
    else:
        degree = {"slight": "조금", "strong": "훨씬"}[values["degree"]]
        if values["evidentiality"] == "seem":
            evidential_noun = "것"
            ending = "같아요" if form_register == "polite_haeyo" else "같아"
            attach = False
        elif values["evidentiality"] == "certain" and form_register == "casual_banmal":
            evidential_noun, ending, attach = "거", "야", True
        else:
            raise ValueError("weather form does not support these evidential semantics")
    atoms = (
        time,
        topic_particle,
        degree,
        comparison,
        predicate,
        evidential_noun,
        ending,
        ".",
    )
    evidential_surface = f"{evidential_noun}{ending}" if attach else f"{evidential_noun} {ending}"
    return _SurfaceProjection(
        text=(f"{time}{topic_particle} {degree} {comparison} {predicate} {evidential_surface}."),
        atoms=atoms,
        atom_roles=(
            "time",
            "particle",
            "degree",
            "comparison",
            "predicate",
            "evidential",
            "evidential",
            "punctuation",
        ),
        register=form_register,
    )


def _render_fatigue(spec: _SurfaceSpec, values: dict[str, str]) -> _SurfaceProjection:
    form_register = {
        "fatigue.standard": "casual_banmal",
        "fatigue.concise": "casual_banmal",
        "fatigue.polite": "polite_haeyo",
    }[spec.form_id]
    if values["lifecycle"] != "active" or values["evidentiality"] != "inferred":
        raise ValueError("fatigue acknowledgement requires an active inferred state")
    expected_degree = {"high": "high", "low": "low"}[values["intensity"]]
    if values["degree"] != expected_degree:
        raise ValueError("fatigue intensity and degree disagree")
    time = {"today": "오늘"}[values["time"]]
    duration = {"all_day": "하루 종일"}[values["duration"]]
    cause = {
        ("work", "fatigue.standard"): "일했으면",
        ("work", "fatigue.concise"): "일했으니",
        ("work", "fatigue.polite"): "일했으면",
    }[(values["cause"], spec.form_id)]
    degree = {
        ("high", "fatigue.standard"): "많이",
        ("high", "fatigue.concise"): "꽤",
        ("high", "fatigue.polite"): "많이",
        ("low", "fatigue.standard"): "조금",
    }[(values["degree"], spec.form_id)]
    state = {"fatigue": "지쳤", "hunger": "배고팠"}[values["state"]]
    evidential = "겠"
    ending = {
        "fatigue.standard": "네",
        "fatigue.concise": "다",
        "fatigue.polite": "네요",
    }[spec.form_id]
    experiencer_atoms = {
        "user": (),
        "third_party": ("친구는",),
    }[spec.experiencer]
    atoms = (
        *experiencer_atoms,
        time,
        duration,
        cause,
        degree,
        state,
        evidential,
        ending,
        ".",
    )
    atom_roles = (
        *(("experiencer",) if experiencer_atoms else ()),
        "time",
        "duration",
        "cause",
        "degree",
        "state",
        "evidential",
        "ending",
        "punctuation",
    )
    prefix = f"{experiencer_atoms[0]} " if experiencer_atoms else ""
    return _SurfaceProjection(
        text=(f"{prefix}{time} {duration} {cause} {degree} {state}{evidential}{ending}."),
        atoms=atoms,
        atom_roles=atom_roles,
        register=form_register,
    )


def _render_food(spec: _SurfaceSpec, values: dict[str, str]) -> _SurfaceProjection:
    form_register = {
        "food.standard": "casual_banmal",
        "food.concise": "casual_banmal",
        "food.polite": "polite_haeyo",
    }[spec.form_id]
    items = {
        "chicken_gomtang": ("닭곰탕", "mild", "warm", "soup"),
        "malatang": ("마라탕", "spicy", "warm", "soup"),
        "naengmyeon": ("냉면", "refreshing", "cold", "noodle"),
    }
    item, item_taste, item_temperature, item_category = items[values["item"]]
    if (
        values["taste"] != item_taste
        or values["temperature"] != item_temperature
        or values["category"] != item_category
        or values["fit"] != "good"
        or values["evidentiality"] != "suggestion"
        or values["desire"] != "active"
    ):
        raise ValueError("food item properties do not match candidate semantics")
    taste = {"mild": "담백하고", "spicy": "얼얼하고", "refreshing": "산뜻하고"}[values["taste"]]
    temperature = {"warm": "따뜻한", "cold": "시원한"}[values["temperature"]]
    category = {
        ("soup", "food.standard"): "국물이면",
        ("soup", "food.concise"): "국물엔",
        ("soup", "food.polite"): "국물이면",
        ("noodle", "food.standard"): "면이면",
    }[(values["category"], spec.form_id)]
    fit, ending = {
        "food.standard": ("잘 맞", "네"),
        "food.concise": ("괜찮", "네"),
        "food.polite": ("잘 맞", "어요"),
    }[spec.form_id]
    evidential = "겠"
    atoms = (taste, temperature, category, item, "이", fit, evidential, ending, ".")
    return _SurfaceProjection(
        text=(f"{taste} {temperature} {category} {item}이 {fit}{evidential}{ending}."),
        atoms=atoms,
        atom_roles=(
            "taste",
            "temperature",
            "category",
            "item",
            "particle",
            "fit",
            "evidential",
            "ending",
            "punctuation",
        ),
        register=form_register,
    )


def _render_relation(spec: _SurfaceSpec, values: dict[str, str]) -> _SurfaceProjection:
    form_register = {
        "relation.standard": "casual_banmal",
        "relation.emphatic": "casual_banmal",
        "relation.polite": "polite_haeyo",
    }[spec.form_id]
    derivations = {
        ("grandfather_of", "all_people"): (
            "genealogy",
            "extreme",
            "palman_daejanggyeong",
        ),
        ("grandfather_of", "small_group"): ("genealogy", "small", "notepad"),
        ("friend_of", "many_people"): ("contact_list", "large", "phonebook"),
    }
    expected = derivations[(values["relation"], values["scope"])]
    if (
        (values["target_record"], values["scale"], values["analogy"]) != expected
        or values["literalness"] != "playful"
        or values["rhetorical"] != "yes"
    ):
        raise ValueError("relation bridge and analogy semantics disagree")
    target, particle = {
        "genealogy": ("족보", "가"),
        "contact_list": ("연락처", "가"),
    }[values["target_record"]]
    analogy = {
        "palman_daejanggyeong": "팔만대장경",
        "notepad": "수첩",
        "phonebook": "전화번호부",
    }[values["analogy"]]
    intensifier = "무슨" if spec.form_id == "relation.emphatic" else ""
    rhetorical_end = "인가요" if form_register == "polite_haeyo" else "인가"
    middle = f"{intensifier} " if intensifier else ""
    atoms = (
        target,
        particle,
        *((intensifier,) if intensifier else ()),
        analogy,
        rhetorical_end,
        "?",
    )
    atom_roles = (
        "target_record",
        "particle",
        *(("intensifier",) if intensifier else ()),
        "analogy",
        "rhetorical_end",
        "punctuation",
    )
    return _SurfaceProjection(
        text=f"{target}{particle} {middle}{analogy}{rhetorical_end}?",
        atoms=atoms,
        atom_roles=atom_roles,
        register=form_register,
    )


def _surface_projection(spec: _SurfaceSpec) -> _SurfaceProjection:
    values = feature_map(spec.semantic_features)
    renderer = {
        "weather_outlook": _render_weather,
        "fatigue_acknowledgement": _render_fatigue,
        "food_recommendation": _render_food,
        "relation_hyperbole": _render_relation,
    }[spec.family]
    return renderer(spec, values)


def _realize(spec: _SurfaceSpec) -> SurfaceCandidate:
    projection = _surface_projection(spec)
    if projection.register != spec.register:
        raise ValueError(f"surface form register disagrees: {spec.candidate_id}")
    return SurfaceCandidate(
        candidate_id=spec.candidate_id,
        family=spec.family,
        text=projection.text,
        response_act=spec.response_act,
        topic=spec.topic,
        experiencer=spec.experiencer,
        semantic_features=spec.semantic_features,
        register=spec.register,
        form_id=spec.form_id,
        atoms=projection.atoms,
        atom_roles=projection.atom_roles,
        deterministic_naturalness_prior=spec.naturalness,
        deterministic_clarity_prior=spec.clarity,
    )


def _spec_from_plan(
    plan: ContentPlan,
    *,
    candidate_id: str,
    form_id: str,
    naturalness: float,
    clarity: float,
    semantic_features: tuple[SemanticFeature, ...] | None = None,
    experiencer: str | None = None,
    register: str | None = None,
) -> _SurfaceSpec:
    return _SurfaceSpec(
        candidate_id=candidate_id,
        family=plan.family,
        response_act=plan.response_act,
        topic=plan.topic,
        experiencer=experiencer or plan.experiencer,
        semantic_features=semantic_features or plan.semantic_features,
        register=register or plan.register,
        form_id=form_id,
        naturalness=naturalness,
        clarity=clarity,
    )


def _weather_specs(plan: ContentPlan) -> tuple[_SurfaceSpec, ...]:
    values = plan.semantic_features
    return (
        _spec_from_plan(
            plan,
            candidate_id="weather.less_cold.seem",
            form_id="weather.standard",
            naturalness=0.95,
            clarity=0.98,
        ),
        _spec_from_plan(
            plan,
            candidate_id="weather.less_cold.concise",
            form_id="weather.concise",
            naturalness=0.93,
            clarity=0.91,
        ),
        _spec_from_plan(
            plan,
            candidate_id="weather.wrong_time",
            form_id="weather.standard",
            naturalness=0.96,
            clarity=0.97,
            semantic_features=replace_features(values, target_time="yesterday"),
        ),
        _spec_from_plan(
            plan,
            candidate_id="weather.wrong_comparison",
            form_id="weather.standard",
            naturalness=0.97,
            clarity=0.97,
            semantic_features=replace_features(values, comparison="more"),
        ),
        _spec_from_plan(
            plan,
            candidate_id="weather.wrong_degree",
            form_id="weather.standard",
            naturalness=0.99,
            clarity=0.99,
            semantic_features=replace_features(values, degree="strong"),
        ),
        _spec_from_plan(
            plan,
            candidate_id="weather.wrong_evidentiality",
            form_id="weather.standard",
            naturalness=0.99,
            clarity=0.99,
            semantic_features=replace_features(values, evidentiality="certain"),
        ),
        _spec_from_plan(
            plan,
            candidate_id="weather.wrong_register",
            form_id="weather.polite",
            naturalness=0.99,
            clarity=0.99,
            register="polite_haeyo",
        ),
    )


def _fatigue_specs(plan: ContentPlan) -> tuple[_SurfaceSpec, ...]:
    values = plan.semantic_features
    return (
        _spec_from_plan(
            plan,
            candidate_id="fatigue.ack.standard",
            form_id="fatigue.standard",
            naturalness=0.96,
            clarity=0.97,
        ),
        _spec_from_plan(
            plan,
            candidate_id="fatigue.ack.concise",
            form_id="fatigue.concise",
            naturalness=0.93,
            clarity=0.92,
        ),
        _spec_from_plan(
            plan,
            candidate_id="fatigue.wrong_state",
            form_id="fatigue.standard",
            naturalness=0.99,
            clarity=0.99,
            semantic_features=replace_features(values, state="hunger"),
        ),
        _spec_from_plan(
            plan,
            candidate_id="fatigue.wrong_intensity",
            form_id="fatigue.standard",
            naturalness=0.99,
            clarity=0.99,
            semantic_features=replace_features(values, intensity="low", degree="low"),
        ),
        _spec_from_plan(
            plan,
            candidate_id="fatigue.wrong_experiencer",
            form_id="fatigue.standard",
            naturalness=0.99,
            clarity=0.99,
            experiencer="third_party",
        ),
        _spec_from_plan(
            plan,
            candidate_id="fatigue.wrong_register",
            form_id="fatigue.polite",
            naturalness=0.99,
            clarity=0.99,
            register="polite_haeyo",
        ),
    )


def _food_specs(plan: ContentPlan) -> tuple[_SurfaceSpec, ...]:
    values = plan.semantic_features
    return (
        _spec_from_plan(
            plan,
            candidate_id="food.chicken_gomtang.standard",
            form_id="food.standard",
            naturalness=0.96,
            clarity=0.98,
        ),
        _spec_from_plan(
            plan,
            candidate_id="food.chicken_gomtang.concise",
            form_id="food.concise",
            naturalness=0.94,
            clarity=0.93,
        ),
        _spec_from_plan(
            plan,
            candidate_id="food.wrong_taste_and_item",
            form_id="food.standard",
            naturalness=0.99,
            clarity=0.99,
            semantic_features=replace_features(values, taste="spicy", item="malatang"),
        ),
        _spec_from_plan(
            plan,
            candidate_id="food.wrong_temperature_category_item",
            form_id="food.standard",
            naturalness=0.99,
            clarity=0.99,
            semantic_features=replace_features(
                values,
                taste="refreshing",
                temperature="cold",
                category="noodle",
                item="naengmyeon",
            ),
        ),
        _spec_from_plan(
            plan,
            candidate_id="food.wrong_register",
            form_id="food.polite",
            naturalness=0.99,
            clarity=0.99,
            register="polite_haeyo",
        ),
    )


def _relation_specs(plan: ContentPlan) -> tuple[_SurfaceSpec, ...]:
    values = plan.semantic_features
    return (
        _spec_from_plan(
            plan,
            candidate_id="relation.genealogy.palman",
            form_id="relation.standard",
            naturalness=0.96,
            clarity=0.96,
        ),
        _spec_from_plan(
            plan,
            candidate_id="relation.genealogy.palman.emphatic",
            form_id="relation.emphatic",
            naturalness=0.93,
            clarity=0.91,
        ),
        _spec_from_plan(
            plan,
            candidate_id="relation.wrong_scale_analogy",
            form_id="relation.standard",
            naturalness=0.99,
            clarity=0.99,
            semantic_features=replace_features(
                values,
                scope="small_group",
                scale="small",
                analogy="notepad",
            ),
        ),
        _spec_from_plan(
            plan,
            candidate_id="relation.wrong_target_relation",
            form_id="relation.standard",
            naturalness=0.99,
            clarity=0.99,
            semantic_features=replace_features(
                values,
                relation="friend_of",
                scope="many_people",
                target_record="contact_list",
                scale="large",
                analogy="phonebook",
            ),
        ),
        _spec_from_plan(
            plan,
            candidate_id="relation.wrong_register",
            form_id="relation.polite",
            naturalness=0.99,
            clarity=0.99,
            register="polite_haeyo",
        ),
    )


def compose_candidates(plan: ContentPlan) -> tuple[SurfaceCandidate, ...]:
    """Compose scene candidates from lexeme/morpheme atoms, not sentence strings."""

    spec_builder = {
        "weather_outlook": _weather_specs,
        "fatigue_acknowledgement": _fatigue_specs,
        "food_recommendation": _food_specs,
        "relation_hyperbole": _relation_specs,
    }.get(plan.family)
    if spec_builder is None:
        raise ValueError(f"unsupported surface family: {plan.family}")
    return tuple(_realize(spec) for spec in spec_builder(plan))


def _candidate_spec(candidate: SurfaceCandidate) -> _SurfaceSpec:
    return _SurfaceSpec(
        candidate_id=candidate.candidate_id,
        family=candidate.family,
        response_act=candidate.response_act,
        topic=candidate.topic,
        experiencer=candidate.experiencer,
        semantic_features=candidate.semantic_features,
        register=candidate.register,
        form_id=candidate.form_id,
        naturalness=candidate.deterministic_naturalness_prior,
        clarity=candidate.deterministic_clarity_prior,
    )


def verify_candidate(
    plan: ContentPlan,
    candidate: SurfaceCandidate,
) -> CandidateVerdict:
    reasons: list[str] = []
    for field in ("family", "response_act", "topic", "experiencer", "register"):
        if getattr(candidate, field) != getattr(plan, field):
            reasons.append(f"gate:{field}_mismatch")

    try:
        plan_features = feature_map(plan.semantic_features)
        candidate_features = feature_map(candidate.semantic_features)
    except ValueError:
        reasons.append("gate:semantic_features_invalid")
    else:
        for axis, expected in plan_features.items():
            if candidate_features.get(axis) != expected:
                reasons.append(f"gate:{axis}_mismatch")
        if set(candidate_features) - set(plan_features):
            reasons.append("gate:unexpected_semantic_feature")

    if not isinstance(plan.required_atoms, tuple) or not all(
        isinstance(role, str) and role for role in plan.required_atoms
    ):
        reasons.append("gate:required_atoms_contract_invalid")
    elif not isinstance(candidate.atom_roles, tuple) or not set(plan.required_atoms).issubset(
        candidate.atom_roles
    ):
        reasons.append("gate:required_atoms_missing")
    if (
        not isinstance(candidate.atoms, tuple)
        or not isinstance(candidate.atom_roles, tuple)
        or len(candidate.atoms) != len(candidate.atom_roles)
    ):
        reasons.append("gate:atom_role_alignment_mismatch")
    if not candidate.atoms or candidate.atoms[-1] not in {".", "?", "!"}:
        reasons.append("gate:morphology_incomplete")

    priors = (
        candidate.deterministic_naturalness_prior,
        candidate.deterministic_clarity_prior,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not isfinite(value)
        or not 0.0 <= value <= 1.0
        for value in priors
    ):
        reasons.append("gate:score_prior_invalid")

    try:
        projection = _surface_projection(_candidate_spec(candidate))
    except (KeyError, TypeError, ValueError):
        reasons.append("gate:surface_projection_failed")
    else:
        if candidate.text != projection.text:
            reasons.append("gate:surface_text_mismatch")
        if candidate.atoms != projection.atoms:
            reasons.append("gate:surface_atoms_mismatch")
        if candidate.atom_roles != projection.atom_roles:
            reasons.append("gate:surface_atom_roles_mismatch")
        if candidate.register != projection.register:
            reasons.append("gate:surface_metadata_mismatch")

    accepted = not reasons
    score = (
        round(
            0.55 * candidate.deterministic_naturalness_prior
            + 0.45 * candidate.deterministic_clarity_prior,
            6,
        )
        if accepted
        else 0.0
    )
    if accepted:
        reasons.append("gate:meaning_and_morphology_preserved")
    return CandidateVerdict(
        candidate_id=candidate.candidate_id,
        accepted=accepted,
        score=score,
        reason_codes=tuple(reasons),
    )


def select_candidate(
    candidates: tuple[SurfaceCandidate, ...],
    verdicts: tuple[CandidateVerdict, ...],
) -> SurfaceCandidate:
    candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
    verdict_ids = tuple(verdict.candidate_id for verdict in verdicts)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("duplicate surface candidate id")
    if len(verdict_ids) != len(set(verdict_ids)):
        raise ValueError("duplicate candidate verdict id")
    if set(candidate_ids) != set(verdict_ids):
        raise ValueError("candidate and verdict ids do not align")
    accepted = {verdict.candidate_id: verdict for verdict in verdicts if verdict.accepted}
    eligible = tuple(candidate for candidate in candidates if candidate.candidate_id in accepted)
    if not eligible:
        raise ValueError("no verified surface candidate")
    return max(
        eligible,
        key=lambda candidate: (
            accepted[candidate.candidate_id].score,
            candidate.candidate_id,
        ),
    )
