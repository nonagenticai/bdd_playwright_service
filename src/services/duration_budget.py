"""Turn an ORDERED film duration into the two numbers this service authors.

WHY THIS EXISTS. A demo_video ordered at 30s delivered 47.662s (159%) on
2026-09-10. It is not a rate problem — narration-to-picture fit is ~100.2%. It is
structural: the ordered duration reaches this service only as a scenario COUNT
ceiling (``max_scenarios_per_feature``), and a single beat's footage is ~24s, so
NO integer beat count lands within +-10% of a 30s order (1 beat = 25.4s / 85%,
2 beats = 47.4s / 159%). Choosing a beat count cannot fix that.

This service is the layer that can: it authors BOTH the narration text and the
dwell, and until now neither had a length budget — which is exactly why
narration ~= clip ~= 23.8s regardless of what was ordered.

The budget is a BOUND, never an extension:

* it can only SHORTEN a clip below the ``HERO_DWELL_MS`` default, never lengthen
  it past it (a 300s order over 2 scenarios must not ask for a 150s dwell, which
  the per-test Playwright timeout would kill mid-hook anyway); and
* below :data:`MIN_SCENARIO_RECORDING_SECONDS` it clamps, because a 2-second clip
  that shows nothing is a worse outcome than a slightly long demo.

Nothing here raises. A missed order is RECORDED (``order_met=False`` on the
``step_coverage`` payload), never a failure: generation succeeded, and turning a
currently-shipping package red is the caller's product decision.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

from src.database.models import DurationBudget
from src.services.generation.step_def_generator import (
    GUARANTEED_RECORDING_SECONDS_PER_SCENARIO,
)

logger = logging.getLogger(__name__)

#: FLOOR — the shortest clip this service will deliberately produce, in seconds.
#:
#: ⚠️ NEW CONSTANT, introduced with the duration budget; it is not a measured
#: value carried over from anywhere. Justification, from this repo's own numbers:
#: the bimodal-duration defect that `HERO_DWELL_MS` was introduced to fix
#: produced 1.24-3.94s clips, which downstream read as `timeline GROSSLY
#: UNDERFULL` and visually are a page-load with a timer, not a demo. A clip has
#: to survive page settle plus ONE complete narrated sentence; at this service's
#: ``narration.words_per_minute`` default of 150 wpm, 8s buys ~20 words, i.e.
#: one full sentence. Below that the clip cannot carry its own narration line, so
#: the order is refused (recorded, not raised) rather than honoured degenerately.
MIN_SCENARIO_RECORDING_SECONDS = 8.0

#: HERO-SHARE CAP — the largest fraction of the ORDERED film duration that any one
#: recorded beat's dwell may claim.
#:
#: WHY. A 44s order delivered +54.8%. The cause is not script length: ONE clip's own
#: footage was ~35s of the 44s order, and in the orchestrator a beat whose footage is
#: that close to the whole film becomes ``min == ideal == max, elasticity 0, pinned``
#: — a shape narration can never shrink. The existing ceiling
#: (:data:`GUARANTEED_RECORDING_SECONDS_PER_SCENARIO`, 22s) cannot prevent it because
#: 22s IS most of a 44s order. The cap has to be relative to the order, not absolute.
#:
#: WHERE IT APPLIES — and this is the part that keeps it from overshooting. It is
#: gated on ``ceiling_applied``: it touches ONLY a beat whose fair share
#: (``ordered / scenarios``) already exceeded the 22s ceiling, i.e. a beat whose
#: order the footage was NEVER going to meet. A proportionate split — 30s over 2
#: scenarios is 15s each, ``order_met`` — is left exactly as it is. So the cap
#: cannot turn a meetable order into a film that ends early; it can only add
#: elasticity to a beat that was already going to miss and would otherwise miss
#: while PINNED.
#:
#: WHY 0.4, and not something that bites harder. Every bound here is pinned to a
#: constant this module already owns, so the number is derived rather than picked:
#:
#: * UPPER EDGE — the cap is a NO-OP for any order at or above 55s, because
#:   ``22 / 0.4 = 55`` and above that the 22s dwell is already under 40% of the
#:   order. It cannot touch long-form work at all.
#: * LOWER EDGE — ``0.4 * 20 = 8.0`` is exactly
#:   :data:`MIN_SCENARIO_RECORDING_SECONDS`, so at a 20s order the cap's own value
#:   coincides with the floor and below that the floor wins outright. The cap can
#:   never produce a degenerate clip at ANY order.
#: * It is therefore active only in the 20-55s band, which is where the measured
#:   defect (44s) lives, and it leaves at least 60% of the order to the other beats
#:   and to the orchestrator's elasticity.
#: * On the measured case it moves the dwell 22s -> 17.6s. That is a 20% reduction
#:   in the picture floor, not a re-plan.
#:
#: ⚠️ IT CAPS THE PICTURE, AND THE NARRATION IS SIZED FROM THE CAPPED NUMBER.
#: Duration corrections in this estate have a measured history of overshooting the
#: other way (+56.8% -> -33.5%, +216% -> -20.4%) because a lever that shortens the
#: footage also shortens the narration, and a film is exactly its narration. That is
#: a real hazard — but the opposite arrangement is WORSE, and it is the one this
#: module shipped with: sizing the word budget from the UNCAPPED
#: ``per_scenario_seconds`` while the ledger records only ``dwell_seconds`` writes a
#: line that outlasts its own footage. On the measured 44s order that is ~22s of
#: speech over 17.6s of picture — the picture stops and the audio keeps going, which
#: is precisely the truncation defect, not a fix for it. And the published clip floor
#: (``expected_recording_seconds`` in ``job_processor``) is ALREADY ``dwell_seconds``,
#: so the orchestrator plans the beat at the capped length either way.
#:
#: So both numbers come from :attr:`DurationBudgetPlan.dwell_seconds`. The beat still
#: loses its pin and gains elasticity. It does NOT then fill the order — but it never
#: could: the cap only fires when ``ceiling_applied``, i.e. when ``order_met`` is
#: already ``False`` and the miss is already recorded on ``step_coverage``. The choice
#: here is only between missing the order with the audio and the picture agreeing, and
#: missing it with them disagreeing.
HERO_BEAT_MAX_ORDER_SHARE = 0.4

#: Sentence boundary for budget-aware trimming (keeps the terminator attached).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

#: :func:`trim_to_word_budget` keeps the sentence that crosses the budget when the
#: sentences before it fill less than this share of the budget ...
TRIM_RESCUE_FLOOR_RATIO = 0.85
#: ... and keeping it lands at or under this multiple of the budget. Measured on the
#: live model (2026-10-20 replay of job 01a156a6-8dd9): asked to write to a budget it
#: lands ~+25% over (24-26 words for 20, 19-22 for 15), so 1.35 keeps a whole second
#: sentence in that range while still cutting a reply that is a different length of
#: text entirely.
TRIM_RESCUE_CEILING_RATIO = 1.35


@dataclass(frozen=True)
class DurationBudgetPlan:
    """The per-scenario second budget derived from an ordered film duration.

    ``per_scenario_seconds`` is the order's fair share after the floor/ceiling
    clamps. ``dwell_seconds`` is that share after the hero-share cap, and it is
    what BOTH the picture and the narration length are sized from, so the two
    cannot drift apart. They differ only when the cap binds.
    """

    ordered_seconds: float
    scenario_count: int
    per_scenario_seconds: float
    narration_word_budget: int
    floor_applied: bool
    ceiling_applied: bool
    #: The PICTURE budget: ``per_scenario_seconds`` after the hero-share cap
    #: (:data:`HERO_BEAT_MAX_ORDER_SHARE`). Equal to ``per_scenario_seconds``
    #: whenever the cap does not bind, which is every order at or above 55s.
    dwell_seconds: float
    hero_cap_applied: bool

    @property
    def dwell_budget_ms(self) -> int:
        """The per-scenario dwell ledger budget, in whole milliseconds."""
        return round(self.dwell_seconds * 1000)

    @property
    def projected_total_seconds(self) -> float:
        """What the order becomes once the floor/ceiling clamps are applied."""
        return round(self.per_scenario_seconds * self.scenario_count, 3)

    @property
    def order_met(self) -> bool:
        """False when a clamp made the delivered length miss the order.

        The ONLY signal that the order could not be honoured. It is reported, not
        raised — see the module docstring.
        """
        return not (self.floor_applied or self.ceiling_applied)

    def as_dict(self) -> DurationBudget:
        """JSON-safe payload carried on the job's ``step_coverage``."""
        return {
            "ordered_seconds": round(self.ordered_seconds, 3),
            "scenario_count": self.scenario_count,
            "per_scenario_seconds": round(self.per_scenario_seconds, 3),
            "narration_word_budget": self.narration_word_budget,
            "floor_seconds": MIN_SCENARIO_RECORDING_SECONDS,
            "floor_applied": self.floor_applied,
            "ceiling_seconds": GUARANTEED_RECORDING_SECONDS_PER_SCENARIO,
            "ceiling_applied": self.ceiling_applied,
            "dwell_seconds": round(self.dwell_seconds, 3),
            "hero_cap_share": HERO_BEAT_MAX_ORDER_SHARE,
            "hero_cap_applied": self.hero_cap_applied,
            "projected_total_seconds": self.projected_total_seconds,
            "order_met": self.order_met,
        }


def plan_duration_budget(
    ordered_seconds: float | None,
    scenario_count: int,
    *,
    words_per_minute: int,
) -> DurationBudgetPlan | None:
    """Split an ordered film duration across the scenarios actually generated.

    Returns ``None`` — meaning "behave exactly as before" — when no duration was
    ordered, when the order is non-positive, or when there are no scenarios to
    divide it across. That ``None`` is the backward-compatibility contract: every
    caller that never sends ``target_duration_seconds`` must keep today's
    behaviour byte for byte.

    Args:
        ordered_seconds: The film duration the caller ordered, if any.
        scenario_count: Scenarios actually generated (NOT the requested cap —
            the LLM routinely emits fewer, and budgeting against a cap that was
            not filled would under-run every clip).
        words_per_minute: The speaking rate used to convert seconds to a word
            budget. Supplied by the caller from
            ``Settings.narration.words_per_minute`` so this module never owns a
            second copy of that number.
    """
    if ordered_seconds is None or ordered_seconds <= 0 or scenario_count <= 0:
        return None

    raw = ordered_seconds / scenario_count
    ceiling_applied = raw > GUARANTEED_RECORDING_SECONDS_PER_SCENARIO
    floor_applied = raw < MIN_SCENARIO_RECORDING_SECONDS
    per_scenario = min(
        max(raw, MIN_SCENARIO_RECORDING_SECONDS),
        GUARANTEED_RECORDING_SECONDS_PER_SCENARIO,
    )
    # HERO-SHARE CAP on the PICTURE only — see HERO_BEAT_MAX_ORDER_SHARE. Gated on
    # ``ceiling_applied``, which is the whole reason it cannot overshoot: it acts
    # ONLY on a beat whose fair share already exceeded the 22s ceiling, i.e. one
    # whose order the footage was never going to meet anyway. A proportionate split
    # (``raw <= 22``, ``order_met``) is left byte for byte alone, so the cap can
    # never turn a meetable order into a film that ends early.
    dwell = per_scenario
    hero_cap_applied = False
    if ceiling_applied:
        hero_cap = HERO_BEAT_MAX_ORDER_SHARE * float(ordered_seconds)
        # The floor is re-applied on top so the cap can never undercut
        # MIN_SCENARIO_RECORDING_SECONDS.
        dwell = max(min(per_scenario, hero_cap), MIN_SCENARIO_RECORDING_SECONDS)
        hero_cap_applied = dwell < per_scenario
    if hero_cap_applied:
        logger.info(
            "hero dwell CLAMPED: %.3fs -> %.3fs (%.0f%% cap on a %.1fs order over %d "
            "scenario(s)); narration word budget sized from the CAPPED %ss so the line "
            "cannot outlast its own footage (the order was already unmeetable: "
            "ceiling_applied)",
            per_scenario,
            dwell,
            HERO_BEAT_MAX_ORDER_SHARE * 100,
            ordered_seconds,
            scenario_count,
            round(dwell, 1),
        )
    # A word budget of 0 would delete the narration outright, which is worse than
    # a slightly long line; the floor above already guarantees room for a
    # sentence, so any positive rate yields at least one word.
    #
    # ⚠️ Sized from ``dwell``, i.e. AFTER the hero-share cap — the same number the
    # dwell ledger is narrowed to and the same number published as the clip floor
    # (``expected_recording_seconds``). Sizing it from the uncapped
    # ``per_scenario`` writes speech that outlasts the footage. ``dwell`` equals
    # ``per_scenario`` on every path where the cap does not bind, so this is a
    # no-op for every order that is not both ceiling-clamped and under 55s.
    word_budget = max(1, int(dwell * words_per_minute / 60))
    return DurationBudgetPlan(
        ordered_seconds=float(ordered_seconds),
        scenario_count=scenario_count,
        per_scenario_seconds=per_scenario,
        narration_word_budget=word_budget,
        floor_applied=floor_applied,
        ceiling_applied=ceiling_applied,
        dwell_seconds=dwell,
        hero_cap_applied=hero_cap_applied,
    )


def narration_word_budgets(
    words_target: object,
    words_per_scenario: object,
    scenario_count: int,
) -> list[int] | None:
    """The ORCHESTRATOR-supplied per-scenario word budget, or ``None``.

    The orchestrator now sends ``narration_words_target`` (total words for the
    film) and optionally ``narration_words_per_scenario`` (the per-beat split)
    alongside ``target_duration_seconds``. Both are OPTIONAL by contract: every
    orchestrator that has not deployed yet sends neither, and this service must
    behave byte for byte as it does today for those payloads — which is what
    ``None`` means here.

    A supplied budget is what the script is written to FILL, so it is preferred
    over the locally derived ``narration_word_budget``: the orchestrator knows the
    delivered speech rate, and this service only knows a configured ``wpm``.

    Precedence and the reason for it:

    * ``narration_words_per_scenario`` wins when it is a list of positive ints of
      exactly ``scenario_count`` length. A length mismatch means the orchestrator
      planned against a different beat count than the LLM actually emitted, so the
      per-beat split describes beats that do not exist — it is DISCARDED (logged)
      in favour of the total, never zipped short.
    * ``narration_words_target`` is then split evenly, with the remainder spread
      one word at a time over the leading beats so the parts sum to the total
      EXACTLY rather than losing words to floor division.
    * Anything else — absent, ``None``, non-numeric, non-positive — yields ``None``.
    """
    if scenario_count <= 0:
        return None

    if isinstance(words_per_scenario, list) and words_per_scenario:
        split = [
            int(w)
            for w in words_per_scenario
            if isinstance(w, int | float) and not isinstance(w, bool) and w > 0
        ]
        if len(split) == len(words_per_scenario) == scenario_count:
            return split
        logger.warning(
            "narration_words_per_scenario DISCARDED: %d usable entries for %d generated "
            "scenario(s); falling back to narration_words_target",
            len(split),
            scenario_count,
        )

    if (
        isinstance(words_target, int | float)
        and not isinstance(words_target, bool)
        and words_target > 0
    ):
        total = int(words_target)
        base, remainder = divmod(total, scenario_count)
        budgets = [base + (1 if i < remainder else 0) for i in range(scenario_count)]
        # Floor division can hand a trailing beat a budget of 0, which would delete
        # its line outright; one word is the same guard `plan_duration_budget` uses.
        return [max(1, w) for w in budgets]

    return None


def raise_to_clip_floor(
    word_budgets: list[int] | None,
    budget: DurationBudgetPlan | None,
    words_target: object,
    ordered_seconds: object,
) -> list[int] | None:
    """Raise each per-scenario word budget to the words its CLIP will actually hold.

    The orchestrator splits ``narration_words_target`` across the scenarios as a
    share of the ORDER. When the order's share is under the recording floor
    (``budget.floor_applied``) every clip records at least ``dwell_seconds`` anyway,
    so a share-sized line under-fills its own clip by construction — and the
    orchestrator then drops whole beats to fit the order, so each SURVIVING beat
    rides a clip of at least that length. Live job 01a156a6-8dd9 (2026-10-20): 30s
    over 5 scenarios = 6s shares clamped to 8s clips, so 15-word lines rode clips the
    orchestrator measured at 10.6s.

    The words-per-second is the ORCHESTRATOR's own (``words_target / ordered_seconds``),
    never this service's configured ``wpm`` — it knows the voice; this does not.
    Returned unchanged when no floor was applied or any input is unusable.
    """
    if not word_budgets or budget is None or not budget.floor_applied:
        return word_budgets
    target = _positive_number(words_target)
    seconds = _positive_number(ordered_seconds)
    if target is None or seconds is None:
        return word_budgets
    words_per_second = target / seconds
    clip_words = round(budget.dwell_seconds * words_per_second)
    return [max(words, clip_words) for words in word_budgets]


def _positive_number(value: object) -> float | None:
    """``value`` as a float when it is a real, positive int/float (``bool`` excluded)."""
    if isinstance(value, int | float) and not isinstance(value, bool) and value > 0:
        return float(value)
    return None


def trim_to_word_budget(text: str, max_words: int) -> str:
    """Trim ``text`` to about ``max_words`` words, preferring a sentence end.

    Not a hard ceiling any more: the one sentence that crosses the budget is kept
    when the text before it would otherwise under-fill the budget (see
    :data:`TRIM_RESCUE_FLOOR_RATIO` / :data:`TRIM_RESCUE_CEILING_RATIO`).

    Narration is spoken over the clip, so a mid-sentence cut is audible; whole
    sentences are dropped first and only a single over-long opening sentence is
    hard-cut. Text already within budget is returned UNCHANGED (identity, not a
    re-join) so the no-budget path and the within-budget path cannot differ by
    whitespace.
    """
    if max_words <= 0:
        return text
    words = text.split()
    if len(words) <= max_words:
        return text

    kept: list[str] = []
    spent = 0
    rescue_ceiling = math.floor(max_words * TRIM_RESCUE_CEILING_RATIO)
    for sentence in _SENTENCE_SPLIT_RE.split(text.strip()):
        count = len(sentence.split())
        if not count:
            continue
        if spent + count > max_words:
            # THE RESCUE. Dropping this sentence would leave the line far UNDER its
            # budget, so keep it when the overrun it costs is small. Without this the
            # "ceiling" removed most of the text to avoid a modest overrun: a 21-word
            # two-sentence polish against a 15-word budget came back as its 7-word
            # first sentence (live job 01a156a6-8dd9, 2026-10-20, every beat). An
            # underrun cannot be recovered downstream except by dragging the voice;
            # a small overrun is trimmed by the orchestrator's own rewriter.
            if spent < max_words * TRIM_RESCUE_FLOOR_RATIO and spent + count <= rescue_ceiling:
                kept.append(sentence)
                spent += count
            break
        kept.append(sentence)
        spent += count
    if kept:
        return " ".join(kept)
    # One opening sentence longer than the whole budget: hard-cut it.
    return " ".join(words[:max_words])
