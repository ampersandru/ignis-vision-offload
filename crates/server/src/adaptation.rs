//! **Adaptation** (ADR 0063 §3, spec 27b-16gb/04 solution 4, `CONTEXT.md`):
//! a 27B load whose VRAM plan refuses its profile's values lowers the ones
//! the operator left to the profile or the hardcoded default, in a fixed
//! order, until the plan fits -- and says which it lowered.
//!
//! The ladder is a pure function of the values it starts from, which of them
//! may adapt, and a closure that plans at given values. The planner
//! ([`ignis_core::plan_vram`]) stays a pure function of its inputs, called
//! again with the next step's; the load (`runtime::qwen27b_load`) supplies
//! the closure, which re-plans the leaf's lines at each step's context,
//! lanes and head. So every rule below is a CPU test with injected budgets.
//!
//! The order, each step taken only while the plan is still refused:
//! 1. the drafter's proposal head: a `shortlist` from the profile falls to
//!    `full`, freeing the shortlist's ~357 MB (it is bound beside the full
//!    head). `auto` resolves to `full` already, so on today's profiles this
//!    never fires;
//! 2. `max_context`, from its value down every multiple of
//!    [`CONTEXT_STEP_TOKENS`] to that floor, keeping the largest at which the
//!    plan fits and the KV pool holds one full sequence and one more
//!    [`SECOND_RESERVATION_TOKENS`]-token reservation (at one lane, where a
//!    second reservation has no lane to serve, the plan's own floor alone);
//! 3. `decode_lanes` to 1, after which the context ladder runs again from the
//!    value it started from. Lanes that cannot drop leave the largest context
//!    the plan itself holds, the second reservation unmet, and the step says
//!    so.
//!
//! A start the plan fits as configured adapts nothing. A plan refused for any
//! other reason than its minimum (a budget above free memory, a named KV pool
//! below its floor) is not adapted either: no step lowers what that refusal
//! names. When not even the floor fits, the start is refused with the last
//! attempt's deficit and every adaptation tried, in order.

use ignis_core::{ProposalHead, VramPlanError};

/// The context ladder's step and its floor, in tokens (ADR 0063 §3): a
/// context adapts down 65,536 at a time and never below 65,536. Neither
/// refuses a start today's code would accept: without adaptation every start
/// below its value is refused already.
pub const CONTEXT_STEP_TOKENS: u32 = 65_536;

/// The reservation an adapted context leaves room for beside one full
/// sequence, at more than one lane (ADR 0063 §3): a second request of this
/// many tokens can run while the first holds the whole context.
pub const SECOND_RESERVATION_TOKENS: u32 = 65_536;

/// The values a 27B load's VRAM plan is made at that adaptation may lower.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Attempt {
    pub max_context: u32,
    pub decode_lanes: u32,
    /// The drafter's proposal head; `None` without a drafter.
    pub draft_head: Option<ProposalHead>,
}

/// Which of an [`Attempt`]'s values the operator left to the profile or the
/// hardcoded default ([`crate::config::Config::adaptable`]): only those
/// adapt. An explicit value never does, and is refused as it always was.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Adaptable {
    pub max_context: bool,
    pub decode_lanes: bool,
    pub draft_head: bool,
}

impl Adaptable {
    /// Every value adapts: a start that names none of them.
    pub const ALL: Self = Self { max_context: true, decode_lanes: true, draft_head: true };
}

/// One adaptation a load took, in the words of `ignis.runtime.vram_plan`'s
/// `adapted` list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    DraftHead { from: ProposalHead, to: ProposalHead },
    /// `second_reservation` is false when the lanes could not drop and the
    /// pool holds the one sequence alone.
    MaxContext { from: u32, to: u32, second_reservation: bool },
    DecodeLanes { from: u32, to: u32 },
}

impl std::fmt::Display for Step {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::DraftHead { from, to } => write!(f, "draft_head {} -> {}", from.as_str(), to.as_str()),
            Self::MaxContext { from, to, second_reservation } => {
                write!(f, "max_context {from} -> {to}")?;
                if !second_reservation {
                    write!(f, " (one sequence only: no room for a second {SECOND_RESERVATION_TOKENS}-token reservation, and decode_lanes is explicit)")?;
                }
                Ok(())
            }
            Self::DecodeLanes { from, to } => write!(f, "decode_lanes {from} -> {to}"),
        }
    }
}

/// `steps` as the `adapted` field of `ignis.runtime.vram_plan` spells them:
/// in the order taken, comma-separated; empty when the load adapted nothing.
pub fn steps_text(steps: &[Step]) -> String {
    steps.iter().map(Step::to_string).collect::<Vec<_>>().join(", ")
}

/// What one plan at an [`Attempt`] came to.
#[derive(Debug)]
pub enum Planned<T> {
    /// The plan fits; its KV pool holds `kv_tokens` tokens.
    Fits { plan: T, kv_tokens: u64 },
    /// The plan refused the attempt.
    Refused(VramPlanError),
}

/// The values a load runs at, the plan made at them, and the steps that got
/// there (none when the configured values fit).
#[derive(Debug)]
pub struct Adapted<T> {
    pub attempt: Attempt,
    pub plan: T,
    pub steps: Vec<Step>,
}

/// One adaptation a refused start tried: a step, or a walk down the context
/// ladder (every context planned, the first being where it started).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Tried {
    Step(Step),
    Contexts(Vec<u32>),
}

impl std::fmt::Display for Tried {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Step(step) => step.fmt(f),
            Self::Contexts(contexts) => write!(
                f,
                "max_context {}",
                contexts.iter().map(u32::to_string).collect::<Vec<_>>().join(" -> ")
            ),
        }
    }
}

/// A start adaptation could not fit: the plan's refusal at the last attempt
/// -- its deficit, as the plan words it -- and every adaptation tried before
/// it, in order. With nothing tried it reads exactly as the plan's refusal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refused {
    pub error: VramPlanError,
    pub tried: Vec<Tried>,
}

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(f)?;
        if !self.tried.is_empty() {
            let tried = self.tried.iter().map(Tried::to_string).collect::<Vec<_>>().join(", ");
            write!(f, "; adaptation tried, in order: {tried}")?;
        }
        Ok(())
    }
}

/// Why [`adapt`] returned no plan: the plan refused every adaptation, or the
/// closure itself failed.
#[derive(Debug)]
pub enum AdaptError<E> {
    Refused(Refused),
    Failed(E),
}

/// The contexts the ladder plans from `from` (ADR 0063 §3): `from` itself,
/// then every multiple of [`CONTEXT_STEP_TOKENS`] below it, down to that
/// floor -- 262,144, 196,608, 131,072, 65,536 on the 16 GB profile; 917,504
/// down 65,536 at a time on the 5090's. A context at or below the floor is
/// its own only candidate.
pub fn contexts(from: u32) -> Vec<u32> {
    let below = (from.saturating_sub(1) / CONTEXT_STEP_TOKENS) * CONTEXT_STEP_TOKENS;
    let mut contexts = vec![from];
    contexts.extend((1..=below / CONTEXT_STEP_TOKENS).rev().map(|n| n * CONTEXT_STEP_TOKENS));
    contexts
}

/// Plan at `start`, and when the plan refuses it below its minimum, adapt
/// what `adaptable` allows, in the order the module doc gives, until a plan
/// fits. `plan` plans at an attempt: [`Planned::Refused`] carries the
/// planner's refusal, an `Err` a failure of its own, which ends the ladder.
pub fn adapt<T, E>(
    start: Attempt,
    adaptable: Adaptable,
    mut plan: impl FnMut(Attempt) -> Result<Planned<T>, E>,
) -> Result<Adapted<T>, AdaptError<E>> {
    let mut plan = |attempt| plan(attempt).map_err(AdaptError::Failed);
    let refused = |error, tried| AdaptError::Refused(Refused { error, tried });
    let mut last = match plan(start)? {
        Planned::Fits { plan, .. } => return Ok(Adapted { attempt: start, plan, steps: Vec::new() }),
        Planned::Refused(error @ VramPlanError::BelowMinimum { .. }) => error,
        Planned::Refused(error) => return Err(refused(error, Vec::new())),
    };
    let mut attempt = start;
    let mut steps = Vec::new();
    let mut tried = Vec::new();

    // 1. The drafter's head: a shortlist the operator did not name.
    if adaptable.draft_head && attempt.draft_head == Some(ProposalHead::Shortlist) {
        let step = Step::DraftHead { from: ProposalHead::Shortlist, to: ProposalHead::Full };
        attempt.draft_head = Some(ProposalHead::Full);
        steps.push(step);
        tried.push(Tried::Step(step));
        match plan(attempt)? {
            Planned::Fits { plan, .. } => return Ok(Adapted { attempt, plan, steps }),
            Planned::Refused(error @ VramPlanError::BelowMinimum { .. }) => last = error,
            Planned::Refused(error) => return Err(refused(error, tried)),
        }
    }

    // 2. The context, at these lanes, then 3. at one lane. The first pass
    // skips the context just refused at them.
    let candidates = if adaptable.max_context { contexts(start.max_context) } else { vec![start.max_context] };
    let mut skip = 1;
    loop {
        let one_lane = attempt.decode_lanes <= 1;
        let mut fallback = None;
        for &context in &candidates[skip..] {
            let at = Attempt { max_context: context, ..attempt };
            match plan(at)? {
                Planned::Fits { plan, kv_tokens }
                    if one_lane || kv_tokens >= u64::from(context) + u64::from(SECOND_RESERVATION_TOKENS) =>
                {
                    if context != start.max_context {
                        steps.push(Step::MaxContext { from: start.max_context, to: context, second_reservation: true });
                    }
                    return Ok(Adapted { attempt: at, plan, steps });
                }
                // The largest context the plan holds, the second reservation
                // unmet: kept only if the lanes cannot drop.
                Planned::Fits { plan, .. } => {
                    if fallback.is_none() {
                        fallback = Some((at, plan));
                    }
                }
                Planned::Refused(error @ VramPlanError::BelowMinimum { .. }) => last = error,
                Planned::Refused(error) => return Err(refused(error, tried)),
            }
        }
        if candidates.len() > 1 {
            tried.push(Tried::Contexts(candidates.clone()));
        }
        if !adaptable.decode_lanes || one_lane {
            return match fallback {
                Some((at, plan)) => {
                    steps.push(Step::MaxContext { from: start.max_context, to: at.max_context, second_reservation: false });
                    Ok(Adapted { attempt: at, plan, steps })
                }
                None => Err(refused(last, tried)),
            };
        }
        let step = Step::DecodeLanes { from: attempt.decode_lanes, to: 1 };
        attempt.decode_lanes = 1;
        steps.push(step);
        tried.push(Tried::Step(step));
        skip = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ignis_core::{KvFormat, KvGeometry, Residency, VramLines, VramMode, VramPlan};
    use std::convert::Infallible;

    /// One hq-e8-2b page: 64 tokens at 9,216 bytes.
    fn page_bytes() -> u64 {
        KvFormat::HqE8_2b.page_bytes(KvGeometry::qwen38_27b())
    }

    /// The shortlist head's bytes beside the full head (plans of 2026-10-20).
    const SHORTLIST_BYTES: u64 = 357_040_380;

    /// A card the tests plan on: the real planner over lines shaped like
    /// the 27B's -- weights, the drafter head's, a workspace whose hq planes
    /// follow the context up to 262,144 keys, a per-lane cost, and two host
    /// retained slots a lane, a KV page each -- inside an injected budget.
    #[derive(Clone, Copy)]
    struct Card {
        budget: u64,
        weights: u64,
        /// The workspace below its 4,096 bytes per context token.
        workspace: u64,
        /// What a lane costs: its state, hq window and round buffers.
        lane: u64,
    }

    /// The 5090's 27B under `rtx5090`: its plan's lines at 8 lanes, from
    /// `vram_plan` events (`.scratch/27b-16gb-study-2026-10-20`).
    const RTX5090: Card = Card {
        budget: 0,
        weights: 18_585_584_640 + 452_595_712 + 335_544_320 + 124_474_800,
        workspace: 1_508_311_552 - 4_096 * 262_144,
        lane: 196_880_384 + 35_651_648 + (4_623_344 + 82_125_920 + 152_139_488 + 78_027_552) / 8,
    };

    /// A 16 GB card's 27B under `rtx50-16gb`: a stand-in for the 16 GB
    /// body's weights (#338), text only.
    const GB16: Card = Card { budget: 0, weights: 12_500_000_000, workspace: 342_290_432, lane: 268_000_000 };

    impl Card {
        fn with_budget(self, budget: u64) -> Self {
            Self { budget, ..self }
        }

        /// Every line but the pool at `attempt`.
        fn held(&self, attempt: Attempt) -> u64 {
            let head = if attempt.draft_head == Some(ProposalHead::Shortlist) { SHORTLIST_BYTES } else { 0 };
            self.weights + head + self.workspace + 4_096 * u64::from(attempt.max_context.min(262_144))
                + self.lane * u64::from(attempt.decode_lanes)
        }

        /// The budget at which `attempt` keeps its context with exactly the
        /// pool `tokens` asks for.
        fn budget_holding(&self, attempt: Attempt, tokens: u32) -> u64 {
            self.held(attempt) + u64::from(tokens.div_ceil(64)) * page_bytes()
        }

        fn plan(&self, attempt: Attempt) -> Result<VramPlan, VramPlanError> {
            let page = page_bytes();
            let arena = move |pages: u32| u64::from(pages) * page;
            ignis_core::plan_vram(&ignis_core::VramRequest {
                mode: VramMode::Explicit { budget_bytes: self.budget, allow_oversubscription: false },
                free_at_start_bytes: u64::MAX,
                lines: VramLines { weights: self.held(attempt), ..VramLines::default() },
                kv_page_bytes: page,
                max_context_tokens: attempt.max_context,
                retained_slots: 2 * attempt.decode_lanes,
                kv_pool: None,
                residency: Residency::Resident,
                embedding_pool_named: false,
                kv_arena_bytes: &arena,
                can_page: true,
            })
        }

        /// The ladder on this card, with every attempt it planned.
        fn adapt(&self, start: Attempt, adaptable: Adaptable) -> (Result<Adapted<VramPlan>, AdaptError<Infallible>>, Vec<Attempt>) {
            let mut attempts = Vec::new();
            let adapted = adapt(start, adaptable, |attempt| {
                attempts.push(attempt);
                Ok::<_, Infallible>(match self.plan(attempt) {
                    Ok(plan) => Planned::Fits { kv_tokens: plan.kv_token_capacity(), plan },
                    Err(error) => Planned::Refused(error),
                })
            });
            (adapted, attempts)
        }
    }

    fn at(max_context: u32, decode_lanes: u32) -> Attempt {
        Attempt { max_context, decode_lanes, draft_head: Some(ProposalHead::Full) }
    }

    fn kept(result: Result<Adapted<VramPlan>, AdaptError<Infallible>>) -> Adapted<VramPlan> {
        match result {
            Ok(adapted) => adapted,
            Err(AdaptError::Refused(refused)) => panic!("refused: {refused}"),
            Err(AdaptError::Failed(never)) => match never {},
        }
    }

    fn refused(result: Result<Adapted<VramPlan>, AdaptError<Infallible>>) -> Refused {
        match result {
            Ok(adapted) => panic!("adapted to {:?}: {}", adapted.attempt, steps_text(&adapted.steps)),
            Err(AdaptError::Refused(refused)) => refused,
            Err(AdaptError::Failed(never)) => match never {},
        }
    }

    fn context_step(from: u32, to: u32) -> Step {
        Step::MaxContext { from, to, second_reservation: true }
    }

    #[test]
    fn the_ladder_goes_down_65536_at_a_time_from_any_value_to_its_floor() {
        assert_eq!(contexts(262_144), [262_144, 196_608, 131_072, 65_536]);
        assert_eq!(contexts(917_504).len(), 14);
        assert_eq!(contexts(917_504)[..3], [917_504, 851_968, 786_432]);
        assert_eq!(contexts(1_048_576).len(), 16);
        assert_eq!(contexts(100_000), [100_000, 65_536], "a value between steps, then the multiples below it");
        assert_eq!(contexts(65_536), [65_536]);
        assert_eq!(contexts(40_960), [40_960], "below the floor: itself alone");
    }

    /// Decision 5: on the 5090, a start that fits today -- 917,504 by a
    /// hair, its pool short of a second reservation -- plans once and adapts
    /// nothing.
    #[test]
    fn a_5090_plan_that_fits_by_a_hair_adapts_nothing() {
        let start = at(917_504, 8);
        let card = RTX5090.with_budget(RTX5090.budget_holding(start, 917_504 + 16 * 64));
        let (result, attempts) = card.adapt(start, Adaptable::ALL);
        let adapted = kept(result);
        assert_eq!((adapted.attempt, adapted.steps.as_slice()), (start, &[][..]));
        assert_eq!(attempts, [start], "one plan, at the configured values");
        assert_eq!(steps_text(&adapted.steps), "", "the event's `adapted` is empty");
    }

    /// 100 MB short of 917,504, 851,968 fits but its pool cannot hold a
    /// second 65,536-token reservation beside it: the load runs 786,432, as
    /// the 5090 did by hand with a 3.3 GB desktop (release 0.7.3).
    #[test]
    fn a_5090_plan_100_mb_short_keeps_the_largest_context_with_a_second_reservation() {
        let start = at(917_504, 8);
        let card = RTX5090.with_budget(RTX5090.budget_holding(start, 917_504 + 16 * 64) - 100_000_000);
        let (result, attempts) = card.adapt(start, Adaptable::ALL);
        let adapted = kept(result);
        assert_eq!(adapted.attempt, at(786_432, 8));
        assert_eq!(adapted.steps, [context_step(917_504, 786_432)]);
        assert_eq!(steps_text(&adapted.steps), "max_context 917504 -> 786432");
        assert!(adapted.plan.kv_token_capacity() >= 786_432 + 65_536);
        assert_eq!(attempts.iter().map(|a| a.max_context).collect::<Vec<_>>(), [917_504, 851_968, 786_432]);
    }

    /// The 16 GB profile's ladder: each budget keeps one step lower, never
    /// touching the lanes while a context at two lanes holds a second
    /// reservation.
    #[test]
    fn the_16gb_ladder_steps_262144_196608_131072_65536() {
        let start = at(262_144, 2);
        let fits = GB16.with_budget(GB16.budget_holding(start, 262_144 + 4 * 64));
        assert!(kept(fits.adapt(start, Adaptable::ALL).0).steps.is_empty(), "262,144 fits as configured");
        for context in [196_608, 131_072, 65_536] {
            let card = GB16.with_budget(GB16.budget_holding(at(context, 2), context + 65_536));
            let adapted = kept(card.adapt(start, Adaptable::ALL).0);
            assert_eq!(adapted.attempt, at(context, 2), "{context}");
            assert_eq!(adapted.steps, [context_step(262_144, context)]);
        }
    }

    /// Decision 6: a context the plan holds at two lanes without a second
    /// reservation is passed over; the lanes drop to one and the ladder runs
    /// again from the profile's value, the one-lane criterion being the
    /// plan's own floor.
    #[test]
    fn lanes_drop_to_one_then_the_ladder_runs_again_from_the_profiles_value() {
        let start = at(262_144, 2);
        // A pool one page short of 65,536 + 65,536 at two lanes: no context
        // there keeps a second reservation. A lane's bytes back hold 196,608.
        let card = GB16.with_budget(GB16.budget_holding(at(65_536, 2), 131_072) - 1);
        let (result, attempts) = card.adapt(start, Adaptable::ALL);
        let adapted = kept(result);
        let one_lane = adapted.attempt;
        assert_eq!(one_lane.decode_lanes, 1);
        assert_eq!(adapted.steps[0], Step::DecodeLanes { from: 2, to: 1 });
        assert!(card.plan(one_lane).is_ok());
        assert!(
            one_lane.max_context == 262_144 || card.plan(Attempt { max_context: one_lane.max_context + 65_536, ..one_lane }).is_err(),
            "the largest context one lane holds: {one_lane:?}"
        );
        let two_lane_contexts: Vec<u32> = attempts.iter().filter(|a| a.decode_lanes == 2).map(|a| a.max_context).collect();
        assert_eq!(two_lane_contexts, [262_144, 196_608, 131_072, 65_536], "every two-lane context first");
        assert_eq!(attempts.iter().find(|a| a.decode_lanes == 1).unwrap().max_context, 262_144, "then from the top");

        // A costlier lane, whose bytes hold 196,608 at one lane where no
        // context fits two: both steps, the lanes' first.
        let card = Card { lane: 2_000_000_000, ..GB16 };
        let card = card.with_budget(card.budget_holding(at(196_608, 1), 196_608 + 2 * 64));
        let adapted = kept(card.adapt(start, Adaptable::ALL).0);
        assert_eq!(adapted.attempt, at(196_608, 1));
        assert_eq!(adapted.steps, [Step::DecodeLanes { from: 2, to: 1 }, context_step(262_144, 196_608)]);
        assert_eq!(steps_text(&adapted.steps), "decode_lanes 2 -> 1, max_context 262144 -> 196608");
    }

    /// Step 1 (decision 2): a profile's `shortlist` falls to `full` first,
    /// and a plan that fits then adapts nothing else.
    #[test]
    fn a_profiles_shortlist_head_falls_to_full_before_anything_else() {
        let start = Attempt { draft_head: Some(ProposalHead::Shortlist), ..at(262_144, 2) };
        let card = GB16.with_budget(GB16.budget_holding(at(262_144, 2), 262_144 + 4 * 64));
        assert!(card.plan(start).is_err(), "the shortlist's bytes do not fit");
        let (result, attempts) = card.adapt(start, Adaptable::ALL);
        let adapted = kept(result);
        assert_eq!(adapted.attempt, at(262_144, 2));
        assert_eq!(adapted.steps, [Step::DraftHead { from: ProposalHead::Shortlist, to: ProposalHead::Full }]);
        assert_eq!(steps_text(&adapted.steps), "draft_head shortlist -> full");
        assert_eq!(attempts, [start, at(262_144, 2)]);
    }

    /// An explicit value never adapts: each of the three, alone, keeps the
    /// value it was given in every plan the ladder makes.
    #[test]
    fn an_explicit_value_never_adapts() {
        let start = Attempt { draft_head: Some(ProposalHead::Shortlist), ..at(262_144, 2) };
        let card = GB16.with_budget(GB16.budget_holding(at(65_536, 1), 65_536 + 2 * 64));

        let (_, attempts) = card.adapt(start, Adaptable { draft_head: false, ..Adaptable::ALL });
        assert!(attempts.iter().all(|a| a.draft_head == Some(ProposalHead::Shortlist)), "{attempts:?}");

        let (result, attempts) = card.adapt(start, Adaptable { max_context: false, ..Adaptable::ALL });
        assert!(attempts.iter().all(|a| a.max_context == 262_144), "{attempts:?}");
        assert!(matches!(result, Err(AdaptError::Refused(_))), "262,144 fits on no lane count here");

        let (result, attempts) = card.adapt(start, Adaptable { decode_lanes: false, ..Adaptable::ALL });
        assert!(attempts.iter().all(|a| a.decode_lanes == 2), "{attempts:?}");
        assert!(matches!(result, Err(AdaptError::Refused(_))), "65,536 at one lane is all this card holds");
    }

    /// Lanes that cannot drop leave the largest context the plan holds, and
    /// the step says the second reservation is not there. A pool that holds
    /// a larger context and its floor holds 65,536 and a second reservation
    /// too, so that largest context is the floor's.
    #[test]
    fn explicit_lanes_keep_the_largest_context_the_plan_holds_and_say_so() {
        let start = at(262_144, 2);
        let card = GB16.with_budget(GB16.budget_holding(at(65_536, 2), 65_536 + 4 * 64));
        let adapted = kept(card.adapt(start, Adaptable { decode_lanes: false, ..Adaptable::ALL }).0);
        assert_eq!(adapted.attempt, at(65_536, 2));
        assert_eq!(adapted.steps, [Step::MaxContext { from: 262_144, to: 65_536, second_reservation: false }]);
        assert!(steps_text(&adapted.steps).starts_with("max_context 262144 -> 65536 (one sequence only"));
        // At one lane already, the floor is the criterion: nothing unmet.
        let card = GB16.with_budget(GB16.budget_holding(at(131_072, 1), 131_072 + 2 * 64));
        let adapted = kept(card.adapt(at(262_144, 1), Adaptable::ALL).0);
        assert_eq!(adapted.steps, [context_step(262_144, 131_072)]);
    }

    /// Decision 7: when not even 65,536 at one lane fits, the start is
    /// refused with that last attempt's deficit and every adaptation tried.
    #[test]
    fn a_budget_below_the_last_step_is_refused_with_its_deficit_and_the_steps_tried() {
        let start = Attempt { draft_head: Some(ProposalHead::Shortlist), ..at(262_144, 2) };
        let card = GB16.with_budget(GB16.budget_holding(at(65_536, 1), 65_536 + 2 * 64) - 1);
        let refused = refused(card.adapt(start, Adaptable::ALL).0);
        let last = card.plan(at(65_536, 1)).expect_err("one byte short");
        assert_eq!(refused.error, last, "the deficit of the last, smallest attempt");
        assert!(matches!(refused.error, VramPlanError::BelowMinimum { max_context_tokens: 65_536, .. }));
        let message = refused.to_string();
        assert!(message.starts_with(&last.to_string()), "{message}");
        assert!(
            message.ends_with(
                "; adaptation tried, in order: draft_head shortlist -> full, \
                 max_context 262144 -> 196608 -> 131072 -> 65536, decode_lanes 2 -> 1, \
                 max_context 262144 -> 196608 -> 131072 -> 65536"
            ),
            "{message}"
        );
    }

    /// Nothing adaptable: the refusal is the plan's own, word for word, as
    /// before adaptation existed.
    #[test]
    fn with_nothing_adaptable_the_refusal_is_the_plans_own() {
        let start = at(262_144, 2);
        let card = GB16.with_budget(GB16.budget_holding(start, 262_144));
        let (result, attempts) = card.adapt(start, Adaptable::default());
        let refused = refused(result);
        assert_eq!(attempts, [start]);
        assert!(refused.tried.is_empty());
        assert_eq!(refused.to_string(), card.plan(start).unwrap_err().to_string());
    }

    /// Decision 5: only a plan refused below its minimum adapts -- a budget
    /// above free memory is refused at once, nothing tried.
    #[test]
    fn a_refusal_other_than_the_minimum_is_not_adapted() {
        let start = at(262_144, 2);
        let mut attempts = 0;
        let result = adapt(start, Adaptable::ALL, |_| {
            attempts += 1;
            Ok::<Planned<()>, Infallible>(Planned::Refused(VramPlanError::BudgetAboveFree { budget_bytes: 2, free_bytes: 1 }))
        });
        let Err(AdaptError::Refused(refused)) = result else { panic!("refused") };
        assert_eq!((attempts, refused.tried.len()), (1, 0));
        assert!(matches!(refused.error, VramPlanError::BudgetAboveFree { .. }));
    }

    /// The closure's own failure ends the ladder where it happened.
    #[test]
    fn a_planning_failure_ends_the_ladder() {
        let result = adapt(at(262_144, 2), Adaptable::ALL, |attempt| {
            if attempt.max_context < 262_144 {
                return Err("the leaf cannot plan that");
            }
            Ok::<Planned<()>, _>(Planned::Refused(VramPlanError::BelowMinimum {
                mode: VramMode::Derived { headroom_bytes: 0 },
                policy: ignis_core::KvPoolPolicy::Resident,
                budget_bytes: 1,
                needed_bytes: 2,
                max_context_tokens: attempt.max_context,
                kv_pool_named: false,
                embedding_pool_named: false,
            }))
        });
        assert!(matches!(result, Err(AdaptError::Failed("the leaf cannot plan that"))));
    }
}
