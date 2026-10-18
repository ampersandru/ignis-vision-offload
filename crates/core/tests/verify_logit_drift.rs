//! The rules the Flash-Next GPU test holds a verify budget past 8 rows to
//! (`support/logit_drift.rs`, GitHub #329 item 5, ADR 0056), on synthetic
//! BF16 rows: no GPU.

#[path = "support/logit_drift.rs"]
mod logit_drift;

use logit_drift::{
    acceptance_within_band, argmax, bf16, bf16_ulp, commit_follows_argmax, log_prob, row_drift, within, LaneCommit,
    ACCEPTANCE_BAND, HEAD_ACCEPTANCE_FLOOR, MEAN_LOG_PROB_NATS, PICK_GAP_ULPS, TOLERANCE_ULPS,
};

fn row(values: &[f32]) -> Vec<u16> {
    values
        .iter()
        .map(|&v| {
            let bits = (v.to_bits() >> 16) as u16;
            assert_eq!(bf16(bits), v, "{v} is not a BF16 value");
            bits
        })
        .collect()
}

/// The engine draws the lowest id among equal maxima; the ulp is BF16's at
/// the value's binade.
#[test]
fn the_pick_is_the_first_maximum_and_the_ulp_is_bf16s() {
    assert_eq!(argmax(&row(&[1.0, 3.0, -2.0, 3.0])), (1, 3.0));
    assert_eq!(argmax(&row(&[-5.0, -4.0])), (1, -4.0));
    assert_eq!(bf16_ulp(1.0), 1.0 / 128.0);
    assert_eq!(bf16_ulp(-20.0), 0.125);
    assert_eq!(bf16_ulp(31.9), 0.125);
    assert_eq!(bf16_ulp(32.0), 0.25);
}

/// The drift is the largest |on - off| in ulps at the plain row's largest
/// magnitude -- a large negative logit sets the scale as a large positive
/// one does -- and the pick gap is the plain row's lead of its pick over
/// the wide row's.
#[test]
fn the_drift_counts_ulps_at_the_rows_scale_and_the_picks_gap() {
    let off = row(&[20.0, 19.75, -40.0, 0.5]);
    let same = row_drift(&off, &off);
    assert_eq!((same.max_abs, same.ulps, same.pick_off, same.pick_on, same.gap_ulps), (0.0, 0.0, 0, 0, 0.0));

    // Scale 40 (ulp 0.25): -40 -> -39.5 is 2 ulps, the pick unmoved.
    let on = row(&[20.0, 19.75, -39.5, 0.5]);
    let d = row_drift(&off, &on);
    assert_eq!((d.max_abs, d.ulps, d.worst, d.pick_on, d.gap_ulps), (0.5, 2.0, 2, 0, 0.0));

    // The wide row picks id 1, which the plain row rates 0.25 (1 ulp) below.
    let on = row(&[19.75, 20.0, -40.0, 0.5]);
    let d = row_drift(&off, &on);
    assert_eq!((d.max_abs, d.ulps, d.pick_off, d.pick_on, d.gap_ulps), (0.25, 1.0, 0, 1, 1.0));
}

/// Within the tolerance every logit stays inside it and a pick moves only
/// where the plain row calls the two picks a tie inside it.
#[test]
fn a_row_passes_inside_the_tolerance_and_names_the_excess_outside_it() {
    let off = row(&[20.0, 19.75, -40.0, 0.5]);
    let near_flip = row_drift(&off, &row(&[19.75, 20.0, -40.0, 0.5]));
    assert_eq!(within(&near_flip, 1.0, 1.0), Ok(()));
    assert!(within(&near_flip, 0.5, 1.0).unwrap_err().contains("a logit moved"));

    // Every logit within 2 ulps (scale 20, ulp 0.125), but the pick moved
    // where the plain row leads by 3: the pick rule refuses it at 2.
    let off = row(&[20.0, 19.625, 0.0]);
    let on = row(&[19.75, 19.875, 0.0]);
    let d = row_drift(&off, &on);
    assert_eq!((d.ulps, d.pick_on, d.gap_ulps), (2.0, 1, 3.0));
    assert_eq!(within(&d, 3.0, 3.0), Ok(()));
    assert!(within(&d, 3.0, 2.0).unwrap_err().contains("the pick moved from 0 to 1"));
}

/// A moved pick is held to its own near-tie bound, not to the logit
/// tolerance: at the policy's 64 ulps a top logit of 20 may move 8 logits,
/// which is no near tie. The plain row leads its pick by 17 ulps here and
/// every logit stays within 17: past [`PICK_GAP_ULPS`], refused.
#[test]
fn a_moved_pick_is_held_to_the_near_tie_bound_not_to_the_logit_tolerance() {
    // Scale 20: one ulp is 0.125, 17 ulps 2.125.
    let off = row(&[20.0, 17.875, 0.0]);
    let on = row(&[17.875, 20.0, 0.0]);
    let d = row_drift(&off, &on);
    assert_eq!((d.ulps, d.pick_off, d.pick_on, d.gap_ulps), (17.0, 0, 1, 17.0));
    assert!(d.ulps <= TOLERANCE_ULPS, "every logit inside the tolerance");
    let err = within(&d, TOLERANCE_ULPS, PICK_GAP_ULPS).unwrap_err();
    assert!(err.contains("the pick moved from 0 to 1, 17 ulps apart"), "{err}");
    // At the bound itself the move is a near tie.
    let off = row(&[20.0, 18.0, 0.0]);
    let d = row_drift(&off, &row(&[18.0, 20.0, 0.0]));
    assert_eq!(d.gap_ulps, PICK_GAP_ULPS);
    assert_eq!(within(&d, TOLERANCE_ULPS, PICK_GAP_ULPS), Ok(()));
}

/// The head's first-draft acceptance at a wide budget stays within the band
/// of the same prompts' at 8 rows, either way; no first draft offered is no
/// acceptance to compare.
#[test]
fn the_wide_budgets_acceptance_stays_in_the_band_around_the_defaults() {
    assert_eq!(acceptance_within_band((80, 100), (85, 100), ACCEPTANCE_BAND), Ok(()));
    assert_eq!(acceptance_within_band((90, 100), (85, 100), ACCEPTANCE_BAND), Ok(()));
    let err = acceptance_within_band((70, 100), (85, 100), ACCEPTANCE_BAND).unwrap_err();
    assert!(err.contains("0.700") && err.contains("0.850"), "{err}");
    assert!(acceptance_within_band((99, 100), (85, 100), ACCEPTANCE_BAND).is_err(), "two-sided");
    assert!(acceptance_within_band((0, 0), (85, 100), ACCEPTANCE_BAND).unwrap_err().contains("no first draft"));
    assert!(acceptance_within_band((80, 100), (0, 0), ACCEPTANCE_BAND).unwrap_err().contains("no first draft"));
}

/// A log-softmax: uniform rows give -ln n, and the probabilities sum to one.
#[test]
fn the_log_probability_is_a_log_softmax() {
    let uniform = row(&[2.0; 4]);
    assert!((log_prob(&uniform, 3) + 4f64.ln()).abs() < 1e-12);
    let r = row(&[3.0, 1.0, -2.0, 0.5]);
    let total: f64 = (0..4).map(|t| log_prob(&r, t).exp()).sum();
    assert!((total - 1.0).abs() < 1e-12, "{total}");
    assert!(log_prob(&r, 0) > log_prob(&r, 3));
}

/// The commit is the round's own argmax: drafts accepted while each is the
/// pick after the column before it, the anchor and those drafts committed,
/// and the pick after the last committed column pending.
#[test]
fn a_round_commits_its_own_picks_and_nothing_else() {
    let picks = [11, 12, 99, 14];
    let ok = |drafts: &[u32], committed: &[u32], next: Option<u32>| {
        commit_follows_argmax(&LaneCommit { anchor: 10, drafts, committed, next, picks: &picks })
    };
    // Two drafts accepted, the third refused; 99 pending.
    assert_eq!(ok(&[11, 12, 13], &[10, 11, 12], Some(99)), Ok(()));
    // Every draft accepted: the last column's pick pending, or none when the budget is spent.
    assert_eq!(ok(&[11, 12, 99], &[10, 11, 12, 99], Some(14)), Ok(()));
    assert_eq!(ok(&[11, 12, 99], &[10, 11, 12, 99], None), Ok(()));
    // Extent 0: the anchor alone, the first column's pick pending.
    assert_eq!(ok(&[], &[10], Some(11)), Ok(()));
    // A refused draft committed, an accepted one dropped, a wrong successor.
    assert!(ok(&[11, 7, 13], &[10, 11, 7], Some(12)).unwrap_err().contains("committed [10, 11, 7]"));
    assert!(ok(&[11, 12, 13], &[10, 11], Some(12)).unwrap_err().contains("accept [10, 11, 12]"));
    assert!(ok(&[11, 12, 13], &[10, 11, 12], Some(13)).unwrap_err().contains("13 stands pending"));
    // Fewer rows than columns.
    let short = LaneCommit { anchor: 10, drafts: &[11, 12, 99, 14], committed: &[10], next: None, picks: &picks };
    assert!(commit_follows_argmax(&short).unwrap_err().contains("4 rows for 4 drafts"));
}

/// The policy's tolerance admits a row moved by exactly it and refuses one
/// ulp more; the log-probability epsilon is a fraction of a nat and the
/// acceptance floor a share.
#[test]
fn the_policy_admits_its_tolerance_and_refuses_an_ulp_past_it() {
    // Scale 16: one ulp is 0.125.
    let off = row(&[16.0, 8.0, 0.0]);
    let moved = |ulps: f32| row_drift(&off, &row(&[16.0, 8.0 + ulps * 0.125, 0.0]));
    assert_eq!(moved(TOLERANCE_ULPS).ulps, TOLERANCE_ULPS);
    assert_eq!(within(&moved(TOLERANCE_ULPS), TOLERANCE_ULPS, PICK_GAP_ULPS), Ok(()));
    assert!(within(&moved(TOLERANCE_ULPS + 1.0), TOLERANCE_ULPS, PICK_GAP_ULPS).is_err());
    assert!(PICK_GAP_ULPS < TOLERANCE_ULPS, "a moved pick's bound is the tighter one");
    assert!(MEAN_LOG_PROB_NATS > 0.0 && MEAN_LOG_PROB_NATS < 0.1, "{MEAN_LOG_PROB_NATS}");
    assert!(HEAD_ACCEPTANCE_FLOOR > 0.0 && HEAD_ACCEPTANCE_FLOOR < 1.0, "{HEAD_ACCEPTANCE_FLOOR}");
    assert!(ACCEPTANCE_BAND > 0.0 && ACCEPTANCE_BAND < HEAD_ACCEPTANCE_FLOOR, "{ACCEPTANCE_BAND}");
}
