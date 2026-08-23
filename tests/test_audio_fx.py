"""Synthetic speech-shaped audio, so the repair logic can be tested on any
machine — no GPU, no model, no reference recordings.

"Speech" here is a train of tone bursts separated by short gaps: bursts are
words, longer gaps are sentence breaks. That is enough structure to prove the
one thing that mattered most in production — that a reference is never cut in
the middle of a burst.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import audio_fx

SR = 24000


def tone(seconds, freq=180.0, amp=0.3):
    t = np.arange(int(seconds * SR)) / SR
    env = np.hanning(max(2, len(t)))          # avoid clicks at burst edges
    return (amp * np.sin(2 * np.pi * freq * t) * env).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SR), dtype=np.float32)


def speech(pattern):
    """pattern: list of ('w', dur) words and ('s', dur) silences."""
    return np.concatenate([tone(d) if kind == "w" else silence(d)
                           for kind, d in pattern]).astype(np.float32)


def cut_word(seconds):
    """A word the recorder stopped in the middle of: ends at full volume."""
    return tone(seconds * 2)[: int(seconds * SR)]


def half_word_start(seconds):
    """A word the recording started in the middle of: begins at full volume."""
    return tone(seconds * 2)[int(seconds * SR):]


def sentence(n_words=5, word=0.35, gap=0.08):
    out = []
    for i in range(n_words):
        out.append(("w", word))
        if i < n_words - 1:
            out.append(("s", gap))
    return out


# --- the reference fix ----------------------------------------------------
def test_long_reference_is_cut_at_a_pause_not_mid_word():
    """The bug this whole module exists for: a 60s reference hard-cut at
    exactly 10.0s landed inside a word, and that half word was then spoken at
    the end of 86-90% of every clip made with that voice."""
    pattern = []
    for _ in range(12):                        # ~24s of speech
        pattern += sentence(4)
        pattern += [("s", 0.35)]               # sentence pause
    x = speech(pattern)
    assert len(x) / SR > 15

    y, info = audio_fx.smart_trim_reference(x, SR, max_sec=10.0,
                                            tail_silence_sec=0.3)
    assert info["trimmed"] is True
    assert info["cut_at_pause"] is True, info
    assert not info["ends_mid_word"]

    # The kept audio ends in real silence, not at full volume.
    assert not audio_fx.ends_abruptly(y, SR)
    tail = y[-int(0.25 * SR):]
    assert float(np.max(np.abs(tail))) < 1e-3, "no trailing silence was added"

    # And it did not throw away most of the allowance to find that pause.
    assert 5.0 < len(y) / SR <= 10.5, info


def test_a_reference_with_no_pause_is_flagged_loudly():
    x = tone(20.0)                              # one unbroken 20s sound
    y, info = audio_fx.smart_trim_reference(x, SR, max_sec=10.0)
    assert info["trimmed"] and not info["cut_at_pause"]
    assert info["ends_mid_word"] is True
    assert any("mid-phrase" in w for w in info["warnings"]), info["warnings"]


def test_a_users_own_clip_that_stops_mid_word_is_cut_back():
    """The second way a reference ends mid-word: nothing trimmed it, the
    customer just stopped recording while still talking. It does exactly the
    same damage, so it gets exactly the same repair."""
    body = np.concatenate([speech(sentence(5)), silence(0.30),
                           speech(sentence(5)), silence(0.30),
                           speech(sentence(5))])
    x = np.concatenate([body, silence(0.30), cut_word(0.25)])
    assert audio_fx.ends_abruptly(x, SR), "the fixture is not mid-word"

    y, info = audio_fx.smart_trim_reference(x, SR, max_sec=30.0)
    assert info["repaired_mid_word"] is True, info
    assert info["cut_at_pause"] is True
    assert any("still speaking" in w for w in info["warnings"]), info["warnings"]
    # The half word is gone, the clip ends in silence, and it is still usable.
    assert not audio_fx.ends_abruptly(y, SR)
    assert len(y) / SR >= 3.0, info
    assert len(y) < len(x)


def test_a_clip_too_short_to_cut_back_is_kept_and_explained():
    """Refusing is right when cutting back would leave a 1s reference — but the
    customer has to be told what to do instead."""
    x = np.concatenate([tone(0.6), silence(0.25), cut_word(1.5)])
    y, info = audio_fx.smart_trim_reference(x, SR, max_sec=30.0,
                                            min_keep_sec=3.0)
    assert info["ends_mid_word"] is True
    assert info["repaired_mid_word"] is False
    assert any("re-record" in w for w in info["warnings"]), info["warnings"]
    assert len(y) > 0


def test_a_clip_that_starts_mid_word_loses_the_partial_opening():
    x = np.concatenate([half_word_start(0.25), silence(0.25),
                        speech(sentence(8)), silence(0.5)])
    _y, info = audio_fx.smart_trim_reference(x, SR, max_sec=30.0)
    assert info["trimmed_lead_sec"] > 0, info


def test_a_long_clip_prefers_a_shorter_cut_to_a_mid_word_one():
    """Only one usable pause, and nothing else inside the search window."""
    x = np.concatenate([tone(4.0), silence(0.30), tone(20.0)])
    y, info = audio_fx.smart_trim_reference(x, SR, max_sec=10.0, hard_max_sec=15.0)
    assert info["cut_at_pause"] is True, info
    assert info["ends_mid_word"] is False
    assert 3.0 <= len(y) / SR <= 5.5, info["cut_sec"]
    assert not audio_fx.ends_abruptly(y, SR)


# --- the target is a target, not a wall -----------------------------------
def test_a_sentence_finishing_just_past_the_target_is_kept():
    """The limit is 10s, the sentence finishes at ~11s, and the pause before it
    is at 4s. Backing off to 4s to respect a round number would cost far more
    than the extra second."""
    x = np.concatenate([tone(4.0), silence(0.30), tone(6.6), silence(0.40),
                        tone(8.0)])
    y, info = audio_fx.smart_trim_reference(x, SR, max_sec=10.0, hard_max_sec=15.0)
    kept = len(y) / SR
    assert info["cut_at_pause"] is True, info
    assert 10.0 < kept <= 15.0, f"kept {kept:.2f}s -- it backed off instead"
    assert info.get("overshoot_sec", 0) > 0, info
    assert not audio_fx.ends_abruptly(y, SR)


def test_a_pause_just_under_the_target_still_wins():
    """Overshoot is allowed, not preferred: 9.5s beats 13s."""
    x = np.concatenate([tone(9.2), silence(0.35), tone(3.2), silence(0.35),
                        tone(6.0)])
    y, info = audio_fx.smart_trim_reference(x, SR, max_sec=10.0, hard_max_sec=15.0)
    kept = len(y) / SR
    # The 9.2s pause wins over the 12.8s one. (The tone envelope fades, so the
    # detected speech-stop sits a little before the nominal boundary.)
    assert 8.0 <= kept <= 10.4, f"kept {kept:.2f}s"
    assert not info.get("overshoot_sec")


def test_nothing_is_ever_kept_past_the_hard_ceiling():
    x = np.concatenate([tone(13.5), silence(0.4), tone(6.0)])
    y, info = audio_fx.smart_trim_reference(x, SR, max_sec=10.0, hard_max_sec=15.0)
    assert len(y) / SR <= 15.0 + 0.4, info
    # No pause at all inside the window -> hard cut, and it says so.
    z = tone(30.0)
    _y2, i2 = audio_fx.smart_trim_reference(z, SR, max_sec=10.0, hard_max_sec=15.0)
    assert i2["ends_mid_word"] is True
    assert any("no pause found anywhere" in w for w in i2["warnings"]), i2["warnings"]


def test_trim_always_appends_the_configured_tail_silence():
    x = speech(sentence(5) + [("s", 0.9)])
    y, _ = audio_fx.smart_trim_reference(x, SR, max_sec=30.0,
                                         tail_silence_sec=0.30)
    assert abs(audio_fx.trailing_silence_sec(y, SR) - 0.30) < 0.12


# --- loudness -------------------------------------------------------------
def test_two_very_different_recordings_end_up_at_the_same_loudness():
    """One measured batch came back between -0.2 dB and -12.6 dB, so the
    narrator's volume jumped between segments of the same video."""
    # The spread actually measured in production: references at -16.1, -21.2
    # and -25.6 dBFS RMS, whose outputs then ran from -0.2 dB to -12.6 dB.
    quiet = speech(sentence(8)) * 0.3
    loud = speech(sentence(8)) * 0.9
    a, ia = audio_fx.normalize_loudness(quiet, SR, target_lufs=-20.0)
    b, ib = audio_fx.normalize_loudness(loud, SR, target_lufs=-20.0)
    assert abs(ia["out_lufs"] - ib["out_lufs"]) < 1.0, (ia, ib)
    assert abs(ia["out_lufs"] - (-20.0)) < 1.5, ia


def test_peak_ceiling_is_respected():
    x = speech(sentence(6)) * 0.95
    y, info = audio_fx.normalize_loudness(x, SR, target_lufs=-6.0,
                                          peak_ceiling_db=-1.0)
    assert audio_fx.true_peak_db(y) <= -1.0 + 0.15, info
    assert float(np.max(np.abs(y))) <= 1.0


def test_a_near_silent_recording_is_not_amplified_into_hiss():
    tiny = speech(sentence(8)) * 0.01           # ~40 dB below target
    _y, info = audio_fx.normalize_loudness(tiny, SR, target_lufs=-20.0)
    assert info["limited_by"] == "gain_cap"
    assert info["met_target"] is False


def test_silence_does_not_get_amplified_into_noise():
    y, info = audio_fx.normalize_loudness(silence(1.0), SR)
    assert info["gain_db"] == 0.0
    assert info["limited_by"] == "silence"
    assert float(np.max(np.abs(y))) == 0.0


def test_lufs_moves_with_level():
    x = speech(sentence(6))
    assert audio_fx.lufs(x * 0.5, SR) < audio_fx.lufs(x, SR)
    assert abs((audio_fx.lufs(x, SR) - audio_fx.lufs(x * 0.5, SR)) - 6.0) < 0.6


# --- output repair --------------------------------------------------------
def test_ends_abruptly_is_the_right_way_round():
    """The naive version called a clip truncated *because* it ended in silence,
    which is exactly what a clip that finished properly does."""
    finished = speech(sentence(5) + [("s", 0.5)])
    assert audio_fx.ends_abruptly(finished, SR) is False
    cut_off = speech(sentence(5))[: -int(0.05 * SR)]
    assert audio_fx.ends_abruptly(cut_off, SR) is True


def test_a_leaked_tail_is_removed_and_the_speech_is_kept():
    body = speech(sentence(6))
    clip = np.concatenate([body, silence(0.30), tone(0.9)])   # bleed at the end
    fixed, removed = audio_fx.remove_tail_after_gap(clip, SR)
    assert removed > 0.5, removed
    assert len(fixed) < len(clip)
    # The real speech survived.
    assert len(fixed) >= len(body) * 0.95
    assert not audio_fx.ends_abruptly(fixed, SR)


def test_nothing_is_removed_when_there_is_no_gap():
    clip = speech(sentence(6))
    fixed, removed = audio_fx.remove_tail_after_gap(clip, SR, min_gap_ms=250)
    assert removed == 0.0
    assert len(fixed) == len(clip)


# --- reference report -----------------------------------------------------
def test_analyze_reference_flags_a_short_clip():
    rep = audio_fx.analyze_reference(speech(sentence(2, word=0.3)), SR)
    assert rep["duration_sec"] < 3
    assert any("under 3s" in w or "clones poorly" in w for w in rep["warnings"])
    assert rep["quality_score"] < 1.0


def test_analyze_reference_flags_clipping():
    x = np.clip(speech(sentence(8)) * 12.0, -1.0, 1.0)
    rep = audio_fx.analyze_reference(x, SR)
    assert rep["clipping_ratio"] > 0.001
    assert any("clipping" in w for w in rep["warnings"])


def test_analyze_reference_flags_noise():
    rng = np.random.default_rng(0)
    noisy = speech(sentence(8)) + rng.normal(0, 0.05, len(speech(sentence(8)))).astype(np.float32)
    rep = audio_fx.analyze_reference(noisy, SR)
    assert rep["snr_db"] < 30


def test_a_good_reference_scores_clean():
    x = speech(sentence(6) + [("s", 0.4)] + sentence(6) + [("s", 0.5)])
    rep = audio_fx.analyze_reference(x, SR)
    assert rep["quality_score"] >= 0.9, rep
    assert rep["warnings"] == [], rep["warnings"]


# --- basics ---------------------------------------------------------------
def test_int16_roundtrip():
    x = speech(sentence(3))
    back = audio_fx.to_mono_float(audio_fx.to_int16(x))
    assert np.max(np.abs(back - x)) < 1e-3


def test_wpm():
    assert audio_fx.wpm(150, 60.0) == 150.0
    assert audio_fx.wpm(10, 0.0) == 0.0


def test_empty_audio_is_survivable():
    empty = np.zeros(0, dtype=np.float32)
    y, info = audio_fx.smart_trim_reference(empty, SR)
    assert len(y) == 0 and info["warnings"]
    assert audio_fx.trailing_silence_sec(empty, SR) == 0.0
    assert audio_fx.ends_abruptly(empty, SR) is False


# --- joining chunks without an audible seam --------------------------------
def test_chunks_at_different_levels_are_matched_not_normalized():
    """XTTS's documented long-form flaw is per-chunk normalization to a fixed
    target: it flattens dynamics and still jumps at boundaries. Pull outliers
    toward the clip's own median instead."""
    a = speech(sentence(5))
    b = speech(sentence(5)) * 0.25          # ~12 dB quieter
    c = speech(sentence(5))
    matched, corrections = audio_fx.match_levels([a, b, c], SR)
    assert corrections[0] == 0.0 and corrections[2] == 0.0, corrections
    assert corrections[1] > 0, "the quiet chunk was not lifted"
    levels = [audio_fx.speech_rms_db(p, SR) for p in matched]
    assert max(levels) - min(levels) < 8.0, levels    # was ~12 dB


def test_level_matching_leaves_natural_variation_alone():
    a = speech(sentence(5))
    b = speech(sentence(5)) * 0.9           # ~1 dB — normal dynamics
    _m, corrections = audio_fx.match_levels([a, b], SR)
    assert corrections == [0.0, 0.0], corrections


def test_join_gives_even_gaps_and_no_hard_edges():
    parts = [speech(sentence(4)),
             np.concatenate([silence(0.6), speech(sentence(4)), silence(0.9)]),
             speech(sentence(4))]
    y, info = audio_fx.join_chunks(parts, SR, gap_sec=0.15, tail_pad_sec=0.30)
    assert info["chunks"] == 3
    # The 0.6s/0.9s of model silence is replaced by the fixed gap, so the
    # result is shorter than a naive concatenation.
    assert len(y) < sum(len(p) for p in parts) + int(0.9 * SR)
    assert info["pad_sec"] == 0.30
    assert abs(audio_fx.trailing_silence_sec(y, SR) - 0.30) < 0.15


def test_join_reports_truncation_but_not_on_every_clip():
    """The check is judged on the raw last chunk. Once trailing silence has
    been stripped every clip ends in speech, so checking the joined audio would
    flag all of them — and a detector that fires on every good clip is worse
    than no detector at all."""
    cut_off = speech(sentence(5))[: -int(0.05 * SR)]
    _y, info = audio_fx.join_chunks([cut_off], SR, tail_pad_sec=0.30)
    assert info["ends_abruptly"] is True, info

    finished = [speech(sentence(4) + [("s", 0.5)]),
                speech(sentence(4) + [("s", 0.5)])]
    _y2, info2 = audio_fx.join_chunks(finished, SR, tail_pad_sec=0.30)
    assert info2["ends_abruptly"] is False, "false alarm on a finished clip"


def test_join_does_not_eat_a_soft_final_consonant():
    """Silence stripping clipping the last sound is a documented way to lose a
    word, so the edge threshold is stricter than the reference one."""
    soft_tail = np.concatenate([speech(sentence(4)), tone(0.25, amp=0.02)])
    y, _info = audio_fx.join_chunks([soft_tail], SR, tail_pad_sec=0.0)
    kept = len(y) / SR
    assert kept >= (len(soft_tail) / SR) - 0.12, "the soft ending was trimmed off"


def test_join_of_a_single_chunk_is_still_padded():
    y, info = audio_fx.join_chunks([speech(sentence(4))], SR, tail_pad_sec=0.30)
    assert info["pad_sec"] == 0.30
    assert not audio_fx.ends_abruptly(y, SR)


def test_join_of_nothing_is_survivable():
    y, info = audio_fx.join_chunks([], SR)
    assert len(y) == 0 and info["chunks"] == 0
