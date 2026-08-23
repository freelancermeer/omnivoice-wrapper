"""The five failure modes the verifier has to tell apart.

Straight from the production correction note: a `w not in sent_words` diff
passes every one of the first four, which is why this file exists.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import verify


def test_repetition_collapse_is_caught():
    sent = "He called it perjury. He called it fraud. He called it contempt."
    spoken = "He called it perjury, fraud, contempt."
    d = verify.word_diff(sent, spoken)
    assert d["dropped"], "the collapsed clause was not reported as dropped"
    assert d["word_accuracy"] < 0.75
    assert not verify.passed(d)


def test_membership_diff_would_have_passed_this():
    """Guards the actual regression: every word still appears somewhere."""
    sent = "He called it perjury. He called it fraud. He called it contempt."
    spoken = "He called it perjury, fraud, contempt."
    naive_dropped = [w for w in verify.words(sent)
                     if w not in set(verify.words(spoken))]
    assert naive_dropped == []          # the old check sees nothing
    assert verify.word_diff(sent, spoken)["dropped"]  # this one does


def test_clean_clip_passes():
    sent = "The judge ordered immediate seizure of the assets."
    d = verify.word_diff(sent, "the judge ordered immediate seizure of the assets")
    assert d["dropped"] == [] and d["inserted"] == []
    assert d["word_accuracy"] == 1.0
    assert verify.passed(d)


def test_reference_bleed_is_caught_even_at_full_accuracy():
    sent = "Donald Trump really is."
    spoken = "donald trump really is forcing him to back down"
    d = verify.word_diff(sent, spoken)
    assert d["word_accuracy"] == 1.0, "everything sent was said"
    assert d["hard_inserted"], "the extra words were not reported"
    assert d["tail_inserted"] == ["forcing", "him", "to", "back", "down"]
    assert not verify.passed(d)
    assert verify.only_tail_is_wrong(d), "this is the repairable shape"


def test_truncation_is_caught():
    sent = "No immunity claim left, no procedural delays, no appeals."
    d = verify.word_diff(sent, "no immunity claim left")
    assert "appeals" in d["dropped"]
    assert d["word_accuracy"] < 0.7
    assert not verify.passed(d)


def test_model_loop_is_caught():
    sent = "The judge ordered a seizure."
    d = verify.word_diff(sent, "the judge ordered a seizure seizure seizure")
    assert d["word_accuracy"] == 1.0
    assert d["hard_inserted"] == ["seizure", "seizure"]
    assert not verify.passed(d)


def test_tail_repair_is_not_claimed_when_words_are_missing():
    d = verify.word_diff("one two three four", "one two four extra words")
    assert not verify.only_tail_is_wrong(d)


def test_number_words_and_hyphens_match_asr():
    d = verify.word_diff("twenty-first", "twenty first")
    assert verify.passed(d)


def test_baseline_wpm():
    assert verify.baseline_wpm("one two three four five six", 3.0) == 120.0
    assert verify.baseline_wpm("", 3.0) is None


def test_rate_is_judged_against_the_voice_itself():
    # A 100 wpm documentary narrator must not be failed by a fixed 140-180 band.
    assert verify.rate_warning(104, baseline=100) is None
    assert verify.rate_warning(150, baseline=100) is not None   # 50% fast for it
    # An energetic 210 wpm voice is fine at 210 and flagged when it collapses.
    assert verify.rate_warning(210, baseline=210) is None
    assert verify.rate_warning(140, baseline=210) is not None
    # 175 vs a 210 baseline is a 17% drift — inside a voice's normal spread
    # (one real voice ranged 160-203 wpm), so it deliberately does not warn.
    assert verify.rate_warning(175, baseline=210) is None
    # No baseline at all: only a catastrophe warns.
    assert verify.rate_warning(175, baseline=None) is None
    assert verify.rate_warning(300, baseline=None) is not None


def test_rate_band_is_tunable():
    assert verify.rate_warning(175, baseline=210, low=0.90) is not None


def test_reference_transcript_mismatch_is_detected():
    m = verify.reference_matches_audio(
        "they care about returning favors and",
        "his own handpicked judges are torching his executive orders")
    assert not m["matches"]
    ok = verify.reference_matches_audio("hello there friend", "hello there friend")
    assert ok["matches"]
