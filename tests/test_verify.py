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


# --- a re-generation costs as much as the original, so spend them wisely ----
def test_a_misheard_proper_noun_does_not_trigger_a_regeneration():
    """The batch saw "Bessent" transcribed as "bessant" five times and
    "Hegseth" as "hexeth" twice. Regenerating those fixes nothing and is judged
    by the same fallible transcriber."""
    d = verify.word_diff("Bessent and Hegseth testified.",
                         "bessant and hexeth testified")
    assert d["missing"] == [] and d["extra"] == []
    assert verify.clean(d)
    assert not verify.worth_regenerating(d)
    assert d["misheard"], "the substitution was not recorded at all"
    assert "check pronunciation" in (verify.pronunciation_note(d) or "")
    # It is still reported as a difference, per the documented diff contract.
    assert d["hard_dropped"] and not verify.passed(d)


def test_a_wrong_number_is_never_written_off_as_a_mishearing():
    d = verify.word_diff("He owes five hundred million dollars.",
                         "he owes five hundred billion dollars")
    assert "million" in d["missing"]
    assert verify.worth_regenerating(d)
    assert not verify.clean(d)


def test_a_dropped_negation_is_never_written_off():
    d = verify.word_diff("He did not know.", "he did now know")
    assert verify.worth_regenerating(d), d


def test_a_genuinely_dropped_word_still_regenerates():
    d = verify.word_diff("He called it perjury. He called it fraud.",
                         "he called it perjury")
    assert verify.worth_regenerating(d)


def test_a_leaked_tail_is_repaired_not_regenerated():
    d = verify.word_diff("Donald Trump really is.",
                         "donald trump really is forcing him to back down")
    assert verify.worth_regenerating(d)          # something IS wrong
    assert verify.only_tail_is_wrong(d)          # but trimming fixes it


def test_a_totally_garbled_clip_is_still_caught():
    """If almost every word came back as a near-match, the clip may really be
    mush rather than merely misheard."""
    d = verify.word_diff("alpha bravo charlie delta",
                         "alpna bravc charlei delfa")
    assert verify.worth_regenerating(d), d


def test_clean_clip_needs_nothing():
    d = verify.word_diff("The judge ordered a seizure.",
                         "the judge ordered a seizure")
    assert verify.clean(d) and not verify.worth_regenerating(d)
    assert verify.pronunciation_note(d) is None


# --- numbers written as digits by the transcriber -------------------------
# Measured on the GPU box: a 240-word clip warned "not spoken: ninety twenty;
# extra: 90 20" and paid for a regeneration, because the script said "ninety
# pages" and Whisper wrote "90 pages". Same number, different notation.

def test_digits_heard_for_number_words_are_not_a_drop():
    d = verify.word_diff("By noon the transcript ran to ninety pages",
                         "by noon the transcript ran to 90 pages")
    assert verify.clean(d), d
    assert not verify.worth_regenerating(d)


def test_multi_word_numbers_survive_the_notation_change():
    d = verify.word_diff("A property worth one hundred fifty million dollars",
                         "a property worth 150 million dollars")
    assert verify.clean(d), d


def test_every_notation_whisper_uses_for_a_number():
    """Measured on the GPU box: each of these cost a warning and a wasted
    regeneration, because the transcriber spells numbers its own way."""
    for sent, heard in [
        ("worth five hundred million dollars", "worth $500 million"),
        ("from eleven thousand square feet", "from 11,000 square feet"),
        ("by two hundred to three hundred percent", "by 200 to 300%"),
        ("owe four point seven million dollars", "owe $4.7 million"),
        ("the January sixth speech", "the January 6th speech"),
        ("on the first of March", "on the 1st of March"),
    ]:
        d = verify.word_diff(sent, heard)
        assert verify.clean(d), (sent, heard, d)


def test_years_use_the_same_reading_as_the_script():
    """2026 is "twenty twenty-six" here, not "two thousand twenty-six"."""
    d = verify.word_diff("It happened in twenty twenty-six",
                         "it happened in 2026")
    assert verify.clean(d), d


def test_a_different_number_is_still_a_real_error():
    """The notation fix must not become a licence to mishear the value."""
    d = verify.word_diff("he owes ninety million", "he owes 90 billion")
    assert verify.worth_regenerating(d), d
    assert "billion" in d["hard_inserted"]


def test_a_number_that_was_genuinely_dropped_is_still_caught():
    d = verify.word_diff("twenty senators voted", "senators voted")
    assert "twenty" in d["hard_dropped"], d


# --- clock times, which Whisper writes with a full stop -------------------
# Measured on the GPU box: "7:45" came back as "7.45", read as a decimal, and
# scored three dropped words and seven inserted ones on a clip where the model
# had said "seven forty-five" perfectly.

def test_a_clock_time_written_with_a_dot_is_not_a_drop():
    d = verify.word_diff(
        "The train leaves at seven forty-five and arrives at ten twenty",
        "the train leaves at 7.45 and arrives at 10.20")
    assert verify.clean(d), d
    assert not verify.worth_regenerating(d)


def test_a_real_decimal_is_never_read_as_a_clock_time():
    """The charitable reading is only accepted when it improves the match."""
    for sent, heard in [
        ("his sons owe four point two five million", "his sons owe 4.25 million"),
        ("pi is three point one four", "pi is 3.14"),
    ]:
        d = verify.word_diff(sent, heard)
        assert verify.clean(d), (sent, heard, d)


def test_the_clock_reading_cannot_hide_a_genuine_error():
    d = verify.word_diff("the meeting at seven forty-five was cancelled",
                         "the meeting at 7.45 was")
    assert "cancelled" in d["hard_dropped"], d


# --- a currency symbol the transcriber inferred ---------------------------
# Measured on the GPU box, and confirmed independently: given "$4.2 million"
# earlier in the sentence, BOTH Whisper and AssemblyAI wrote the following
# bare "3 million" as "$3 million". That expands to a "dollars" the script
# never contained, and it cost a full regeneration.

def test_a_currency_symbol_the_transcriber_added_is_not_an_insertion():
    d = verify.word_diff(
        "Revenue fell from four point two million dollars in the first "
        "quarter to just under three million in the second",
        "Revenue fell from $4.2 million in the first quarter to just under "
        "$3 million in the second")
    assert verify.clean(d), d
    assert not verify.worth_regenerating(d)


def test_a_currency_word_the_model_really_dropped_is_still_caught():
    d = verify.word_diff("he owes four million dollars", "he owes 4 million")
    assert "dollars" in d["hard_dropped"], d
    assert verify.worth_regenerating(d)


def test_one_dropped_currency_word_out_of_two_is_still_caught():
    """The spurious mark may be removed; a real omission may not be hidden."""
    d = verify.word_diff(
        "it cost five hundred dollars and two hundred dollars",
        "it cost $500 and 200")
    assert "dollars" in d["hard_dropped"], d


def test_matching_currency_needs_no_special_reading():
    d = verify.word_diff("it cost five hundred dollars", "it cost $500")
    assert verify.clean(d), d


# --- a bare inflection is the transcriber, not the model ------------------
# "memory" against "memories" scores 0.714 on raw similarity, just under the
# 0.72 threshold, so it was buying a full re-generation that could not fix it.

def test_a_plural_the_transcriber_heard_is_not_a_defect():
    for sent, heard in [
        ("memory eleven", "memories 11"),
        ("the report says", "the reports says"),
        ("she asked", "she asks"),
    ]:
        d = verify.word_diff(sent, heard)
        assert verify.clean(d), (sent, heard, d)
        assert not verify.worth_regenerating(d), (sent, heard)


def test_inflection_matching_never_excuses_a_meaning_change():
    """CRITICAL_WORDS is checked first, so numbers and negations are safe."""
    for sent, heard in [
        ("he owes ninety million", "he owes 90 billion"),
        ("it is not official", "it is now official"),
        ("the first attempt", "the third attempt"),
        ("nothing was found", "something was found"),
        ("one hundred dollars", "one hundred"),
    ]:
        d = verify.word_diff(sent, heard)
        assert not verify.clean(d), (sent, heard, d)


def test_stemming_needs_a_word_left_over():
    """Short words must not be stemmed down to nothing and matched."""
    assert verify._stem("is") == "is"
    assert verify._stem("as") == "as"
    assert verify._stem("memories") == "memory"
    assert verify._stem("reports") == "report"


# --- the transcriber's spelling is not the model's mistake ----------------
# From a 19-clip batch: every "defect" the verifier found was one of these,
# and not one genuinely dropped word among them. Each was buying a full
# regeneration plus a re-verification of the whole clip.

def test_reported_spelling_pairs_are_not_defects():
    for sent, heard in [
        ("advisers", "advisors"), ("organization", "organisation"),
        ("seventies", "seventys"), ("behavior", "behaviour"),
        ("allen", "alan"), ("client", "clients"), ("claimed", "claim"),
    ]:
        assert verify.likely_asr_artifact(sent, heard), (sent, heard)


def test_a_spacing_difference_is_not_a_dropped_word():
    """"lockup" against "lock up": one token against two, same sound."""
    d = verify.word_diff("the lockup was ordered", "the lock up was ordered")
    assert verify.clean(d), d
    assert not verify.worth_regenerating(d)


def test_the_phonetic_key_keeps_the_first_letter():
    """So no amount of similarity can collide million with billion."""
    assert verify._phonetic_key("allen") == verify._phonetic_key("alan")
    assert verify._phonetic_key("million") != verify._phonetic_key("billion")
    assert verify._phonetic_key("") == ""


def test_sound_alike_matching_never_excuses_a_meaning_change():
    for sent, heard in [
        ("he owes ninety million", "he owes 90 billion"),
        ("it is not official", "it is now official"),
        ("nothing was found", "something was found"),
        ("the first attempt", "the third attempt"),
        ("one hundred dollars", "one hundred"),
        ("twenty senators voted", "senators voted"),
    ]:
        d = verify.word_diff(sent, heard)
        assert not verify.clean(d), (sent, heard, d)


# --- the speaking-rate band ----------------------------------------------
# A clean 63-clip batch warned on 194, 195 and 204 wpm against a 147 wpm
# baseline. All three were ordinary clips.

def test_ordinary_rate_variation_does_not_warn():
    for wpm in (194, 195, 204):
        assert verify.rate_warning(wpm, 147.0) is None, wpm


def test_a_real_rate_drift_still_warns():
    assert verify.rate_warning(230, 147.0) is not None      # +56%
    assert verify.rate_warning(95, 147.0) is not None       # -35%


def test_a_voice_with_no_baseline_only_catches_catastrophes():
    assert verify.rate_warning(204, None) is None
    assert verify.rate_warning(310, None) is not None
    assert verify.rate_warning(40, None) is not None
