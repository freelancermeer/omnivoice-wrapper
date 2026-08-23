"""Every example in these tests is text the server was actually sent."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import textnorm

CURLY_APOS = "’"
EM_DASH = "—"
GRINNING = "\U0001F600"


def norm(t, lang="English"):
    return textnorm.normalize_text(t, lang, level="full")[0]


# --- numbers, currency, percent -------------------------------------------
def test_currency():
    assert norm("A property worth $500 million.") == \
        "A property worth five hundred million dollars."
    assert "four point seven million dollars" in norm("His sons owe $4.7 million each.")
    assert norm("It cost $1.") == "It cost one dollar."


def test_percent():
    assert norm("Trump inflated values by 200 to 300%.") == \
        "Trump inflated values by two hundred to three hundred percent."


def test_thousands_separator_is_not_a_pause():
    out = norm("His penthouse went from 11,000 square feet to 30,000.")
    assert "eleven thousand" in out and "thirty thousand" in out
    assert "," not in out.replace(".", "")


def test_ordinals():
    # "1st"/"2nd"/"3rd" used to come out as "onest"/"twond"/"threerd" because
    # the digit was substituted inside the token.
    assert norm("The 1st and 2nd and 3rd.") == "The first and second and third."
    assert norm("the 21st century") == "the twenty-first century"


def test_dates():
    assert norm("What about the January 6th speech?") == \
        "What about the January sixth speech?"
    assert norm("What about the January 6 speech?") == \
        "What about the January sixth speech?"


def test_american_number_style_by_default():
    """num2words says "one hundred and fifty"; American narration does not."""
    assert norm("It was 150 dollars.") == "It was one hundred fifty dollars."
    textnorm.NUM_STYLE = "uk"
    try:
        assert "hundred and fifty" in norm("It was 150 dollars.")
    finally:
        textnorm.NUM_STYLE = "us"


def test_years_read_as_years():
    assert "twenty twenty-four" in norm("Back in 2024 he said so.")


def test_phone_numbers_go_digit_by_digit():
    out = norm("Call 555-123-4567 now.")
    assert "five five five" in out
    assert "-" not in out
    out2 = norm("Call 5551234567 now.")
    assert "five five five" in out2


# --- punctuation ----------------------------------------------------------
def test_em_dash_becomes_a_pause_not_a_run_on():
    out = norm("he admitted the fake elector plot" + EM_DASH
               + "the deliberate plan to file.")
    assert "plot, the deliberate" in out
    assert EM_DASH not in out and "plotthe" not in out


def test_colon_becomes_a_pause_but_a_clock_survives():
    assert norm("desperate to ignore: private acts get zero immunity.") == \
        "desperate to ignore, private acts get zero immunity."
    # A clock survives the pause rule and is then spoken, not left as "3:30"
    # (which reached the model as "three:thirty").
    assert norm("Be there at 3:30.") == "Be there at three thirty."
    assert norm("Be there at 9:00.") == "Be there at nine o'clock."
    assert norm("Be there at 9:05.") == "Be there at nine oh five."


def test_quotes_are_silent_and_curly_marks_are_folded():
    out = norm('Notes like "this should be higher" were in the margins.')
    assert '"' not in out
    assert "Trump's" in norm("Trump" + CURLY_APOS + "s team")


def test_curly_apostrophe_no_longer_disables_number_handling():
    """The old guard bailed out on any codepoint > 0x2FF, so one curly
    apostrophe or em dash switched off number normalization for the whole
    script - and 26 of 63 real segments contained an em dash."""
    out = textnorm.normalize_text("Trump" + CURLY_APOS + "s 11,000 square feet",
                                  None, level="full")[0]
    assert "eleven thousand" in out
    out2 = textnorm.normalize_text("the plot" + EM_DASH + "all 11,000 of them",
                                   None, level="full")[0]
    assert "eleven thousand" in out2


def test_unknown_characters_are_dropped_not_fatal():
    out = norm("Hello " + GRINNING + " world.")
    assert "Hello" in out and "world" in out


def test_non_latin_text_is_left_alone():
    src = "这是 123 个测试"
    assert textnorm.normalize_text(src, None, level="full")[0] == src


def test_level_off_is_a_true_bypass():
    src = "$500 million" + EM_DASH + "now"
    assert textnorm.normalize_text(src, "English", level="off")[0] == src


# --- lexicon --------------------------------------------------------------
def test_pronounceable_acronyms_only():
    out = norm("They are sabotaging MAGA, says the former NSC official.")
    assert "Maga" in out
    # Ordinary acronyms are already spoken correctly by the model - leave them.
    assert "NSC" in out


def test_user_lexicon_overrides():
    out = textnorm.normalize_text("Bessent spoke.", "English", level="full",
                                  lexicon={"Bessent": "Bess ent"})[0]
    assert "Bess ent" in out


# --- sentence splitting / chunking ----------------------------------------
def test_abbreviations_do_not_split_sentences():
    s = "Surrender to U.S. Marshals, said Don Jr. He then left."
    parts = textnorm.split_sentences(s)
    assert len(parts) == 2, parts
    assert parts[0].startswith("Surrender to U.S. Marshals")


def test_initials_do_not_split_sentences():
    assert len(textnorm.split_sentences("Donald J. Trump spoke today.")) == 1


def test_ordinary_sentences_do_split():
    assert len(textnorm.split_sentences("One. Two. Three.")) == 3


def test_chunks_are_balanced_not_greedy():
    """Greedy packing left a 100-char chunk beside a 6-char one, and short
    inputs are the rushed ones (190-203 wpm in the measured batch)."""
    text = ("Alpha bravo charlie delta echo foxtrot. " * 4
            + "Golf hotel india.")
    chunks = textnorm.chunk_text(text, max_chars=100)
    assert len(chunks) >= 2
    shortest, longest = min(map(len, chunks)), max(map(len, chunks))
    assert shortest > 25, chunks
    assert longest - shortest < 80, chunks


def test_no_tiny_tail_chunk():
    text = "A" * 95 + ". Ok."
    chunks = textnorm.chunk_text(text, max_chars=100)
    assert all(len(c) > 10 for c in chunks), chunks


def test_a_very_long_sentence_is_still_split():
    text = "word " * 200          # no terminators at all
    chunks = textnorm.chunk_text(text.strip(), max_chars=100)
    assert len(chunks) > 1
    assert max(len(c) for c in chunks) < 260


def test_short_text_is_one_chunk():
    assert textnorm.chunk_text("Hello there.", 100) == ["Hello there."]


def test_word_count():
    assert textnorm.word_count("one two three") == 3
    assert textnorm.word_count("") == 0


# --- whatever the customer actually pasted ---------------------------------
# Scripts arrive out of Google Docs, Notion, a CMS or an LLM. Unsupported
# typography is documented to make neural vocoders emit static, so none of it
# should ever reach the model.
def test_markdown_is_stripped_not_spoken():
    # The heading gains a full stop: on the page it is a visual break, and out
    # loud it has to be an audible one or it runs into the next sentence.
    assert norm("## The Verdict\n\nThis is **damning**. He *knew*.") == \
        "The Verdict. This is damning. He knew."


def test_bullets_and_numbering_are_stripped():
    out = norm("- First point\n- Second point\n1. Numbered\n2. Also numbered")
    # Each bullet becomes its own sentence, so the reader pauses between them
    # instead of running four list items together in one breath.
    assert out == "First point. Second point. Numbered. Also numbered."


def test_a_hard_wrapped_paragraph_is_not_broken_into_sentences():
    """Text pasted out of a PDF wraps mid-sentence. Adding a full stop at every
    line ending would put a pause in the middle of a clause."""
    out = norm("He admitted the fake elector plot was the\n"
               "deliberate plan to file phony slates.")
    assert out == "He admitted the fake elector plot was the deliberate plan to file phony slates."


def test_markdown_links_keep_the_words_and_drop_the_target():
    assert norm("Read [the full report](https://example.com/report) now.") == \
        "Read the full report now."


def test_html_tags_and_entities():
    assert norm("<p>Hello &amp; welcome</p><br>") == "Hello and welcome"


def test_all_caps_speaker_labels_go_but_ordinary_colons_stay():
    assert norm("NARRATOR: He walked in.\nHOST: And then?") == \
        "He walked in. And then?"
    # The dangerous false positive: this is a sentence, not a speaker label.
    assert norm("One thing: this matters.") == "One thing, this matters."


def test_stage_directions_and_footnote_markers_are_removed():
    assert norm("He said [laughs] it was fine [1].") == "He said it was fine."


def test_urls_are_read_as_a_domain_not_a_query_string():
    assert norm("Visit https://www.example.com/path?x=1 today.") == \
        "Visit example dot com today."


def test_emails_are_readable():
    assert "at example dot co dot uk" in norm("Email me at meer@example.co.uk please.")


def test_runaway_punctuation_is_calmed_down():
    out = norm("What?!?!?! Really....")
    assert "?!" not in out
    assert "...." not in out


def test_all_caps_runs_are_reported_not_silently_changed():
    out, notes = textnorm.normalize_text("THIS IS VERY IMPORTANT NEWS today.",
                                         "English", level="full")
    assert "THIS IS VERY IMPORTANT NEWS" in out, "emphasis was changed"
    assert any("ALL-CAPS" in n for n in notes), notes


def test_phone_numbers_are_grouped_like_a_person_says_them():
    out = norm("Call 555-123-4567 now.")
    assert out == "Call five five five, one two three, four five six seven now."
    # Same answer without the separators.
    assert norm("Call 5551234567 now.") == out


def test_sanitising_never_touches_ordinary_prose():
    src = ("He admitted the plot was not official. The judge, who had already "
           "ruled twice, disagreed; the appeal failed.")
    out = norm(src)
    assert "He admitted the plot was not official." in out
    assert "the judge, who had already ruled twice, disagreed" in out.lower()


def test_a_pasted_document_survives_end_to_end():
    src = ("# Segment 4\n\n"
           "NARRATOR: A property Trump claimed was worth $500 million was "
           "appraised at $150 million [2].\n\n"
           "- His penthouse went from 11,000 square feet to 30,000\n"
           "- See [the filing](https://example.com/f) for more\n")
    out = norm(src)
    for expected in ("five hundred million dollars",
                     "one hundred fifty million dollars",
                     "eleven thousand", "thirty thousand", "the filing"):
        assert expected in out, (expected, out)
    for banned in ("#", "[", "]", "NARRATOR", "http", "- "):
        assert banned not in out, (banned, out)


# --- chunk sizing: short chunks are the dangerous ones ---------------------
def test_no_chunk_is_ever_left_undersized():
    """Upstream #229/#206: the cloned voice switches speaker — sometimes
    gender — on short generations, and "longer sentences appear more stable".
    A three-word chunk is also where the measured batch was most rushed."""
    text = ("Absolutely damning. Yes. He knew. "
            + "This is a much longer sentence that carries most of the text. " * 3)
    chunks = textnorm.chunk_text(text, max_chars=200)
    assert all(len(c) >= 60 for c in chunks), [len(c) for c in chunks]


def test_short_fragments_are_merged_not_emitted_alone():
    chunks = textnorm.chunk_text("Me. Yes. No. Stop. Go.", max_chars=200)
    assert len(chunks) == 1, chunks


def test_merging_respects_the_ceiling():
    long_one = "word " * 60                       # ~300 chars, one sentence
    chunks = textnorm.chunk_text(long_one.strip() + ". Short.", max_chars=200)
    assert max(len(c) for c in chunks) <= int(200 * 1.6) + 20, [len(c) for c in chunks]


def test_chunking_still_splits_genuinely_long_text():
    text = "This is a complete sentence of moderate length. " * 20
    chunks = textnorm.chunk_text(text, max_chars=200)
    assert len(chunks) > 3
    assert all(len(c) <= 320 for c in chunks), [len(c) for c in chunks]


def test_space_before_punctuation_is_opt_in():
    src = "Hola, esto es una prueba."
    assert norm(src) == "Hola, esto es una prueba."
    textnorm.SPACE_BEFORE_PUNCT = True
    try:
        out, notes = textnorm.normalize_text(src, "Spanish", level="full")
        assert " ," in out and " ." in out, out
        assert any("space-before-punctuation" in n for n in notes)
    finally:
        textnorm.SPACE_BEFORE_PUNCT = False
