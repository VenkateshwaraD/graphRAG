import pytest

from app.rag.providers.fake import FakeLLM
from app.rag.router import route_question


def test_override_valid():
    """A valid mode override wins and skips the LLM entirely.

    Author: venkateshwara doijode
    """
    llm = FakeLLM()
    assert route_question("anything", llm, override="traverse") == "traverse"
    assert not llm.calls  # override skips the LLM


def test_override_invalid():
    """An invalid mode override raises ValueError (-> 400 at API).

    Author: venkateshwara doijode
    """
    with pytest.raises(ValueError):
        route_question("anything", FakeLLM(), override="bogus")


def test_llm_route_parsed():
    """A clean single-token LLM reply parses to its route.

    Author: venkateshwara doijode
    """
    llm = FakeLLM(canned="traverse")
    assert route_question("who teaches what?", llm) == "traverse"


def test_llm_verbose_answer_tolerated():
    """A sentence containing a route token still parses.

    Author: venkateshwara doijode
    """
    llm = FakeLLM(canned="I would choose global for this.")
    assert route_question("summarize policies", llm) == "global"


def test_default_fallback():
    """Unparseable LLM output falls back to hybrid, not an error.

    Author: venkateshwara doijode
    """
    llm = FakeLLM(canned="unparseable nonsense")
    assert route_question("q", llm) == "hybrid"


def test_first_token_by_position_wins():
    """When several route tokens appear, the earliest wins —
    deterministic regardless of set iteration order.

    Author: venkateshwara doijode
    """
    llm = FakeLLM(canned="use traverse, not local search")
    assert route_question("q", llm) == "traverse"


def test_word_boundary_blocks_substring():
    """'globalization' must not parse as the 'global' route.

    Author: venkateshwara doijode
    """
    llm = FakeLLM(canned="globalization effects")
    assert route_question("q", llm) == "hybrid"
