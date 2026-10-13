from anxin.checkin import classify, decide, distress_hit


class NoLLM:
    """Fails the test if the LLM is consulted when a rule should decide."""

    def chat(self, *a, **k):
        return "UNCLEAR"


def test_distress_keywords_escalate():
    for t in ["I can't get up", "My hip really hurts", "Help me please", "I'm not hurt but I can't get up",
              "Please help, I don't know where I am"]:
        assert distress_hit(t), t
        assert classify(t, NoLLM()) == ("HELP", "keyword")


def test_negated_keywords_go_to_the_llm():
    for t in ["I don't need help, I'm okay.", "It doesn't hurt, I'm fine.", "No pain, just embarrassed."]:
        assert not distress_hit(t), t


def test_silence_escalates():
    assert classify("", NoLLM()) == ("NO_RESPONSE", "silence")
    assert decide("NO_RESPONSE") == decide("UNCLEAR") == decide("HELP") == "ALERT_URGENT"
    assert decide("OK") == "ALERT_WATCH"
