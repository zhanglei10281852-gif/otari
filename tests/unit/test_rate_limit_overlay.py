"""Stored rate-limit rules added to ``config.rate_limits``."""

from gateway.core.config import GatewayConfig, RateLimitRule
from gateway.models.rate_limits import StoredRateLimitRule
from gateway.services.rate_limits._overlay import apply_stored_rules, config_file_rules, rule_of


def _rule(name: str, rpm: int = 1) -> RateLimitRule:
    return RateLimitRule(name=name, per="key", rpm=rpm)


def test_stored_rules_follow_the_config_rules() -> None:
    config = GatewayConfig(rate_limits=[_rule("file")])

    apply_stored_rules(config, [_rule("stored")])

    assert [rule.name for rule in config.rate_limits] == ["file", "stored"]
    assert [rule.name for rule in config_file_rules(config)] == ["file"]


def test_a_config_rule_wins_a_name_collision() -> None:
    config = GatewayConfig(rate_limits=[_rule("shared", rpm=5)])

    skipped = apply_stored_rules(config, [_rule("shared", rpm=1)])

    assert skipped == ["shared"]
    assert [(rule.name, rule.rpm) for rule in config.rate_limits] == [("shared", 5)]


def test_applying_again_drops_a_rule_no_longer_stored() -> None:
    config = GatewayConfig(rate_limits=[_rule("file")])
    apply_stored_rules(config, [_rule("stored")])

    apply_stored_rules(config, [])

    assert [rule.name for rule in config.rate_limits] == ["file"]


def test_rule_of_carries_the_rows_generation() -> None:
    row = StoredRateLimitRule(
        name="stored", per="key", rpm=1, tpm_admission="used", lease_sec=900.0, generation="a" * 32
    )

    assert rule_of(row).generation == "a" * 32


def test_rule_of_treats_a_row_without_a_generation_as_legacy() -> None:
    """A row written before generations existed keeps the name-only counter keys."""
    row = StoredRateLimitRule(name="old", per="key", rpm=1, tpm_admission="used", lease_sec=900.0)

    assert rule_of(row).generation is None


def test_applying_carries_each_stored_rules_generation() -> None:
    """The reload every replica runs must propagate the generation the request path keys on."""
    config = GatewayConfig(rate_limits=[_rule("file")])
    stored = _rule("stored")
    stored._generation = "b" * 32

    apply_stored_rules(config, [stored])

    assert config.rate_limits[1].generation == "b" * 32
