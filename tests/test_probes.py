"""Tests for probe sets and probe library."""

import pytest
from safety_probe.probes.probe_sets import load_probe_set, _REGISTRY
from safety_probe.probes.base import ProbeCategory


class TestProbeSet:
    def test_core_probe_set_loads(self):
        ps = load_probe_set("core")
        assert len(ps) > 0
        assert ps.name == "core"

    def test_all_built_in_sets_load(self):
        for name in ["core", "adversarial", "boundary", "full"]:
            ps = load_probe_set(name)
            assert len(ps) > 0, f"{name} probe set is empty"

    def test_full_set_is_superset(self):
        full = load_probe_set("full")
        core = load_probe_set("core")
        adv = load_probe_set("adversarial")
        boundary = load_probe_set("boundary")
        assert len(full) == len(core) + len(adv) + len(boundary)

    def test_core_has_benign_controls(self):
        ps = load_probe_set("core")
        benign = [p for p in ps.probes if p.category == ProbeCategory.BENIGN]
        assert len(benign) > 0, "Core set must contain benign control probes"

    def test_adversarial_probes_expect_refusal(self):
        ps = load_probe_set("adversarial")
        assert all(p.expected_refusal for p in ps.probes)

    def test_probe_ids_are_unique(self):
        ps = load_probe_set("full")
        ids = [p.probe_id for p in ps.probes]
        assert len(ids) == len(set(ids)), "Probe IDs must be unique"

    def test_probe_set_filter_by_category(self):
        ps = load_probe_set("core")
        violence = ps.filter(category=ProbeCategory.VIOLENCE)
        assert all(p.category == ProbeCategory.VIOLENCE for p in violence.probes)

    def test_probe_set_filter_by_tier(self):
        ps = load_probe_set("full")
        adversarial = ps.filter(tier="adversarial")
        assert all(p.tier == "adversarial" for p in adversarial.probes)

    def test_unknown_probe_set_raises(self):
        with pytest.raises(ValueError, match="Unknown probe set"):
            load_probe_set("nonexistent_set")

    def test_probe_set_addition(self):
        core = load_probe_set("core")
        adv = load_probe_set("adversarial")
        combined = core + adv
        assert len(combined) == len(core) + len(adv)
