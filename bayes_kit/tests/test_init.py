from __future__ import annotations

import bayes_kit


def test_public_exports() -> None:
    for name in bayes_kit.__all__:
        assert hasattr(bayes_kit, name)
