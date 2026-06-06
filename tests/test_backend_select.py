from __future__ import annotations

import novadit


def test_import_novadit_package() -> None:
    assert novadit.__version__ == "0.1.0"
