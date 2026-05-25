"""Pytest yapilandirmasi ve paylasimli fixture'lar.

Tum test modulleri tarafindan kullanilan ortak fixture'lar burada tanimlaniyor.
pyproject.toml icindeki `pythonpath = ["."]` ayari sayesinde sys.path
manupulasyonuna gerek yoktur.
"""

import pytest


@pytest.fixture
def tmp_ban_db(tmp_path):
    """Ban DB'yi gecici dizine yonlendir.

    Her test icin izole bir ban veritabani saglar. Test sonunda
    orijinal yol geri yuklenir. Bu fixture testler arasi yan etkileri onler.
    """
    import main

    orig = main.BANNED_DB_FILE
    main.BANNED_DB_FILE = str(tmp_path / "test_banned.json")
    yield tmp_path
    main.BANNED_DB_FILE = orig
