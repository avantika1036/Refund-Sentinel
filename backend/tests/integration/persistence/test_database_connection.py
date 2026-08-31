from __future__ import annotations

import pytest
from sqlalchemy import text

from backend.app.persistence.database import engine


@pytest.mark.integration
def test_postgres_connection():
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1")).scalar_one()

    assert result == 1