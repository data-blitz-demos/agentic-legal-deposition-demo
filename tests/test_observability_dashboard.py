# Copyright (c) 2026 Data-Blitz Inc. All rights reserved.
# License: Proprietary. See NOTICE.md.
# Author: Paul Harvener.

import json
from pathlib import Path


def test_thought_stream_persisted_panel_is_table_with_instant_query() -> None:
    root = Path(__file__).resolve().parents[1]
    dashboard = json.loads(
        (
            root
            / "deploy"
            / "observability"
            / "grafana"
            / "provisioning"
            / "dashboards"
            / "json"
            / "attorneyos-observability.json"
        ).read_text(encoding="utf-8")
    )
    panel = next(
        item for item in dashboard['panels'] if item['title'] == 'Thought Stream Events (Persisted)'
    )

    assert panel['type'] == 'table'
    assert panel['datasource'] == 'Prometheus'
    assert panel['targets'][0]['format'] == 'table'
    assert panel['targets'][0]['instant'] is True
    assert 'deposition_thought_stream_persisted_events' in panel['targets'][0]['expr']
