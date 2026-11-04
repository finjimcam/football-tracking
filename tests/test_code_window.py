"""Issue #3: code window templates, tagging instances, labels, trims, CSV, video."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from test_lines import FPS, MID, N, build_synthetic  # noqa: E402

from app.coding import DEFAULT_TEMPLATE, CodingError, validate_template  # noqa: E402


@pytest.fixture
def root(tmp_path):
    r = build_synthetic(tmp_path)
    (r / "config" / f"{MID}_reference.json").write_text(json.dumps({"lineups": {
        "team_a": {"lines": [[{"name": "Ederson", "number": 31}], [{"name": "Walker", "number": 2}]]},
        "team_b": {"lines": [[{"name": "Lloris", "number": 1}]]}}}))
    return r


@pytest.fixture
def client(root):
    from fastapi.testclient import TestClient

    from app.jobs import JobRunner
    from app.server import create_app
    return TestClient(create_app(root, runner=JobRunner(root, nice_heavy=False)))


def test_default_template_is_valid():
    assert validate_template(DEFAULT_TEMPLATE)["buttons"][0]["key"] == "q"


@pytest.mark.parametrize("bad, msg", [
    ({"name": "x", "buttons": []}, "at least one button"),
    ({"name": "x", "buttons": [{"label": "A", "key": "q"}, {"label": "B", "key": "Q"}]}, "already used"),
    ({"name": "x", "buttons": [{"label": "A", "key": " "}]}, "player controls"),
    ({"name": "x", "buttons": [{"label": "A", "lead_s": 500}]}, "between 0 and"),
    ({"name": "x", "buttons": [{"label": "A", "color": "red"}]}, "#RRGGBB"),
    ({"name": "../evil", "buttons": [{"label": "A"}]}, "Template name"),
    ({"name": "x", "buttons": [{"label": "A", "key": "1"}],
      "label_groups": [{"name": "Outcome", "labels": [{"label": "Won", "key": "1"}]}]}, "already used"),
])
def test_template_validation_rejects(bad, msg):
    with pytest.raises(CodingError, match=msg):
        validate_template(bad)


def test_templates_save_list_load_delete(client, root):
    names = [t["name"] for t in client.get("/api/code/templates").json()]
    assert names == ["Default"]
    tpl = {"name": "Pressing", "buttons": [{"label": "High press", "key": "p", "color": "#ffd84d", "lead_s": 2, "lag_s": 8}],
           "label_groups": [{"name": "Team", "source": "teams"},
                            {"name": "Outcome", "labels": ["Won", {"label": "Lost", "key": "2"}]}]}
    r = client.put("/api/code/templates/Pressing", json=tpl)
    assert r.status_code == 200, r.text
    saved = r.json()
    assert saved["buttons"][0] == {"id": "high_press", "label": "High press", "key": "p", "color": "#FFD84D",
                                   "lead_s": 2.0, "lag_s": 8.0}
    assert (root / "config" / "code_windows" / "pressing.json").exists()
    assert [t["name"] for t in client.get("/api/code/templates").json()] == ["Default", "Pressing"]
    assert client.get("/api/code/templates/Pressing").json()["label_groups"][1]["labels"][1]["key"] == "2"
    assert client.put("/api/code/templates/Other", json=tpl).status_code == 400  # name mismatch
    bad = dict(tpl, buttons=[])
    assert client.put("/api/code/templates/Pressing", json=bad).status_code == 400
    assert client.delete("/api/code/templates/Pressing").status_code == 204
    assert client.get("/api/code/templates/Pressing").status_code == 404
    assert client.delete("/api/code/templates/Default").status_code == 404  # built in, nothing saved


def test_tag_label_trim_delete_and_csv(client, root):
    base = f"/api/matches/{MID}/instances"
    d = client.get(base).json()
    assert d["instances"] == []
    assert [x["label"] for x in d["match_labels"]["teams"]] == ["Man City", "Tottenham"]
    assert {x["label"]: x["team"] for x in d["match_labels"]["lineup"]} == {
        "31 Ederson": "team_a", "2 Walker": "team_a", "1 Lloris": "team_b"}

    dur = N / FPS
    r = client.post(base, json={"code": "Shot", "color": "#ef5b5b", "t": 2.0, "lead_s": 8, "lag_s": 5,
                                "template": "Default"})
    assert r.status_code == 201, r.text
    shot = r.json()
    assert (shot["start"], shot["end"], shot["color"]) == (0.0, min(dur, 7.0), "#EF5B5B")  # clamped at 0
    assert shot["clock"] == "00:02"  # clock_sync kickoff_frame 0
    late = client.post(base, json={"code": "Pass", "t": dur - 1, "lead_s": 3, "lag_s": 5}).json()
    assert late["end"] == pytest.approx(dur)  # clamped at the end of the video
    assert client.post(base, json={"code": "Pass", "t": dur + 5}).status_code == 400
    assert client.post(base, json={"code": "", "t": 1}).status_code == 400

    u = f"{base}/{shot['id']}"
    assert client.patch(u, json={"set_label": {"group": "Team", "label": "Man City"}}).json()["labels"] == {"Team": "Man City"}
    client.patch(u, json={"set_label": {"group": "Outcome", "label": "On target"}})
    assert client.patch(u, json={"set_label": {"group": "Team", "label": None}}).json()["labels"] == {"Outcome": "On target"}
    t = client.patch(u, json={"start": 1.0, "end": 4.5}).json()
    assert (t["start"], t["end"]) == (1.0, 4.5)
    assert client.patch(u, json={"start": 5.0}).status_code == 400  # would start after it ends
    assert client.patch(f"{base}/nope", json={"start": 1}).status_code == 404

    d = client.get(base).json()
    assert [i["code"] for i in d["instances"]] == ["Shot", "Pass"] and d["template"] == "Default"
    stored = json.loads((root / "config" / f"{MID}_instances.json").read_text())
    assert len(stored["instances"]) == 2 and "clock" not in stored["instances"][0]

    csv = client.get(f"{base}.csv")
    assert csv.status_code == 200 and "attachment" in csv.headers["content-disposition"]
    lines = csv.text.strip().splitlines()
    assert lines[0] == "code,time_s,match_clock,start_s,end_s,duration_s,Outcome"
    assert lines[1].startswith("Shot,2.0,00:02,1.0,4.5,3.5,On target")

    assert client.delete(f"{base}/{shot['id']}").status_code == 204
    assert client.delete(f"{base}/{shot['id']}").status_code == 404
    assert [i["code"] for i in client.get(base).json()["instances"]] == ["Pass"]


def test_video_is_served_with_ranges(client):
    r = client.get(f"/api/matches/{MID}/video", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100
    assert client.get("/api/matches/nope/video").status_code == 404


def test_code_tab_is_wired_into_the_app():
    js = (ROOT / "app" / "static" / "app.js").read_text()
    assert "import { viewCode } from './code.js'" in js and "parts[0] === 'code'" in js
    assert "issue: 3" not in js  # no longer a placeholder
