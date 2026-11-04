"""Issue #16: library of teams, players, competitions, seasons and matches (tools/library.py, Library tab)."""
import json

import pytest

from library import (LibraryError, delete, empty, import_match, link_match, load, match_lineup, save,
                     set_squad, split_competition, upsert)

CFG = {"team_a_name": "Man City", "team_b_name": "Tottenham",
       "teams": {"team_a": {"name": "Man City", "short": "MCI", "kit": "sky blue"},
                 "team_b": {"name": "Tottenham", "short": "TOT"}},
       "colors": {"team_a": "#6CABDD", "team_b": "#132257"},
       "date": "2022-02-19", "competition": "Premier League 2021-22, Gameweek 26",
       "venue": "Etihad Stadium", "result": "Man City 2-3 Tottenham (HT 1-1)"}
REF = {"lineups": {
    "team_a": {"lines": [[{"name": "Ederson", "number": 31}], [{"name": "Walker", "number": 2}, {"name": "Dias", "number": 3}],
                         [{"name": "Rodri", "number": 16}], [{"name": "Foden", "number": 47}]],
               "subs": [{"name": "Mahrez", "number": 26}]},
    "team_b": {"lines": [[{"name": "Lloris", "number": 1}], [{"name": "Romero", "number": 4}], [{"name": "Kane", "number": 10}]]}}}


def test_split_competition():
    assert split_competition("Premier League 2021-22, Gameweek 26") == ("Premier League", "2021-22", "Gameweek 26")
    assert split_competition("Scottish Cup 2023/24") == ("Scottish Cup", "2023-24", None)
    assert split_competition("Friendly") == ("Friendly", None, None)


def test_import_match_builds_everything_and_is_idempotent():
    lib = empty()
    s = import_match(lib, "m1", CFG, REF, "/videos/m1.mp4", ["config/m1_match.json"])
    assert s["added"] == {"teams": 2, "players": 9, "competitions": 1, "seasons": 1}
    assert lib["seasons"][s["season"]] == {"competition": "premier-league", "name": "2021-22"}
    assert lib["teams"]["man-city"] == {"name": "Man City", "short": "MCI", "kit": "sky blue", "color": "#6CABDD"}
    m = lib["matches"]["m1"]
    assert (m["team_a"], m["team_b"], m["home"], m["round"], m["video"]) == ("man-city", "tottenham", "team_a", "Gameweek 26", "/videos/m1.mp4")
    a = m["lineups"]["team_a"]
    assert [(e["player"], e["number"], e["position"], e["starter"]) for e in a][:2] == [("ederson", 31, "GK", True), ("walker", 2, "DEF", True)]
    assert a[-1] == {"player": "mahrez", "number": 26, "position": None, "starter": False}
    assert lib["players"]["foden"]["positions"] == ["FWD"] and lib["players"]["rodri"]["positions"] == ["MID"]
    assert [e["number"] for e in lib["squads"][s["season"]]["man-city"]] == [31, 2, 3, 16, 47, 26]
    # hand edits survive a re-import, and nothing is duplicated
    lib["teams"]["man-city"]["short"] = "CITY"
    s2 = import_match(lib, "m1", CFG, REF)
    assert s2["added"] == {"teams": 0, "players": 0, "competitions": 0, "seasons": 0}
    assert lib["teams"]["man-city"]["short"] == "CITY" and len(lib["players"]) == 9
    assert lib["matches"]["m1"]["video"] == "/videos/m1.mp4"   # kept when the re-import doesn't know it


def test_squads_numbers_per_season_and_new_players():
    lib = empty()
    import_match(lib, "m1", CFG, REF)
    nxt = upsert(lib, "seasons", {"competition": "premier-league", "name": "2022-23"})
    # next season: Walker changes number, a new signing is typed in by name
    out = set_squad(lib, nxt, "man-city", [{"player": "walker", "number": 4}, {"name": "Erling Haaland", "number": 9, "positions": ["FWD"]},
                                            {"name": "  "}])
    assert [(e["player"], e["number"]) for e in out] == [("walker", 4), ("erling-haaland", 9)]
    assert lib["players"]["erling-haaland"]["positions"] == ["FWD"]
    old = lib["squads"]["premier-league-2021-22"]["man-city"]
    assert next(e["number"] for e in old if e["player"] == "walker") == 2   # last season untouched
    # a name already in the library links to that player instead of making a duplicate
    set_squad(lib, nxt, "tottenham", [{"name": "kane", "number": 10}])
    assert lib["squads"][nxt]["tottenham"][0]["player"] == "kane"
    with pytest.raises(LibraryError, match="used twice"):
        set_squad(lib, nxt, "man-city", [{"player": "walker", "number": 9}, {"player": "foden", "number": 9}])
    with pytest.raises(LibraryError, match="1-99"):
        set_squad(lib, nxt, "man-city", [{"player": "walker", "number": 100}])
    with pytest.raises(LibraryError, match="position"):
        set_squad(lib, nxt, "man-city", [{"player": "walker", "number": 2, "positions": ["Striker"]}])


def test_delete_refuses_while_in_use_and_validation():
    lib = empty()
    import_match(lib, "m1", CFG, REF)
    with pytest.raises(LibraryError, match="Still used"):
        delete(lib, "teams", "man-city")
    with pytest.raises(LibraryError, match="Still used"):
        delete(lib, "competitions", "premier-league")
    tid = upsert(lib, "teams", {"name": "Arsenal", "color": "#EF0107"})
    delete(lib, "teams", tid)
    assert tid not in lib["teams"]
    with pytest.raises(LibraryError, match="colour"):
        upsert(lib, "teams", {"name": "X", "color": "red"})
    with pytest.raises(LibraryError, match="competition"):
        upsert(lib, "seasons", {"competition": "nope", "name": "2020-21"})
    with pytest.raises(LibraryError, match="itself"):
        link_match(lib, "m1", {"team_b": "man-city"})


def test_match_lineup_adds_the_rest_of_the_squad_as_subs():
    lib = empty()
    s = import_match(lib, "m1", CFG, REF)
    set_squad(lib, s["season"], "man-city", lib["squads"][s["season"]]["man-city"] + [{"name": "Jack Grealish", "number": 10}])
    lu = match_lineup(lib, "m1", "team_a")
    assert [p["number"] for p in lu] == [31, 2, 3, 16, 47, 26, 10]
    assert lu[0]["gk"] and lu[5]["sub"] and lu[6]["sub"] and not lu[1]["sub"]
    assert match_lineup(lib, "nope", "team_a") == []


def test_save_and_load_round_trip(tmp_path):
    p = tmp_path / "library.json"
    assert load(p) == empty()
    lib = empty()
    import_match(lib, "m1", CFG, REF)
    save(p, lib)
    assert load(p) == lib
    p.write_text("{broken")
    with pytest.raises(LibraryError):
        load(p)


def test_library_api_and_identify_uses_it(tmp_path):
    import sys
    from pathlib import Path

    from fastapi.testclient import TestClient
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from test_lines import MID, build_synthetic

    from app.jobs import JobRunner
    from app.server import create_app
    root = build_synthetic(tmp_path)
    cfg_p = root / "config" / f"{MID}_match.json"
    cfg = json.loads(cfg_p.read_text())
    cfg.update({k: CFG[k] for k in ("teams", "competition", "date", "result")})
    cfg_p.write_text(json.dumps(cfg))
    (root / "config" / f"{MID}_reference.json").write_text(json.dumps(REF))
    c = TestClient(create_app(root, runner=JobRunner(root, nice_heavy=False)))
    d = c.get("/api/library").json()
    assert d["library"]["teams"] == {} and d["videos"][0]["match_id"] == MID and not d["videos"][0]["linked"]
    r = c.post(f"/api/library/import/{MID}")
    assert r.status_code == 200, r.text
    assert r.json()["imported"]["added"]["players"] == 9
    assert json.loads((root / "config" / "library.json").read_text())["matches"][MID]["team_a"] == "man-city"
    # edit a team, add a squad player, a bad edit is rejected and leaves the file alone
    assert c.put("/api/library/teams/man-city", json={"short": "MC", "name": "Manchester City"}).json()["library"]["teams"]["man-city"]["short"] == "MC"
    season = r.json()["imported"]["season"]
    squad = r.json()["library"]["squads"][season]["man-city"]
    res = c.put(f"/api/library/squads/{season}/man-city", json={"entries": squad + [{"name": "Jack Grealish", "number": 10}]})
    assert res.status_code == 200, res.text
    before = (root / "config" / "library.json").read_text()
    assert c.put(f"/api/library/squads/{season}/man-city", json={"entries": [{"player": "walker", "number": 2}, {"player": "dias", "number": 2}]}).status_code == 400
    assert (root / "config" / "library.json").read_text() == before
    assert c.post("/api/library/widgets", json={"name": "x"}).status_code == 400
    assert c.delete("/api/library/teams/man-city").status_code == 400
    assert c.put("/api/library/matches/not-a-match", json={}).status_code == 404
    assert c.put(f"/api/library/matches/{MID}", json={"round": "GW 27"}).json()["library"]["matches"][MID]["round"] == "GW 27"
    # the Lines tab's Identify picker now offers the squad, so a substitute can be named
    lineup = c.get(f"/api/matches/{MID}/lines").json()["lineup"]
    assert [p["number"] for p in lineup][-2:] == [26, 10] and lineup[-1]["name"] == "Jack Grealish"
