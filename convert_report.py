#!/usr/bin/env python3
"""
convert_report.py — MiCuz FC Pipeline Converter
================================================
Converts tracking_data.json and tracking_data_teams.json (output of the
HP EliteBook 840 G7 pipeline: YOLO → ByteTrack → Events → Analytics → Groq LLM)
into report.json — the format expected by the MiCuz FC PWA.

Usage:
    python convert_report.py \
        --tracking tracking_data.json \
        --teams tracking_data_teams.json \
        --matchday MD1 \
        --opponent "Marol Bloodline" \
        --date "28 Mar 2026" \
        --out report.json

All flags except --tracking are optional (defaults apply).

Output JSON schema is documented at the bottom of this file.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from datetime import date


# ══════════════════════════════════════════════════════════════════
#  SQUAD REFERENCE — must match app squad data
# ══════════════════════════════════════════════════════════════════
SQUAD_REF = {
    22: {"name": "Victor Shriyan",   "pos": "Freeplay"},
    29: {"name": "Joshua Lopez",     "pos": "Defender"},
    12: {"name": "Dhanush Nadar",    "pos": "Defender"},
     9: {"name": "Melwin Vaz",       "pos": "Forward"},
    11: {"name": "Sheldon Rego",     "pos": "Forward"},
    14: {"name": "Kishore Shetty",   "pos": "Goalkeeper"},
     7: {"name": "Reuben Pereira",   "pos": "Forward"},
    19: {"name": "Merwyn D'Souza",   "pos": "Forward"},
    25: {"name": "Keith Briganza",   "pos": "Defender"},
    69: {"name": "Dylan Rodrigues",  "pos": "Goalkeeper"},
     8: {"name": "Denver Rodrigues", "pos": "Defender"},
}


# ══════════════════════════════════════════════════════════════════
#  CORE CONVERTER
# ══════════════════════════════════════════════════════════════════
def load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(val, default=0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def safe_int(val, default=0) -> int:
    try:
        return int(round(float(val)))
    except (TypeError, ValueError):
        return default


def compute_match_rating(team_stats: dict, result: str) -> float:
    """
    Compute a single overall match rating (0–10) inspired by SportIQ.
    Weights: possession 15%, pass_completion 20%, shots_on_target/shots 15%,
             pressing_intensity 15%, tackles_won 15%, result_bonus 20%.
    """
    possession    = safe_float(team_stats.get("possession", 50))
    pass_pct      = safe_float(team_stats.get("passCompletion", 70))
    shots         = safe_int(team_stats.get("shots", 1)) or 1
    on_target     = safe_int(team_stats.get("shotsOnTarget", 0))
    pressing      = safe_float(team_stats.get("pressingIntensity", 50))
    tackles       = safe_int(team_stats.get("tacklesWon", 0))
    total_tackles = safe_int(team_stats.get("tackles", 1)) or 1

    sot_ratio    = on_target / shots
    tackle_ratio = tackles / total_tackles

    result_bonus = {"Win": 1.0, "Draw": 0.5, "Loss": 0.0}.get(result, 0.5)

    raw = (
        (possession / 100) * 0.15
        + (pass_pct / 100)  * 0.20
        + sot_ratio          * 0.15
        + (pressing / 100)   * 0.15
        + tackle_ratio        * 0.15
        + result_bonus        * 0.20
    )
    return round(raw * 10, 1)


def compute_player_rating(p: dict) -> float:
    """
    Compute a player match rating (0–10).
    Axes (mirrors hexagonal radar): involvement, shooting, goal_threat,
    speed, defending, stamina.
    """
    involvement = min(safe_float(p.get("passes", 0)) / 30.0, 1.0)
    shooting    = min(safe_float(p.get("shots", 0)) / 5.0, 1.0)
    goal_threat = min(safe_float(p.get("goals", 0)) / 2.0, 1.0) * 0.5 + \
                  min(safe_float(p.get("shotsOnTarget", 0)) / 3.0, 1.0) * 0.5
    max_speed   = safe_float(p.get("maxSpeed", 0))
    speed       = min(max_speed / 28.0, 1.0)          # 28 km/h ≈ full mark
    defending   = (min(safe_float(p.get("tackles", 0)) / 5.0, 1.0) * 0.5 +
                   min(safe_float(p.get("interceptions", 0)) / 5.0, 1.0) * 0.5)
    distance    = safe_float(p.get("distance", 0))    # km
    stamina     = min(distance / 3.5, 1.0)            # 3.5 km ≈ full mark for 5-a-side

    raw = (
        involvement * 0.18
        + shooting   * 0.14
        + goal_threat* 0.18
        + speed      * 0.15
        + defending  * 0.20
        + stamina    * 0.15
    )
    return round(raw * 10, 1)


def determine_result(goals_for: int, goals_against: int) -> str:
    if goals_for > goals_against:
        return "Win"
    elif goals_for == goals_against:
        return "Draw"
    return "Loss"


def convert_players(tracking: dict) -> list:
    """
    Extract per-player Tier 1 stats from tracking_data.json.

    Expected input structure (tracking_data.json):
    {
      "players": [
        {
          "jersey": 22,
          "distance_km": 2.8,
          "sprint_count": 14,
          "max_speed_kmh": 26.1,
          "passes_attempted": 32,
          "passes_completed": 27,
          "interceptions": 3,
          "tackles": 2,
          "tackles_won": 2,
          "shots": 4,
          "shots_on_target": 2,
          "goals": 1,
          "heatmap": [[x, y, intensity], ...],     // optional
          "pass_map": [{"x1":..,"y1":..,"x2":..,"y2":..,"success":true}, ...] // optional
        }, ...
      ]
    }
    """
    out = []
    # Handle raw pipeline list output
    if isinstance(tracking, list):
        raw_players = tracking
    else:
        raw_players = tracking.get("players", [])
    for rp in raw_players:
        jersey = safe_int(rp.get("jersey", rp.get("number", rp.get("num", 0))))
        ref    = SQUAD_REF.get(jersey, {"name": f"Player #{jersey}", "pos": "—"})

        passes_att  = safe_int(rp.get("passes_attempted", rp.get("passes", 0)))
        passes_comp = safe_int(rp.get("passes_completed", rp.get("passesCompleted", passes_att)))
        pass_pct    = round(passes_comp / passes_att * 100) if passes_att else 0

        player = {
            "num":              jersey,
            "name":             rp.get("name", ref["name"]),
            "pos":              rp.get("position", ref["pos"]),
            # Tier 1 pipeline stats
            "distance":         round(safe_float(rp.get("distance_km", rp.get("distance", 0))), 2),
            "sprints":          safe_int(rp.get("sprint_count", rp.get("sprints", 0))),
            "maxSpeed":         round(safe_float(rp.get("max_speed_kmh", rp.get("maxSpeed", 0))), 1),
            "passes":           passes_att,
            "passesCompleted":  passes_comp,
            "passCompletion":   pass_pct,
            "interceptions":    safe_int(rp.get("interceptions", 0)),
            "tackles":          safe_int(rp.get("tackles", 0)),
            "tacklesWon":       safe_int(rp.get("tackles_won", rp.get("tacklesWon", 0))),
            "shots":            safe_int(rp.get("shots", 0)),
            "shotsOnTarget":    safe_int(rp.get("shots_on_target", rp.get("shotsOnTarget", 0))),
            "goals":            safe_int(rp.get("goals", 0)),
            # Radar axes (computed from above)
            "radar": {
                "involvement":  min(round(passes_att / 30.0 * 100), 100),
                "shooting":     min(round(safe_float(rp.get("shots", 0)) / 5.0 * 100), 100),
                "goalThreat":   min(round((safe_float(rp.get("goals", 0)) / 2.0 * 0.5 +
                                           safe_float(rp.get("shots_on_target", rp.get("shotsOnTarget", 0))) / 3.0 * 0.5) * 100), 100),
                "speed":        min(round(safe_float(rp.get("max_speed_kmh", rp.get("maxSpeed", 0))) / 28.0 * 100), 100),
                "defending":    min(round((safe_float(rp.get("tackles", 0)) / 5.0 * 0.5 +
                                           safe_float(rp.get("interceptions", 0)) / 5.0 * 0.5) * 100), 100),
                "stamina":      min(round(safe_float(rp.get("distance_km", rp.get("distance", 0))) / 3.5 * 100), 100),
            },
            # Spatial data (pass through if present)
            "heatmap":  rp.get("heatmap", []),
            "passMap":  rp.get("pass_map", rp.get("passMap", [])),
            "shotMap":  rp.get("shot_map", rp.get("shotMap", [])),
        }
        player["rating"] = compute_player_rating(player)
        out.append(player)
    return out


def convert_team_stats(teams, home_team: str = "MiCuz FC") -> tuple:
    """
    Extract team-level stats from tracking_data_teams.json.

    Expected input structure (tracking_data_teams.json):
    {
      "teams": {
        "home": {
          "name": "MiCuz FC",
          "possession_pct": 54.2,
          "shots": 8,
          "shots_on_target": 5,
          "goals": 3,
          "passes_attempted": 127,
          "passes_completed": 104,
          "interceptions": 9,
          "tackles": 18,
          "tackles_won": 14,
          "sprint_count": 47,
          "distance_km": 12.4,
          "pressing_intensity": 67.3,   // % of opposition touches pressed within 3s
          "pressing_events": 22,
          "heatmap": {...},              // optional
          "pressZone": {...}             // optional
        },
        "away": { ... same structure ... }
      },
      "events": [
        {
          "minute": 7,
          "type": "goal",               // goal | shot | card | sub | foul | corner | press
          "title": "Goal — Victor Shriyan",
          "detail": "Low shot, bottom-left corner",
          "team": "home",               // home | away
          "player_jersey": 22
        }, ...
      ]
    }
    """
    # Handle both dict format (expected) and list format (raw pipeline output)
    if isinstance(teams, list):
        team_a = [d for d in teams if str(d.get("team","")).upper() in ("A","0","HOME") or d.get("team_id") == 0]
        team_b = [d for d in teams if str(d.get("team","")).upper() in ("B","1","AWAY") or d.get("team_id") == 1]
        if not team_a and not team_b:
            # Try splitting by first/second half of unique track IDs
            track_ids = list(set(d.get("tracker_id", d.get("track_id", 0)) for d in teams))
            mid = len(track_ids) // 2
            team_a_ids = set(track_ids[:mid])
            team_a = [d for d in teams if d.get("tracker_id", d.get("track_id")) in team_a_ids]
            team_b = [d for d in teams if d.get("tracker_id", d.get("track_id")) not in team_a_ids]
        ball_a = sum(1 for d in team_a if d.get("has_ball") or d.get("ball_possession"))
        ball_b = sum(1 for d in team_b if d.get("has_ball") or d.get("ball_possession"))
        total_ball = ball_a + ball_b
        home_poss = round(ball_a / total_ball * 100, 1) if total_ball else 50.0
        home = {
            "possession_pct": home_poss, "goals": 0,
            "shots": 0, "shots_on_target": 0,
            "passes_attempted": sum(1 for d in team_a if d.get("pass_made")),
            "passes_completed": sum(1 for d in team_a if d.get("pass_success")),
            "interceptions": sum(1 for d in team_a if d.get("interception")),
            "tackles": sum(1 for d in team_a if d.get("tackle")),
            "tackles_won": sum(1 for d in team_a if d.get("tackle_won")),
            "sprint_count": sum(1 for d in team_a if d.get("sprint")),
            "distance_km": safe_float(sum(safe_float(d.get("distance",0)) for d in team_a)) / 1000,
            "pressing_intensity": 50.0, "pressing_events": 0
        }
        away = {
            "possession_pct": round(100 - home_poss, 1), "goals": 0,
            "shots": 0, "shots_on_target": 0,
            "passes_attempted": sum(1 for d in team_b if d.get("pass_made")),
            "passes_completed": sum(1 for d in team_b if d.get("pass_success")),
            "interceptions": sum(1 for d in team_b if d.get("interception")),
            "tackles": sum(1 for d in team_b if d.get("tackle")),
            "tackles_won": sum(1 for d in team_b if d.get("tackle_won")),
            "sprint_count": sum(1 for d in team_b if d.get("sprint")),
            "distance_km": safe_float(sum(safe_float(d.get("distance",0)) for d in team_b)) / 1000,
            "pressing_intensity": 50.0, "pressing_events": 0
        }
    else:
        raw_teams = teams.get("teams", teams)
        home = raw_teams.get("home", raw_teams.get("micuz", raw_teams.get("MiCuz FC", {})))
        away = raw_teams.get("away", raw_teams.get("opposition", {}))

    poss_home = safe_float(home.get("possession_pct", home.get("possession", 50)))
    poss_away = round(100 - poss_home, 1)

    goals_for     = safe_int(home.get("goals", 0))
    goals_against = safe_int(away.get("goals", 0))

    passes_att  = safe_int(home.get("passes_attempted", home.get("passes", 0)))
    passes_comp = safe_int(home.get("passes_completed", home.get("passesCompleted", passes_att)))
    pass_pct    = round(passes_comp / passes_att * 100) if passes_att else 0

    total_tackles = safe_int(home.get("tackles", 1)) or 1
    tackles_won   = safe_int(home.get("tackles_won", home.get("tacklesWon", 0)))

    distance_total = safe_float(home.get("distance_km", home.get("distance", 0)))
    player_count   = 5  # 5-a-side
    dist_per_player = round(distance_total / player_count, 2) if distance_total else 0

    ts = {
        "possession":         round(poss_home, 1),
        "possessionAway":     poss_away,
        "shots":              safe_int(home.get("shots", 0)),
        "shotsOnTarget":      safe_int(home.get("shots_on_target", home.get("shotsOnTarget", 0))),
        "passes":             passes_att,
        "passesCompleted":    passes_comp,
        "passCompletion":     pass_pct,
        "interceptions":      safe_int(home.get("interceptions", 0)),
        "tackles":            total_tackles,
        "tacklesWon":         tackles_won,
        "sprints":            safe_int(home.get("sprint_count", home.get("sprints", 0))),
        "distanceCovered":    round(distance_total, 2),
        "distancePerPlayer":  dist_per_player,
        "pressingIntensity":  round(safe_float(home.get("pressing_intensity",
                                                         home.get("pressingIntensity", 0))), 1),
        "pressureEvents":     safe_int(home.get("pressing_events",
                                                 home.get("pressureEvents", 0))),
        # Raw heatmap / pressing zone data passed through
        "heatmap":            home.get("heatmap", {}),
        "pressZone":          home.get("pressZone", home.get("press_zone", {})),
    }
    events = teams.get("events", [])
    return ts, goals_for, goals_against, events


def build_report(
    tracking: dict,
    teams: dict,
    matchday: str,
    opponent: str,
    match_date: str,
    match_type: str = "League",
) -> dict:
    team_stats, goals_for, goals_against, events = convert_team_stats(teams)
    players = convert_players(tracking)

    result   = determine_result(goals_for, goals_against)
    rating   = compute_match_rating(team_stats, result)

    # Resolve Groq LLM narrative if present in either input file
    llm_report = (
        tracking.get("llm_report")
        or tracking.get("groq_report")
        or teams.get("llm_report")
        or teams.get("groq_report")
        or ""
    )

    report = {
        # ── Identity ──────────────────────────────────────────
        "matchday":     matchday,
        "opponent":     opponent,
        "date":         match_date,
        "type":         match_type,
        "venue":        "Five Nine Turf, Marol, Mumbai",
        "tournament":   "MFL Season 2",
        # ── Result ────────────────────────────────────────────
        "goalsFor":     goals_for,
        "goalsAgainst": goals_against,
        "result":       result,
        "matchRating":  rating,
        # ── Tier 1 auto-stats ─────────────────────────────────
        "teamStats":    team_stats,
        "possession":   team_stats["possession"],
        "players":      players,
        # ── Events ────────────────────────────────────────────
        "events": [
            {
                "minute": safe_int(e.get("minute", 0)),
                "type":   e.get("type", "event"),
                "title":  e.get("title", ""),
                "detail": e.get("detail", ""),
                "team":   e.get("team", "home"),
                "player": e.get("player_jersey", e.get("player", None)),
            }
            for e in events
        ],
        # ── LLM narrative ─────────────────────────────────────
        "llmReport": llm_report,
        # ── Meta ──────────────────────────────────────────────
        "generatedAt":  date.today().isoformat(),
        "schemaVersion": "2.0",
        # ── Tier 2 placeholder (populated via app Data Entry) ─
        "tier2": {
            "yellowCards":     0,
            "yellowCardsOpp":  0,
            "cornersUs":       0,
            "cornersOpp":      0,
            "freeKicksUs":     0,
            "freeKicksOpp":    0,
            "goalScorers":     [],
            "subTimings":      [],
        },
    }
    return report


# ══════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description="MiCuz FC — Pipeline JSON Converter v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_report.py --tracking tracking_data.json --teams tracking_data_teams.json --matchday MD1 --opponent "Marol Bloodline" --out report.json
  python convert_report.py --tracking td.json --teams tt.json --matchday MD3 --opponent "Cecilians Of Marol" --date "15 Apr 2026" --type League --out md3_report.json
        """,
    )
    parser.add_argument("--tracking", required=True,  help="Path to tracking_data.json")
    parser.add_argument("--teams",    required=True,  help="Path to tracking_data_teams.json")
    parser.add_argument("--matchday", default="MD1",  help="Matchday label (e.g. MD1, Playoff QF)")
    parser.add_argument("--opponent", default="Opposition", help="Opponent team name")
    parser.add_argument("--date",     default=date.today().strftime("%d %b %Y"), help="Match date (e.g. '28 Mar 2026')")
    parser.add_argument("--type",     default="League", help="Match type: League | Playoff QF | Playoff SF | Final")
    parser.add_argument("--out",      default="report.json", help="Output file path")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"[MiCuz FC Converter] Loading tracking data…")
    tracking = load_json(args.tracking)
    teams    = load_json(args.teams)
    print(f"[MiCuz FC Converter] Building report.json…")
    report = build_report(
        tracking=tracking,
        teams=teams,
        matchday=args.matchday,
        opponent=args.opponent,
        match_date=args.date,
        match_type=args.type,
    )
    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[MiCuz FC Converter] ✓ Written: {out_path} ({out_path.stat().st_size // 1024} KB)")
    print(f"  Matchday : {report['matchday']}")
    print(f"  Opponent : {report['opponent']}")
    print(f"  Result   : {report['goalsFor']}–{report['goalsAgainst']} ({report['result']})")
    print(f"  Rating   : {report['matchRating']}/10")
    print(f"  Players  : {len(report['players'])} tracked")
    print(f"  Events   : {len(report['events'])} logged")


if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════════════════════
#  OUTPUT JSON SCHEMA — report.json v2.0
# ══════════════════════════════════════════════════════════════════
#
# {
#   "matchday":     "MD1",                         // string
#   "opponent":     "Marol Bloodline",             // string
#   "date":         "28 Mar 2026",                 // string
#   "type":         "League",                      // League | Playoff QF | Playoff SF | Final
#   "venue":        "Five Nine Turf, Marol, Mumbai",
#   "tournament":   "MFL Season 2",
#   "goalsFor":     3,                             // integer
#   "goalsAgainst": 1,                             // integer
#   "result":       "Win",                         // Win | Draw | Loss
#   "matchRating":  7.4,                           // float 0–10 (SportIQ-style)
#   "possession":   54.2,                          // float % (MiCuz share)
#   "teamStats": {
#     "possession":         54.2,
#     "possessionAway":     45.8,
#     "shots":              8,
#     "shotsOnTarget":      5,
#     "passes":             127,
#     "passesCompleted":    104,
#     "passCompletion":     82,                    // integer %
#     "interceptions":      9,
#     "tackles":            18,
#     "tacklesWon":         14,
#     "sprints":            47,
#     "distanceCovered":    12.4,                  // km, team total
#     "distancePerPlayer":  2.48,                  // km average per player
#     "pressingIntensity":  67.3,                  // % of opp touches pressed <3s
#     "pressureEvents":     22,
#     "heatmap":            {},                    // raw heatmap data (pass-through)
#     "pressZone":          {}                     // pressing zone data (pass-through)
#   },
#   "players": [
#     {
#       "num":             22,
#       "name":            "Victor Shriyan",
#       "pos":             "Freeplay",
#       "distance":        2.8,                    // km
#       "sprints":         14,
#       "maxSpeed":        26.1,                   // km/h
#       "passes":          32,
#       "passesCompleted": 27,
#       "passCompletion":  84,                     // integer %
#       "interceptions":   3,
#       "tackles":         2,
#       "tacklesWon":      2,
#       "shots":           4,
#       "shotsOnTarget":   2,
#       "goals":           1,
#       "rating":          7.8,                    // float 0–10
#       "radar": {
#         "involvement":  93,                      // integer 0–100 (hex radar axes)
#         "shooting":     80,
#         "goalThreat":   85,
#         "speed":        93,
#         "defending":    60,
#         "stamina":      80
#       },
#       "heatmap":  [[x, y, intensity], ...],      // array of [0–1 normalised x, y, intensity]
#       "passMap":  [{"x1":0.3,"y1":0.5,"x2":0.6,"y2":0.4,"success":true}, ...],
#       "shotMap":  [{"x":0.75,"y":0.48,"onTarget":true,"goal":false}, ...]
#     }
#     // ... one entry per tracked player
#   ],
#   "events": [
#     {
#       "minute":  7,
#       "type":    "goal",                         // goal|shot|card|sub|foul|corner|press
#       "title":   "Goal — Victor Shriyan",
#       "detail":  "Low shot, bottom-left corner",
#       "team":    "home",                         // home | away
#       "player":  22                              // jersey number or null
#     }
#     // ...
#   ],
#   "llmReport":  "Groq LLM narrative text…",     // string (from pipeline)
#   "generatedAt": "2026-03-28",
#   "schemaVersion": "2.0",
#   "tier2": {
#     "yellowCards":    0,                         // populated via app Data Entry screen
#     "yellowCardsOpp": 0,
#     "cornersUs":      0,
#     "cornersOpp":     0,
#     "freeKicksUs":    0,
#     "freeKicksOpp":   0,
#     "goalScorers":    [{"player": "Victor #22", "minute": 7}],
#     "subTimings":     [{"off": "Sheldon #11", "on": "Reuben #7", "minute": 24}]
#   }
# }
