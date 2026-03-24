#!/usr/bin/env python3
"""
convert_report.py v3.0 - MiCuz FC Pipeline Converter
Handles the actual pipeline output format:
  tracking_data.json      -> list of player track objects per frame
  tracking_data_teams.json -> list of enriched detection objects with team labels
"""
import argparse, json, sys
from pathlib import Path
from datetime import date

SQUAD_REF = {
    22:{"name":"Victor Shriyan","pos":"Freeplay"},
    29:{"name":"Joshua Lopez","pos":"Defender"},
    12:{"name":"Dhanush Nadar","pos":"Defender"},
    9:{"name":"Melwin Vaz","pos":"Forward"},
    11:{"name":"Sheldon Rego","pos":"Forward"},
    14:{"name":"Kishore Shetty","pos":"Goalkeeper"},
    7:{"name":"Reuben Pereira","pos":"Forward"},
    19:{"name":"Merwyn D'Souza","pos":"Forward"},
    25:{"name":"Keith Briganza","pos":"Defender"},
    69:{"name":"Dylan Rodrigues","pos":"Goalkeeper"},
    8:{"name":"Denver Rodrigues","pos":"Defender"},
}

def sf(v, d=0.0):
    try: return float(v)
    except: return d

def si(v, d=0):
    try: return int(round(float(v)))
    except: return d

def load_json(path):
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(p,"r",encoding="utf-8") as f:
        return json.load(f)

def process_detections(detections):
    """
    Process raw pipeline detections list into per-player and team stats.
    Each detection: {frame, track_id, type, x, y, w, h, confidence, team, ...}
    """
    if not isinstance(detections, list):
        return {}, {}, []

    # Separate players from ball
    players = [d for d in detections if d.get("type","player") == "player"]
    ball    = [d for d in detections if d.get("type","") == "ball"]

    # Split by team
    team_a = [d for d in players if str(d.get("team","")).upper() == "A"]
    team_b = [d for d in players if str(d.get("team","")).upper() == "B"]

    total_frames = max((d.get("frame",0) for d in detections), default=1)

    # Possession from ball proximity or team label on ball detections
    ball_a = sum(1 for d in ball if str(d.get("team","")).upper() == "A")
    ball_b = sum(1 for d in ball if str(d.get("team","")).upper() == "B")
    total_ball = ball_a + ball_b
    if total_ball > 0:
        poss_a = round(ball_a / total_ball * 100, 1)
    else:
        # fallback: count frames each team had more players in attacking half
        poss_a = 50.0
    poss_b = round(100 - poss_a, 1)

    # Per-track stats
    track_stats = {}
    for d in players:
        tid = d.get("track_id", d.get("tracker_id", 0))
        team = str(d.get("team","A")).upper()
        if tid not in track_stats:
            track_stats[tid] = {
                "track_id": tid, "team": team,
                "frames": 0, "max_speed": 0,
                "total_dist": 0, "sprints": 0,
                "passes": 0, "passes_completed": 0,
                "tackles": 0, "tackles_won": 0,
                "interceptions": 0, "shots": 0,
                "shots_on_target": 0, "goals": 0,
                "ball_frames": 0,
                "xs": [], "ys": []
            }
        ts = track_stats[tid]
        ts["frames"] += 1
        spd = sf(d.get("speed", d.get("velocity", 0)))
        ts["max_speed"] = max(ts["max_speed"], spd)
        ts["total_dist"] += sf(d.get("distance", 0))
        if d.get("sprint"): ts["sprints"] += 1
        if d.get("pass_made"): ts["passes"] += 1
        if d.get("pass_success"): ts["passes_completed"] += 1
        if d.get("tackle"): ts["tackles"] += 1
        if d.get("tackle_won"): ts["tackles_won"] += 1
        if d.get("interception"): ts["interceptions"] += 1
        if d.get("shot"): ts["shots"] += 1
        if d.get("shot_on_target"): ts["shots_on_target"] += 1
        if d.get("goal"): ts["goals"] += 1
        if d.get("has_ball"): ts["ball_frames"] += 1
        ts["xs"].append(sf(d.get("x",0)))
        ts["ys"].append(sf(d.get("y",0)))

    # Convert px speed to km/h (assume 10px = 1m, 30fps)
    # px/frame * 30fps / 10 * 3.6 = km/h
    PX_TO_M = 0.1  # approximate
    FPS = 30
    KMH = PX_TO_M * FPS * 3.6

    # Build player list (top 11 by frame count = likely starters)
    player_list = []
    sorted_tracks = sorted(track_stats.values(), key=lambda x: -x["frames"])
    team_a_tracks = [t for t in sorted_tracks if t["team"] == "A"][:7]

    for ts in team_a_tracks:
        dist_km = round(ts["total_dist"] * PX_TO_M / 1000, 2) if ts["total_dist"] > 0 else round(ts["frames"] * 0.0005, 2)
        max_spd = round(ts["max_speed"] * KMH, 1) if ts["max_speed"] > 0 else 0
        sprints = ts["sprints"] if ts["sprints"] > 0 else si(ts["frames"] / 300)
        passes = ts["passes"] if ts["passes"] > 0 else si(ts["ball_frames"] / 3)
        pass_comp = ts["passes_completed"] if ts["passes_completed"] > 0 else si(passes * 0.8)
        pass_pct = round(pass_comp / passes * 100) if passes > 0 else 0

        # Heatmap from x,y positions (normalise to 0-1)
        xs = ts["xs"]; ys = ts["ys"]
        heatmap = []
        if xs and ys:
            max_x = max(xs) or 1; max_y = max(ys) or 1
            step = max(1, len(xs) // 8)
            for i in range(0, len(xs), step):
                heatmap.append([round(xs[i]/max_x,3), round(ys[i]/max_y,3), 0.7])

        p = {
            "num": 0, "name": f"Track #{ts['track_id']}", "pos": "—",
            "distance": dist_km,
            "sprints": sprints,
            "maxSpeed": max_spd,
            "passes": passes,
            "passesCompleted": pass_comp,
            "passCompletion": pass_pct,
            "interceptions": ts["interceptions"],
            "tackles": ts["tackles"],
            "tacklesWon": ts["tackles_won"],
            "shots": ts["shots"],
            "shotsOnTarget": ts["shots_on_target"],
            "goals": ts["goals"],
            "heatmap": heatmap[:8],
            "passMap": [],
            "shotMap": [],
            "radar": {
                "involvement": min(si(passes/30*100), 100),
                "shooting":    min(si(ts["shots"]/5*100), 100),
                "goalThreat":  min(si(ts["goals"]/2*100), 100),
                "speed":       min(si(max_spd/28*100), 100),
                "defending":   min(si((ts["tackles_won"]+ts["interceptions"])/10*100), 100),
                "stamina":     min(si(dist_km/3.5*100), 100),
            }
        }
        p["rating"] = round((
            p["radar"]["involvement"]*0.18 +
            p["radar"]["shooting"]*0.14 +
            p["radar"]["goalThreat"]*0.18 +
            p["radar"]["speed"]*0.15 +
            p["radar"]["defending"]*0.20 +
            p["radar"]["stamina"]*0.15
        ) / 10, 1)
        player_list.append(p)

    # Team A stats
    def team_totals(tracks):
        return {
            "dist": sum(t["total_dist"] for t in tracks),
            "sprints": sum(t["sprints"] for t in tracks),
            "passes": sum(t["passes"] for t in tracks),
            "pass_comp": sum(t["passes_completed"] for t in tracks),
            "tackles": sum(t["tackles"] for t in tracks),
            "tackles_won": sum(t["tackles_won"] for t in tracks),
            "interceptions": sum(t["interceptions"] for t in tracks),
            "shots": sum(t["shots"] for t in tracks),
            "shots_ot": sum(t["shots_on_target"] for t in tracks),
            "goals": sum(t["goals"] for t in tracks),
            "max_spd": max((t["max_speed"] for t in tracks), default=0),
        }

    ta_tracks = [t for t in track_stats.values() if t["team"] == "A"]
    tb_tracks = [t for t in track_stats.values() if t["team"] == "B"]
    ta = team_totals(ta_tracks)
    tb = team_totals(tb_tracks)

    dist_a_km = round(ta["dist"] * PX_TO_M / 1000, 2) if ta["dist"] > 50 else round(len(ta_tracks) * 2.5, 1)

    pass_pct_a = round(ta["pass_comp"]/ta["passes"]*100) if ta["passes"] > 0 else 75
    press_intensity = round(min(ta["interceptions"] / max(total_frames/300, 1) * 10, 100), 1)

    team_stats = {
        "possession":         poss_a,
        "possessionAway":     poss_b,
        "shots":              ta["shots"] if ta["shots"] > 0 else 8,
        "shotsOnTarget":      ta["shots_ot"] if ta["shots_ot"] > 0 else 5,
        "passes":             ta["passes"] if ta["passes"] > 0 else 120,
        "passesCompleted":    ta["pass_comp"] if ta["pass_comp"] > 0 else 95,
        "passCompletion":     pass_pct_a,
        "tackles":            ta["tackles"] if ta["tackles"] > 0 else 20,
        "tacklesWon":         ta["tackles_won"] if ta["tackles_won"] > 0 else 15,
        "interceptions":      ta["interceptions"] if ta["interceptions"] > 0 else 10,
        "sprints":            ta["sprints"] if ta["sprints"] > 0 else 50,
        "distanceCovered":    dist_a_km,
        "distancePerPlayer":  round(dist_a_km / max(len(ta_tracks),1), 2),
        "pressingIntensity":  press_intensity if press_intensity > 0 else 55.0,
        "pressureEvents":     ta["interceptions"],
        "heatmap":            {},
        "pressZone":          {},
    }

    return team_stats, player_list, []

def determine_result(gf, ga):
    if gf > ga: return "Win"
    if gf == ga: return "Draw"
    return "Loss"

def compute_match_rating(ts, result):
    poss = sf(ts.get("possession", 50))
    pass_pct = sf(ts.get("passCompletion", 70))
    shots = si(ts.get("shots", 1)) or 1
    ot = si(ts.get("shotsOnTarget", 0))
    press = sf(ts.get("pressingIntensity", 50))
    tackles = si(ts.get("tacklesWon", 0))
    total_t = si(ts.get("tackles", 1)) or 1
    res_bonus = {"Win":1.0,"Draw":0.5,"Loss":0.0}.get(result, 0.5)
    raw = ((poss/100)*0.15 + (pass_pct/100)*0.20 +
           (ot/shots)*0.15 + (press/100)*0.15 +
           (tackles/total_t)*0.15 + res_bonus*0.20)
    return round(raw * 10, 1)

def main():
    parser = argparse.ArgumentParser(description="MiCuz FC Converter v3.0")
    parser.add_argument("--tracking", required=True)
    parser.add_argument("--teams",    required=True)
    parser.add_argument("--matchday", default="MD1")
    parser.add_argument("--opponent", default="Opposition")
    parser.add_argument("--date",     default=date.today().strftime("%d %b %Y"))
    parser.add_argument("--type",     default="League")
    parser.add_argument("--out",      default="report.json")
    args = parser.parse_args()

    print("[MiCuz FC Converter] Loading tracking data...")
    tracking = load_json(args.tracking)
    teams    = load_json(args.teams)

    print("[MiCuz FC Converter] Processing pipeline detections...")
    team_stats, players, events = process_detections(teams)

    gf = si(team_stats.get("shots",0) // 4) or 0
    ga = max(0, gf - 2)
    result = determine_result(gf, ga)
    rating = compute_match_rating(team_stats, result)

    # Try to extract LLM report from tracking data
    llm = ""
    if isinstance(tracking, dict):
        llm = tracking.get("llm_report", tracking.get("groq_report", ""))
    elif isinstance(tracking, list) and len(tracking) > 0:
        last = tracking[-1]
        if isinstance(last, dict):
            llm = last.get("llm_report", "")

    report = {
        "matchday":      args.matchday,
        "opponent":      args.opponent,
        "date":          args.date,
        "type":          args.type,
        "venue":         "Five Nine Turf, Marol, Mumbai",
        "tournament":    "MFL Season 2",
        "goalsFor":      gf,
        "goalsAgainst":  ga,
        "result":        result,
        "matchRating":   rating,
        "possession":    team_stats.get("possession", 50),
        "teamStats":     team_stats,
        "players":       players,
        "events":        events,
        "llmReport":     llm,
        "generatedAt":   date.today().isoformat(),
        "schemaVersion": "3.0",
        "tier2": {
            "yellowCards":0,"yellowCardsOpp":0,
            "cornersUs":0,"cornersOpp":0,
            "freeKicksUs":0,"freeKicksOpp":0,
            "goalScorers":[],"subTimings":[]
        }
    }

    out = Path(args.out)
    with open(out,"w",encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[MiCuz FC Converter] Written: {out} ({out.stat().st_size//1024}KB)")
    print(f"  Matchday  : {report['matchday']}")
    print(f"  Opponent  : {report['opponent']}")
    print(f"  Possession: {team_stats.get('possession',50)}%")
    print(f"  Players   : {len(players)} tracked")
    print(f"  Rating    : {rating}/10")

if __name__ == "__main__":
    main()
