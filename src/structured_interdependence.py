"""Structured (constructive) interdependence metric for Collab-Overcooked traces.

Following Biswas et al. (arXiv 2502.06976), but computed directly from the
Python action traces in the existing experiment JSON logs — no STRIPS
encoding layer.

An interdependence event = a counter handoff:
    agent_j executes place_obj_on_counter() at t0 holding object X, then
    agent_i executes pickup(X, counter) at t1 > t0.

A handoff is *constructive* if BOTH:
    - goal-reaching: object X (or its downstream cooked product) was delivered
    - non-looping:   agent_i did not previously hold X before t0

Usage:
    python structured_interdependence.py <episode.json>
    python structured_interdependence.py <directory>     # aggregates all *.json
"""
import json
import os
import sys
import glob


def load_episode(path):
    with open(path, "r") as f:
        return json.load(f)


def parse_action_name(action_str):
    """Return the function name preceding '(' in 'name(args)'."""
    return action_str.split("(", 1)[0].strip()


def parse_first_arg(action_str):
    """Return the first comma-separated argument inside parentheses, or None."""
    if "(" not in action_str or ")" not in action_str:
        return None
    inside = action_str.split("(", 1)[1].rsplit(")", 1)[0]
    if not inside.strip():
        return None
    return inside.split(",", 1)[0].strip()


def infer_placed_object(agent_trace, t_place):
    """Walk back to the most recent pickup/fill action by this agent."""
    prior = [e for e in agent_trace if e["timestamp"] < t_place]
    for entry in reversed(prior):
        name = parse_action_name(entry["action"])
        if name in ("pickup", "fill_dish_with_food"):
            obj = parse_first_arg(entry["action"])
            if obj:
                return obj
    return None


def find_handoffs(episode):
    a0, a1 = episode["total_action_list"]
    handoffs = []
    for giver_id, giver_trace, receiver_id, receiver_trace in [
        (0, a0, 1, a1),
        (1, a1, 0, a0),
    ]:
        for ge in giver_trace:
            if parse_action_name(ge["action"]) != "place_obj_on_counter":
                continue
            obj = infer_placed_object(giver_trace, ge["timestamp"])
            if obj is None:
                continue
            for re in receiver_trace:
                if re["timestamp"] <= ge["timestamp"]:
                    continue
                if parse_action_name(re["action"]) != "pickup":
                    continue
                pickup_obj = parse_first_arg(re["action"])
                pickup_place = (re["action"].split(",", 1)[1].rsplit(")", 1)[0].strip()
                                if "," in re["action"] else "")
                if pickup_obj == obj and "counter" in pickup_place:
                    handoffs.append({
                        "giver": giver_id,
                        "receiver": receiver_id,
                        "object": obj,
                        "t_place": ge["timestamp"],
                        "t_pickup": re["timestamp"],
                    })
                    break
    return handoffs


def delivered_objects(episode):
    """Walk traces to find the object held when deliver_soup() ran."""
    delivered = set()
    for trace in episode["total_action_list"]:
        for e in trace:
            if parse_action_name(e["action"]) != "deliver_soup":
                continue
            t = e["timestamp"]
            prior = [x for x in trace if x["timestamp"] < t]
            for prev in reversed(prior):
                if parse_action_name(prev["action"]) != "pickup":
                    continue
                place = (prev["action"].split(",", 1)[1].rsplit(")", 1)[0].strip()
                         if "," in prev["action"] else "")
                # Only count pickups from utensils (not dispenser / counter)
                if "ingredient_dispenser" in place or "counter" in place or "dish_dispenser" in place:
                    continue
                obj = parse_first_arg(prev["action"])
                if obj:
                    delivered.add(obj)
                break
    return delivered


def is_constructive(handoff, episode, delivered):
    obj = handoff["object"]
    # Goal-reaching: the raw object name appears in the delivered cooked name
    # (e.g. 'sweet_potato' is part of 'boiled_sweet_potato')
    goal = any(obj in d for d in delivered)
    # Non-looping: receiver did not hold obj before the handoff started
    receiver_trace = episode["total_action_list"][handoff["receiver"]]
    held_before = any(
        parse_action_name(e["action"]) == "pickup"
        and parse_first_arg(e["action"]) == obj
        and e["timestamp"] < handoff["t_place"]
        for e in receiver_trace
    )
    return goal and not held_before


def score(episode):
    handoffs = find_handoffs(episode)
    delivered = delivered_objects(episode)
    cons, noncons = 0, 0
    enriched = []
    for h in handoffs:
        h["constructive"] = is_constructive(h, episode, delivered)
        enriched.append(h)
        if h["constructive"]:
            cons += 1
        else:
            noncons += 1

    # Solo-completion: did one agent execute pickup-from-dispenser AND deliver_soup,
    # AND no constructive handoff was needed?
    solo_agent = None
    for idx, trace in enumerate(episode["total_action_list"]):
        actions = [e["action"] for e in trace]
        has_dispenser_pickup = any("ingredient_dispenser" in a for a in actions)
        has_deliver = any(parse_action_name(a) == "deliver_soup" for a in actions)
        if has_dispenser_pickup and has_deliver:
            solo_agent = idx
            break

    return {
        "delivered": episode.get("total_order_finished", 0) >= 1
                    if isinstance(episode.get("total_order_finished"), int)
                    else len(episode.get("total_order_finished", [])) >= 1,
        "Int_cons": cons,
        "Int_non_cons": noncons,
        "solo_completion": solo_agent is not None and cons == 0,
        "completing_agent": solo_agent if cons == 0 else None,
        "handoff_events": enriched,
    }


def score_path(path):
    episode = load_episode(path)
    out = score(episode)
    out["dish"] = os.path.basename(os.path.dirname(path)) or os.path.basename(path)
    out["file"] = path
    return out


def aggregate(results):
    n = len(results) or 1
    delivered = sum(1 for r in results if r["delivered"])
    total_cons = sum(r["Int_cons"] for r in results)
    total_noncons = sum(r["Int_non_cons"] for r in results)
    solo = sum(1 for r in results if r["solo_completion"])
    return {
        "n_episodes": len(results),
        "delivered_rate": delivered / n,
        "mean_Int_cons": total_cons / n,
        "mean_Int_non_cons": total_noncons / n,
        "solo_completion_rate": solo / n,
        "any_handoff_rate": sum(1 for r in results if r["handoff_events"]) / n,
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    target = sys.argv[1]
    if os.path.isdir(target):
        files = sorted(glob.glob(os.path.join(target, "**/*.json"), recursive=True))
        results = [score_path(f) for f in files]
        out = {"summary": aggregate(results), "episodes": results}
    else:
        out = score_path(target)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
