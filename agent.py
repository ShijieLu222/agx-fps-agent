"""AGX FPS v9 candidate adapted from the official public starter.

Source: https://agx.cool/docs/schema/fps.example.agent.py
Retrieved: 2026-11-15. Contract: rsi-fps-5.5.
Local changes: geometry-weighted routes, objective continuity, usable-weapon
selection, own-clock checks and a rifle-first defensive buy policy.
Duel pacing, contact memory and objective braking adapt ideas from the
user-supplied AGX_FPS_v3_3_Duel.zip (2026-11-15); no performance gain is proven.
v3 addressed range/exhausted-ammo stalls and objective/navigation failures.
v4 targets stopping, unseen-damage response and route failures observed in the
owner's second match. Offline checks do not establish a win-rate gain.
v5 adds final-waypoint arrival control; its
closed-loop fixtures are approximations, not official match evaluations.
v6 addresses false airborne arrivals and repeated box-climbing failures from
the owner's Dust2 match. Navigation source edges remain unverified candidates.
v7 confirms brief head-only sightings before firing and bounds lost-sight
braking, without treating spectator positions as legal target observations.
v8 separates navigation gaze from movement and searches after unseen damage.
Public spawn areas are search directions, never enemy positions or fire targets.
v9 adds map-derived guard stations, persistent witnessed retreats, sound-aware
preparation, execution-boundary shot clocks and distance-aware aiming/cadence.
"""
import math
import gc
import heapq
from itertools import count


# Frozen rsi-fps-5.5 catalogue, metres. Observation visibility can exceed a
# weapon's projectile range. Unknown weapons never authorize speculative fire.
WEAPON_RANGE = {
    **{name: 208.0768 for name in
       ("ak47", "m4a1", "m4a4", "awp", "galilar", "scar20", "sg553", "ssg08")},
    **{name: 104.0384 for name in
       ("pistol", "usp", "elite", "p250", "fiveseven", "deagle", "tec9")},
    "mac10": 91.44, "mp9": 91.44, "mp7": 91.44, "bizon": 91.44,
    "nova": 76.2, "xm1014": 76.2, "sawedoff": 35.56, "mag7": 35.56,
}


def clip(value):
    return max(-1.0, min(1.0, value))


def distance(a, b):
    return math.dist(a, b)


def angle(value):
    return (value + math.pi) % (2 * math.pi) - math.pi


def clock_ready(value, tick, ready_flag=False):
    """Read a public deadline, or a bool readiness/busy flag if supplied.

    Missing optional clocks preserve compatibility. Numeric clocks are absolute
    match ticks; do not confuse a past nonzero deadline with an active timer.
    """
    if value is None:
        return True
    if isinstance(value, bool):
        return value if ready_flag else not value
    return (type(value) in (int, float) and math.isfinite(value)
            and value <= tick)


def current_slot(me, fallback):
    slot = me.get("active_slot")
    return slot if type(slot) is int and slot in (0, 1, 2) else fallback


def counterstop(me):
    """Release both movement axes and let ground friction finish stopping.

    Reversing tiny velocities every 3-tick decision sustained a 0.78105 m/s
    oscillation in the owner's real match. Neutral input avoids that feedback
    without guessing engine acceleration. The fire/objective gates still check
    observed speed and grounding; neutral input alone never authorizes either.
    """
    return {"move_forward": 0.0, "move_right": 0.0}


class Player:
    def __init__(self):
        self.nodes, self.edges, self.sites = {}, {}, []
        self.spawn_views = {}
        self.guard_plans = {}
        self.pressure_points, self.zone_points = {}, {}
        self.walk_edges = set()
        self.source_edges = set()
        self.pending_map = None
        self.map_identity = None
        self.path, self.goal = [], None
        self.round = None
        self.last_position, self.stuck = None, 0
        self.anchor, self.anchor_tick = None, 0
        self.escape_until, self.escape_yaw, self.escapes = 0, 0.0, 0
        self.clear_route_history()
        self.reset_navigation()
        self.reset_tactics()

    def clear_route_history(self):
        self.route_threats = {}
        self.nav_side, self.nav_round = None, 0

    def remember_threat(self, o):
        """Cost a traversed approach after own unseen damage, not an enemy map.

        Keep a bounded, side-specific preference for four rounds. Edges remain
        traversable; a costly detour is not always safer than the short route.
        """
        if not self.path_nodes or self.nav_from is None:
            return
        there = self.path_nodes[0]
        edge = (self.nav_from, there)
        if edge not in self.walk_edges:
            return
        key = (o["side"],) + edge
        length = distance(self.nodes[edge[0]], self.nodes[edge[1]])
        previous = self.route_threats.get(key, (0, 0.0))
        old = previous[1] if previous[0] >= o["round_index"] else 0.0
        self.route_threats[key] = (o["round_index"] + 4,
                                   min(120.0, old + max(20.0, length)))
        if len(self.route_threats) > 256:
            oldest = min(self.route_threats, key=lambda k: self.route_threats[k][0])
            del self.route_threats[oldest]

    def reset_navigation(self):
        self.path_nodes, self.nav_from = [], None
        self.retry_node = None
        self.pending_leg = None
        self.nav_trace, self.trace_airborne = [], False
        self.recovery_trace = []
        self.recovery_until, self.recovery_edge = 0, None
        self.return_failures = set()
        self.jump_attempts, self.last_jump_tick = {}, -10000
        self.failed_edges = {}
        self.nav_tick = 0
        self.leg_target, self.leg_best, self.leg_progress_tick = None, math.inf, 0
        self.last_move_tick = None
        self.arrival_goal = None

    def track_ground(self, me, pos):
        """Retain a short actually walked corridor, never reverse a jump/drop.

        An unsuccessful jump may land back at its launch point. Only that
        spatially matched landing can reconnect the preceding ground trail.
        """
        if not me["grounded"]:
            self.trace_airborne = True
            return
        if self.nav_trace:
            previous = self.nav_trace[-1]
            if (math.dist(pos[:2], previous[:2]) > (0.2 if self.trace_airborne else 0.8)
                    or abs(pos[2] - previous[2]) > (0.2 if self.trace_airborne else 0.35)):
                self.nav_trace = []
        self.trace_airborne = False
        if not self.nav_trace or distance(pos, self.nav_trace[-1]) > 0.12:
            self.nav_trace.append(list(pos))
            self.nav_trace = self.nav_trace[-128:]

    def witnessed_return(self, here):
        """Return points to a reached anchor only along our continuous trail."""
        if self.trace_airborne or not self.nav_trace or here not in self.nodes:
            return []
        target = self.nodes[here]
        for index in range(len(self.nav_trace)-1, -1, -1):
            point = self.nav_trace[index]
            if (math.dist(point[:2], target[:2]) < 1.0
                    and abs(point[2]-target[2]) <= 0.6):
                return [list(p) for p in reversed(self.nav_trace[index:])]
        return []

    def penalize_leg(self):
        """Remember an observed failed traversal, never an unseen obstacle."""
        if not self.path_nodes:
            return
        there = self.path_nodes[0]
        # A search starting at the still-unreached destination must not erase
        # the failed leg's real origin or grant it a fresh jump allowance.
        if (self.nav_from is None and self.pending_leg is not None
                and there == self.pending_leg[1]):
            self.nav_from = self.pending_leg[0]
        if self.nav_from is None:
            # A spawn-to-first-node connector has no reached-node anchor.
            # Recover only along an explicit walk corridor containing our feet,
            # never by connecting to an arbitrary nearby or other-floor node.
            if there not in self.nodes or self.last_position is None:
                return
            candidates = []
            for here, target in self.walk_edges:
                if target != there:
                    continue
                a, b, pos = self.nodes[here], self.nodes[there], self.last_position
                dx, dy = b[0]-a[0], b[1]-a[1]
                squared = dx*dx + dy*dy
                if squared < 0.01:
                    continue
                along = ((pos[0]-a[0])*dx + (pos[1]-a[1])*dy) / squared
                if not 0.0 < along < 1.0:
                    continue
                projected = [a[i] + along*(b[i]-a[i]) for i in range(3)]
                lateral = math.dist(pos[:2], projected[:2])
                # Do not walk back up a ledge whose expected floor was lost.
                if lateral <= 0.5 and abs(pos[2]-projected[2]) <= 0.6:
                    candidates.append((lateral, distance(pos, a), here))
            if not candidates:
                return
            # Key avoids comparing mixed integer/string navigation identifiers.
            here = min(candidates, key=lambda row: row[:2])[2]
            key = (here, there)
            length = distance(self.nodes[here], self.nodes[there])
            old = self.failed_edges.get(key, (0, 0.0))[1]
            self.failed_edges[key] = (self.nav_tick + 1800,
                                     min(500.0, old + max(60.0, 5*length)))
            self.retry_node = here
            return
        if there is None or there == self.nav_from:
            return
        key = (self.nav_from, there)
        self.pending_leg = key
        # Backtrack only over an explicitly reversible walk at a compatible
        # foot height. A completed descent cannot be reversed by a virtual
        # start connector. Continue toward that leg's published destination;
        # do not snap to arbitrary nearby nodes on the lower floor.
        can_backtrack = (key in self.walk_edges
                         and self.last_position is not None
                         and abs(self.last_position[2]
                                 - self.nodes[self.nav_from][2]) <= 0.6)
        witnessed = ([] if (key in self.return_failures
                            or (key not in self.walk_edges and key not in self.source_edges)) else
                     self.witnessed_return(self.nav_from))
        self.retry_node = self.nav_from if can_backtrack or witnessed else there
        self.recovery_trace = witnessed if not can_backtrack else []
        if self.recovery_trace:
            self.recovery_edge = key
            length = sum(distance(a, b) for a, b in zip(witnessed, witnessed[1:]))
            self.recovery_until = self.nav_tick + min(360, 90 + int(60 * length))
        old = self.failed_edges.get(key, (0, 0.0))[1]
        length = distance(self.nodes[self.nav_from], self.nodes[there])
        self.failed_edges[key] = (self.nav_tick + 1800,
                                 min(500.0, old + max(60.0, 5 * length)))
        if len(self.failed_edges) > 256:
            oldest = min(self.failed_edges, key=lambda k: self.failed_edges[k][0])
            del self.failed_edges[oldest]

    def reset_tactics(self):
        self.contact_until = 0
        self.enemy_memory = None
        self.target_seat = None
        self.last_fire = False
        self.burst_start = None
        self.fire_slot = None
        self.known_bomb = None
        self.engagement_started = None
        self.advance_until = 0
        self.search_site = None
        self.search_arrival = None
        self.searched_sites = set()
        self.last_hp = None
        self.last_safety_tick = None
        self.last_legal_visible = False
        self.breadcrumbs = []
        self.retreat_path = []
        self.retreat_until = 0
        self.damage_cooldown = 0
        self.sight_tracks = {}
        self.last_combat_tick = None
        self.last_visible_tick = None
        self.lost_hold_used, self.hold_last_tick = 0, None
        self.peek_advance_until = 0
        self.unconfirmed_head_since = None
        self.damage_watch_start = None
        self.damage_watch_yaw, self.damage_watch_sign = 0.0, 1.0
        self.sound_yaw, self.sound_until = None, 0
        self.head_targets = set()
        self.guard_site, self.guard_choice = None, 0
        self.guard_bad = set()
        self.guard_started, self.guard_best = None, math.inf
        self.guard_escape_count = 0
        self.attack_site = None
        self.team_status = {}
        self.support_zone, self.support_until = None, 0
        self.pressure_key, self.pressure_index = None, -1
        self.pressure_arrival, self.pressure_started = None, 0
        self.pressure_bad = set()

    def observe_team(self, o):
        """Only an observed alive->dead transition is a new support event.

        Teammate observations have zones, not enemy coordinates or HP. Do not
        assign invented meanings to other agents' numeric signals.
        """
        for mate in o.get("teammates", []):
            seat = mate.get("seat")
            old = self.team_status.get(seat)
            zone = mate.get("zone")
            if old and old[0] and mate.get("alive") is False:
                zone = zone if zone in self.zone_points else old[1]
                if zone in self.zone_points:
                    self.support_zone = zone
                    self.support_until = o["match_tick"] + 720
                    self.pressure_key = None
            self.team_status[seat] = (mate.get("alive"), zone)

    def pressure_goal(self, o, index, base):
        """Short entrance check, then push connected public approach nodes.

        Cycle back through the staging position after two peeks. A failed
        forward position is retired for this cycle; combat is not a stall.
        """
        key = (o["side"], index, self.support_zone if o["match_tick"] < self.support_until else None)
        points = self.pressure_points.get(key[:2], [])
        tick, m = o["match_tick"], o["me"]
        pos = [m[k] for k in ("x", "y", "z")]
        if self.pressure_key != key:
            self.pressure_key, self.pressure_index = key, -1
            self.pressure_arrival, self.pressure_started = None, tick
            self.pressure_bad = set()
        goal = base if self.pressure_index < 0 else points[self.pressure_index]
        arrived = (m["grounded"] and distance(pos, goal) < 1.2
                   and abs(pos[2]-goal[2]) < .6)
        if arrived:
            if self.pressure_arrival is None:
                self.pressure_arrival = tick
        else:
            self.pressure_arrival = None
        if o["visible"] or self.last_move_tick is None or tick-self.last_move_tick > 6:
            self.pressure_started = tick
        stalled = (self.pressure_index >= 0 and not arrived
                   and tick-self.pressure_started > 240
                   and tick-self.leg_progress_tick > 180)
        dwell = 45 if o["side"] == "attack" or self.pressure_index >= 0 else 90
        if stalled or (self.pressure_arrival is not None and tick-self.pressure_arrival >= dwell):
            if stalled:
                self.pressure_bad.add(self.pressure_index)
            choices = [i for i in range(self.pressure_index+1, len(points))
                       if i not in self.pressure_bad]
            self.pressure_index = choices[0] if choices else -1
            self.pressure_arrival, self.pressure_started = None, tick
            goal = base if self.pressure_index < 0 else points[self.pressure_index]
        return list(goal)

    def approach_view(self, o):
        """A public opposing spawn area to check, not a tracked opponent."""
        return self.spawn_views.get("CT" if o["side"] == "attack" else "T")

    def observe_sounds(self, o):
        """An anonymous stereo cue can prioritize looking, never authorize fire."""
        if o["me"]["blind_ticks"] or o["visible"]:
            self.sound_yaw, self.sound_until = None, 0
            return
        if o["match_tick"] < self.sound_until:
            return  # Do not chase a moving relative bearing on every callback.
        self.sound_yaw = None
        cues = [s for s in o.get("sounds", [])
                if s.get("type") in ("footstep", "fire", "shot", "reload", "plant", "defuse")
                and s.get("direction") in ("left", "right", "center")
                and s.get("distance_band") in ("near", "mid")]
        if cues:
            cue = min(cues, key=lambda s: s["distance_band"] != "near")
            offset = {"left": math.pi/3, "right": -math.pi/3, "center": 0.}[cue["direction"]]
            self.sound_yaw = angle(o["me"]["yaw"] + offset)
            self.sound_until = o["match_tick"] + 30

    def build_guard_plans(self, row):
        """Derive stations/entrances from public map hints and navigation nodes.

        Published stations and box outlines are candidates, not a guarantee of
        collision clearance or safety. Route failures retire candidates; only
        actually walked ground is ever used for an emergency reverse.
        """
        hints, boxes = row.get("bot_waypoints", {}), row.get("boxes", [])
        if not self.nodes or not (self.spawn_views or boxes or hints):
            return
        pads = (row.get("spawns") or {}).get("CT", [])
        reachable = set()
        for pad in pads:
            p = pad.get("position")
            if p:
                reachable.add(min(self.nodes, key=lambda n: distance(p, self.nodes[n])))
        pending = list(reachable)
        while pending:
            here = pending.pop()
            for there, _ in self.edges[here]:
                if there not in reachable:
                    reachable.add(there)
                    pending.append(there)
        for index, site in enumerate(self.sites):
            center = site["center"]
            name = site.get("id", str(index))
            published = hints.get("hold" + name, [])
            watch = [list(p) for p in hints.get("watch" + name + "_CT", [])]
            if not watch:
                near = min(self.nodes, key=lambda n: distance(center, self.nodes[n]))
                approach = self.spawn_views.get("T", center)
                neighbors = [self.nodes[n] for n, _ in self.edges.get(near, [])
                             if math.dist(center[:2], self.nodes[n][:2]) >= 3]
                neighbors.sort(key=lambda p: distance(p, approach))
                watch = [[p[0], p[1], p[2] + 1.18] for p in neighbors[:3]]
                if not watch and "T" in self.spawn_views:
                    watch = [list(self.spawn_views["T"])]
            if not watch:
                continue
            candidates = []
            for n, p in self.nodes.items():
                if pads and n not in reachable:
                    continue
                dist = distance(p, center)
                if not 1.4 <= dist <= max(18., site["radius"] * 2.5):
                    continue
                if abs(p[2] - center[2]) > 1.5 or not self.edges.get(n):
                    continue
                hint_distance = min((distance(p, h) for h in published), default=0.)
                if published and hint_distance > 1.6:
                    continue
                blocked, cover_gap = False, math.inf
                for box in boxes:
                    if not all(k in box for k in ("x", "y", "z", "w", "d", "h")):
                        continue
                    if not box["z"] - .3 <= p[2] < box["z"] + box["h"] - .3:
                        continue
                    dx = abs(p[0]-box["x"]) - box["w"]/2
                    dy = abs(p[1]-box["y"]) - box["d"]/2
                    if dx < .35 and dy < .35:
                        blocked = True
                        break
                    if box["z"] + box["h"] >= p[2] + 1.2:
                        cover_gap = min(cover_gap, math.hypot(max(0, dx), max(0, dy)))
                if blocked:
                    continue
                score = (hint_distance * 2 + abs(p[2]-center[2]) * 4
                         + abs(dist - max(3., site["radius"])) * .15
                         - (2. if .35 <= cover_gap <= 3. else 0.))
                candidates.append((score, str(n), list(p)))
            candidates.sort()
            stations = []
            for _, _, point in candidates:
                if all(distance(point, old) >= 2. for old in stations):
                    stations.append(point)
                if len(stations) == 4:
                    break
            if stations:
                self.guard_plans[index] = {"stations": stations, "watch": watch}

        # Public geometry supplies bounded advance targets, never enemy tracks.
        # Reuse the spawn-connected graph and reject obvious box interiors.
        for side, opposing in (("defend", "T"), ("attack", "CT")):
            toward = self.spawn_views.get(opposing)
            if toward is None:
                continue
            for index, site in enumerate(self.sites):
                c = site["center"]
                vx, vy = toward[0]-c[0], toward[1]-c[1]
                norm = math.hypot(vx, vy)
                if norm < 1.:
                    continue
                candidates = []
                for n, p in self.nodes.items():
                    if (pads and n not in reachable) or not self.edges.get(n):
                        continue
                    dx, dy = p[0]-c[0], p[1]-c[1]
                    dist = math.hypot(dx, dy)
                    projection = (dx*vx+dy*vy)/norm
                    if not (4 <= dist <= 24 and projection >= dist*.35 and abs(p[2]-c[2]) <= 1.5):
                        continue
                    if any(all(k in b for k in ("x","y","z","w","d","h"))
                           and b["z"]-.3 <= p[2] < b["z"]+b["h"]-.3
                           and abs(p[0]-b["x"]) < b["w"]/2+.35
                           and abs(p[1]-b["y"]) < b["d"]/2+.35 for b in boxes):
                        continue
                    candidates.append((p, projection, dist))
                points = []
                for desired in (10., 18.):
                    options = [(abs(d-desired)+(d-proj)*.5+abs(p[2]-c[2])*3, str(p), p)
                               for p, proj, d in candidates
                               if all(distance(p, q) > 4 for q in points)]
                    if options:
                        points.append(list(min(options)[2]))
                self.pressure_points[(side, index)] = points

    def guard_goal(self, o, site, plan):
        m, tick = o["me"], o["match_tick"]
        pos = [m[k] for k in ("x", "y", "z")]
        if self.guard_site != site:
            self.guard_site, self.guard_bad = site, set()
            self.guard_choice = (o["seat"] % 5 // 2) % len(plan["stations"])
            self.guard_started, self.guard_best = tick, math.inf
            self.guard_escape_count = self.escapes
        goal = plan["stations"][self.guard_choice]
        dist = distance(pos, goal)
        if dist < self.guard_best - .4:
            self.guard_best, self.guard_started = dist, tick
        if (o["visible"] or self.last_move_tick is None
                or tick-self.last_move_tick > 6):
            self.guard_started = tick
        # Do not count combat, planting or externally held observations as
        # failed travel. At most one candidate is retired per three seconds.
        if (self.pressure_index < 0 and dist > 1.2 and tick - self.guard_started > 180
                and not o["visible"] and self.last_move_tick is not None
                and tick-self.last_move_tick <= 6
                and (tick-self.leg_progress_tick > 180
                     or self.escapes-self.guard_escape_count >= 2)):
            self.guard_bad.add(self.guard_choice)
        if self.guard_choice in self.guard_bad:
            choices = [i for i in range(len(plan["stations"])) if i not in self.guard_bad]
            if not choices:
                # Hold this reached ground instead of cycling failed stations.
                return list(pos)
            self.guard_choice = min(choices, key=lambda i: distance(pos, plan["stations"][i]))
            self.guard_started, self.guard_best = tick, math.inf
            self.guard_escape_count = self.escapes
            self.path, self.goal, self.arrival_goal = [], None, None
            self.retry_node, self.pending_leg = None, None
            self.recovery_trace = []
            goal = plan["stations"][self.guard_choice]
        return list(goal)

    def guard_action(self, o, action, plan, moving=False):
        m, tick = o["me"], o["match_tick"]
        points = [p for p in plan["watch"]
                  if math.hypot(p[0]-m["x"], p[1]-m["y"]) >= 3.]
        if not points:
            fallback = self.approach_view(o)
            points = [fallback] if fallback is not None else []
        if not points:
            return dict(action, **counterstop(m)) if not moving else action
        # Most time on the primary entrance, brief checks of secondary lanes.
        # Never continuously turn our back through a full 360-degree scan.
        phase = o["round_tick"] % 360
        primary = (o["seat"] % 5 // 2) % len(points)
        index = primary if phase < 270 else (primary+1+(phase-270)//45) % len(points)
        point = points[index]
        yaw, pitch = self.aim_error(m, point)
        if self.sound_yaw is not None and tick < self.sound_until:
            yaw, pitch = angle(self.sound_yaw-m["yaw"]), -m["pitch"]
        result = dict(action, yaw_rate=clip(yaw/.15), pitch_rate=clip(pitch/.1))
        if not moving:
            result.update(counterstop(m))
            result.update(fire=False, crouch=True, jump=False)
        return result

    def prepare_approach(self, o, action):
        """Reduce entry speed for a nearby route corner or a fresh sound cue."""
        if (o["visible"] or action.get("use") or action.get("jump")
                or not o["me"]["grounded"] or o["match_tick"] < self.escape_until
                or self.recovery_trace):
            return action
        corner = False
        if len(self.path) >= 2:
            m = o["me"]
            p, q = self.path[:2]
            dx, dy = p[0]-m["x"], p[1]-m["y"]
            ax, ay = q[0]-p[0], q[1]-p[1]
            if 1. <= math.hypot(dx, dy) <= 5. and math.hypot(ax, ay) > 1.:
                turn = angle(math.atan2(ay, ax)-math.atan2(dy, dx))
                corner = abs(turn) >= .65
        sound = self.sound_yaw is not None and o["match_tick"] < self.sound_until
        if corner or sound:
            action = dict(action)
            for key in ("move_forward", "move_right"):
                if key in action:
                    action[key] *= .55
            action["walk"] = True
        return action

    def watch_navigation(self, o, action, recovering=False):
        """Turn the view without replacing the world direction of movement.

        The route controller already expresses movement in the CURRENT view
        basis. Only view controls change here; the next callback recomputes
        movement from the new observed yaw. Never synthesize visible targets.
        """
        m, tick = o["me"], o["match_tick"]
        if (m["blind_ticks"] or o["visible"] or not m["grounded"]
                or action.get("use") or action.get("jump")):
            return action
        elapsed = (tick - self.damage_watch_start
                   if self.damage_watch_start is not None else 108)
        if 0 <= elapsed < 108:
            # Check one side, then the other. Repeated hits do not restart
            # the first sector forever. No attacker bearing is invented.
            offset = self.damage_watch_sign * (math.pi / 2 if elapsed < 36
                                               else -math.pi / 2)
            yaw = angle(self.damage_watch_yaw + offset - m["yaw"])
            return dict(action, yaw_rate=clip(yaw / 0.15),
                        pitch_rate=clip(-m["pitch"] / 0.1))
        if (self.sound_yaw is not None and tick < self.sound_until
                and not recovering and self.enemy_memory is None
                and tick >= self.escape_until and not self.recovery_trace):
            yaw = angle(self.sound_yaw - m["yaw"])
            return dict(action, yaw_rate=clip(yaw / 0.15),
                        pitch_rate=clip(-m["pitch"] / 0.1))
        if (recovering or tick < self.escape_until or self.recovery_trace
                or self.enemy_memory is not None):
            return action
        # Defenders use guard_action's entrance selection. On attack, alternate
        # the published defending approach and the route before first contact.
        point = self.approach_view(o) if o["side"] == "attack" else None
        moving = math.hypot(action.get("move_forward", 0.0),
                            action.get("move_right", 0.0)) > 0.1
        if (point is None or not moving or o["round_tick"] % 120 >= 72
                or math.dist([m["x"], m["y"]], point[:2]) < 4.0):
            return action
        yaw, pitch = self.aim_error(m, point)
        return dict(action, yaw_rate=clip(yaw / 0.15),
                    pitch_rate=clip(pitch / 0.1))

    @staticmethod
    def head_only(enemy):
        parts = enemy.get("parts", [])
        return "head" in parts and not any(p in parts for p in ("chest", "abdomen"))

    def observe_targets(self, o):
        """Track only consecutive, current legal sightings at decision cadence."""
        tick = o["match_tick"]
        current = {}
        for enemy in ([] if o["me"]["blind_ticks"] else o["visible"]):
            seat = enemy.get("seat")
            pose = (bool(enemy["crouch"]), self.head_only(enemy))
            old = self.sight_tracks.get(seat)
            since = (old[2] if old and old[0] in (tick-3, tick)
                     and old[1] == pose else tick)
            current[seat] = (tick, pose, since)
        # An absent or blinded target cannot accumulate stable time in memory.
        self.sight_tracks = current
        self.head_targets.intersection_update(current)
        if current:
            # Suppressing combat during a navigation window is not a quiet
            # encounter. Only an actual gap in legal sightings rearms holding.
            if self.last_visible_tick is None or tick-self.last_visible_tick >= 60:
                self.lost_hold_used = 0
                self.unconfirmed_head_since = None
            self.last_visible_tick = tick

    def target_ready(self, enemy, tick):
        if not any(p in enemy.get("parts", []) for p in ("chest", "abdomen", "head")):
            return False
        if not self.head_only(enemy):
            return True
        track = self.sight_tracks.get(enemy.get("seat"))
        return bool(track and track[0] == tick and tick-track[2] >= 6)

    def bound_head_wait(self, o, enemies):
        """Changing posture/identity cannot renew an unproductive head duel."""
        tick = o["match_tick"]
        visible = enemies
        if visible and all(self.head_only(e) and not self.target_ready(e, tick)
                           for e in visible):
            if self.unconfirmed_head_since is None:
                self.unconfirmed_head_since = tick
            elif tick-self.unconfirmed_head_since >= 90:
                self.peek_advance_until = tick + 30
                self.unconfirmed_head_since = tick
        elif o["visible"] or self.last_visible_tick is None or tick-self.last_visible_tick >= 60:
            self.unconfirmed_head_since = None

    def lost_contact_hold(self, o, action):
        """Briefly stay stopped without sight, with a non-refreshing budget."""
        m, tick = o["me"], o["match_tick"]
        if (o["visible"] or m["blind_ticks"] or not m["grounded"]
                or self.enemy_memory is None or self.last_combat_tick is None
                or tick-self.last_combat_tick > 9
                or m["carrying_bomb"]
                or (o["side"] == "defend" and o["bomb"]["state"] == "planted")
                or tick < self.advance_until or tick < self.peek_advance_until):
            return None
        if self.lost_hold_used >= 30:
            # Brief repeated glimpses must not indefinitely alternate braking
            # and tiny movement requests. A stable head/body can still defend.
            self.peek_advance_until = tick + 30
            self.contact_until, self.enemy_memory = 0, None
            return None
        elapsed = max(0, min(3, tick-(self.hold_last_tick or self.last_combat_tick)))
        self.lost_hold_used += elapsed
        self.hold_last_tick = tick
        return self.quiet_action(o, dict(action, **counterstop(m),
                                         jump=False, fire=False, walk=False))

    def observe_safety(self, o):
        """Remember only our own continuous, grounded recent traversal."""
        m, tick = o["me"], o["match_tick"]
        hp = m.get("hp")
        damaged = (isinstance(hp, (int, float)) and self.last_hp is not None
                   and hp < self.last_hp - 0.5)
        pos = [m["x"], m["y"], m["z"]]
        if (not m["grounded"] or self.last_safety_tick is None
                or tick - self.last_safety_tick > 6
                or (self.breadcrumbs and
                    (distance(pos, self.breadcrumbs[-1][1]) > 0.8
                     or abs(pos[2]-self.breadcrumbs[-1][1][2]) > 0.15))):
            self.breadcrumbs = []
            self.retreat_path = []
        if m["grounded"]:
            # Spatial memory, not a two-second queue: standing still must not
            # overwrite the path that actually led to our guard position.
            if not self.breadcrumbs or distance(pos, self.breadcrumbs[-1][1]) >= 0.12:
                self.breadcrumbs.append((tick, pos))
                self.breadcrumbs = self.breadcrumbs[-64:]
        self.last_hp = hp if isinstance(hp, (int, float)) else None
        self.last_safety_tick = tick
        self.last_legal_visible = bool(o["visible"] and not m["blind_ticks"])
        return damaged

    def unseen_damage_escape(self, o, action, damaged):
        """Briefly reverse a witnessed walk, never guess a hidden attacker."""
        m, tick = o["me"], o["match_tick"]
        pos = [m["x"], m["y"], m["z"]]
        if self.last_legal_visible:
            self.retreat_path = []
            self.retreat_until = 0
            self.damage_watch_start = None
            return None
        if (damaged and m["grounded"] and not m["blind_ticks"]
                and (self.damage_watch_start is None
                     or tick - self.damage_watch_start >= 108)):
            self.damage_watch_start, self.damage_watch_yaw = tick, m["yaw"]
            point = self.approach_view(o)
            offset = self.aim_error(m, point)[0] if point is not None else 0.0
            if self.sound_yaw is not None and tick < self.sound_until:
                offset = angle(self.sound_yaw - m["yaw"])
            self.damage_watch_sign = (1.0 if offset > 0 else -1.0) if abs(offset) > 0.1 else (1.0 if o["seat"] % 2 else -1.0)
        if damaged and tick >= self.damage_cooldown and m["grounded"]:
            reverse, length, previous = [], 0.0, pos
            for _, point in reversed(self.breadcrumbs[:-1]):
                # No reversal over a jump/drop or to a different floor. Use
                # successive samples, not a straight shortcut around a corner.
                if abs(point[2]-pos[2]) > 0.35:
                    break
                segment = distance(previous, point)
                if segment > 0.8:
                    break
                if segment < 0.12:
                    continue
                reverse.append(point)
                length += segment
                previous = point
                if length >= 3.0:
                    break
            if length >= 0.3:
                self.retreat_path = reverse
                self.retreat_until = tick + 54
                if self.pressure_index >= 0:
                    self.pressure_bad.add(self.pressure_index)
                    self.pressure_index, self.pressure_arrival = -1, None
                if self.guard_site is not None:
                    self.guard_bad.add(self.guard_choice)
                self.damage_cooldown = tick + 180
                self.remember_threat(o)
                self.enemy_memory, self.target_seat = None, None
                self.contact_until = 0
                self.escape_until = 0
                self.path = []
                self.last_move_tick = None
        if (not self.retreat_path or tick >= self.retreat_until
                or not m["grounded"]):
            self.retreat_path = []
            return None
        while self.retreat_path and distance(pos, self.retreat_path[0]) < 0.18:
            self.retreat_path.pop(0)
        if not self.retreat_path:
            return None
        point = self.retreat_path[0]
        if abs(point[2]-pos[2]) > 0.35 or distance(pos, point) > 1.0:
            self.retreat_path = []
            return None
        yaw = angle(math.atan2(-(point[1]-pos[1]), point[0]-pos[0])-m["yaw"])
        self.last_fire, self.burst_start = False, None
        # Backtrack the witnessed corridor while searching, rather than keeping
        # a direction that has already failed to reveal the attacker.
        return self.watch_navigation(o, {"weapon_slot": current_slot(m, action["weapon_slot"]),
                "move_forward": 0.85 * math.cos(yaw),
                "move_right": -0.85 * math.sin(yaw),
                "yaw_rate": 0.0, "pitch_rate": 0.0, "fire": False,
                "crouch": False, "walk": False, "jump": False}, recovering=True)

    def load_map(self, row):
        self.clear_route_history()
        self.reset_navigation()
        self.sites = row.get("sites") or []
        self.spawn_views = {}
        self.guard_plans = {}
        self.pressure_points = {}
        self.zone_points = {k: list(p) for k, p in row.get("route_landmarks", {}).items()
                            if isinstance(p, (list, tuple)) and len(p) == 3
                            and all(type(v) in (int, float) and math.isfinite(v) for v in p)}
        for side, pads in (row.get("spawns") or {}).items():
            positions = [p.get("position") for p in pads if isinstance(p, dict)]
            positions = [p for p in positions if isinstance(p, (list, tuple))
                         and len(p) == 3 and all(type(v) in (int, float)
                                                and math.isfinite(v) for v in p)]
            if positions:
                self.spawn_views[side] = [sum(p[i] for p in positions) / len(positions)
                                         for i in range(3)]
                self.spawn_views[side][2] += 1.18
        nav = row.get("nav") or {}
        self.nodes = {n["id"]: n["position"]
                      for n in nav.get("nodes", [])}
        self.edges = {key: [] for key in self.nodes}
        self.walk_edges = set()
        self.source_edges = set()
        for edge in nav.get("edges", []):
            if (edge["from"] in self.nodes and edge["to"] in self.nodes
                    and edge["kind"] != "ladder"):
                cost = distance(self.nodes[edge["from"]],
                                self.nodes[edge["to"]])
                if edge["kind"] == "source-adjacency":
                    self.source_edges.add((edge["from"], edge["to"]))
                    # These edges are not collision-validated. Prefer a longer
                    # gradual ascent to a short series of tall box steps. This
                    # soft cost is a routing preference, NOT a jump-height law.
                    rise = self.nodes[edge["to"]][2] - self.nodes[edge["from"]][2]
                    cost += 30.0 * max(0.0, rise - 0.35) ** 2
                self.edges[edge["from"]].append((edge["to"], cost))
                if edge["kind"] == "walk":
                    self.edges[edge["to"]].append((edge["from"], cost))
                    self.walk_edges.add((edge["from"], edge["to"]))
                    self.walk_edges.add((edge["to"], edge["from"]))

        self.build_guard_plans(row)

    def route(self, start, goal):
        self.nav_from = None
        self.failed_edges = {key: value for key, value in self.failed_edges.items()
                             if value[0] > self.nav_tick}
        self.route_threats = {key: value for key, value in self.route_threats.items()
                             if value[0] >= self.nav_round}
        if not self.nodes:
            self.path_nodes = [None]
            return [goal]
        first = (self.retry_node if self.retry_node in self.nodes else
                 min(self.nodes, key=lambda k: distance(start, self.nodes[k])))
        last = min(self.nodes,
                   key=lambda k: distance(goal, self.nodes[k]))
        # Euclidean distance is admissible for these geometric edge lengths.
        # A counter keeps heap ties independent of int/string node ID types.
        order = count()
        queue = [(distance(self.nodes[first], self.nodes[last]),
                  next(order), 0.0, first)]
        parent, best = {first: None}, {first: 0.0}
        while queue:
            _, _, cost, here = heapq.heappop(queue)
            if cost != best[here]:
                continue
            if here == last:
                break
            for there, length in self.edges[here]:
                penalty = self.failed_edges.get((here, there), (0, 0.0))[1]
                penalty += self.route_threats.get(
                    (self.nav_side, here, there), (0, 0.0))[1]
                candidate = cost + length + penalty
                if candidate < best.get(there, math.inf):
                    best[there] = candidate
                    parent[there] = here
                    estimate = candidate + distance(self.nodes[there],
                                                    self.nodes[last])
                    heapq.heappush(queue, (estimate, next(order),
                                          candidate, there))
        if last not in parent:
            if self.retry_node in self.nodes:
                # Recovery must not bypass a disconnected/directed graph by
                # drawing a direct line from a lower floor to the old goal.
                self.path_nodes = [first]
                return [self.nodes[first]]
            self.path_nodes = [None]
            return [goal]
        path, path_nodes, cursor = [goal], [None], last
        while cursor is not None:
            path.append(self.nodes[cursor])
            path_nodes.append(cursor)
            cursor = parent[cursor]
        self.path_nodes = list(reversed(path_nodes))
        return list(reversed(path))

    def buy(self, o):
        m = o["me"]
        rifle = "ak47" if o["side"] == "attack" else "m4a1"
        price = 2700 if o["side"] == "attack" else 2900
        if m["weapons"][0] is None and m["cash"] >= price:
            return {"buy": rifle}
        # A bounded force-buy after repeated losses; preserve early-round
        # saving instead of buying a cheap primary on every low-budget round.
        budget_gun, budget_price = (("galilar", 1800) if o["side"] == "attack"
                                    else ("mp9", 1250))
        loss = o.get("loss_level", 0)
        if (m["weapons"][0] is None and type(loss) in (int, float) and loss >= 2
                and m["cash"] >= budget_price + (650 if m["armor"] < 50 else 0)):
            return {"buy": budget_gun}
        if m["armor"] < 50 and m["cash"] >= 650:
            return {"buy": "armor"}
        if (m["weapons"][0] is not None and not m.get("helmet", False)
                and m["cash"] >= 1000):
            return {"buy": "helmet"}
        if (o["side"] == "defend" and not m["has_kit"]
                and m["cash"] >= 400):
            return {"buy": "defusekit"}
        return {}

    def weapon_slot(self, me, contact=False, tick=None):
        primary, secondary = me["weapons"][:2]
        # The public reload deadline describes work already in progress. Do not
        # cancel an active primary reload just because contact begins midway.
        if (primary and tick is not None and current_slot(me, None) == 0
                and not clock_ready(primary.get("reload_until"), tick)):
            return 0
        if (primary and primary.get("ammo", 0) == 0
                and secondary and secondary.get("ammo", 0) > 0
                and (contact or primary.get("reserve", 0) == 0)):
            return 1
        # Outside contact, restore and reload a primary that still has reserve.
        for slot in (0, 1):
            gun = me["weapons"][slot]
            if gun and (gun.get("ammo", 0) > 0 or gun.get("reserve", 0) > 0):
                return slot
        return 0 if me["weapons"][0] is not None else 1

    def select_enemy(self, o, enemies=None):
        m = o["me"]
        enemies = o["visible"] if enemies is None else enemies

        def squared(enemy):
            return ((enemy["x"] - m["x"]) ** 2
                    + (enemy["y"] - m["y"]) ** 2
                    + (enemy["z"] - m["z"]) ** 2)

        nearest = min(enemies, key=squared)
        held = next((e for e in enemies
                     if self.target_seat is not None
                     and e.get("seat") == self.target_seat), None)
        # Keep an available target unless an alternative is at least 25% closer.
        enemy = (held if held is not None
                 and squared(nearest) >= squared(held) * 0.75 ** 2
                 else nearest)
        self.target_seat = enemy.get("seat")
        return enemy

    def reachable_enemies(self, o, gun):
        """Only current legal sight within the held gun's conservative range."""
        if not gun or gun.get("ammo", 0) <= 0 or o["me"]["blind_ticks"]:
            return []
        reach = max(0.0, WEAPON_RANGE.get(gun.get("id"), 0.0) - 0.75)
        m = o["me"]
        eye = [m["x"], m["y"], m["z"] + (0.95 if m["crouch"] else 1.62)]
        return [e for e in o["visible"]
                if any(p in e.get("parts", []) for p in ("head", "chest", "abdomen"))
                and distance(eye, self.aim_point(e)) < reach]

    def unknown_bomb_goal(self, o, pos):
        """Search public sites; never infer the hidden bomb coordinate."""
        announced = next((i for i, site in enumerate(self.sites)
                          if site.get("id") is not None
                          and site.get("id") == o["bomb"].get("site")), None)
        if announced is not None:
            return list(self.sites[announced]["center"])
        tick = o["match_tick"]
        if self.search_site is not None:
            site = self.sites[self.search_site]
            if distance(pos, site["center"]) < 2.0:
                if self.search_arrival is None:
                    self.search_arrival = tick
                elif tick - self.search_arrival >= 90:
                    self.searched_sites.add(self.search_site)
                    self.search_site, self.search_arrival = None, None
            else:
                self.search_arrival = None
        if self.search_site is None:
            available = [i for i in range(len(self.sites))
                         if i not in self.searched_sites]
            if not available:
                self.searched_sites.clear()
                available = list(range(len(self.sites)))
            self.search_site = min(available, key=lambda i:
                                   distance(pos, self.sites[i]["center"]))
        return list(self.sites[self.search_site]["center"])

    def should_engage(self, o, enemies, goal):
        """Retakers clear visible threats first; carriers bound distant duels."""
        if not enemies:
            self.engagement_started = None
            return False
        m, tick = o["me"], o["match_tick"]
        if o["side"] == "defend" and o["bomb"]["state"] == "planted":
            return True
        if (tick < self.peek_advance_until
                and all(self.head_only(e) and not self.target_ready(e, tick) for e in enemies)):
            return False
        pos = [m["x"], m["y"], m["z"]]
        nearest = min(distance(pos, [e["x"], e["y"], e["z"]]) for e in enemies)
        objective = (m["carrying_bomb"] or
                     (o["side"] == "defend" and o["bomb"]["state"] == "planted"))
        if not objective or nearest <= 30.0 or goal is None:
            if self.engagement_started is None:
                self.engagement_started = tick
            return True
        if tick < self.advance_until:
            return False
        remaining = o.get("phase_remaining_ticks", max(0, 7260-o["round_tick"])) / 60
        interaction = 3.0
        if o["bomb"]["state"] == "planted":
            fuse = o["bomb"].get("fuse_end_tick")
            if isinstance(fuse, (int, float)):
                remaining = max(0.0, (fuse-tick) / 60)
            interaction = 5.0 if m["has_kit"] else 10.0
        # This is a planning margin, not a claimed exact physics travel time.
        urgent = remaining <= distance(pos, goal) / 3.0 + interaction + 8.0
        if self.engagement_started is None:
            self.engagement_started = tick
        if urgent or tick-self.engagement_started >= 120:
            self.advance_until = tick + 120
            self.engagement_started = None
            return False
        return True

    def aim_point(self, enemy, me=None, gun=None, tick=None):
        heights = {"head": 1.64, "chest": 1.18, "abdomen": 0.88}
        part = next((p for p in ("chest", "abdomen", "head")
                     if p in enemy["parts"]), "chest")
        if me is not None and gun is not None and tick is not None and "head" in enemy["parts"]:
            key = enemy.get("seat")
            track = self.sight_tracks.get(key)
            stable = bool(track and track[0] == tick and tick-track[2] >= 6)
            dist = math.dist([me[k] for k in ("x", "y", "z")],
                             [enemy[k] for k in ("x", "y", "z")])
            speed = math.hypot(me.get("vx", 0), me.get("vy", 0))
            inaccuracy = gun.get("inaccuracy", 0.0)
            eligible = (gun.get("id") in ("ak47", "m4a1", "m4a4", "galilar", "sg553")
                        and stable and me["grounded"] and speed < .25)
            if eligible and dist <= 30 and inaccuracy <= .012:
                self.head_targets.add(key)
            elif not eligible or dist > 40 or inaccuracy > .03:
                self.head_targets.discard(key)
            if key in self.head_targets:
                part = "head"
        elif me is not None:
            self.head_targets.discard(enemy.get("seat"))
        height = heights[part] * (0.6 if enemy["crouch"] else 1.0)
        return [enemy["x"], enemy["y"], enemy["z"] + height]

    def aim_error(self, me, point, gun=None):
        m = me
        dx, dy = point[0] - m["x"], point[1] - m["y"]
        eye = 0.95 if m["crouch"] else 1.62
        dz = point[2] - m["z"] - eye
        punch = gun.get("aim_punch", {}) if gun else {}
        yaw = angle(math.atan2(-dy, dx) - m["yaw"]
                    - 2 * punch.get("yaw", 0.0))
        pitch = math.atan2(dz, max(0.1, math.hypot(dx, dy)))
        pitch -= m["pitch"] + 2 * punch.get("pitch", 0.0)
        return yaw, pitch

    def quiet_action(self, o, action):
        # Missing sight releases the trigger immediately. Memory only turns the
        # view toward an actually observed old position; it never authorizes fire.
        self.last_fire, self.burst_start = False, None
        tick, m = o["match_tick"], o["me"]
        if tick < self.peek_advance_until:
            self.contact_until, self.enemy_memory = 0, None
        if self.enemy_memory and tick - self.enemy_memory[1] > 180:
            self.enemy_memory, self.target_seat = None, None
        if not action.get("use") and not action.get("jump"):
            if tick < self.contact_until:
                action["crouch"] = True
            if (self.enemy_memory is not None and not m["blind_ticks"]
                    and tick >= self.escape_until):
                yaw, pitch = self.aim_error(m, self.enemy_memory[0])
                action.update(yaw_rate=clip(yaw / 0.15),
                              pitch_rate=clip(pitch / 0.1))
        return self.watch_navigation(o, action)

    def combat(self, o, action, gun, enemies=None):
        m = o["me"]
        tick = o["match_tick"]
        self.observe_targets(o)
        enemy = self.select_enemy(o, enemies)
        point = self.aim_point(enemy, m, gun, tick)
        self.last_combat_tick, self.hold_last_tick = tick, tick
        self.enemy_memory = (point, tick)
        yaw, pitch = self.aim_error(m, point, gun)
        if self.fire_slot != action["weapon_slot"]:
            self.last_fire, self.burst_start = False, None
            self.fire_slot = action["weapon_slot"]
        if self.burst_start is not None and tick - self.burst_start >= 42:
            self.burst_start = None
        ready = bool(gun and gun["ammo"] > 0
                     and current_slot(m, action["weapon_slot"]) == action["weapon_slot"]
                     and clock_ready(m.get("switch_until"), tick)
                     and clock_ready(gun.get("reload_until"), tick)
                     and clock_ready(gun.get("shot_ready"), tick + 1, ready_flag=True)
                     and clock_ready(m.get("shot_ready"), tick + 1, ready_flag=True))
        # Shot requests execute at tick+1, as verified against v8's actual
        # fire events. Only numeric shot deadlines use that boundary; reload,
        # switching, bool readiness and current legality stay conservative.
        dist = distance([m[k] for k in ("x", "y", "z")], point)
        semi = action["weapon_slot"] == 1
        close_controlled = dist <= 30 and gun.get("inaccuracy", 0.) <= .03
        burst_ticks = 12 if dist > 80 else 24
        trigger = (not self.last_fire if semi else close_controlled or
                   self.burst_start is None or tick - self.burst_start < burst_ticks)
        head = self.head_only(enemy) or enemy.get("seat") in self.head_targets
        tolerance = min(.025, math.atan2(.10 if head else .25, max(1., dist)))
        fire = bool(ready and trigger and self.target_ready(enemy, tick) and m["grounded"]
                    and math.hypot(m.get("vx", 0.0), m.get("vy", 0.0)) < 0.25
                    and abs(yaw) < tolerance and abs(pitch) < tolerance)
        if fire and not semi and self.burst_start is None:
            self.burst_start = tick
        self.last_fire = fire
        action.update(counterstop(m))
        action.update(yaw_rate=clip(yaw / 0.15),
                      pitch_rate=clip(pitch / 0.1),
                      crouch=bool(m["crouch"]) if fire else True,
                      fire=fire)
        return action

    def objective_action(self, me, slot, use):
        # Do not counter-steer at the objective: low-speed reversal oscillated
        # indefinitely in a live match. Neutral controls let ground friction
        # settle, then keep requesting use with the weapon already held.
        self.last_fire, self.burst_start = False, None
        action = {"weapon_slot": current_slot(me, slot)}
        if (not me.get("grounded", False)
                or math.hypot(me.get("vx", 0.0), me.get("vy", 0.0)) > 0.1):
            action.update(move_forward=0.0, move_right=0.0)
        else:
            action["use"] = use
        return action

    def move(self, o, goal, action, stop_at_goal=True):
        m = o["me"]
        pos = [m["x"], m["y"], m["z"]]
        tick = o["match_tick"]
        self.nav_tick = tick
        self.nav_side, self.nav_round = o["side"], o["round_index"]
        # Time spent fighting or using an objective is not failed movement.
        if self.last_move_tick is None or tick - self.last_move_tick > 6:
            self.anchor, self.anchor_tick = pos, tick
            self.leg_target = None
            if self.recovery_trace and self.recovery_edge is not None:
                self.return_failures.add(self.recovery_edge)
                self.retry_node = self.recovery_edge[1]
                self.path = []
            self.nav_trace, self.recovery_trace = [], []
        self.last_move_tick = tick
        self.track_ground(m, pos)
        if self.recovery_trace:
            # Follow the observed corridor before restarting the graph search;
            # selecting an anchor never teleports us across a corner or floor.
            while self.recovery_trace and distance(pos, self.recovery_trace[0]) < 0.18:
                self.recovery_trace.pop(0)
            if self.recovery_trace:
                point = self.recovery_trace[0]
                if (tick >= self.recovery_until or not m["grounded"] or distance(pos, point) > 1.0
                        or abs(pos[2]-point[2]) > 0.6):
                    self.recovery_trace = []
                    # A failed return is not permission to replace the walked
                    # corridor with a straight line to its starting anchor.
                    if self.recovery_edge is not None:
                        self.return_failures.add(self.recovery_edge)
                        self.retry_node = self.recovery_edge[1]
                    self.nav_trace = []
                    self.path = []
                else:
                    yaw = angle(math.atan2(-(point[1]-pos[1]), point[0]-pos[0])-m["yaw"])
                    self.anchor, self.anchor_tick = pos, tick
                    return dict(action, move_forward=0.65*math.cos(yaw),
                                move_right=-0.65*math.sin(yaw), yaw_rate=clip(yaw/0.15),
                                pitch_rate=clip(-m["pitch"]/0.1), walk=False, jump=False)
        if tick < self.escape_until:
            yaw = angle(self.escape_yaw - m["yaw"])
            return dict(action, move_forward=0.9 * math.cos(yaw),
                        move_right=-0.9 * math.sin(yaw),
                        yaw_rate=clip(yaw / 0.15),
                        pitch_rate=clip(-m["pitch"] / 0.1), walk=False,
                        jump=False)
        if self.goal != goal or not self.path:
            self.path, self.goal = self.route(pos, goal), list(goal)
            self.leg_target = None
        # Nearby XY while airborne/below a box is not a completed traversal.
        # The tolerance accommodates sampled ground, not another floor.
        while (len(self.path) > 1 and m["grounded"] and
               math.dist(pos[:2], self.path[0][:2]) < 1.0 and
               abs(pos[2] - self.path[0][2]) <= 0.6):
            self.path.pop(0)
            if self.path_nodes:
                reached = self.path_nodes.pop(0)
                if reached is not None:
                    self.nav_from = reached
                    self.pending_leg = None
                    self.nav_trace = [list(pos)]
                    if reached == self.retry_node:
                        self.retry_node = None
        point = self.path[0]
        if (self.nav_from is not None and self.path_nodes
                and self.path_nodes[0] is not None
                and self.path_nodes[0] != self.nav_from):
            self.pending_leg = (self.nav_from, self.path_nodes[0])
        dx, dy = point[0]-pos[0], point[1]-pos[1]
        leg_distance = distance(pos, point)
        # Only the final destination gets arrival handling. A nearby goal across
        # a wall/floor, an intermediate corner or a failed-route recovery anchor
        # must still follow the published path. Hysteresis avoids repeatedly
        # powering through the same point after a small inertial displacement.
        final_leg = all(distance(p, goal) < 0.25 for p in self.path)
        goal_distance = distance(pos, goal)
        if self.arrival_goal != tuple(goal):
            self.arrival_goal = None
        if (stop_at_goal and final_leg and m["grounded"]
                and goal_distance < (1.2 if self.arrival_goal else 0.65)):
            self.arrival_goal = tuple(goal)
            self.last_move_tick, self.leg_target = None, None
            self.last_position, self.stuck = pos, 0
            self.anchor, self.anchor_tick = pos, tick
            return dict(action, move_forward=0.0, move_right=0.0,
                        yaw_rate=0.15, pitch_rate=clip(-m["pitch"] / 0.1),
                        walk=False, jump=False)
        self.arrival_goal = None
        if self.leg_target != point:
            self.leg_target, self.leg_best = list(point), leg_distance
            self.leg_progress_tick = tick
        elif leg_distance < self.leg_best - 0.4:
            self.leg_best, self.leg_progress_tick = leg_distance, tick
        yaw = angle(math.atan2(-dy, dx) - m["yaw"])
        moved = (self.last_position is None
                 or distance(pos, self.last_position) > 0.01)
        self.stuck = 0 if moved else self.stuck + 1
        self.last_position = pos
        # A small terminal taper reduces overshoot without slowing intermediate
        # route legs or changing the measured-speed gate for plant/defuse.
        throttle = 0.75 * min(1.0, max(0.2, goal_distance / 1.5)) if final_leg else 0.75
        action.update(move_forward=throttle * math.cos(yaw),
                      move_right=-throttle * math.sin(yaw),
                      yaw_rate=clip(yaw / 0.15),
                      pitch_rate=clip(-m["pitch"] / 0.1), walk=False)
        if self.anchor is None or math.dist(pos[:2], self.anchor[:2]) > 0.6:
            self.anchor, self.anchor_tick = pos, tick
        if (tick >= self.escape_until and leg_distance >= 0.65
                and (tick - self.anchor_tick > 60
                     or tick - self.leg_progress_tick > 240)):
            # Commit to leaving a blocked corner; small sidesteps must not
            # immediately reset recovery and send us into the same obstacle.
            self.escapes += 1
            self.penalize_leg()
            turn = (1 if (o["seat"] + self.escapes) % 2 else -1)
            self.escape_yaw = m["yaw"] + turn * (1.2 + 0.4 * (self.escapes % 3))
            self.escape_until = tick if self.recovery_trace else tick + 45
            self.anchor, self.anchor_tick = pos, self.escape_until
            self.path = []
            self.leg_target = None
        if tick < self.escape_until:
            yaw = angle(self.escape_yaw - m["yaw"])
            action.update(move_forward=0.9 * math.cos(yaw),
                          move_right=-0.9 * math.sin(yaw),
                          yaw_rate=clip(yaw / 0.15),
                          jump=False)
        elif point[2] - pos[2] > 0.45 and math.hypot(dx, dy) < 1.8:
            target = self.path_nodes[0] if self.path_nodes else None
            # Key the allowance by destination too: replanning a connector
            # cannot make an already-failed box appear to be a new attempt.
            key = target if target is not None else tuple(point)
            attempts = self.jump_attempts.get(key, 0)
            jump = (m["grounded"] and attempts < 2
                    and tick-self.last_jump_tick >= 60 and not self.recovery_trace)
            action["jump"] = bool(jump)
            if jump:
                self.jump_attempts[key] = attempts + 1
                self.last_jump_tick = tick
        return action

    def act(self, o):
        identity = (o.get("map_id"), o.get("static_schema_hash"))
        if identity != self.map_identity:
            # Never navigate using geometry from another map or schema revision.
            self.map_identity = identity
            self.clear_route_history()
            self.nodes, self.edges, self.sites = {}, {}, []
            self.spawn_views = {}
            self.guard_plans = {}
            self.pressure_points, self.zone_points = {}, {}
            self.walk_edges = set()
            self.source_edges = set()
            self.pending_map = None
            self.path, self.goal = [], None
            self.last_position, self.stuck = None, 0
            self.anchor, self.anchor_tick = None, 0
            self.escape_until, self.escape_yaw, self.escapes = 0, 0.0, 0
            self.reset_navigation()
            self.reset_tactics()
        if isinstance(o.get("map"), dict):
            self.pending_map = o["map"]
        # The opening observation is large. Build the navigation graph on a
        # later buy callback so JSON decoding and graph work use separate budgets.
        if self.pending_map is not None and ("map" not in o or o["phase"] == "live"):
            self.load_map(self.pending_map)
            self.pending_map = None
        if self.round != o["round_index"]:
            self.round = o["round_index"]
            self.path, self.goal = [], None
            self.last_position, self.stuck = None, 0
            self.anchor, self.anchor_tick = None, 0
            self.escape_until, self.escapes = 0, 0
            self.reset_navigation()
            self.reset_tactics()
        m = o["me"]
        if not m["alive"] or o["phase"] not in ("buy", "live"):
            # Record a fatal unseen encounter once, including a 100-to-0 hit.
            # Do not attribute an attacker or infer a private enemy location.
            if (not m["alive"] and self.last_hp is not None and self.last_hp > 0
                    and not self.last_legal_visible
                    and self.last_safety_tick is not None
                    and 0 < o["match_tick"]-self.last_safety_tick <= 6
                    and o["match_tick"] >= self.damage_cooldown):
                self.remember_threat(o)
            self.reset_tactics()
            return {}
        if o["phase"] == "buy":
            self.reset_tactics()
            return self.buy(o)
        tick = o["match_tick"]
        self.observe_sounds(o)
        self.observe_team(o)
        visible = bool(o["visible"] and not m["blind_ticks"])
        self.observe_targets(o)
        damaged = self.observe_safety(o)
        if visible:
            self.contact_until = tick + 90
        slot = self.weapon_slot(m, contact=tick < self.contact_until, tick=tick)
        gun = m["weapons"][slot]
        enemies = self.reachable_enemies(o, gun)
        self.bound_head_wait(o, enemies)
        confirmed = [e for e in enemies if self.target_ready(e, tick)]
        engage_enemies = confirmed or enemies
        pos = [m["x"], m["y"], m["z"]]
        bomb = o["bomb"]
        planted = bomb["state"] == "planted"
        if planted and bomb.get("position") is not None:
            # This coordinate came from this player's legal observation only.
            self.known_bomb = list(bomb["position"])
        elif bomb["state"] in ("defused", "exploded"):
            self.known_bomb = None
        bomb_position = self.known_bomb if planted else bomb.get("position")
        # Clear a visible threat before defusing, including at the bomb itself.
        # With no visible threat preserve the active slot throughout interaction.
        defusing = (o["side"] == "defend" and planted
                    and bomb_position is not None)
        planting = o["side"] == "attack" and m["carrying_bomb"] and not planted
        if defusing and distance(pos, bomb_position) < 2.0 and not visible:
            return self.objective_action(m, slot, "defuse")
        if (planting and any(distance(pos, site["center"]) < site["radius"] * 0.5
                             for site in self.sites)):
            return self.objective_action(m, slot, "plant")
        action = {"weapon_slot": slot}
        escape = self.unseen_damage_escape(o, action, damaged)
        if escape is not None:
            return escape
        if visible:
            action["fire"] = False
        if (gun and gun["ammo"] == 0 and gun["reserve"]
                and current_slot(m, slot) == slot
                and clock_ready(m.get("switch_until"), o["match_tick"])
                and clock_ready(gun.get("reload_until"), o["match_tick"])):
            action["reload"] = True
        if not self.sites:
            # Missing static data cannot justify inventing a navigation target.
            if enemies:
                return self.combat(o, action, gun, engage_enemies)
            return self.quiet_action(o, action)
        # Attackers rally on a shared site; defenders cover sites by team seat.
        index = (o["round_index"] if o["side"] == "attack"
                 else o["seat"] % 5) % len(self.sites)
        supporting = (o["side"] == "defend" and not planted
                      and self.support_zone in self.zone_points and tick < self.support_until)
        if supporting:
            support = self.zone_points[self.support_zone]
            index = min(range(len(self.sites)), key=lambda i: distance(support, self.sites[i]["center"]))
        if o["side"] == "attack" and not m["carrying_bomb"] and not planted:
            carrier_pos = bomb.get("position") if bomb.get("state") == "held" else None
            if carrier_pos is not None and bomb.get("carrier") != o["seat"]:
                near = min(range(len(self.sites)), key=lambda i: distance(carrier_pos, self.sites[i]["center"]))
                if distance(carrier_pos, self.sites[near]["center"]) < 30:
                    self.attack_site = near
            if self.attack_site is not None:
                index = self.attack_site
        site = self.sites[index]
        guard_plan = self.guard_plans.get(index) if o["side"] == "defend" and not planted else None
        goal = self.guard_goal(o, index, guard_plan) if guard_plan else list(site["center"])
        if supporting:
            # Move to the reported region, not an invented killer coordinate.
            support = self.zone_points[self.support_zone]
            if not guard_plan or distance(support, site["center"]) > site["radius"] * 2:
                goal = list(support)
                guard_plan = None
        if not planted and not planting:
            goal = self.pressure_goal(o, index, goal)
            if self.pressure_index >= 0 and self.approach_view(o) is not None:
                guard_plan = {"watch": [self.approach_view(o)]}
        if planted and bomb_position is not None:
            goal = list(bomb_position)
        elif planted and o["side"] == "defend":
            goal = self.unknown_bomb_goal(o, pos)
        if o["side"] == "attack" and bomb["state"] == "world" and bomb.get("position"):
            goal = list(bomb["position"])
            if distance(pos, goal) < 1.2:
                return self.quiet_action(
                    o, dict(action, use="pickup", pickup_id=bomb["id"]))
        if self.should_engage(o, engage_enemies, goal):
            # Prefer a confirmed shootable target over an unconfirmed nearer
            # glimpse. Engagement urgency and actual selection use the same set.
            return self.combat(o, action, gun, engage_enemies)
        hold = self.lost_contact_hold(o, action)
        if hold is not None:
            return hold
        if visible:
            # An unshootable target or an objective deadline must not keep the
            # policy crouched in place or override the next route heading.
            self.contact_until = 0
            self.enemy_memory = None
            self.target_seat = None
        if (o["side"] == "attack" and planted
                and distance(pos, goal) < 8):
            approach = self.approach_view(o)
            if approach is not None:
                return self.guard_action(o, action, {"watch": [approach]})
            return self.quiet_action(o, dict(action, **counterstop(m)))
        if (guard_plan and distance(pos, goal) < 1.2 and m["grounded"]
                and abs(pos[2]-goal[2]) < .6):
            return self.guard_action(o, action, guard_plan)
        # Buy already gates movement; do not delay later seats after live starts.
        action = self.move(o, goal, action, stop_at_goal=not (planting or defusing))
        if (guard_plan and distance(pos, goal) < 8 and not action.get("jump")
                and m["grounded"] and tick >= self.escape_until and not self.recovery_trace):
            action = self.guard_action(o, action, guard_plan, moving=True)
        return self.quiet_action(o, self.prepare_approach(o, action))


_player = Player()


def init(context):
    global _player
    # Observations and this policy use acyclic containers; reference counting
    # releases them without repeatedly scanning the large public map.
    gc.disable()
    _player = Player()


def act(observation):
    return _player.act(observation)
