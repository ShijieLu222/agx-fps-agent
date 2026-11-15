"""AGX FPS v4 adapted from the official public starter.

Source: https://agx.cool/docs/schema/fps.example.agent.py
Retrieved: 2026-11-15. Contract: rsi-fps-5.5.
Local changes: geometry-weighted routes, objective continuity, usable-weapon
selection, own-clock checks and a rifle-first defensive buy policy.
Duel pacing, contact memory and objective braking adapt ideas from the
user-supplied AGX_FPS_v3_3_Duel.zip (2026-11-15); no performance gain is proven.
v3 addressed range/exhausted-ammo stalls and objective/navigation failures.
v4 targets stopping, unseen-damage response and route failures observed in the
owner's second match. Offline checks do not establish a win-rate gain.
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
        self.walk_edges = set()
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
        self.failed_edges = {}
        self.nav_tick = 0
        self.leg_target, self.leg_best, self.leg_progress_tick = None, math.inf, 0
        self.last_move_tick = None

    def penalize_leg(self):
        """Remember an observed failed traversal, never an unseen obstacle."""
        if not self.path_nodes:
            return
        there = self.path_nodes[0]
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
        # Backtrack only over an explicitly reversible walk at a compatible
        # foot height. A completed descent cannot be reversed by a virtual
        # start connector. Continue toward that leg's published destination;
        # do not snap to arbitrary nearby nodes on the lower floor.
        can_backtrack = (key in self.walk_edges
                         and self.last_position is not None
                         and abs(self.last_position[2]
                                 - self.nodes[self.nav_from][2]) <= 0.6)
        self.retry_node = self.nav_from if can_backtrack else there
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
            self.breadcrumbs.append((tick, pos))
            self.breadcrumbs = [(t, p) for t, p in self.breadcrumbs[-64:]
                                if tick-t <= 120]
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
            return None
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
        # Keep watching the current direction while backing along our path.
        # No stale contact-memory turn or crouch is applied to this action.
        return {"weapon_slot": current_slot(m, action["weapon_slot"]),
                "move_forward": 0.85 * math.cos(yaw),
                "move_right": -0.85 * math.sin(yaw),
                "yaw_rate": 0.0, "pitch_rate": 0.0, "fire": False,
                "crouch": False, "walk": False, "jump": False}

    def load_map(self, row):
        self.clear_route_history()
        self.reset_navigation()
        self.sites = row.get("sites") or []
        nav = row.get("nav") or {}
        self.nodes = {n["id"]: n["position"]
                      for n in nav.get("nodes", [])}
        self.edges = {key: [] for key in self.nodes}
        self.walk_edges = set()
        for edge in nav.get("edges", []):
            if (edge["from"] in self.nodes and edge["to"] in self.nodes
                    and edge["kind"] != "ladder"):
                cost = distance(self.nodes[edge["from"]],
                                self.nodes[edge["to"]])
                self.edges[edge["from"]].append((edge["to"], cost))
                if edge["kind"] == "walk":
                    self.edges[edge["to"]].append((edge["from"], cost))
                    self.walk_edges.add((edge["from"], edge["to"]))
                    self.walk_edges.add((edge["to"], edge["from"]))

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
        if m["armor"] < 50 and m["cash"] >= 650:
            return {"buy": "armor"}
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
                if distance(eye, self.aim_point(e)) < reach]

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
        """Bound distant duels when carrying or racing the planted-bomb clock."""
        if not enemies:
            self.engagement_started = None
            return False
        m, tick = o["me"], o["match_tick"]
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

    def aim_point(self, enemy):
        heights = {"head": 1.64, "chest": 1.18, "abdomen": 0.88}
        part = next((p for p in ("chest", "head", "abdomen")
                     if p in enemy["parts"]), "chest")
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
        return action

    def combat(self, o, action, gun, enemies=None):
        m = o["me"]
        tick = o["match_tick"]
        point = self.aim_point(self.select_enemy(o, enemies))
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
                     and clock_ready(gun.get("shot_ready"), tick, ready_flag=True)
                     and clock_ready(m.get("shot_ready"), tick, ready_flag=True))
        # Pistol release follows actual fire requests, not an unrelated clock.
        # Rifles fire for at most 24 ticks within each 42-tick burst cycle.
        semi = action["weapon_slot"] == 1
        trigger = (not self.last_fire if semi else
                   self.burst_start is None or tick - self.burst_start < 24)
        fire = bool(ready and trigger and m["grounded"]
                    and math.hypot(m.get("vx", 0.0), m.get("vy", 0.0)) < 0.25
                    and abs(yaw) < 0.025 and abs(pitch) < 0.025)
        if fire and not semi and self.burst_start is None:
            self.burst_start = tick
        self.last_fire = fire
        action.update(counterstop(m))
        action.update(yaw_rate=clip(yaw / 0.15),
                      pitch_rate=clip(pitch / 0.1), crouch=True,
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

    def move(self, o, goal, action):
        m = o["me"]
        pos = [m["x"], m["y"], m["z"]]
        tick = o["match_tick"]
        self.nav_tick = tick
        self.nav_side, self.nav_round = o["side"], o["round_index"]
        # Time spent fighting or using an objective is not failed movement.
        if self.last_move_tick is None or tick - self.last_move_tick > 6:
            self.anchor, self.anchor_tick = pos, tick
            self.leg_target = None
        self.last_move_tick = tick
        if tick < self.escape_until:
            yaw = angle(self.escape_yaw - m["yaw"])
            return dict(action, move_forward=0.9 * math.cos(yaw),
                        move_right=-0.9 * math.sin(yaw),
                        yaw_rate=clip(yaw / 0.15),
                        pitch_rate=clip(-m["pitch"] / 0.1), walk=False,
                        jump=bool(m["grounded"] and tick % 30 == 0))
        if self.goal != goal or not self.path:
            self.path, self.goal = self.route(pos, goal), list(goal)
            self.leg_target = None
        # Navmesh samples can be above the actual foot position on a slope.
        while (len(self.path) > 1 and
               math.dist(pos[:2], self.path[0][:2]) < 1.0 and
               abs(pos[2] - self.path[0][2]) < 2.0):
            self.path.pop(0)
            if self.path_nodes:
                reached = self.path_nodes.pop(0)
                if reached is not None:
                    self.nav_from = reached
                    if reached == self.retry_node:
                        self.retry_node = None
        point = self.path[0]
        dx, dy = point[0]-pos[0], point[1]-pos[1]
        leg_distance = distance(pos, point)
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
        action.update(move_forward=0.75 * math.cos(yaw),
                      move_right=-0.75 * math.sin(yaw),
                      yaw_rate=clip(yaw / 0.15),
                      pitch_rate=clip(-m["pitch"] / 0.1), walk=False)
        if self.anchor is None or math.dist(pos[:2], self.anchor[:2]) > 0.6:
            self.anchor, self.anchor_tick = pos, tick
        if (tick >= self.escape_until and leg_distance >= 1.0
                and (tick - self.anchor_tick > 60
                     or tick - self.leg_progress_tick > 240)):
            # Commit to leaving a blocked corner; small sidesteps must not
            # immediately reset recovery and send us into the same obstacle.
            self.escapes += 1
            self.penalize_leg()
            turn = (1 if (o["seat"] + self.escapes) % 2 else -1)
            self.escape_yaw = m["yaw"] + turn * (1.2 + 0.4 * (self.escapes % 3))
            self.escape_until = tick + 45
            self.anchor, self.anchor_tick = pos, self.escape_until
            self.path = []
            self.leg_target = None
        if tick < self.escape_until:
            yaw = angle(self.escape_yaw - m["yaw"])
            action.update(move_forward=0.9 * math.cos(yaw),
                          move_right=-0.9 * math.sin(yaw),
                          yaw_rate=clip(yaw / 0.15),
                          jump=bool(m["grounded"] and tick % 30 == 0))
        elif point[2] - pos[2] > 0.45 and math.hypot(dx, dy) < 1.8:
            action["jump"] = bool(m["grounded"] and tick % 24 == 0)
        return action

    def act(self, o):
        identity = (o.get("map_id"), o.get("static_schema_hash"))
        if identity != self.map_identity:
            # Never navigate using geometry from another map or schema revision.
            self.map_identity = identity
            self.clear_route_history()
            self.nodes, self.edges, self.sites = {}, {}, []
            self.walk_edges = set()
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
        visible = bool(o["visible"] and not m["blind_ticks"])
        damaged = self.observe_safety(o)
        if visible:
            self.contact_until = tick + 90
        slot = self.weapon_slot(m, contact=tick < self.contact_until, tick=tick)
        gun = m["weapons"][slot]
        enemies = self.reachable_enemies(o, gun)
        pos = [m["x"], m["y"], m["z"]]
        bomb = o["bomb"]
        planted = bomb["state"] == "planted"
        if planted and bomb.get("position") is not None:
            # This coordinate came from this player's legal observation only.
            self.known_bomb = list(bomb["position"])
        elif bomb["state"] in ("defused", "exploded"):
            self.known_bomb = None
        bomb_position = self.known_bomb if planted else bomb.get("position")
        # Objective requests are independent of route selection. Preserve the
        # active slot and sustain use even when an enemy appears or ammo is empty.
        if (o["side"] == "defend" and planted
                and bomb_position is not None
                and distance(pos, bomb_position) < 2.0):
            return self.objective_action(m, slot, "defuse")
        if (o["side"] == "attack" and m["carrying_bomb"] and not planted
                and any(distance(pos, site["center"]) < site["radius"] * 0.5
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
                return self.combat(o, action, gun, enemies)
            return self.quiet_action(o, action)
        # Attackers rally on a shared site; defenders cover sites by team seat.
        index = (o["round_index"] if o["side"] == "attack"
                 else o["seat"] % 5) % len(self.sites)
        site = self.sites[index]
        goal = list(site["center"])
        if planted and bomb_position is not None:
            goal = list(bomb_position)
        elif planted and o["side"] == "defend":
            goal = self.unknown_bomb_goal(o, pos)
        if o["side"] == "attack" and bomb["state"] == "world" and bomb.get("position"):
            goal = list(bomb["position"])
            if distance(pos, goal) < 1.2:
                return self.quiet_action(
                    o, dict(action, use="pickup", pickup_id=bomb["id"]))
        if self.should_engage(o, enemies, goal):
            return self.combat(o, action, gun, enemies)
        if visible:
            # An unshootable target or an objective deadline must not keep the
            # policy crouched in place or override the next route heading.
            self.contact_until = 0
            self.enemy_memory = None
            self.target_seat = None
        if (o["side"] == "attack" and planted
                and distance(pos, goal) < 8):
            return self.quiet_action(o, dict(action, yaw_rate=0.15, walk=True))
        # Buy already gates movement; do not delay later seats after live starts.
        return self.quiet_action(o, self.move(o, goal, action))


_player = Player()


def init(context):
    global _player
    # Observations and this policy use acyclic containers; reference counting
    # releases them without repeatedly scanning the large public map.
    gc.disable()
    _player = Player()


def act(observation):
    return _player.act(observation)
