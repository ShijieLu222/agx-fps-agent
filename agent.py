"""AGX FPS v3 adapted from the official public starter.

Source: https://agx.cool/docs/schema/fps.example.agent.py
Retrieved: 2026-11-15. Contract: rsi-fps-5.5.
Local changes: geometry-weighted routes, objective continuity, usable-weapon
selection, own-clock checks and a rifle-first defensive buy policy.
Duel pacing, contact memory and objective braking adapt ideas from the
user-supplied AGX_FPS_v3_3_Duel.zip (2026-11-15); no performance gain is proven.
v3 addresses range/exhausted-ammo stalls and objective/navigation failures in
the owner's first v2 match. Offline replay does not establish a win-rate gain.
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
    """Brake observed horizontal velocity in the player's current view basis."""
    vx, vy = me.get("vx", 0.0), me.get("vy", 0.0)
    if math.hypot(vx, vy) <= 0.1:
        return {"move_forward": 0.0, "move_right": 0.0}
    cosine, sine = math.cos(me["yaw"]), math.sin(me["yaw"])
    return {"move_forward": clip(-(vx * cosine - vy * sine) * 0.4),
            "move_right": clip(-(vx * sine + vy * cosine) * 0.4)}


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
        self.reset_navigation()
        self.reset_tactics()

    def reset_navigation(self):
        self.path_nodes, self.nav_from = [], None
        self.retry_node = None
        self.failed_edges = {}
        self.nav_tick = 0
        self.leg_target, self.leg_best, self.leg_progress_tick = None, math.inf, 0
        self.last_move_tick = None

    def penalize_leg(self):
        """Remember an observed failed traversal, never an unseen obstacle."""
        if not self.path_nodes or self.nav_from is None:
            return
        there = self.path_nodes[0]
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

    def load_map(self, row):
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
        # Switching or moving can break objective use. Brake first, then keep
        # requesting use with the slot already held, irrespective of combat.
        self.last_fire, self.burst_start = False, None
        action = {"weapon_slot": current_slot(me, slot)}
        if math.hypot(me.get("vx", 0.0), me.get("vy", 0.0)) > 0.1:
            action.update(counterstop(me))
        else:
            action["use"] = use
        return action

    def move(self, o, goal, action):
        m = o["me"]
        pos = [m["x"], m["y"], m["z"]]
        tick = o["match_tick"]
        self.nav_tick = tick
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
            self.reset_tactics()
            return {}
        if o["phase"] == "buy":
            self.reset_tactics()
            return self.buy(o)
        tick = o["match_tick"]
        visible = bool(o["visible"] and not m["blind_ticks"])
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
        # Give each teammate room to leave its spawn before entering a narrow route.
        if o["round_tick"] < 360 + (o["seat"] % 5) * 90:
            return self.quiet_action(o, action)
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
