"""AGX FPS baseline adapted from the official public starter.

Source: https://agx.cool/docs/schema/fps.example.agent.py
Retrieved: 2026-11-15. Contract: rsi-fps-5.5.
Local changes: guard absent map data, invalidate map caches by identity,
and avoid requesting reload during plant/defuse interaction.
This baseline has not been evaluated in a ranked match.
"""
import math
import gc
from collections import deque


def clip(value):
    return max(-1.0, min(1.0, value))


def distance(a, b):
    return math.sqrt(sum((x - y)**2 for x, y in zip(a, b)))


def angle(value):
    return (value + math.pi) % (2 * math.pi) - math.pi


class Player:
    def __init__(self):
        self.nodes, self.edges, self.sites = {}, {}, []
        self.pending_map = None
        self.map_identity = None
        self.path, self.goal = [], None
        self.round = None
        self.last_position, self.stuck = None, 0
        self.anchor, self.anchor_tick = None, 0
        self.escape_until, self.escape_yaw, self.escapes = 0, 0.0, 0

    def load_map(self, row):
        self.sites = row.get("sites") or []
        nav = row.get("nav") or {}
        self.nodes = {n["id"]: n["position"]
                      for n in nav.get("nodes", [])}
        self.edges = {key: [] for key in self.nodes}
        for edge in nav.get("edges", []):
            if (edge["from"] in self.nodes and edge["to"] in self.nodes
                    and edge["kind"] != "ladder"):
                self.edges[edge["from"]].append(edge["to"])
                if edge["kind"] == "walk":
                    self.edges[edge["to"]].append(edge["from"])

    def route(self, start, goal):
        if not self.nodes:
            return [goal]
        first = min(self.nodes,
                    key=lambda k: distance(start, self.nodes[k]))
        last = min(self.nodes,
                   key=lambda k: distance(goal, self.nodes[k]))
        queue, parent = deque([first]), {first: None}
        while queue and last not in parent:
            here = queue.popleft()
            for there in self.edges[here]:
                if there not in parent:
                    parent[there] = here
                    queue.append(there)
        if last not in parent:
            return [goal]
        path, cursor = [goal], last
        while cursor is not None:
            path.append(self.nodes[cursor])
            cursor = parent[cursor]
        return list(reversed(path))

    def buy(self, o):
        m = o["me"]
        if (o["side"] == "defend" and not m["has_kit"]
                and m["cash"] >= 400):
            return {"buy": "defusekit"}
        rifle = "ak47" if o["side"] == "attack" else "m4a1"
        price = 2700 if o["side"] == "attack" else 2900
        if m["weapons"][0] is None and m["cash"] >= price:
            return {"buy": rifle}
        if m["armor"] < 50 and m["cash"] >= 650:
            return {"buy": "armor"}
        return {}

    def combat(self, o, action, gun):
        m = o["me"]
        enemy = min(o["visible"], key=lambda e:
                    (e["x"]-m["x"])**2 + (e["y"]-m["y"])**2)
        dx, dy = enemy["x"]-m["x"], enemy["y"]-m["y"]
        heights = {"head": 1.64, "chest": 1.18, "abdomen": 0.88}
        part = next((p for p in ("chest", "head", "abdomen")
                     if p in enemy["parts"]), "chest")
        height = heights[part] * (0.6 if enemy["crouch"] else 1.0)
        eye = 0.95 if m["crouch"] else 1.62
        dz = enemy["z"] + height - m["z"] - eye
        punch = gun.get("aim_punch", {}) if gun else {}
        yaw = angle(math.atan2(-dy, dx) - m["yaw"]
                    - 2 * punch.get("yaw", 0.0))
        pitch = math.atan2(dz, max(0.1, math.hypot(dx, dy)))
        pitch -= m["pitch"] + 2 * punch.get("pitch", 0.0)
        ready = bool(gun and gun["ammo"]
                     and not gun["reload_until"])
        # Release the trigger between pistol shots.
        trigger = (action["weapon_slot"] == 0
                   or (o["match_tick"] // 3) % 4 == 0)
        action.update(yaw_rate=clip(yaw / 0.15),
                      pitch_rate=clip(pitch / 0.1), crouch=True,
                      fire=ready and trigger and abs(yaw) < 0.04
                      and abs(pitch) < 0.04)
        return action

    def move(self, o, goal, action):
        m = o["me"]
        pos = [m["x"], m["y"], m["z"]]
        tick = o["match_tick"]
        if tick < self.escape_until:
            yaw = angle(self.escape_yaw - m["yaw"])
            return dict(action, move_forward=0.9 * math.cos(yaw),
                        move_right=-0.9 * math.sin(yaw),
                        yaw_rate=clip(yaw / 0.15),
                        pitch_rate=clip(-m["pitch"] / 0.1), walk=False,
                        jump=bool(m["grounded"] and tick % 30 == 0))
        if self.goal != goal or not self.path:
            self.path, self.goal = self.route(pos, goal), list(goal)
        # Navmesh samples can be above the actual foot position on a slope.
        while (len(self.path) > 1 and
               math.dist(pos[:2], self.path[0][:2]) < 1.0 and
               abs(pos[2] - self.path[0][2]) < 2.0):
            self.path.pop(0)
        point = self.path[0]
        dx, dy = point[0]-pos[0], point[1]-pos[1]
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
        if tick >= self.escape_until and tick - self.anchor_tick > 60:
            # Commit to leaving a blocked corner; small sidesteps must not
            # immediately reset recovery and send us into the same obstacle.
            self.escapes += 1
            turn = (1 if (o["seat"] + self.escapes) % 2 else -1)
            self.escape_yaw = m["yaw"] + turn * (1.2 + 0.4 * (self.escapes % 3))
            self.escape_until = tick + 120
            self.anchor, self.anchor_tick = pos, self.escape_until
            self.path = []
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
            self.pending_map = None
            self.path, self.goal = [], None
            self.last_position, self.stuck = None, 0
            self.anchor, self.anchor_tick = None, 0
            self.escape_until, self.escape_yaw, self.escapes = 0, 0.0, 0
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
        m = o["me"]
        if not m["alive"] or o["phase"] not in ("buy", "live"):
            return {}
        if o["phase"] == "buy":
            return self.buy(o)
        slot = 0 if m["weapons"][0] is not None else 1
        gun = m["weapons"][slot]
        action = {"weapon_slot": slot}
        if gun and gun["ammo"] == 0 and gun["reserve"]:
            action["reload"] = True
        if not self.sites:
            # Missing static data cannot justify inventing a navigation target.
            if o["visible"] and not m["blind_ticks"]:
                return self.combat(o, action, gun)
            return action
        pos = [m["x"], m["y"], m["z"]]
        bomb = o["bomb"]
        patrol = 0 if m["carrying_bomb"] else o["round_tick"] // 1800
        index = (o["round_index"] + o["seat"] % 2 + patrol) % len(self.sites)
        site = self.sites[index]
        goal = list(site["center"])
        planted = bomb["state"] == "planted"
        if planted and bomb.get("position"):
            goal = list(bomb["position"])
        if o["side"] == "attack" and bomb["state"] == "world" and bomb.get("position"):
            goal = list(bomb["position"])
            if distance(pos, goal) < 1.2:
                return dict(action, use="pickup", pickup_id=bomb["id"])
        if (o["side"] == "defend" and planted
                and bomb.get("position")
                and distance(pos, goal) < 2.0):
            return {"weapon_slot": slot, "use": "defuse"}
        radius = site["radius"] * 0.5
        at_site = distance(pos, site["center"]) < radius
        if m["carrying_bomb"] and at_site:
            return {"weapon_slot": slot, "use": "plant"}
        if o["visible"] and not m["blind_ticks"]:
            return self.combat(o, action, gun)
        if (o["side"] == "attack" and planted
                and distance(pos, goal) < 8):
            return dict(action, yaw_rate=0.15, walk=True)
        # Give each teammate room to leave its spawn before entering a narrow route.
        if o["round_tick"] < 360 + (o["seat"] % 5) * 90:
            return action
        return self.move(o, goal, action)


_player = Player()


def init(context):
    global _player
    # Observations and this policy use acyclic containers; reference counting
    # releases them without repeatedly scanning the large public map.
    gc.disable()
    _player = Player()


def act(observation):
    return _player.act(observation)
