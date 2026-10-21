"""
Student Agent for Text Adventure Games

This is your submission file. Implement the StudentAgent class to play
text adventure games using the MCP server you also implement.

Your agent should:
1. Connect to the MCP server via the provided client
2. Use the ReAct pattern (Thought -> Action -> Observation)
3. Call MCP tools to interact with the game
4. Maximize the game score within the step limit

Required method:
    async def run(self, client, game, max_steps, seed, verbose) -> RunResult

The 'client' is a FastMCP Client already connected to your MCP server.
Use it to call tools like: await client.call_tool("play_action", {"action": "look"})

Tips:
- Start by looking around and understanding your environment
- Keep track of visited locations to avoid loops
- Pick up useful items (lamp, sword, etc.)
- The seed parameter should be used to set your LLM's seed for reproducibility
"""
from collections import defaultdict
from http import client
import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional
import random

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

# =============================================================================
# LLM Configuration - DO NOT MODIFY
# =============================================================================

LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"

_hf_token = os.getenv("HF_TOKEN")
if not _hf_token:
    raise ValueError("HF_TOKEN not found. Set it in your .env file.")

LLM_CLIENT = InferenceClient(token=_hf_token)


def call_llm(prompt: str, system_prompt: str, seed: int, max_tokens: int = 300) -> str:
    """Call the LLM with the given prompt."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    
    response = LLM_CLIENT.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
        seed=seed,
    )
    
    return response.choices[0].message.content


@dataclass
class RunResult:
    """Result of running the agent. Do not modify this class."""
    final_score: int
    max_score: int
    moves: int
    locations_visited: set[str]
    game_completed: bool
    error: Optional[str] = None
    history: list[tuple[str, str, str]] = field(default_factory=list)

ALL_DIR = set([
    "north", "south", "east", "west", "up", "down",
    "northeast", "northwest", "southeast", "southwest",
    "go north", "go south", "go east", "go west", "go up", "go down",
    "n", "s", "e", "w", "u", "d", "ne", "nw", "se", "sw",
    "enter", "exit"
])

ABBR_DIR_TO_FULL = {
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "u": "up", "d": "down",
}

OPPOSITE_DIR = {
    "north": "south", "south": "north", "east": "west", "west": "east",
    "northeast": "southwest", "southwest": "northeast", "northwest": "southeast", "southeast": "northwest",
    "up": "down", "down": "up", 
    "enter": "exit", "exit": "enter"
}

LIGHT_ACTIONS = set([
    "turn on lamp", "turn on light"
])

FAILURE_PHRASES = [
    "i don't know", "i do not know", 
    "you can't see", "can't see",
    "you can't go", "you cannot go", "can't go that way",
    "don't know the word", "nothing happens", "nothing special",
    "you can't do that", "there is no", "there's no",
    "you already have", "is already open",
    "is locked", "is boarded",
    "can't see", "not see", "not go any",
    "not turn on", "not turn off", "not here",
    "not word need", "never see", "never try",
    "not allowed", "not know", "not turn on",
    "that not turn", "that not make sense",
    "not see that",
    "only see way to go",
    "grunk not see", "not good idea",
    "not work", "stuck to wall",
    "too heavy", "can not",
    "not in stream"
    "not tall enough", "not good at",
    "eaten by",

    "no door", "is locked", "is boarded", # door
    "can't go that way", "can't that way", "can't cross"
    "doesn't lead", "didn't lead",
    "only see way to go back",
]

DARK_PHRASES = [
    "so much dark", "too dark", "is dark",
    "dark hole", "in the dark", "pitch dark",
]

NON_DIR_ACTIONS = [
    "push <item>",
    "pull <item>",
    "turn <item>",
    "move <item>",
    "open <item>",
    "light <item>",
    "examine <item>",
    "catch <item>",
    "climb",
    "jump",
    "swim",
]

# Observed and assumed objects
ALL_OBJECTS = [
    "door", "window", "mailbox", "box", "bag", "bucket", "bowl",
    "lamp", "lantern", "torch", "fountain",
    "key", "sword", "knife", "rope", "bottle", "leaflet",
    "coin", "shield", "sign", "note", "letter",
    "table", "chair", "bench", "stairs", "staircase", "wall", "floor",
    "button", "curtain", "painting", "picture", "statue",
    "pig", 
    # "bridge", "tree", "stone", "carpet",
    # "stream", "river", "water",
    # "animal", "monster", "dragon",
]


OBJECT_ACTIONS = {
    "door": ["open", "close", "push", "pull"], "window": ["open", "close"],
    "mailbox": ["open", "close", "examine"], "box": ["open", "close", "examine"],
    "bag": ["open", "close", "examine", "take"], "bucket": ["open", "examine", "take"],
    "lamp": ["take", "turn on", "turn off"], "lantern": ["take", "turn on", "turn off"],
    "key": ["take"], "sword": ["take"], "torch": ["take", "light"],
    "knife": ["take", "examine"], "rope": ["take", "examine"], 
    "bottle": ["take", "examine", "open"],
    "leaflet": ["take", "examine", "read"], "coin": ["take", "examine"],
    "sign": ["examine", "read"], "note": ["take", "read"],
    "letter": ["take", "read"], "table": ["examine", "move"], 
    "chair": ["examine", "move", "push"], "bench": ["examine"],
    "stairs": ["examine", "climb"], "staircase": ["examine", "climb"],
    "wall": ["examine"], "floor": ["examine"],
    "button": ["push", "examine"], "curtain": ["open", "pull", "examine"],
    "bowl": ["examine"], "painting": ["examine", "take"],
    "picture": ["examine", "move"],"statue": ["examine", "move", "push"], "fountain": ["examine"],

    # "coins": ["take", "examine"], # "gold": ["take", "examine"],
    # "hat": ["take", "examine", "wear"], # "shield": ["take", "examine"],
    # "shoe": ["take", "examine"], # "pants": ["take", "examine", "wear"],
    # "carpet": ["examine", "move", "pull"], # "bridge": ["examine", "cross"],
    # "tree": ["examine", "climb"], # "stone": ["examine", "take", "move", "push"],
    # "well": ["examine"], # "stream": ["examine", "cross"],
    # "river": ["examine", "cross"], # "water": ["examine"],
    # "pig": ["examine"], # "animal": ["examine"], # "monster": ["examine", "attack"],
    # "dragon": ["examine", "attack"],
}

# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are an expert text adventure game player. Your goal is to explore, collect treasures, and maximize your score.

AVAILABLE TOOLS (use these via MCP):
1. play_action - Execute game commands (north, take lamp, open mailbox, etc.)
2. memory - Get current game state, score, and recent history
3. get_map - See explored locations and connections
4. inventory - Check what you're carrying

VALID GAME COMMANDS for play_action:
- Movement: north, south, east, west, up, down, enter, exit
- Objects: take <item>, drop <item>, open <thing>, close <thing>, examine <thing>
- Light: turn on lamp, turn off lamp
- Combat: attack <enemy> with <weapon>
- Other: inventory, look, read <thing>, wait

FORBIDDEN (will NOT work): check, inspect, search, grab, use, help

RESPOND IN THIS EXACT FORMAT (no markdown):
THOUGHT: <brief reasoning about what to do next>
TOOL: <tool_name>
ARGS: <JSON arguments>

Examples:
THOUGHT: I need to see what's around me.
TOOL: play_action
ARGS: {"action": "look"}

THOUGHT: Let me check my current state and score.
TOOL: memory
ARGS: {}

THOUGHT: The mailbox might contain something useful.
TOOL: play_action
ARGS: {"action": "open mailbox"}

STRATEGY:
1. Start by looking around and checking memory
2. Explore systematically - try all directions
3. Pick up useful items (lamp, sword, etc.)
4. Open containers (mailbox, window, etc.)
5. Use get_map to avoid getting lost
6. Turn on lamp before dark areas!
7. DO NOT repeat the same action multiple times in a row.

GOAL:
- Maximize score.
- Explore as many locations as possible.
"""


# =============================================================================
# Student Agent Implementation
# =============================================================================

class StudentAgent:
    """
    MCP ReAct Agent - A complete working example.
    
    This agent demonstrates:
    - ReAct loop (Thought -> Tool -> Observation)
    - Loop detection
    - Action validation
    - Score tracking via memory tool
    """
    
    def __init__(self):
        """Initialize the agent state."""
        self.history: list[dict] = []
        self.recent_actions: list[str] = []
        self.recent_locations: list[str] = []
        self.explored_locations: list[str] = []
        self.observations: list[str] = []
        self.score: int = 0
        self.step: int = 0
        self.moves: int = 0

        self.location_exits: dict[str, dict[str, str]] = defaultdict(dict)
        self.location_objects: dict[str, set[str]] = defaultdict(set)
        self.location_tried_actions: dict[str, set[str]] = defaultdict(set)
        self.location_failed_actions: dict[str, set[str]] = defaultdict(set)
        self.location_exits: dict[str, dict[str, str]] = defaultdict(dict)
        
        self.inventory_items: list[str] = []
        self.loop_detected_step: int = 0
        self.failure_prevention_step: int = 0
        self.is_in_darkness: bool = False
        self.is_in_loop: bool = True

        self.rng = random.Random(42)
    
    async def run(self, client, game: str, max_steps: int, seed: int, verbose: bool = False) -> RunResult:
        """Run the agent for a game session."""
        history = []
        
        # Get list of available tools
        tools = await client.list_tools()
        tool_names = [t.name for t in tools]
        
        # Get initial observation
        result = await client.call_tool("play_action", {"action": "look"})
        observation = self._extract_result(result)
                
        # Track informations from initial observation
        self._update_agent_state_from_action("look", observation, is_initial=True)
        
        if "inventory" in tool_names:
            result = await client.call_tool("inventory", {})
            inventory = self._extract_result(result)
            self._extract_items_from_inventory(inventory)
            self.inventory_refresh_step = 0

        if verbose:
            print(f"\n{observation}")
        
        # Main ReAct loop
        for step in range(1, max_steps + 1):
            self.step = step

            # Refresh inventory information every 10 steps to ensure none new items were missed
            if hasattr(self, 'inventory_refresh_step') and self.step - self.inventory_refresh_step >= 10:
                result = await client.call_tool("inventory", {})
                inventory = self._extract_result(result)
                self._extract_items_from_inventory(inventory)
                self.inventory_refresh_step = self.step

            # Build and validate prompt
            prompt = self._build_prompt(observation)
            response = call_llm(prompt, SYSTEM_PROMPT, seed + step)
            thought, tool_name, tool_args = self._parse_response(response, tool_names)
            tool_name, tool_args = self._validate_tool_call(tool_name, tool_args, tool_names)
            
            if tool_name == "play_action":
                action = tool_args.get("action", "look")

                # Failed action detection and prevention
                if self._validate_action(action) in self.location_failed_actions.get(self.explored_locations[-1], set()):
                    if self.step - self.failure_prevention_step >= 6:
                        possible_actions = self._get_untried_not_failed_action()
                        forced_action = self.rng.choice(possible_actions)
                        tool_args = {"action": forced_action}
                        self.failure_prevention_step = self.step
                        if verbose:
                            forced = tool_args.get("action", "look")
                            print(f"[WARNING] '{action}' has already failed at this location - forcing {forced}")
                    elif self.step - self.failure_prevention_step == 1:
                        tool_args = {"action": 'look'}
                        if verbose:
                            forced = tool_args.get("action", "look")
                            print(f"[WARNING] '{action}' has already failed at this location - forcing {forced}")

                # Loop detection and prevention
                message, is_in_loop = self._is_in_loop(action)
                if is_in_loop:
                    self.is_in_loop = True
                    self.loop_detected_step = self.step
                    # possible_actions = self._get_untried_not_failed_action()
                    # forced_action = self.rng.choice(possible_actions)
                    if verbose:
                        print(f"[WARNING] {message} - forcing look")
                    # tool_args = {"action": forced_action}
                    tool_args = {"action": "look"}
                self.moves += 1

            if verbose:
                print(f"\n--- Step {step} ---")
                print(f"[THOUGHT] {thought}")
                print(f"[TOOL] {tool_name}({tool_args})")
            
            # Execute the tool
            try:
                result = await client.call_tool(tool_name, tool_args)
                observation = self._extract_result(result)
                if verbose:
                    print(f"[RESULT] {observation[:200]}...")
                
                if tool_name == "play_action":
                    action = tool_args.get("action", "look")
                    self._update_agent_state_from_action(action, observation)
                elif tool_name == "inventory":
                    self._extract_items_from_inventory(observation)
                    self.inventory_refresh_step = self.step

            except Exception as e:
                observation = f"Error: {e}"
                if verbose:
                    print(f"[ERROR] {e}")

            # Update history
            self.history.append({
                "step": step,
                "thought": thought,
                "tool": tool_name,
                "args": tool_args,
                "result": observation[:200]
            })
            if len(self.history) > 10:
                self.history = self.history[-10:]
            
            # Track score from observation
            old_score = self.score
            self._update_score(observation)

            # Update inventory
            if self.score > old_score:
                if "inventory" in tool_names:
                    result = await client.call_tool("inventory", {})
                    inventory = self._extract_result(result)
                    self._extract_items_from_inventory(inventory)
                    self.inventory_refresh_step = self.step
            
            # Record in result history
            history.append((thought, f"{tool_name}({tool_args})", observation[:100]))
            
            # Check for game over
            if self._is_game_over(observation):
                if verbose:
                    print("\n*** GAME OVER ***")
                break
        
        return RunResult(
            final_score=self.score,
            max_score=350,
            moves=self.moves,
            locations_visited=set(self.explored_locations),
            game_completed=self._is_game_over(observation),
            history=history,
        )
    
    # =============================================================================
    # Observation tracking
    # =============================================================================

    def _update_agent_state_from_action(self, action: str, observation: str, is_initial: bool = False):
        # Extract location and observation
        if is_initial:
            location_after_action = self._extract_location("look", observation, "Unknown", is_initial=True)
            self.explored_locations.append(location_after_action)
            location_of_action = self.explored_locations[-1]
        else:
            location_of_action = self.explored_locations[-1]
            location_after_action = self._extract_location(action, observation, location_of_action)
            if location_after_action != location_of_action:
                self.explored_locations.append(location_after_action)

        self.recent_locations.append(location_after_action)
        if len(self.recent_locations) > 10:
            self.recent_locations = self.recent_locations[-10:]
        
        self.observations.append(observation)
        
        # Store objects from current location
        self.location_objects[location_after_action].update(self._extract_objects_from_observation(observation))

        # Process action
        action = self._validate_action(action)
        self.recent_actions.append(action)
        if len(self.recent_actions) > 10:
            self.recent_actions = self.recent_actions[-10:]

        # Update tried actions for this locaiton
        self.location_tried_actions[location_of_action].add(action)
        
        # Extract exits
        is_dir_action = action in ALL_DIR   
        if is_dir_action and not location_after_action == location_of_action:
            self.location_exits[location_of_action][action] = location_after_action
            opposite = OPPOSITE_DIR.get(action)
            self.location_exits[location_after_action][opposite] = location_of_action

        # Check and process failed action
        if self._is_failed_action(action, observation):
            self.location_failed_actions[location_of_action].add(action)

        # Check for darkness
        self.is_in_darkness = self._is_in_darkness(observation)
    
    # =============================================================================
    # Detection of failed actions and loops
    # =============================================================================

    def _is_failed_action(self, action: str, observation: str) -> bool:
        """Detection of failure phrases from action and observation"""
        observation = observation.lower().strip()

        # Failed if observation contains failure phrases
        for phrase in FAILURE_PHRASES:
            if phrase in observation:
                return True

        # Failed if observation is among the previous observations
        if action not in ["look", "inventory"] and len(self.observations) >= 1:
                previous_observations = [o.lower().strip()[:50] for o in self.observations[-6:]]
                if observation.strip()[:50] in previous_observations:
                    return True
        
        return False
    
    def _is_in_loop(self, new_action: str) -> bool:
        """Detect if stuck in a loop based on recent actions and observations."""
        # Do not always detect loop and initiate automatic help. Let agent decide after one notice
        if self.step - self.loop_detected_step < 4:
            return None, False
        # Do not check for loop if insufficient information
        if len(self.recent_locations) < 3 or len(self.recent_actions) < 3:
            return None, False
        
        # Keep actions and locations synced. The mismatch comes from fact that action[i] was done at locations[i-1]
        locations = self.recent_locations[-6:]
        actions = self.recent_actions[-6:] + [new_action]

        n = min(len(actions)-1, len(locations))
        location_action_pairs = list(zip(locations[-n:], actions[-n:]))

        # Base check
        if len(set(actions[-3:])) == 1:
            return "Same actions over last 3 tries", True

        # Same locations in loop
        for loop_size in [2, 3, 4]:
            if len(locations) >= loop_size * 2:
                tmp_locations = locations[-loop_size*2:]
                if tmp_locations[:loop_size] == tmp_locations[loop_size:] and len(set(tmp_locations)) >= 2:
                    return "Same locations in loop", True

        # Since locations can incorrectly extracted, we can check the previous constraint by just checking dir actions
        for loop_size in [2, 3, 4]:
            if len(actions) >= loop_size * 2:
                tmp_actions = actions[-loop_size*2:]
                if all(a in ALL_DIR for a in tmp_actions):
                    if tmp_actions[:loop_size] == tmp_actions[loop_size:] and len(set(tmp_actions)) >= 2:
                        return "Same dir actions in loop", True

        # Same action at same location at least 3 times. Similar to first check, yet has larger window of actions
        for pair in set(location_action_pairs):
            if location_action_pairs.count(pair) >= 3:
                return "Same actions at least 3 times on the same location", True
            
        # Extension of previous to check the (non dir) actions such as examine <item>, take <item>, where <item> can differ
        non_dir_actions = [a for (l, a) in location_action_pairs if l == locations[-1] and a not in NON_DIR_ACTIONS]
        if len(non_dir_actions) >= 3:
            tmp_actions = [a.split()[0] for a in non_dir_actions if a.split()] # they are usually a multiple words
            for action in set(tmp_actions):
                if action.count(action) >= 3:
                    return "Same non dir actions over (different) items", True
        
        # Same repetitive observations over last five
        observations = [o.lower().strip()[:50] for o in self.observations[-6:]]
        for observation in observations:
            if observations.count(observation) >= 3:
                return "Same observations at least 3 times over last 6 tries", True
            
        return None, False

    # =============================================================================
    # Decide action once loop or failed action occurs
    # =============================================================================

    def _get_untried_not_failed_action(self):
        location = self.explored_locations[-1]

        tried_actions = self.location_tried_actions.get(location, set())
        failed_actions = self.location_failed_actions.get(location, set())
        objects = self.location_objects.get(location, set())
        # if len(self.observations) == 1 self._extract_objects_from_observation(self.)
        # Exclude last action
        last_action = set() if len(self.recent_actions) == 0 else set(self.recent_actions[-1]) 
        excluded_actions = tried_actions | failed_actions | last_action

        # Take all light actions, direction actions and look action
        light_actions = [a for a in LIGHT_ACTIONS if a not in excluded_actions] if self.is_in_darkness else []
        look_actions = [a for a in ["look"] if a not in excluded_actions]
        dir_actions = [a for a in self._extract_dirs_from_observation(self.observations[-1]) if a not in excluded_actions]
        
        # Take actions with objects present in the room
        object_actions = []
        for object in objects:
            actions = OBJECT_ACTIONS.get(object, None)
            if actions == None: 
                continue
            for action in actions:
                if action not in excluded_actions:
                    object_actions.append(f"{action} {object}")

        rand = self.rng.randint(0, 99)
        if rand < 30 and light_actions:
            return light_actions
        elif rand < 80 and dir_actions:
            return dir_actions
        elif rand < 90 and look_actions:
            return look_actions
        elif object_actions:
            return object_actions

        # Take only untried and not failed actions
        for pool in [light_actions, dir_actions, object_actions, look_actions]:
            if pool:
                return pool

        # If all failed, take direction actions
        all_dirs = [a for a in ALL_DIR if a not in ABBR_DIR_TO_FULL]
        return all_dirs

    # =============================================================================
    # Functions for help when in loop or when failed action
    # =============================================================================

    def _is_in_darkness(self, observation: str) -> bool:
        """Detection of darkness phrases in observation"""
        observation = observation.lower().strip()

        for phrase in DARK_PHRASES:
            if phrase in observation:
                return True
        return False

    # =============================================================================
    # Extraction of information from observations and inventory
    # =============================================================================

    def _extract_location(self, action: str, observation: str, previous_location: str, is_initial: bool = False) -> str:
        """Extract location from observation."""
        action = self._validate_action(action)
        if action not in ALL_DIR and not is_initial and not action == "look":
            return previous_location
        
        lines = observation.strip().split("\n")

        # If single line or empty, keep previous location (stayed in same room)
        if len(lines) < 2:
            if "Hole" in observation:
                print('less than two lines')
            return previous_location

        location = lines[0].strip()
        if not location:
            for line in lines[1:]:
                location = line.strip()
                if location:
                    break
            if not location:
                return previous_location

        if len(location) > 30:
            return previous_location
        return location
    
    def _extract_dirs_from_observation(self, observation: str) -> list[str]:
        """Extract available directions mentioned in the observation"""
        observation = observation.lower()
        all_dirs = ABBR_DIR_TO_FULL.values()
        
        suggested_dirs = []
        for dir in all_dirs:
            if re.search(r'\b' + dir + r'\b', observation):
                suggested_dirs.append(dir)
        return suggested_dirs

    def _extract_objects_from_observation(self, observation: str) -> set[str]:
        """Extract likely objects from an observation."""
        observation = observation.lower()
        objects = set()

        for object in ALL_OBJECTS:
            if object in observation:
                objects.add(object)

        return objects

    def _extract_items_from_inventory(self, inventory: str):
        """Parse inventory text to extract and collect item names."""
        inventory = inventory.strip()

        if "empty-handed" in inventory.lower():
            self.inventory_items = []
            return

        items = inventory.split(":", 1)[1].strip()
        items = [item.strip() for item in items.split(",") if item.strip()]
        self.inventory_items = [item.lower() for item in items]

    # =============================================================================
    # Prompt building
    # =============================================================================

    
    def _build_prompt(self, observation: str) -> str:
        """Build the prompt for the LLM with context."""
        parts = []
        
        parts.append(f"Current Score: {self.score}")
        
        # Recent history
        if self.history:
            parts.append("\nRecent actions:")
            for entry in self.history[-3:]:
                action = entry.get("args", {}).get("action", entry["tool"])
                result_short = entry["result"][:80] + "..." if len(entry["result"]) > 80 else entry["result"]
                parts.append(f"  > {action} -> {result_short}")
            
            # Warn about repeated actions
            if self.recent_actions and len(set(self.recent_actions[-3:])) == 1:
                parts.append(f"\n[WARNING: You've been doing '{self.recent_actions[-1]}' repeatedly. TRY SOMETHING DIFFERENT!]")
        
        parts.append(f"\nCurrent situation:\n{observation}")
        parts.append("\nWhat do you do next?")
        
        return "\n".join(parts)
    
    # Not used
    # def _build_prompt(self, observation: str) -> str:
    #     """Build the prompt for the LLM with context."""
    #     parts = []
        
    #     parts.append(f"Current Score: {self.score}")
    #     location = self.explored_locations[-1] if self.explored_locations else "Unknown"
    #     parts.append(f"Location: {location}")
        
    #     # Inventory
    #     if self.inventory_items:
    #         parts.append(f"Inventory: {', '.join(self.inventory_items)}")
    #     else:
    #         parts.append("Inventory: You are empty-handed.")
        
    #     # Exits from current location
    #     exits = self.location_exits.get(location, {})
    #     if exits:
    #         exit_strs = [f"{d} -> {dest}" for d, dest in exits.items()]
    #         parts.append(f"Known exits from this location: {', '.join(exit_strs)}")
        
    #     # Objects at this location
    #     objects = self.location_objects.get(location, set())
    #     if objects:
    #         parts.append(f"Objects here: {', '.join(sorted(objects))}")

    #     # Agent in loop
    #     if self.is_in_loop:
    #         parts.append(f"WARNING: You are in a LOOP. You HAVE TO escape out of it and try OTHER actions!")
        
    #     # Failed actions at this location
    #     failed = self.location_failed_actions.get(location, set())
    #     if failed:
    #         parts.append(f"WARNING: Actions that DON'T work here: {', '.join(sorted(failed))}")
        
    #     # Agent in darkness
    #     if self.is_in_darkness:
    #         parts.append("WARNING: It is dark! Use 'turn on lamp' or 'turn on lantern' if you have one.")
        
    #     # Recent history
    #     if self.history:
    #         parts.append("\nRecent actions:")
    #         for entry in self.history[-3:]:
    #             action = entry.get("args", {}).get("action", entry["tool"])
    #             result_short = entry["result"][:80] + "..." if len(entry["result"]) > 80 else entry["result"]
    #             parts.append(f"  > {action} -> {result_short}")
            
    #         if self.recent_actions and len(set(self.recent_actions[-3:])) == 1:
    #             parts.append(f"\n[WARNING: You've been doing '{self.recent_actions[-1]}' repeatedly. TRY SOMETHING DIFFERENT!]")
        
    #     # Unique locations
    #     unique_locations = set(self.explored_locations)
    #     parts.append(f"\nExplored {len(unique_locations)} unique locations so far.")
        
    #     # Suggest unexplored directions
    #     tried_dirs = self.location_tried_actions.get(location, set())
    #     all_dirs = [a for a in ALL_DIR if a not in ABBR_DIR_TO_FULL]
    #     untried_dirs = [d for d in all_dirs if d not in tried_dirs and d not in failed]
    #     if untried_dirs:
    #         parts.append(f"Untried directions: {', '.join(untried_dirs)}")
        
    #     parts.append(f"\nCurrent situation:\n{observation}")
    #     parts.append("\nWhat do you do next?")
        
    #     return "\n".join(parts)
    
    # =============================================================================
    # Response parsing and validation of tools and actions
    # =============================================================================
    
    def _parse_response(self, response: str, valid_tools: list[str]) -> tuple[str, str, dict]:
        """Parse the LLM response to extract thought, tool, and arguments."""
        thought = "No reasoning provided"
        tool_name = "play_action"
        tool_args = {"action": "look"}
        
        lines = response.strip().split("\n")
        
        for line in lines:
            line_clean = line.strip()
            line_upper = line_clean.upper()
            
            if line_upper.startswith("THOUGHT:"):
                thought = line_clean.split(":", 1)[1].strip()
            
            elif line_upper.startswith("TOOL:"):
                raw_tool = line_clean.split(":", 1)[1].strip().lower()
                raw_tool = raw_tool.replace("**", "").replace("*", "").replace("`", "")
                raw_tool = raw_tool.split()[0] if raw_tool else "play_action"
                tool_name = raw_tool
            
            elif line_upper.startswith("ARGS:"):
                args_part = line_clean.split(":", 1)[1].strip()
                try:
                    args_part = args_part.replace("'", '"')
                    tool_args = json.loads(args_part)
                except json.JSONDecodeError:
                    match = re.search(r'"action"\s*:\s*"([^"]+)"', args_part)
                    if match:
                        tool_args = {"action": match.group(1)}
                    else:
                        tool_args = {"action": "look"}
        
        return thought, tool_name, tool_args
    
    def _validate_tool_call(self, tool_name: str, tool_args: dict, valid_tools: list[str]) -> tuple[str, dict]:
        """Validate and fix common tool call issues."""
        # Fix tool name
        if tool_name not in valid_tools:
            if tool_name in ["action", "do", "command"]:
                tool_name = "play_action"
            elif tool_name in ["map", "location", "locations"]:
                tool_name = "get_map"
            elif tool_name in ["mem", "state", "status"]:
                tool_name = "memory"
            elif tool_name in ["inv", "items"]:
                tool_name = "inventory"
            else:
                tool_name = "play_action"

        # Fix action verbs
        if tool_name == "play_action":
            action = tool_args.get("action", "look")

            invalid_verb_map = {
                "check": "examine", "inspect": "examine", 
                "use": "examine", "investigate": "examine",
                "grab": "take", "pick up": "take", "pick": "take",
                "search": "look",
            }

            words = action.lower().split()
            if words and words[0] in invalid_verb_map:
                replacement = invalid_verb_map[words[0]]
                words[0] = replacement
                action = " ".join(words)

            action = action.lower().strip()
            action = re.sub(r'[*`]', '', action)
            action = " ".join(action.split())

            tool_args["action"] = action

        return tool_name, tool_args
    
    def _validate_action(self, action: str) -> str:
        action = action.lower().strip()
        action = " ".join(action.split())

        # Expand common abbreviations
        words = action.split()
        if words and words[0] in ABBR_DIR_TO_FULL:
            words[0] = ABBR_DIR_TO_FULL[words[0]]
            action = " ".join(words)

        # Strip "go " prefix ("go west" -> "west")
        if action.startswith("go "):
            action = action[3:]

        return action
    
    # =============================================================================
    # Helpers
    # =============================================================================

    def _extract_result(self, result) -> str:
        """Extract text from MCP tool result."""
        if hasattr(result, 'content') and result.content:
            return result.content[0].text
        if isinstance(result, list) and result:
            return result[0].text if hasattr(result[0], 'text') else str(result[0])
        return str(result)
    
    def _update_score(self, text: str) -> None:
        """Update score from game text."""
        old_score = self.score
        patterns = [
            r'Score:\s*(\d+)',
            r'score[:\s]+(\d+)',
            r'\[Score:\s*(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                self.score = max(self.score, int(match.group(1)))

    def _is_game_over(self, text: str) -> bool:
        """Check if the game is over."""
        game_over_phrases = [
            "game over",
            "you have died",
            "you are dead",
            "*** you have died ***",
        ]
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in game_over_phrases)


# =============================================================================
# Local Testing
# =============================================================================

async def test_agent():
    """Test the agent locally."""
    from fastmcp import Client
    
    agent = StudentAgent()
    
    async with Client("mcp_server.py") as client:
        result = await agent.run(
            client=client,
            game="zork1",
            max_steps=20,
            seed=42,
            verbose=True,
        )
        
        print(f"\n{'=' * 50}")
        print(f"Final Score: {result.final_score}")
        print(f"Moves: {result.moves}")
        print(f"Locations: {len(result.locations_visited)}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_agent())