"""
Student MCP Server for Text Adventure Games

This is your MCP server submission. Implement the tools that your agent
will use to play text adventure games.

Required tool:
    play_action(action: str) -> str
        Execute a game command and return the result.

Recommended tools:
    memory() -> str
        Return current game state, score, and recent history.
    
    inventory() -> str  
        Return the player's current inventory.
    
    get_map() -> str
        Return a map of explored locations.

Test your server with:
    fastmcp dev submission_template/mcp_server.py

Then open the MCP Inspector in your browser to test the tools interactively.
"""

import sys
import os

# Add parent directory to path to import games module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import FastMCP
from games.zork_env import TextAdventureEnv


# =============================================================================
# Create the MCP Server
# =============================================================================

mcp = FastMCP("Student Text Adventure Server")

ALL_DIR = frozenset([
    "north", "south", "east", "west", "up", "down",
    "northeast", "northwest", "southeast", "southwest",
    "go north", "go south", "go east", "go west", "go up", "go down",
    "n", "s", "e", "w", "u", "d", "ne", "nw", "se", "sw",
    "enter"
])

ABBR_DIR_TO_FULL = {
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "u": "up", "d": "down",
}

# =============================================================================
# Game State Management
# =============================================================================

class GameManager:
    """
    Manages the text adventure game state.
    
    TODO: Extend this class to track:
    - Action history (for memory tool)
    - Explored locations (for mapping)
    - Current score and moves
    """
    
    def __init__(self):
        self.env: TextAdventureEnv = None
        self.state = None
        self.game_name: str = ""
        self.history: list[tuple[str, str]] = []
        self.explored_locations: dict[str, set[str]] = {}
        self.current_location: str = "Unknown"
        self.last_score: int = 0
        self.current_step: int = 0

    def _extract_location(self, action: str, observation: str) -> str:
        """Extract location name from observation (usually first line)."""
        action = self._validate_action(action)
        if action not in ALL_DIR and self.current_step:
            return self.current_location
        
        lines = observation.strip().split("\n")

        # If single line or empty, keep previous location (stayed in same room)
        if len(lines) < 2:
            return self.current_location

        location = lines[0].strip()
        if not location:
            for line in lines[1:]:
                location = line.strip()
                if location:
                    break
            if not location:
                return self.current_location

        if len(location) > 30:
            return self.current_location
        return location
    
    def initialize(self, game: str = "zork1"):
        """Initialize or reset the game."""
        self.game_name = game
        self.env = TextAdventureEnv(game)
        self.state = self.env.reset()
        self.history = []
        self.explored_locations = {}
        self.current_location = "Unknown"
        self.last_score = 0
        self.current_step: int = 0

        return self.state.observation
    
    def step(self, action: str) -> str:
        """Execute an action and return the result."""
        if self.env is None:
            self.initialize()
        
        self.state = self.env.step(action)
        result = self.state.observation
        
        # Track history
        self.history.append((action, result))
        if len(self.history) > 50:
            self.history = self.history[-50:]
        
        # Update map
        new_location = self._extract_location(action, result)
        if action in ALL_DIR:
            if self.current_location not in self.explored_locations:
                self.explored_locations[self.current_location] = set()
            if new_location != self.current_location:
                self.explored_locations[self.current_location].add(f"{action} -> {new_location}")
        self.current_location = new_location

        self.last_score = self.state.score
        self.current_step += 1
        
        return result
    
    def get_score(self) -> int:
        """Get current score."""
        return self.state.score if self.state else 0
    
    def get_moves(self) -> int:
        """Get number of moves taken."""
        return self.state.moves if self.state else 0
    
    def get_map(self) -> str:
        """Get a map of explored locations."""
        if not self.explored_locations:
            return "Map: No locations explored yet. Try moving around!"
        
        lines = ["Explored Locations and Exits:"]
        for loc, exits in sorted(self.explored_locations.items()):
            lines.append(f"\n* {loc}")
            for exit_info in sorted(exits):
                lines.append(f"    -> {exit_info}")

        lines.append(f"\n[Current] {self.current_location}")
        return "\n".join(lines)
    
    def get_memory(self) -> str:
        """Get a summary of current game state."""
        recent = self.history[-5:] if self.history else []
        recent_str = (
            "\n".join(f"  > {a}\n    {r[:80]}..." for a, r in recent)
            if recent else "  (none yet)"
        )

        return (
            f"Current State:\n"
            f"- Location: {self.current_location}\n"
            f"- Score: {self.state.score} points\n"
            f"- Moves: {self.state.moves}\n"
            f"- Game: {self.game_name}\n\n"
            f"Recent Actions:\n{recent_str}\n\n"
            f"Current Observation:\n{self.state.observation}"
        )
    
    def get_inventory(self) -> str:
        """Get current inventory."""
        items = self.state.inventory if hasattr(self.state, 'inventory') and self.state.inventory else []
        
        if not items:
            return "Inventory: You are empty-handed."
        
        item_names = []
        for item in items:
            item_str = str(item)
            item_lower = item_str.lower()
            if "parent" in item_lower:
                idx = item_lower.index("parent")
                name = item_str[:idx].strip()
                if ":" in name:
                    name = name.split(":", 1)[1].strip()
                item_names.append(name)
            elif ":" in item_str:
                name = item_str.split(":")[1].strip()
                item_names.append(name)
            else:
                item_names.append(item_str)
        
        return f"Inventory: {', '.join(item_names)}"
    
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

# Global game manager
_game = GameManager()


def get_game() -> GameManager:
    """Get or initialize the game manager."""
    global _game
    if _game.env is None:
        # Get game from environment variable (set by evaluator)
        game = os.environ.get("GAME", "zork1")
        _game.initialize(game)
    return _game


# =============================================================================
# MCP Tools - IMPLEMENT THESE
# =============================================================================

@mcp.tool()
def play_action(action: str) -> str:
    """
    Execute a game command and return the result.
    
    This is the main tool for interacting with the game.
    
    Args:
        action: The command to execute (e.g., "north", "take lamp", "open mailbox")
        
    Returns:
        The game's response to the action
        
    Valid commands include:
        - Movement: north, south, east, west, up, down, enter, exit 
        - Objects: take <item>, drop <item>, open <thing>, examine <thing>
        - Other: look, inventory, read <thing>, turn on lamp
    """
    game = get_game()
    result = game.step(action)
    
    # Add score info
    score_info = f"\n\n[Score: {game.state.score} | Moves: {game.state.moves}]"
    
    if game.state.reward > 0:
        score_info = f"\n\n+{game.state.reward} points! (Total: {game.state.score})"
    
    done_info = ""
    if game.state.done:
        done_info = "\n\nGAME OVER"
    
    return result + score_info + done_info


# TODO: Implement additional tools to help your agent

@mcp.tool()
def memory() -> str:
    """
    Get a summary of the current game state.

    Use this to orient yourself and recall recent actions before deciding your next action.

    Returns:
        - Current location, score, move count, and game name
        - Last 5 actions and their outcomes
        - Current observation (i.e description of the room, visible items, etc.)
    """
    return get_game().get_memory()

@mcp.tool()
def get_map() -> str:
    """
    Get a map of all explored locations and their connections.

    Use this when you feel lost, want to find unexplored areas, or need to
    plan a route. Prefer unexplored directions over revisiting known locations.

    Returns:
        - Each visited location and the exits that led to new location
        - Directions that were tried and unsuccesful per each location
        - Your current location
    """
    return get_game().get_map()


@mcp.tool()
def inventory() -> str:
    """
    Get a list of items you are currently carrying.

    Use this before trying to use, drop or interact with an item to confirm you have it or check whether your inventory is full.

    Returns:
        A list of items in your inventory, or a message indicating you are empty-handed.
    """
    return get_game().get_inventory()


# @mcp.tool()
# def get_valid_actions() -> str:
#     """
#     Get a list of likely valid actions from the current location.
#     
#     Returns:
#         List of actions that might work here
#     """
#     # This is a hint: Jericho provides get_valid_actions()
#     game = get_game()
#     if game.env and game.env.env:
#         valid = game.env.env.get_valid_actions()
#         return "Valid actions: " + ", ".join(valid[:20])
#     return "Could not determine valid actions"


# =============================================================================
# Run the server
# =============================================================================

if __name__ == "__main__":
    # This runs the server with stdio transport (for MCP clients)
    mcp.run()