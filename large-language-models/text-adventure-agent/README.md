---
title: Text Adventure Agent Submission
emoji: "\U0001F5FA"
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "5.12.0"
app_file: app.py
pinned: false
license: mit
---

# Text Adventure Agent Submission

## Overview

This is my submission for the Text Adventure Agent assignment. My agent uses the ReAct pattern to play text adventure games via MCP.

## Approach

I started from the `example_submission` implementation and played multiple games to observe common patterns and failure modes, which guided the improvements described below:

#### Location extraction
Previous implementation was too simple, so responses like `Grunk just see...` were classified as location. My version checks if the action was not a movement command, if the response is single-line or longer than 30 chars, in which case location stays the same. Otherwise, location is extracted as the first line of the response.

#### Observation parsing
I first predefined all movement actions and common game objects and then extended agent to parse each observation to extract and match mentioned directions and objects. This later served as a potential clue and feed into both the state tracking and the loop/failed action prevention systems.

#### State tracking system
After each action (and observation), I collected and updated the action and location data. I maintained the state information of objects at the location, the exits from the location, and the actions attempted there. I also checked whether the action had failed and whether the agent was in a state of darkness.

#### Failed actions detection
I curated a list of common failure phrases and marked an action as failed if its observation contained any of them. An action was also treated as a failure if it produced the same observation as one of the previous actions. If the agent attempts to retry a previously failed action at the same location, it intervenes and selects an alternative action.

#### Loop detection
My version expands loop detection by identifying the following as signs that the agent is stuck: (1) repeated actions, (2) cyclical locations (e.g., A-B-A-B), (3) cyclical directional actions (e.g. 'n'->'s'->'n'->'s') (4) repeated location-action pairs, and (5) repeated observations.

#### Failed action and loop prevention
When a failed action is detected, the system automatically selects the next action using a weighted random policy. It builds a candidate set of untried and non-failed actions at the current location, and samples from several action pools: light actions (when in darkness), movement actions (extracted from the latest observation), 'look' action, and object actions (constructed from most common verbs (e.g. 'examine', 'take') and  objects present in the room). If no valid candidates remain, it falls back to choosing a movement action. 
The same was initially applied for loop prevention, however, in practice, simply selecting 'look' action yielded better results.

#### Final note
All of the above changes were applied to the agent module. In the MCP server module, only the location extraction logic was updated, and the tool docstrings were improved.

## Files

| File | Description |
|------|-------------|
| `agent.py` | ReAct agent with `StudentAgent` class |
| `mcp_server.py` | MCP server with game interaction tools |
| `app.py` | Gradio interface for HF Space |
| `requirements.txt` | Additional dependencies |

## How to Submit

1. Fork the template Space: `https://huggingface.co/spaces/LLM-course/text-adventure-template`
2. Clone your fork locally
3. Implement your agent in `agent.py` and `mcp_server.py`
4. Test locally (see below)
5. Push your changes to your Space
6. Submit your Space URL on the course platform

## Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Test the MCP server interactively
fastmcp dev mcp_server.py

# Run your agent on a game
python run_agent.py --agent . --game lostpig -v -n 20

# Run evaluation
python -m evaluation.evaluate -s . -g lostpig -t 3
```
