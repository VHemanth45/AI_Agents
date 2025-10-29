# Ai_planner

Ai_planner is an experimental CrewAI-based travel-planning toolkit that composes multiple agents and tasks to build rich, actionable trip itineraries. It wires together CrewAI agents (via the `crewai` package), LangChain tools (using an Ollama LLM backend), simple search and calculator tools, and task definitions to produce multi-step planning outputs.

Key ideas:
- Define domain-specific Agent classes in `agents.py`.
- Define Task templates in `tasks.py` for itinerary planning, city selection, and city guides.
- Provide small utility tools in `search_tools.py` and `calculator.py` that agents can call.
- Run an interactive TripCrew from `main.py`.

Status: experimental — see project files to adapt for your setup.

## Features

- Compose multi-agent crews (CrewAI) for travel planning.
- Custom agent roles: Expert Travel Agent, City Selection Expert, Local Tour Guide.
- Tools for web search (SERPER) and lightweight calculations.
- Example interactive runner (`main.py`) to run a TripCrew from the command line.

## Requirements

- Python 3.10 or 3.11
- Poetry (recommended) or pip
- Ollama (local LLM server) for running the `crewai-llama3` model (used via LangChain's Ollama LLM)
- A SERPER API key if you want the search tool to work

Dependencies (from project manifest):

- crewai
- unstructured
- pyowm
- tools
- python-dotenv

See `pyproject.toml` for the pinned project configuration.

## Quick start (recommended: Poetry)

1. Clone the repository:

```zsh
git clone <repo-url>
cd AI_Agents
```

2. Install dependencies with Poetry:

```zsh
poetry install
# or activate the virtualenv and use pip
```

3. Prepare environment variables (create a `.env` file or export variables):

```env
# .env
SERPER_API_KEY=your_serper_api_key_here
# If you use any CrewAI-specific keys add them here
```

4. Run an Ollama local server and ensure the `crewai-llama3` model is available.

This repo includes `llama3crew.sh` which contains the commands used to pull/create an Ollama model. Example:

```zsh
# Make sure you have ollama installed and running
./llama3crew.sh
```

5. Run the interactive TripCrew:

```zsh
python main.py
```

The script prompts for origin, cities, date range, and interests and then runs the configured crew.

## Usage example (programmatic)

Use the `TripCrew` class defined in `main.py` from your own script to programmatically run a crew:

```python
from main import TripCrew

trip = TripCrew(
    origin="San Francisco, CA",
    cities="Paris; Rome; Barcelona",
    date_range="2025-06-01 to 2025-06-10",
    interests="history, food, museums"
)
result = trip.run()
print(result)
```

## Environment variables

- SERPER_API_KEY — required for `SearchTools.search_internet` to use the Serper Google Search API.
- If you run an Ollama server, ensure it's accessible at the URL used by `agents.py` (default: `http://localhost:11434`).

## Files of interest

- `pyproject.toml` — project metadata and dependencies
- `main.py` — example interactive TripCrew runner
- `agents.py` — Agent definitions that create CrewAI `Agent` objects and configure the Ollama LLM
- `tasks.py` — `TravelTasks` templates (plan_itinerary, identify_city, gather_city_info)
- `search_tools.py` — web search helper (requires `SERPER_API_KEY`)
- `calculator.py` — a small calculator tool used by agents
- `llama3crew.sh` and `Llama3Modelfile` — helper script and model definition for preparing a local Ollama model


## Getting help

- Open an issue in this repository for bugs or feature requests.
- Inspect the code in `agents.py` and `tasks.py` for examples of how to extend agents and tasks.

## Maintainer

- Author: Hemanth (from `pyproject.toml` authors list)

## Notes & Security

- This project is experimental and intended for developer use. It runs code that may call external services (SERPER API) and a local Ollama server. Keep API keys secret and do not commit them to version control.
- The sample `calculator` tool uses Python's `eval()` for quick demos; treat this as unsafe for untrusted inputs. Replace with a safe math parser in production.
