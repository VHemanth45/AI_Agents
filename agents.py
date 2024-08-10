from crewai import Agent
from textwrap import dedent
from langchain.llms import Ollama

from search_tools import SearchTools
from calculator import CalculatorTools
# This is an example of how to define custom agents.
# You can define as many agents as you want.
# You can also define custom tasks in tasks.py
class TravelAgents:
    def __init__(self):
        self.Ollama = Ollama(base_url='http://localhost:11434',model="llamacrew")

    def expert_travel_agent(self):
        return Agent(
            role="Expert Travel Agent",
            backstory=dedent(
                f"""Expert in travel planning and logistics. 
                I have decades of expereince making travel iteneraries.
                But always checks from avaiable resources."""),
            goal=dedent(f"""
                        Create a 7-day travel itinerary with detailed per-day plans,
                        include budget, packing suggestions, and safety tips on my own as I am an expert.
                        """),
            tools=[
                SearchTools.search_internet,
                CalculatorTools.calculate
            ],
            verbose=True,
            llm=self.Ollama,
        )

    def city_selection_expert(self):
        return Agent(
            role="City Selection Expert",
            backstory=dedent(
                f"""Expert at analyzing travel data to pick ideal destinations"""),
            goal=dedent(
                f"""Select the best cities based on weather, season, prices, and traveler interests on my own as I am an expert"""),
            tools=[SearchTools.search_internet],
            verbose=True,
            llm=self.Ollama,
        )

    def local_tour_guide(self):
        return Agent(
            role="Local Tour Guide",
            backstory=dedent(f"""Knowledgeable local guide with extensive information
        about the city, it's attractions and customs"""),
            goal=dedent(
                f"""Provide the BEST insights about the selected city on my own as I am an expert"""),
            tools=[SearchTools.search_internet],
            verbose=True,
            llm=self.Ollama,
        )