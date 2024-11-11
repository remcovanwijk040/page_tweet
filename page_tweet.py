import streamlit as st

from haystack import Pipeline
from haystack.utils import Secret
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.converters import HTMLToDocument
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator

from dotenv import load_dotenv
import os

# load_dotenv()  # Load environment variables from .env file

# Access the API key from the environment variable
# api_key = os.getenv("XAI_API_KEY")
api_key_st = st.secrets["XAI_API_KEY"]
fetcher = LinkContentFetcher()
converter = HTMLToDocument()
prompt_template = """
Jij bent een social media-assistent die informatie van websites kan halen en samenvatten. Gegeven een de documenten die je krijgt, is jouw taak om boeiende berichten te schrijven waarin je de belangrijkste punten uit de tekst verwerkt naar een interessante post voor X. ik heb een X account waar ik dagelijks updates geef over de laatste ontwikkelingen op het gebied van kunstmatige intelligentie, mijn posts gaan daarom altijd over dit onderwerp.
De informatie die je aangeboden krijgt zijn altijd internetpagina's. Focus je voor het antwoordt alleen op de tekst van het artikel en niet op reacties onder een artikel.

Thread Creëren: Als de inhoud meer dan 250 tekens lang is, maak dan een thread van meerdere berichten, waarbij elk bericht binnen de tekenlimiet blijft en er tegelijkertijd een samenhangend geheel wordt gevormd. De eerste post moet de aandacht van de lezer trekken zodat hij het draadje verder wil lezen.

Aansprekende Stijl: Maak de berichten boeiend en gemakkelijk te begrijpen, gebruikmakend van een beknopte taal. Focus op het vastleggen van de belangrijkste inzichten, citaten of statistieken. Schrijf in het Nederlands.

Hashtags & Vermeldingen: Waar relevant, voeg hashtags of vermeldingen toe om de zichtbaarheid te vergroten. Zorg ervoor dat de hashtags natuurlijk in de context passen.

Link naar pagina: Voeg in de laatste post altijd de link naar de website toe die meegegeven is in de URLS fetcher.

Volgens de inhoud van deze website:
{% for document in documents %}
  {{document.content}}
{% endfor %}

Antwoord:

Eindig met {{link}}
"""
prompt_builder = PromptBuilder(template=prompt_template)

llm = OpenAIGenerator(
    #api_key=Secret.from_env_var("XAI_API_KEY"),
    api_key=api_key_st,
    api_base_url="https://api.x.ai/v1",
    model="grok-beta",
)


pipeline = Pipeline()
pipeline.add_component("fetcher", fetcher)
pipeline.add_component("converter", converter)
pipeline.add_component("prompt", prompt_builder)
pipeline.add_component("llm", llm)

pipeline.connect("fetcher.streams", "converter.sources")
pipeline.connect("converter.documents", "prompt.documents")
pipeline.connect("prompt.prompt", "llm.prompt")


url = "https://techcrunch.com/2024/11/08/chatgpt-told-2m-people-to-get-their-election-news-elsewhere-and-rejected-250k-deepfakes/"

result = pipeline.run({"fetcher": {"urls": [url]},
                       "prompt": {"link": [url]}})

print(result["llm"]["replies"][0])

st.title("Social Media Post Generator")

url = st.text_input("Voer hier de URL in:")

if st.button("Genereer post"):
    result = pipeline.run({"fetcher": {"urls": [url]},
                             "prompt": {"link": [url]}})
    st.text_area("Resultaat:", result["llm"]["replies"][0],
                  height=800)
