import os
import datetime
import requests
from dotenv import load_dotenv
from tavily import TavilyClient

from strands import Agent, tool
from strands.models.gemini import GeminiModel
from strands.models.ollama import OllamaModel

# -------------------------------------------------
# Configuration
# -------------------------------------------------

COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
PRICE_HISTORY_DAYS = 7
CURRENCY = "usd"
MAX_NEWS_RESULTS = 5

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# -------------------------------------------------
# Utility
# -------------------------------------------------

def convert_timestamp(timestamp_ms):
    return datetime.datetime.fromtimestamp(
        timestamp_ms / 1000
    ).strftime("%Y-%m-%d %H:%M:%S")


@tool
def current_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# -------------------------------------------------
# API Tools
# -------------------------------------------------

@tool
def fetch_crypto_news(crypto: str):

    query = f"latest news about {crypto} cryptocurrency"

    results = tavily.search(
        query=query,
        max_results=MAX_NEWS_RESULTS
    )

    return results.get("results", [])


@tool
def fetch_price_data(crypto: str):

    url = f"{COINGECKO_BASE_URL}/coins/{crypto}/market_chart"

    params = {
        "vs_currency": CURRENCY,
        "days": PRICE_HISTORY_DAYS
    }

    response = requests.get(url, params=params)

    prices = response.json()["prices"]

    formatted = []

    for ts, price in prices:
        formatted.append({
            "time": convert_timestamp(ts),
            "price": price
        })

    return formatted


# -------------------------------------------------
# Model
# -------------------------------------------------

# model = GeminiModel(
#     client_args={"api_key": os.getenv("GEMINI_API_KEY")},
#     model_id="gemini-2.5-flash"
# )

model = OllamaModel(
    host="http://localhost:11434",  # Ollama server address
    model_id="ministral-3:3b"               # Specify which model to use
)

# -------------------------------------------------
# Specialist Agents
# -------------------------------------------------

news_agent = Agent(
    model=model,
    tools=[fetch_crypto_news, current_time],
    system_prompt="""
You are a crypto news analyst.

Fetch recent news and summarize important developments
that could influence the cryptocurrency market.
"""
)


sentiment_agent = Agent(
    model=model,
    system_prompt="""
Classify crypto news sentiment as:

bullish
neutral
bearish

Explain the reasoning briefly.
"""
)


price_agent = Agent(
    model=model,
    tools=[fetch_price_data, current_time],
    system_prompt="""
Retrieve crypto price history and summarize
important price movements.

Valid crypto IDs: bitcoin, ethereum, solana, etc.
"""
)


risk_agent = Agent(
    model=model,
    tools=[current_time],
    system_prompt="""
Identify major investment risks such as:

- volatility
- regulation
- liquidity
- macroeconomic conditions
"""
)


report_agent = Agent(
    model=model,
    tools=[current_time],
    system_prompt="""
Generate a crypto market report.

Sections:
- Market Overview
- News Sentiment
- Price Analysis
- Risk Assessment
- Investment Outlook
"""
)


debate_agent = Agent(
    model=model,
    system_prompt="""
You are a skeptical financial analyst.

Your job is to challenge the investment report.

Identify:
- missing risks
- weak assumptions
- overconfident conclusions

Provide a critique of the analysis.
"""
)

# -------------------------------------------------
# Agent Tools
# -------------------------------------------------

@tool
def news_analysis_tool(crypto: str):
    return news_agent(f"Analyze news for {crypto}")


@tool
def sentiment_tool(news: str):
    return sentiment_agent(f"Analyze sentiment of this news: {news}")


@tool
def price_tool(crypto: str):
    return price_agent(f"Get price history for {crypto}")


@tool
def risk_tool(context: str):
    return risk_agent(context)


@tool
def report_tool(context: str):
    return report_agent(context)


@tool
def debate_tool(report: str):
    return debate_agent(
        f"Critically evaluate this investment report:\n\n{report}"
    )


# -------------------------------------------------
# Coordinator Agent
# -------------------------------------------------

crypto_coordinator = Agent(
    model=model,
    tools=[
        news_analysis_tool,
        sentiment_tool,
        price_tool,
        risk_tool,
        report_tool,
        debate_tool
    ],
    system_prompt="""
You are a cryptocurrency research coordinator.

Create a full crypto market analysis.

Suggested workflow:

1. Analyze latest news
2. Determine market sentiment
3. Retrieve price history
4. Evaluate market risks
5. Generate investment report
6. Ask the debate agent to critique the report

Use tools only when necessary and avoid repeating calls.
"""
)

# -------------------------------------------------
# Run Analysis
# -------------------------------------------------

if __name__ == "__main__":

    result = crypto_coordinator(
        "Create a full market analysis for Bitcoin."
    )

    print(result)
