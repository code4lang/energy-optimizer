import requests
import pandas as pd
import statsmodels.api as sm
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from influxdb_client import InfluxDBClient
from geopy.geocoders import Nominatim

# 1. Set Up AI Model (Assumes Ollama is serving LLaMA3 locally)
llm = OllamaLLM(model="llama3")

# 2. Memory with Vector Store
vectorstore = FAISS(embedding_function=OpenAIEmbeddings())
past_recommendations = [
    {"query": "Best insulation methods", "response": "Use spray foam insulation."},
    {"query": "Solar panel efficiency", "response": "Monocrystalline panels are most efficient."}
]
vectorstore.add_documents(past_recommendations)

# 3. Live Energy Data
client = InfluxDBClient(url="http://localhost:8086", token="your_token", org="your_org")
query = 'from(bucket:"energy_data") |> range(start: -1h) |> filter(fn: (r) => r["_measurement"] == "power_usage")'
tables = client.query_api().query(query)

# 4. Weather Data
geolocator = Nominatim(user_agent="energy_optimizer")
location = geolocator.geocode("Paris, France")
latitude, longitude = location.latitude, location.longitude

weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
weather_data = requests.get(weather_url).json()
temperature = weather_data["current_weather"]["temperature"]
humidity = weather_data["current_weather"].get("relative_humidity", 50)
solar_radiation = weather_data["current_weather"].get("solar_radiation", 300)

# 5. Forecast Energy Usage
data = pd.read_csv("energy_consumption.csv", parse_dates=["date"], index_col="date")
data["temperature"] = temperature
data["humidity"] = humidity
data["solar_radiation"] = solar_radiation

model_forecast = sm.tsa.SARIMAX(data["consumption"],
                                exog=data[["temperature", "humidity", "solar_radiation"]],
                                order=(2, 1, 2), seasonal_order=(1, 1, 1, 12))
results = model_forecast.fit()
forecast = results.get_forecast(steps=30, exog=[[temperature, humidity, solar_radiation]])
forecast_mean = forecast.predicted_mean

# 6. Reinforcement-like Regression (Mocked)
df = pd.read_csv("recommendation_impact.csv")
X = df[["insulation_upgrade", "smart_thermostat", "solar_panels"]]
y = df["energy_savings"]

from sklearn.linear_model import LinearRegression
rl_model = LinearRegression().fit(X, y)
new_measures = np.array([[1, 1, 0]])  # Example config
predicted_savings = rl_model.predict(new_measures)

# 7. Generate Recommendation
template = """
You are an energy optimization assistant. Given current energy usage and weather data:

- Insulation upgrade: potential savings = {insulation_savings} units
- Smart thermostat: potential savings = {smart_thermostat_savings} units
- Solar panels: potential savings = {solar_panels_savings} units

Provide your setup to personalize your optimization plan.
"""
prompt = PromptTemplate(template=template)
recommendation = RetrievalQA(llm=llm, prompt=prompt)

print(recommendation.generate_response({
    "insulation_savings": round(predicted_savings[0] * 0.4, 2),
    "smart_thermostat_savings": round(predicted_savings[0] * 0.35, 2),
    "solar_panels_savings": round(predicted_savings[0] * 0.25, 2)
}))

