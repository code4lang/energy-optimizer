import time
import random
import json
import requests
from datetime import datetime
from influxdb_client import InfluxDBClient, Point, WritePrecision

# --- CONFIGURATION ---
bucket = "energy_data"
org = "ecovie"
token = "your_token_here"
url = "http://localhost:8086"
location = "Paris, France"
log_to_file = True  # Set to False if you don't want to log locally

# --- InfluxDB client ---
client = InfluxDBClient(url=url, token=token, org=org)
write_api = client.write_api()  # Corrected: use default write options

# --- Weather fetch ---
def get_weather(latitude=48.8566, longitude=2.3522):
    try:
        api = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
        response = requests.get(api, timeout=5)
        weather = response.json()["current_weather"]
        return weather.get("temperature", 20.0), weather.get("windspeed", 5.0), weather.get("relative_humidity", 50)
    except:
        return 20.0, 5.0, 50  # fallback values

# --- Energy model ---
def generate_fake_power(temperature):
    base_load = 300
    temp_adjustment = (25 - temperature) * 1.5
    fluctuation = random.uniform(-20, 20)
    return base_load + temp_adjustment + fluctuation

print("🔁 Streaming energy + weather data… Press Ctrl+C to stop.")
try:
    while True:
        now = datetime.utcnow()
        temperature, windspeed, humidity = get_weather()
        power_usage = generate_fake_power(temperature)

        point = (
            Point("power_usage")
            .tag("location", "Paris")
            .field("value", round(power_usage, 2))
            .field("temperature", temperature)
            .field("windspeed", windspeed)
            .field("humidity", humidity)
            .time(now, WritePrecision.NS)
        )

        write_api.write(bucket=bucket, org=org, record=point)

        if log_to_file:
            log_line = {
                "time": now.isoformat(),
                "power_usage": round(power_usage, 2),
                "temperature": temperature,
                "windspeed": windspeed,
                "humidity": humidity
            }
            with open("energy_weather_log.jsonl", "a") as f:
                f.write(json.dumps(log_line) + "\n")

        print(f"✅ Sent: {round(power_usage,2)}W | temp={temperature}°C | wind={windspeed}km/h | humidity={humidity}%")
        time.sleep(10)

except KeyboardInterrupt:
    print("\n⏹️ Streaming stopped.")
finally:
    client.close()

