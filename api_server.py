import streamlit as st
import requests
import random

st.title("Real-Time Rainfall & Alerts")

# ------------------ Helpers ------------------

def geocode_location(location):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": location, "format": "json", "limit": 1}
    res = requests.get(url, params=params, headers={"User-Agent": "rain-app"})
    data = res.json()

    if not data:
        return None, None

    return float(data[0]["lat"]), float(data[0]["lon"])


def get_rainfall(lat, lon):
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=rain,showers,precipitation,precipitation_probability"
    )

    res = requests.get(url)
    data = res.json()

    rain_list = data["hourly"]["rain"]
    showers_list = data["hourly"]["showers"]
    precip_list = data["hourly"]["precipitation"]

    # Choose the highest among rain, showers, precipitation
    combined = [
        max(r, s, p)
        for r, s, p in zip(rain_list, showers_list, precip_list)
    ]

    # Get the rainfall at the CURRENT hour
    current_rainfall = combined[0]

    # If it is ZERO, pick the highest in next 24 hours (better accuracy)
    if current_rainfall == 0:
        current_rainfall = max(combined[:24])

    return float(current_rainfall)



def find_safe_place(lat, lon):
    safe_locations = []

    for _ in range(5):  # check 5 nearby locations
        new_lat = lat + random.uniform(-0.05, 0.05)
        new_lon = lon + random.uniform(-0.05, 0.05)
        rain = get_rainfall(new_lat, new_lon)
        safe_locations.append((rain, new_lat, new_lon))

    safe_locations.sort(key=lambda x: x[0])
    return safe_locations[0]  # lowest rainfall location

# ------------------ Streamlit UI ------------------

location = st.text_input("Enter location to check rainfall:")

if st.button("Check Rainfall"):
    if not location:
        st.error("Please enter a location")
    else:
        lat, lon = geocode_location(location)

        if lat is None:
            st.error("Location not found!")
        else:
            rainfall = get_rainfall(lat, lon)

            st.write(f"**Current Rainfall at {location}: {rainfall} mm**")

            # ---- Alerts ----
            if rainfall > 20:
                st.error("DANGER ⚠️ — Very high rainfall. Landslide risk!")
            elif rainfall > 10:
                st.warning("Alert 🚨 — Heavy rainfall detected.")
            else:
                st.success("Safe 🌤️ — Rainfall is low.")

            # ---- Suggest a safer place ----
            safe_rain, safe_lat, safe_lon = find_safe_place(lat, lon)

            st.write("### 🟢 Recommended Safer Nearby Location")
            st.write(f"**Rainfall:** {safe_rain} mm")
            st.write(f"**Latitude:** {safe_lat}")
            st.write(f"**Longitude:** {safe_lon}")


            # -------------- Optional Alert via Email / SMS --------------
            st.write("---")
            st.write("### 🔔 Send Alert Notification (Optional)")

            contact = st.text_input("Enter email or phone number:")
            if st.button("Send Alert"):
                if rainfall > 10:
                    st.success("Alert sent! (Mock — Connect API like Twilio/SMTP here)")
                else:
                    st.info("Rainfall is low — No alert required.")
