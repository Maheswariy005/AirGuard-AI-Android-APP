# AirGuard AI

**A Lightweight On-Device Machine Learning Framework for Indoor Air Quality Risk Classification Under Connectivity-Constrained Deployments**

College Research Project — Android Application

---

## How to Install the APK on Your Android Phone

1. **Transfer the APK** — Copy `AirGuardAI.apk` to your Android phone via USB cable, Bluetooth, or Google Drive.
2. **Enable Unknown Sources** — On your phone, open Settings → Security (or Apps & Notifications → Special app access → Install unknown apps). Enable installation from your file manager app or browser.
3. **Install** — Open your file manager, find `AirGuardAI.apk`, and tap it to install.
4. **Launch** — Find "AirGuard AI" in your app drawer and open it.

---

## What Each Screen Does

### Dashboard
The home screen shows the most recent sensor reading, current indoor air quality risk level (color-coded), connectivity status, last data sync time, and a 24-point AQI trend chart. Tap **Sync Latest Data** to fetch new readings from the Google Spreadsheet.

### Readings
Scrollable list of all stored sensor readings with filter chips (All / Today / This Week / This Month). Each card shows the timestamp, AQI, risk level badge, temperature, humidity, pressure, and gas resistance.

### Analysis
ML prediction for the most recent reading with confidence percentage and feature contribution breakdown. Also shows AQI distribution pie chart, temperature/humidity trend, and a 14-day risk event history bar chart.

### Model Info
Documentation of the on-device Random Forest model: classification thresholds, feature importance chart, AQI-to-risk-level mapping table, and model specifications.

### Settings
Sync data manually, toggle auto-sync on device boot, view cache statistics, and clear locally stored readings.

---

## How Data Sync Works

The app fetches sensor readings from a Google Spreadsheet published as CSV:

```
https://docs.google.com/spreadsheets/d/13BOzm1LVNGvoJ2wspWgEztHdRzOS6QmOCoNKwiSwB4E/export?format=csv&gid=0
```

Readings are stored locally on the device as a JSON file (`sensor_readings.json` in the app's private storage). The app works fully offline after the first sync — no internet connection is required to view stored data or run ML predictions.

---

## How the Machine Learning Model Works

The app includes a **Random Forest classifier** implemented entirely in Kotlin — no TensorFlow Lite or external ML libraries required.

- **Algorithm**: Random Forest with 15 decision trees
- **Input features (8)**: Temperature, Humidity, Pressure, AQI, Gas Resistance (kΩ), Altitude, Dew Point, Heat Index
- **Output classes (6)**: Good, Moderate, Unhealthy (Sensitive), Unhealthy, Very Unhealthy, Hazardous
- **Classification basis**: US EPA AQI breakpoints adapted to the sensor's 10–100 integer output scale
- **Dominant feature**: AQI (~65% importance), followed by Gas Resistance (~18%)
- **Runs 100% on-device**: No network call needed for inference

### AQI-to-Risk Mapping

| Sensor AQI | Risk Level | EPA Equivalent |
|---|---|---|
| ≤ 20 | Good | 0–50 |
| 21–40 | Moderate | 51–100 |
| 41–60 | Unhealthy (Sensitive) | 101–150 |
| 61–80 | Unhealthy | 151–200 |
| 81–90 | Very Unhealthy | 201–300 |
| 91–100 | Hazardous | 301+ |

---

## Offline Functionality

- Opens and runs without internet after the first sync
- All previously synced readings are stored locally
- ML predictions run on-device with no network required
- Dashboard shows a stale data banner if data is more than 1 hour old
- Last sync time is always displayed

---

## Technical Stack

- **Language**: Kotlin
- **Min Android**: API 26 (Android 8.0)
- **Architecture**: MVVM with LiveData + ViewBinding
- **Charts**: MPAndroidChart v3.1.0
- **Networking**: OkHttp 4.12
- **Serialization**: Gson 2.11
- **ML**: Pure Kotlin Random Forest (no native dependencies)
