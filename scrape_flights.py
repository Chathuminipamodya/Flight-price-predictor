from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import random
from datetime import datetime, timedelta

# -----------------------------------------------
# TARGET — 18 routes x 150 rows = 2700 total
# -----------------------------------------------
ROWS_PER_ROUTE = 150

# -----------------------------------------------
# Create browser function
# -----------------------------------------------
def create_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--start-maximized")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    d = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    d.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return d

def close_driver(d):
    try:
        d.quit()
    except:
        pass

# -----------------------------------------------
# Extract flights from page
# -----------------------------------------------
def extract_flights(driver, origin, destination, date):
    results = []
    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'li.pIav2d'))
        )
        time.sleep(3)

        flight_items = driver.find_elements(By.CSS_SELECTOR, 'li.pIav2d')

        for item in flight_items:
            try:
                # Airline
                try:
                    airline = item.find_element(
                        By.CSS_SELECTOR, '.Ir0Voe .sSHqwe'
                    ).text.strip()
                except:
                    airline = "Unknown"

                # Times
                try:
                    times = item.find_elements(
                        By.CSS_SELECTOR, 'span[aria-label]'
                    )
                    dep_time = times[0].text.strip() if len(times) > 0 else "Unknown"
                    arr_time = times[1].text.strip() if len(times) > 1 else "Unknown"
                except:
                    dep_time = "Unknown"
                    arr_time = "Unknown"

                # Duration
                try:
                    duration = item.find_element(
                        By.CSS_SELECTOR, '.gvkrdb'
                    ).text.strip()
                except:
                    duration = "Unknown"

                # Stops
                try:
                    stops = item.find_element(
                        By.CSS_SELECTOR, '.EfT7Ae span'
                    ).text.strip()
                except:
                    stops = "Unknown"

                # Price
                try:
                    price = item.find_element(
                        By.CSS_SELECTOR, '.FpEdX span'
                    ).text.strip()
                except:
                    try:
                        price = item.find_element(
                            By.CSS_SELECTOR, '.YMlIz.FpEdX span'
                        ).text.strip()
                    except:
                        price = "Unknown"

                # Skip rows with no useful data
                if airline == "Unknown" and price == "Unknown":
                    continue

                results.append({
                    "origin"         : origin,
                    "destination"    : destination,
                    "date"           : date,
                    "airline"        : airline,
                    "departure_time" : dep_time,
                    "arrival_time"   : arr_time,
                    "duration"       : duration,
                    "stops"          : stops,
                    "price"          : price,
                })

            except:
                continue

    except Exception as e:
        print(f"  ⚠️ Extract error: {str(e)[:60]}")

    return results

# -----------------------------------------------
# Load existing CSV — YOUR DATA IS SAFE
# -----------------------------------------------
try:
    existing = pd.read_csv("flights_raw.csv")
    # Remove garbage rows from old scraper
    existing = existing[
        ~existing['airline'].astype(str).str.contains(
            "Unknown|Prices include|Sorted by|round trip|Optional",
            na=True
        )
    ]
    existing = existing[
        ~existing['price'].astype(str).str.contains(
            "Unknown|round trip|Prices",
            na=True
        )
    ]
    all_flights = existing.to_dict('records')
    print(f"✅ Loaded {len(all_flights)} existing clean rows")
except:
    all_flights = []
    print("🆕 No existing file found — starting fresh")

# -----------------------------------------------
# All 18 routes
# -----------------------------------------------
routes = [
    ("Colombo", "Dubai"),
    ("Colombo", "Singapore"),
    ("Colombo", "Bangkok"),
    ("Colombo", "London"),
    ("Colombo", "Kuala Lumpur"),
    ("Colombo", "Doha"),
    ("Colombo", "Mumbai"),
    ("Colombo", "Delhi"),
    ("Colombo", "Paris"),
    ("Colombo", "Frankfurt"),
    ("Colombo", "Tokyo"),
    ("Colombo", "Sydney"),
    ("Colombo", "Male"),
    ("Colombo", "Abu Dhabi"),
    ("Colombo", "Riyadh"),
    ("Colombo", "Muscat"),
    ("Colombo", "Bahrain"),
    ("Colombo", "Kathmandu"),
]

# -----------------------------------------------
# Next 30 days
# -----------------------------------------------
dates = []
for i in range(1, 31):
    d = datetime.today() + timedelta(days=i)
    dates.append(d.strftime("%Y-%m-%d"))

# -----------------------------------------------
# Helper — count rows for a specific route
# -----------------------------------------------
def count_route_rows(flights, origin, destination):
    return sum(
        1 for f in flights
        if f['origin'] == origin and f['destination'] == destination
    )

# -----------------------------------------------
# Show current status before starting
# -----------------------------------------------
print("\n📊 CURRENT STATUS:")
print("-" * 45)
total_needed = 0
for origin, destination in routes:
    have = count_route_rows(all_flights, origin, destination)
    need = max(0, ROWS_PER_ROUTE - have)
    total_needed += need
    status = "✅ Done" if need == 0 else f"⏳ Need {need} more"
    print(f"  {origin} → {destination:<15} | Have: {have:<4} | {status}")
print("-" * 45)
print(f"  Total rows now    : {len(all_flights)}")
print(f"  Total still needed: {total_needed}")
print(f"  Target total      : {len(routes) * ROWS_PER_ROUTE}")
print("-" * 45)

# -----------------------------------------------
# Start browser
# -----------------------------------------------
driver = create_driver()
search_count = 0

# -----------------------------------------------
# Main loop
# -----------------------------------------------
print("\n🚀 Starting collection...\n")

for origin, destination in routes:

    route_count = count_route_rows(all_flights, origin, destination)

    # Skip if already complete
    if route_count >= ROWS_PER_ROUTE:
        print(f"✅ {origin} → {destination} already complete ({route_count} rows) — skipping")
        continue

    print(f"\n📍 {origin} → {destination}")
    print(f"   Have {route_count} rows | Need {ROWS_PER_ROUTE - route_count} more")

    for date in dates:

        # Stop this route if target reached
        route_count = count_route_rows(all_flights, origin, destination)
        if route_count >= ROWS_PER_ROUTE:
            print(f"   🎯 Target reached! ({route_count} rows) Moving to next route.")
            break

        for attempt in range(3):
            try:
                url = (
                    f"https://www.google.com/travel/flights?"
                    f"q=Flights+from+{origin}+to+{destination}+on+{date}"
                    f"&curr=USD"
                )
                driver.get(url)
                search_count += 1
                time.sleep(random.uniform(7, 12))

                new_rows = extract_flights(driver, origin, destination, date)

                if len(new_rows) == 0:
                    print(f"   ⚠️  No data found for {date} — skipping")
                    break

                all_flights.extend(new_rows)
                route_count = count_route_rows(all_flights, origin, destination)

                print(
                    f"   ✅ {date} | "
                    f"+{len(new_rows)} flights | "
                    f"Route: {route_count}/{ROWS_PER_ROUTE} | "
                    f"Grand total: {len(all_flights)}"
                )

                # Save every 100 rows
                if len(all_flights) % 100 == 0:
                    pd.DataFrame(all_flights).to_csv("flights_raw.csv", index=False)
                    print(f"   💾 Checkpoint saved: {len(all_flights)} rows")

                # Long break every 15 searches
                if search_count % 15 == 0:
                    wait = random.uniform(30, 50)
                    print(f"\n   😴 Break for {wait:.0f}s to avoid blocking...\n")
                    time.sleep(wait)

                break  # success — next date

            except Exception as e:
                err = str(e)
                if any(x in err for x in [
                    "invalid session", "disconnected",
                    "session deleted", "not connected"
                ]):
                    print(f"\n   🔴 Chrome crashed! Restarting... (attempt {attempt+1}/3)")
                    close_driver(driver)
                    pd.DataFrame(all_flights).to_csv("flights_raw.csv", index=False)
                    print(f"   💾 Saved {len(all_flights)} rows before restart")
                    time.sleep(5)
                    driver = create_driver()
                    print("   🟢 Browser restarted successfully!")
                    time.sleep(8)
                    # Will retry automatically
                else:
                    print(f"   ❌ Error: {err[:80]}")
                    time.sleep(5)
                    break

# -----------------------------------------------
# Final save
# -----------------------------------------------
close_driver(driver)
df = pd.DataFrame(all_flights)
df.to_csv("flights_raw.csv", index=False)

# -----------------------------------------------
# Final summary
# -----------------------------------------------
print("\n" + "=" * 50)
print("🎉 ALL DONE!")
print("=" * 50)
print(f"Total rows: {len(df)}")
print("\nFinal rows per route:")
print("-" * 45)
summary = df.groupby(['origin','destination']).size().reset_index(name='count')
for _, row in summary.iterrows():
    bar = "█" * (row['count'] // 10)
    print(f"  {row['origin']} → {row['destination']:<15} | {row['count']:>4} rows | {bar}")
print("-" * 45)
print(f"File saved: flights_raw.csv")
print("=" * 50)