import asyncio
import csv
import os
import pandas as pd
from playwright.async_api import async_playwright

DATA_FOLDER = r"E:\final\2024"
OUTPUT_FILE = r"E:\final\2024_tweets_raw.csv"
DONE_FILE = r"E:\final\tweets_done.txt"

TIMEOUT = 2000   

async def fetch_tweet(page, tweet_id):
    url = f"https://x.com/i/status/{tweet_id}"
    try:
        await page.goto(url, timeout=TIMEOUT)
        await page.wait_for_selector("[data-testid='tweetText']", timeout=TIMEOUT)
        element = await page.query_selector("[data-testid='tweetText']")
        text = await element.inner_text()
        return text
    except Exception:
        return None


async def main():
   
    done = set()
    if os.path.exists(DONE_FILE):
        with open(DONE_FILE, "r") as f:
            done = set(line.strip() for line in f.readlines())

 
    all_tweet_ids = []
    for file in os.listdir(DATA_FOLDER):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(DATA_FOLDER, file))
            for _, row in df.iterrows():
                ids = str(row["urls"]).split(",")
                ids = [x.strip() for x in ids if x.strip()]
                for tid in ids:
                    all_tweet_ids.append((row.get("official_id", None),
                                          row.get("calendar_week", None),
                                          tid))

    print(f"🔍 Total tweet IDs collected: {len(all_tweet_ids)}")
    print(f"📄 Already done: {len(done)}")

    async with async_playwright() as p:
        browser = await p.chromium.launch_persistent_context(
            user_data_dir=r"C:\Users\ASUS\AppData\Local\Google\Chrome\User Data\Profile 2",
            headless=False,
            channel="chrome"
        )
        page = await browser.new_page()

       
        write_header = not os.path.exists(OUTPUT_FILE)

        with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f_out, \
             open(DONE_FILE, "a", encoding="utf-8") as f_done:

            writer = csv.writer(f_out)
            if write_header:
                writer.writerow(["official_id", "calendar_week", "tweet_id", "text"])

           
            for idx, (official_id, week, tweet_id) in enumerate(all_tweet_ids):

               
                if tweet_id in done:
                    continue

                print(f"[{idx+1}/{len(all_tweet_ids)}] Fetching {tweet_id} ...")

                text = await fetch_tweet(page, tweet_id)

              
                writer.writerow([official_id, week, tweet_id, text])
                f_out.flush()

               
                f_done.write(tweet_id + "\n")
                f_done.flush()

                
                await asyncio.sleep(0.5)

        await browser.close()

    print("🎉 All tweets scraped with checkpoint system!")
    print("📁 Saved to:", OUTPUT_FILE)

if __name__ == "__main__":
    asyncio.run(main())
