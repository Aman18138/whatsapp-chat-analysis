import os
import re
from collections import Counter

import pandas as pd
from urlextract import URLExtract
from wordcloud import WordCloud
import emoji

extract = URLExtract()

# Resolve the stopwords file relative to this script, not the process's current
# working directory, so it loads correctly regardless of where streamlit is run from.
_STOP_WORDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stop_hinglish.txt")


def _load_stop_words():
    try:
        with open(_STOP_WORDS_PATH, "r", encoding="utf-8") as f:
            # A *set of individual words*, not the raw text blob. Checking
            # "word not in raw_text" (the original bug) does a substring search
            # against the whole file's text, which silently filters out almost
            # every short/common word and lets some real stopwords slip through.
            return set(f.read().split())
    except FileNotFoundError:
        return set()


STOP_WORDS = _load_stop_words()

# Different WhatsApp versions/languages phrase the "media omitted" placeholder
# differently ("<Media omitted>", "image omitted", "video omitted", ...). Matching
# on the exact original string (with a hardcoded trailing "\n") missed most of these.
MEDIA_OMITTED_PATTERN = re.compile(
    r"<Media omitted>|image omitted|video omitted|audio omitted|sticker omitted|"
    r"GIF omitted|document omitted",
    re.IGNORECASE,
)

# emoji>=2.0 removed `emoji.UNICODE_EMOJI` in favor of `emoji.EMOJI_DATA`; support both
# so this works regardless of which version of the `emoji` package is installed.
if hasattr(emoji, "EMOJI_DATA"):
    EMOJI_SET = set(emoji.EMOJI_DATA.keys())
elif hasattr(emoji, "UNICODE_EMOJI"):
    EMOJI_SET = set(emoji.UNICODE_EMOJI["en"].keys())
else:
    EMOJI_SET = set()


def fetch_stats(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    num_messages = df.shape[0]

    words = []
    for message in df["message"]:
        words.extend(message.split())

    num_media_messages = df["message"].str.contains(MEDIA_OMITTED_PATTERN, na=False).sum()

    links = []
    for message in df["message"]:
        links.extend(extract.find_urls(message))

    return num_messages, len(words), int(num_media_messages), len(links)


def most_busy_users(df):
    x = df["user"].value_counts().head()
    # Explicitly name the columns after reset_index() instead of relying on
    # pandas' auto-generated names, which differ between pandas versions
    # ('index'/'user' vs 'user'/'count') and previously produced a rename that
    # silently did nothing on newer pandas.
    busy_df = (df["user"].value_counts() / df.shape[0] * 100).round(2).reset_index()
    busy_df.columns = ["name", "percent"]
    return x, busy_df


def create_wordcloud(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    temp = df[
        (df["user"] != "group_notification")
        & (~df["message"].str.contains(MEDIA_OMITTED_PATTERN, na=False))
    ].copy()

    def remove_stop_words(message):
        return " ".join(word for word in message.lower().split() if word not in STOP_WORDS)

    temp["message"] = temp["message"].apply(remove_stop_words)
    text = temp["message"].str.cat(sep=" ").strip()

    if not text:
        # Nothing left to plot (e.g. an all-media or all-stopword conversation);
        # let the caller decide how to communicate that instead of crashing.
        return None

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color="white")
    return wc.generate(text)


def most_common_words(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    temp = df[
        (df["user"] != "group_notification")
        & (~df["message"].str.contains(MEDIA_OMITTED_PATTERN, na=False))
    ]

    words = []
    for message in temp["message"]:
        for word in message.lower().split():
            if word not in STOP_WORDS:
                words.append(word)

    return pd.DataFrame(Counter(words).most_common(20), columns=["word", "count"])


def emoji_helper(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    emojis = []
    for message in df["message"]:
        emojis.extend([c for c in message if c in EMOJI_SET])

    counts = Counter(emojis)
    return pd.DataFrame(counts.most_common(len(counts)), columns=["emoji", "count"])


def monthly_timeline(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    timeline = df.groupby(["year", "month_num", "month"]).count()["message"].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline["month"][i] + "-" + str(timeline["year"][i]))
    timeline["time"] = time

    return timeline


def daily_timeline(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    return df.groupby("only_date").count()["message"].reset_index()


def week_activity_map(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    return df["day_name"].value_counts()


def month_activity_map(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    return df["month"].value_counts()


def activity_heatmap(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    user_heatmap = df.pivot_table(
        index="day_name", columns="period", values="message", aggfunc="count"
    ).fillna(0)

    # Order rows Monday->Sunday instead of the default alphabetical sort
    # ("Friday, Monday, Saturday, ...") which is hard to read at a glance.
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    ordered = [d for d in day_order if d in user_heatmap.index]
    return user_heatmap.reindex(ordered)
