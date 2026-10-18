import re
import pandas as pd
from dateutil import parser as dateutil_parser

# Matches the timestamp WhatsApp puts at the start of every message line, in either
# export style:
#   Android/typical:  12/31/23, 11:59 pm - Name: message
#   iOS/bracketed:    [12/31/23, 11:59:59 PM] Name: message
# Handles optional seconds, optional AM/PM, and '/', '.', or '-' as date separators.
DATE_LINE_PATTERN = re.compile(
    r"""^
    (?P<prefix>\[)?
    (?P<date>\d{1,4}[/.\-]\d{1,2}[/.\-]\d{1,4})
    ,?\s
    (?P<time>\d{1,2}:\d{2}(?::\d{2})?\s?(?:[APap][Mm])?)
    (?(prefix)\]\s|\s-\s)
    """,
    re.VERBOSE | re.MULTILINE,
)

# WhatsApp exports sometimes contain invisible directionality marks (LRM/RLM) or a
# narrow no-break space before AM/PM which break naive parsing if left in place.
_INVISIBLE_CHARS = {
    "\u200e": "",
    "\u200f": "",
    "\u202f": " ",
    "\xa0": " ",
}


def _clean_invisible_chars(text: str) -> str:
    for bad, good in _INVISIBLE_CHARS.items():
        text = text.replace(bad, good)
    return text


def _detect_dayfirst(raw_dates):
    """
    Decide whether this export's dates are day-first (31/12/25, most locales)
    or month-first (12/31/25, US-style) by scanning every date in the file.

    If any date's *first* number is > 12, it can't be a month, so the format
    must be day-first. If any date's *second* number is > 12, the format must
    be month-first. A single hardcoded assumption previously misread
    month-first exports like "6/13/26" (June 13) as day=6/month=13, an
    invalid date that got silently mis-shuffled into the wrong month.
    """
    saw_first_gt_12 = False
    saw_second_gt_12 = False
    for raw in raw_dates:
        date_part = raw.split(" ", 1)[0]
        parts = re.split(r"[/.\-]", date_part)
        if len(parts) < 2:
            continue
        try:
            a, b = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if a > 12:
            saw_first_gt_12 = True
        if b > 12:
            saw_second_gt_12 = True

    if saw_second_gt_12 and not saw_first_gt_12:
        return False  # month/day/year (US-style)
    if saw_first_gt_12 and not saw_second_gt_12:
        return True  # day/month/year
    # Every date in the file is ambiguous (both numbers <= 12) or the two
    # signals conflict (shouldn't normally happen in one export) — fall back
    # to day-first, WhatsApp's most common default outside the US/Canada.
    return True


def preprocess(data: str) -> pd.DataFrame:
    data = _clean_invisible_chars(data)

    matches = list(DATE_LINE_PATTERN.finditer(data))
    if not matches:
        raise ValueError(
            "Couldn't find any WhatsApp-style timestamps in this file. "
            "Make sure you exported the chat as a .txt file (Settings > Export Chat > "
            "Without Media) and uploaded that file."
        )

    raw_dates = []
    messages = []
    for i, m in enumerate(matches):
        raw_dates.append(f"{m.group('date')} {m.group('time')}")
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(data)
        messages.append(data[start:end])

    df = pd.DataFrame({"user_message": messages, "message_date_raw": raw_dates})

    # Auto-detect day-first vs month-first from the file itself instead of
    # assuming one locale; rows that still fail to parse (corrupted export,
    # stray text that happened to look like a timestamp, etc.) are dropped
    # rather than crashing.
    dayfirst = _detect_dayfirst(df["message_date_raw"])
    parsed = []
    for raw in df["message_date_raw"]:
        try:
            parsed.append(dateutil_parser.parse(raw, dayfirst=dayfirst))
        except (ValueError, OverflowError):
            parsed.append(pd.NaT)
    df["date"] = parsed
    df.drop(columns=["message_date_raw"], inplace=True)
    df = df.dropna(subset=["date"]).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "Timestamps were found but none of them could be parsed as valid dates. "
            "Please double-check the exported chat file."
        )

    users = []
    messages = []
    for message in df["user_message"]:
        # maxsplit=1 ensures only the sender name (before the FIRST ': ') is split
        # off; without it, any ': ' occurring later inside the message text itself
        # (e.g. "Note: call mom") would get incorrectly split too.
        entry = re.split(r"([\w\W]+?):\s", message, maxsplit=1)
        if entry[1:]:
            users.append(entry[1].strip())
            messages.append("".join(entry[2:]).strip())
        else:
            users.append("group_notification")
            messages.append(entry[0].strip())

    df["user"] = users
    df["message"] = messages
    df.drop(columns=["user_message"], inplace=True)

    df["only_date"] = df["date"].dt.date
    df["year"] = df["date"].dt.year
    df["month_num"] = df["date"].dt.month
    df["month"] = df["date"].dt.month_name()
    df["day"] = df["date"].dt.day
    df["day_name"] = df["date"].dt.day_name()
    df["hour"] = df["date"].dt.hour
    df["minute"] = df["date"].dt.minute

    period = []
    for hour in df["hour"]:
        if hour == 23:
            period.append(f"{hour}-00")
        elif hour == 0:
            period.append(f"00-{hour + 1}")
        else:
            period.append(f"{hour}-{hour + 1}")
    df["period"] = period

    return df
