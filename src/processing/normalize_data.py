from datetime import datetime


def current_hour() -> datetime:
    t = datetime.now()
    return t.replace(hour=t.hour, minute=0, second=0, microsecond=0)
