from datetime import datetime


async def closest_hour(t: datetime) -> datetime:
    return t.replace(
        hour=t.hour if t.minute <= 30 else 0 if t.hour == 23 else t.hour + 1,
        minute=0,
        second=0,
        microsecond=0,
    )


def current_hour() -> datetime:
    t = datetime.now()
    return t.replace(hour=t.hour, minute=0, second=0, microsecond=0)


async def flatten_json(nested_json: dict, exclude: list[str] = None) -> dict:
    """
    Flatten a list of nested dicts.
    """
    if exclude is None:
        exclude = [""]
    out = {}

    def flatten(
        x: (list, dict, str), name: str = "", exclude: list[str] = exclude
    ) -> None:
        if type(x) is dict:
            for a in x:
                if a not in exclude:
                    flatten(x[a], f"{name}{a}_")
        elif type(x) is list:
            if len(x) == 1:
                flatten(x[0], f"{name}")
            else:
                i = 0
                for a in x:
                    flatten(a, f"{name}{i}_")
                    i += 1
        else:
            out[name[:-1]] = x

    flatten(nested_json)
    return out


async def next_hour(t: datetime) -> datetime:
    return t.replace(
        day=t.day + 1 if t.hour == 23 else t.day,
        hour=0 if t.hour == 23 else t.hour + 1,
        minute=0,
        second=0,
        microsecond=0,
    )
