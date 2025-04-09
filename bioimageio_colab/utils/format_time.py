from datetime import datetime, timezone


def format_time(last_deployed_time_s, tz: timezone = timezone.utc) -> str:
    current_time = datetime.now(tz)
    last_deployed_time = datetime.fromtimestamp(last_deployed_time_s, tz)

    duration = current_time - last_deployed_time

    days = duration.days
    seconds = duration.seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60

    duration_parts = []
    if days > 0:
        duration_parts.append(f"{days}d")
    if hours > 0:
        duration_parts.append(f"{hours}h")
    if minutes > 0:
        duration_parts.append(f"{minutes}m")
    if remaining_seconds > 0:
        duration_parts.append(f"{remaining_seconds}s")

    duration_str = " ".join(duration_parts)

    return {
        "last_deployed_at": last_deployed_time.strftime("%Y/%m/%d %H:%M:%S"),
        "duration_since": duration_str,
    }
