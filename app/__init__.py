"""Inference API package. Dashboard entry point: dashboard:server."""


def __getattr__(name):
    # Preserve existing manually configured Render command: gunicorn app:server.
    # Import lazily so the API does not require Dash in its Docker image.
    if name == 'server':
        from dashboard import server
        return server
    raise AttributeError(name)
