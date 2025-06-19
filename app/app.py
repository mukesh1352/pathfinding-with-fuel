from app.app import app


# Vercel handler
def handler(request, response):
    return app(request.environ, start_response=lambda *args: None)
