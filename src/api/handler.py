"""AWS Lambda entrypoint. Mangum adapts the ASGI app to the Lambda runtime."""
from mangum import Mangum

from src.api.main import app

handler = Mangum(app)
