from app import app

# Vercel Handler
def handler(event, context):
    from mangum import Mangum
    return Mangum(app)(event, context)
