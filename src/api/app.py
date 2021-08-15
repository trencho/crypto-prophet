from uvicorn import run

from api.config import create_app

app = create_app()

if __name__ == '__main__':
    run('app:app', host='0.0.0.0', debug=True, reload=True)
