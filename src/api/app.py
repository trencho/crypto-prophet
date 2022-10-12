from uvicorn import run

from api.config import app

app = app

if __name__ == '__main__':
    run('app:app', host='0.0.0.0', debug=True, reload=True)
