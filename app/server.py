import aiohttp
import asyncio
import uvicorn
from fastai.vision.all import *
import pathlib
from fastai import *
from fastai.vision import *
from PIL import Image
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url = 'https://www.dropbox.com/s/9rypsvbd0sjx9ri/export.pkl?dl=1'
export_file_name = 'export.pkl'

classes = ['glass', 'plastic', 'metal', 'paper', 'cardboard']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

def read_model(model_file):
    temp = pathlib.WindowsPath
    pathlib.WindowsPath = pathlib.PosixPath
    learn = load_learner(model_file)
    return learn

def yer(filename):
    if filename.name[0:2] == "ca":  # cardboard
        return "cardboard"
    if filename.name[0:2] == "gl":  # glass
        return "glass"
    if filename.name[0:2] == "me":  # metal
        return "metal"
    if filename.name[0:2] == "pa":  # paper
        return "paper"
    if filename.name[0:2] == "pl":  # plastic
        return "plastic"
    if filename.name[0:2] == "tr":  # trash
        return "trash"


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, Path(export_file_name))
    try:
        learn = read_model(export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())

    prediction = learn.predict(img_bytes)[0]
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
