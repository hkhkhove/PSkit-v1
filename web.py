from fastapi import FastAPI, File, UploadFile,Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
import random,string
from typing import List
import pskit,manage_tasks, read_results
import time
from fastapi.responses import FileResponse
import multiprocessing
import shutil
from fastapi.templating import Jinja2Templates
from fastapi import Request

app = FastAPI()

app.mount("/BioWeb", StaticFiles(directory="BioWeb", html=True), name="static")
templates = Jinja2Templates(directory="./templates")

def run_task(config):
    task_id=config["task_id"]
    task_name=config["task_name"]

    manage_tasks.init_db()
    manage_tasks.update_status(task_id, task_name,"running", "")
    try:
        pskit.main(config)
        shutil.make_archive(base_name=config["output_dir"], format='zip', root_dir=config["output_dir"])
        manage_tasks.update_status(task_id, task_name,"finished", "")
    except Exception as e:
        manage_tasks.update_status(task_id, task_name,"failed", str(e))

@app.get("/")
async def serve_vue():
    with open("BioWeb/index.html") as f:
        return HTMLResponse(f.read())

@app.post("/upload")
async def create_task(
    task_name:str = Form(...),
    prot_id: str = Form(None),
    chain_id: str = Form(None),
    map_d_threshold: float = Form(None),
    map_k_number: int = Form(None),
    split_start: int = Form(None),
    split_end: int = Form(None),
    annotate_threshold: float = Form(None),
    prot_seq: str = Form(None),
    ligand_typ: str = Form(None),
    feat_typ: List[str] = Form(None),
    files: List[UploadFile] = File(None)):

    current_time = int(time.time() * 1000)
    task_id=str(current_time)+''.join(random.choices(string.digits, k=8))
    input_dir=f'./task/uploads/{task_id}'
    output_dir=f'./task/results/{task_id}'
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    config = {
        "task_id": task_id,
        "task_name": task_name,
        "prot_id": prot_id,
        "chain_id": chain_id,
        "map_d_threshold": map_d_threshold,
        "map_k_number": map_k_number,
        "split_start": split_start,
        "split_end": split_end,
        "annotate_threshold": annotate_threshold,
        "prot_seq": prot_seq,
        "ligand_typ": ligand_typ,
        "feat_typ": feat_typ,
        "input_dir": input_dir,
        "output_dir": output_dir
    }
    print(config)
    
    if files!=None:
        for file in files:
            content = await file.read()
            file_path=os.path.join(input_dir, file.filename)
            with open(file_path, "wb") as f:
                f.write(content)

    process = multiprocessing.Process(target=run_task, args=(config,))
    process.start()

    return {"task_id": task_id}

@app.get("/download/{task_id}")
async def download_results(task_id: str):
    zip_file_path = os.path.join('./task/results/', f'{task_id}.zip')
    return FileResponse(path=zip_file_path, filename=f'{task_id}.zip', media_type='application/zip')

@app.get("/result/{task_id}", response_class=HTMLResponse)
async def get_results(request: Request, task_id: str):
    return templates.TemplateResponse("result.html", {"request":request,"task_id": task_id})

@app.get("/api/result/{task_id}")
async def get_task_status(task_id: str):
    task = manage_tasks.get_task(task_id)
    if task["status"] == "finished":
        results= read_results.read(task["name"], task_id)
        if results:
            task["results"]=results
        else:
            task["status"]="failed"
            task["error"]="No results found, please contact the administrator."

    return task

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web:app", host="0.0.0.0", port=8082, reload=True)

