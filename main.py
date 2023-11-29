from typing import Union
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from fastapi import FastAPI,Request,UploadFile, File , WebSocket
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, StreamingResponse
from PIL import Image
import numpy as np
import insightface
import io,os,cv2,glob , json , websockets , asyncio , time ,base64,asyncio
from util.filereader import fileInputList 
from util.webStream import webcam_stream
from util.webStream import faecId

#model = FaceAnalysis(allowed_modules=['detection','recongnition'],providers=['CPUExecutionProvider'])
model = FaceAnalysis(providers=['CPUExecutionProvider'])
model.prepare(ctx_id=0, det_size=(640, 640))

app = FastAPI()
templates = Jinja2Templates(directory="static")
app.mount("/resources/css", StaticFiles(directory="resources/css"), name="css")
app.mount("/resources/js", StaticFiles(directory="resources/js"), name="js")


image_extensions = [".jpg", ".jpeg", ".png"]
path = f"resources/image/input/*"


@app.get("/")
async def index(request : Request):
    return templates.TemplateResponse("index.html",{"request": request})

@app.get("/can")
async def can(request : Request):
    return StreamingResponse(webcam_stream(request), media_type="multipart/x-mixed-replace;boundary=frame")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    face_detected = False
    count = 0

    await websocket.accept()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0 and count == 10:
            if not face_detected:
                cap.release()
                face_detected = True
                for (x, y, w, h) in faces:
                    detected_face = frame[y:y + h * 2, x:x + w * 2]
                    file = cv2.cvtColor(detected_face, cv2.COLOR_RGB2BGR) 
                    result = await faecId(file)
                    print(result)
                    time.sleep(1)
                    result_json = json.dumps(result)
                    await websocket.send_text(result_json)
            else:
                break
        else:
            face_detected = False
            for (x, y, w, h) in faces:
                shoulder_height = y + h
                cv2.rectangle(frame, (x, y), (x + w, shoulder_height), (255, 255, 255), 2)
            _, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            encoded_blob = base64.b64encode(frame_bytes).decode('utf-8')   
            data = {"image" : encoded_blob}
            json_data = json.dumps(data)
            await websocket.send_bytes(json_data)
        time.sleep(1/30)
        count = count + 1




# @app.post("/test2")
# async def test2(request: Request, file: File):
#     file_list = glob.glob(path)
#     images = []
#     faecs = []
#     feats = []
#     file_date = await file.read()
#     buffer = io.BytesIO(file_date)
#     pil_img = Image.open(buffer)
#     cv_img = np.array(pil_img)
#     cv_img = cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
#     feace1 = model.get(cv_img)
#     feat1 = np.array(feace1[0].normed_embedding, dtype=np.float32)
#     for f in file_list:
#         f = f.replace("\\","/")
#         img = cv2.imread(f)
#         images.append(img)

#     for i in images:
#         faec = model.get(i)
#         faecs.append(faec)

#     for faec in faecs:
#         feats.append(faec[0].normed_embedding)
#     feats = np.array(feats, dtype=np.float32)
#     result = 0
#     for feat in feats:
#         sims = np.dot(feat1,feat)
#         if np.any(sims > 0.55):
#             result = 1
#             print(sims)
#             break
#         elif np.all(sims < 0.54):
#             continue 

#     return templates.TemplateResponse("index.html",{"request": request, "result" : result})


