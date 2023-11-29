import cv2, time, glob , io , json , asyncio
import numpy as np
from insightface.app import FaceAnalysis
from fastapi import File, Request, FastAPI , UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

model = FaceAnalysis(providers=['CPUExecutionProvider'])
model.prepare(ctx_id=0, det_size=(640, 640))

image_extensions = [".jpg", ".jpeg", ".png"]
path = f"resources/image/input/*"
app = FastAPI()
templates = Jinja2Templates(directory="static")
app.mount("/resources/css", StaticFiles(directory="resources/css"), name="css")
app.mount("/resources/js", StaticFiles(directory="resources/js"), name="js")

async def webcam_stream(request : Request):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    face_detected = False
    result = -1
    count = 0

    while True:
        ret, frame = cap.read()  # 웹캠에서 프레임 읽기
        if not ret:
            break
    
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 프레임을 그레이스케일로 변환하여 얼굴 검출 수행
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0 and count == 30:  # 얼굴이 감지되면
          
            if not face_detected:  # 얼굴이 처음 감지되는 경우
                # 웹캠 중지
                cap.release()
                face_detected = True  # 얼굴 감지 상태를 True로 설정
                # 감지된 얼굴 영역을 처리하는 함수 호출
                for (x, y, w, h) in faces:
                    detected_face = frame[y:y+h * 2, x:x+w * 2]
                    file = cv2.cvtColor(detected_face,cv2.COLOR_RGB2BGR)                
                    result = await faecId(file)
                    if result >= 0 :
                        print(result)
                        result_json = json.dumps({"result": result}).encode('utf-8')
                        yield (
                            b'--frame\r\n'
                            b'Content-Type: application/json\r\n\r\n' + result_json + b'\r\n\r\n'
                                )
            else:
                break  # 이미 얼굴이 감지된 상태면 루프 종료
        else:
            # 얼굴이 감지되지 않은 경우 웹캠 계속 실행
            face_detected = False  # 얼굴 감지 상태를 False로 설정
            # 검출된 얼굴 영역에 사각형 그리기
            for (x, y, w, h) in faces:
                shoulder_height = y + h 
                cv2.rectangle(frame, (x, y), (x + w, shoulder_height), (255, 255, 255), 2)
            
            _, jpeg = cv2.imencode('.jpg', frame)  # JPEG 형식으로 프레임 인코딩
            frame_bytes = jpeg.tobytes()  # 프레임을 바이트로 변환하여 스트리밍
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n'
                    )
            
        time.sleep(0.1)
        count = count + 1     
        

async def faecId(file : np.array):
    file_list = glob.glob(path)
    images = []
    faecs = []
    feats = []
    await asyncio.sleep(2)
    file = cv2.cvtColor(file,cv2.COLOR_RGB2BGR)
    feace1 = model.get(file)
    feat1 = np.array(feace1[0].normed_embedding, dtype=np.float32)
    for f in file_list:
        f = f.replace("\\","/")
        img = cv2.imread(f)
        images.append(img)

    for i in images:
        faec = model.get(i)
        faecs.append(faec)

    for faec in faecs:
        feats.append(faec[0].normed_embedding)

    feats = np.array(feats, dtype=np.float32)
    result = "타인"
    for feat in feats:
        sims = np.dot(feat1,feat)
        if np.any(sims > 0.55):
            result = "동일인"
            break
        elif np.all(sims < 0.54):
            continue 
    print(f"result: {result}")
    return {"result":result}