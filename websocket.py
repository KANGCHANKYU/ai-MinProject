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
import io,os,cv2,glob , json , websockets , asyncio
from util.filereader import fileInputList 
from util.webStream import webcam_stream
from util.webStream import faecId

app = FastAPI()

