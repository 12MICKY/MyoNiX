cat > app.py << 'EOF'
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn, cv2, numpy as np, mediapipe as mp, time, base64
from collections import deque

app = FastAPI()

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

counter=0
stage=None
last_rep_time=0
angle_buffer=deque(maxlen=5)
running=False

HTML="""
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>AI FITNESS</title>

<style>
body{background:#000;color:white;font-family:sans-serif;text-align:center}
.container{width:900px;max-width:95%;margin:auto}
.title{font-size:42px;color:#22c55e;margin:10px}
video{display:none}
canvas{width:100%;border-radius:14px}
.counter{font-size:120px;color:#22c55e}
.timer{font-size:26px;margin-top:5px}
.btn{margin:6px;padding:12px 25px;border:none;border-radius:12px;font-size:18px;color:white;cursor:pointer}
.start{background:#22c55e}
.stop{background:#eab308}
.reset{background:#ef4444}
</style>
</head>

<body>
<div class="container">
<div class="title">AI FITNESS</div>

<video id="cam" autoplay playsinline></video>
<img id="stream" style="width:100%;border-radius:14px"/>

<div id="count" class="counter">0</div>
<div id="timer" class="timer">00:00</div>

<button class="btn start" onclick="start()">START</button>
<button class="btn stop" onclick="stop()">STOP</button>
<button class="btn reset" onclick="resetAll()">RESET</button>
</div>

<script>
const video=document.getElementById("cam")
const count=document.getElementById("count")
const timer=document.getElementById("timer")
const stream=document.getElementById("stream")

navigator.mediaDevices.getUserMedia({video:true})
.then(s=>video.srcObject=s)

const ws=new WebSocket("wss://"+location.host+"/ws")
const canvas=document.createElement("canvas")

let sec=0
let running=false

setInterval(()=>{
if(running){
sec++
let m=Math.floor(sec/60).toString().padStart(2,'0')
let s=(sec%60).toString().padStart(2,'0')
timer.innerText=m+":"+s
}
},1000)

function start(){running=true;ws.send("START")}
function stop(){running=false;ws.send("STOP")}
function resetAll(){sec=0;running=false;ws.send("RESET")}

ws.onmessage=(e)=>{
const data=JSON.parse(e.data)
count.innerText=data.count
if(data.image){stream.src="data:image/jpeg;base64,"+data.image}
}

setInterval(()=>{
if(!video.videoWidth) return
canvas.width=video.videoWidth
canvas.height=video.videoHeight
canvas.getContext("2d").drawImage(video,0,0)
const img=canvas.toDataURL("image/jpeg",0.6)
ws.send(img)
},120)

</script>
</body>
</html>
"""

@app.get("/",response_class=HTMLResponse)
def home():
    return HTML

@app.websocket("/ws")
async def ws_endpoint(ws:WebSocket):
    global counter,stage,last_rep_time,running
    await ws.accept()

    while True:
        data=await ws.receive_text()

        if data=="START":
            running=True
            continue
        if data=="STOP":
            running=False
            continue
        if data=="RESET":
            counter=0
            stage=None
            last_rep_time=0
            running=False
            await ws.send_json({"count":counter})
            continue

        imgdata=data.split(",")[1]
        img_bytes=base64.b64decode(imgdata)
        npimg=np.frombuffer(img_bytes,np.uint8)
        img=cv2.imdecode(npimg,1)
        if img is None: continue

        img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results=pose.process(img_rgb)

        # ===== DRAW SKELETON =====
        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        # ===== COUNT REP =====
        if running and results.pose_landmarks:
            lm=results.pose_landmarks.landmark
            hip=[lm[24].x,lm[24].y]
            knee=[lm[26].x,lm[26].y]
            ankle=[lm[28].x,lm[28].y]

            angle=abs(np.degrees(
                np.arctan2(ankle[1]-knee[1],ankle[0]-knee[0])-
                np.arctan2(hip[1]-knee[1],hip[0]-knee[0])
            ))

            angle_buffer.append(angle)
            smooth=sum(angle_buffer)/len(angle_buffer)
            now=time.time()

            if smooth<85:
                stage="DOWN"
            if smooth>165 and stage=="DOWN":
                if now-last_rep_time>0.7:
                    counter+=1
                    last_rep_time=now
                    stage="UP"

        # ===== SEND IMAGE BACK =====
        _,buffer=cv2.imencode('.jpg',img)
        img_b64=base64.b64encode(buffer).decode()

        await ws.send_json({
            "count":counter,
            "image":img_b64
        })

if __name__=="__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        ssl_keyfile="key.pem",
        ssl_certfile="cert.pem"
    )
EOF

python app.py
