# import cv2 as cv
# import mediapipe as mp
# import time
# import threading
# import json
# import os
# import uuid
# from datetime import datetime
# from flask import Flask, render_template, Response, request, jsonify, stream_with_context
# from flask_socketio import SocketIO, emit
# from openai import OpenAI
# from dotenv import load_dotenv

# load_dotenv()

# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'gesture_control_2025'
# socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# client = OpenAI(
#     api_key=os.getenv("OPENROUTER_API_KEY"),
#     base_url="https://openrouter.ai/api/v1"
# )

# CHATS_DIR = "chats"
# os.makedirs(CHATS_DIR, exist_ok=True)

# BaseOptions           = mp.tasks.BaseOptions
# HandLandmarker        = mp.tasks.vision.HandLandmarker
# HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
# VisionRunningMode     = mp.tasks.vision.RunningMode

# wCam, hCam = 1280, 720
# cap = cv.VideoCapture(0)
# cap.set(3, wCam)
# cap.set(4, hCam)

# HAND_CONNECTIONS = [
#     (0,1),(1,2),(2,3),(3,4),
#     (0,5),(5,6),(6,7),(7,8),
#     (0,9),(9,10),(10,11),(11,12),
#     (0,13),(13,14),(14,15),(15,16),
#     (0,17),(17,18),(18,19),(19,20),
#     (5,9),(9,13),(13,17)
# ]
# tip_ids = [4, 8, 12, 16, 20]

# GESTURE_MAP = {
#     'one':   ('one',   '1', 'Facebook',  'https://facebook.com'),
#     'two':   ('two',   '2', 'Instagram', 'https://instagram.com'),
#     'three': ('three', '3', 'WhatsApp',  'https://web.whatsapp.com'),
#     'four':  ('four',  '4', 'Telegram',  'https://web.telegram.org'),
#     'five':  ('five',  '5', 'Chat Bot',  'https://www.google.com/'), 
# }

# state = {
#     'menu_open':      False,
#     'last_emit_time': 0,
# }
# COOLDOWN  = 2.0
# HOLD_TIME = 1.2

# gesture_hold = {'name': None, 'start': 0.0}

# output_frame = None
# frame_lock   = threading.Lock()


# def list_chats():
#     chats = []
#     for fname in sorted(os.listdir(CHATS_DIR), reverse=True):
#         if fname.endswith(".json"):
#             path = os.path.join(CHATS_DIR, fname)
#             with open(path) as f:
#                 data = json.load(f)
#             chats.append({
#                 "id":         data.get("id"),
#                 "title":      data.get("title", "New Chat"),
#                 "created_at": data.get("created_at", ""),
#             })
#     return chats


# def load_chat(chat_id):
#     path = os.path.join(CHATS_DIR, f"{chat_id}.json")
#     if not os.path.exists(path):
#         return None
#     with open(path) as f:
#         return json.load(f)


# def save_chat(chat):
#     path = os.path.join(CHATS_DIR, f"{chat['id']}.json")
#     with open(path, "w", encoding='utf-8') as f:
#         json.dump(chat, f, indent=2, ensure_ascii=False)


# def new_chat():
#     chat_id = str(uuid.uuid4())[:8]
#     chat = {
#         "id":         chat_id,
#         "title":      "New Chat",
#         "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
#         "messages":   []
#     }
#     save_chat(chat)
#     return chat


# def derive_title(messages):
#     for m in messages:
#         if m["role"] == "user":
#             return m["content"][:40] + ("…" if len(m["content"]) > 40 else "")
#     return "New Chat"


# def get_fingers_up(landmarks):
#     fingers = []
#     fingers.append(1 if landmarks[tip_ids[0]].x < landmarks[tip_ids[0]-1].x else 0)
#     for i in range(1, 5):
#         fingers.append(1 if landmarks[tip_ids[i]].y < landmarks[tip_ids[i]-2].y else 0)
#     return fingers

# def is_fist(landmarks):
#     return get_fingers_up(landmarks) == [0, 0, 0, 0, 0]

# def classify_gesture(landmarks):
#     if is_fist(landmarks):
#         return 'fist'
#     f = get_fingers_up(landmarks)
#     patterns = {
#         'one':   [0,1,0,0,0],
#         'two':   [0,1,1,0,0],
#         'three': [0,1,1,1,0],
#         'four':  [0,1,1,1,1],
#         'five':  [1,1,1,1,1],
#     }
#     for name, pat in patterns.items():
#         if f == pat:
#             return name
#     return None

# def draw_landmarks(frame, hand_landmarks):
#     h, w, _ = frame.shape
#     for s, e in HAND_CONNECTIONS:
#         x1 = int(hand_landmarks[s].x * w); y1 = int(hand_landmarks[s].y * h)
#         x2 = int(hand_landmarks[e].x * w); y2 = int(hand_landmarks[e].y * h)
#         cv.line(frame, (x1,y1), (x2,y2), (0, 200, 255), 2)
#     for lm in hand_landmarks:
#         cx, cy = int(lm.x * w), int(lm.y * h)
#         cv.circle(frame, (cx,cy), 5, (0, 255, 120), cv.FILLED)
#         cv.circle(frame, (cx,cy), 7, (255, 255, 255), 1)

# def draw_hud(frame, gesture_name, hold_progress, menu_open):
#     h, w, _ = frame.shape
#     cv.rectangle(frame, (0,0), (w,75), (0,0,0), -1)
#     cv.rectangle(frame, (0,73), (w,75), (0,150,60), -1)
#     cv.putText(frame, 'CONTROL SYSTEM', (200,50),
#                cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,210,80), 2)
#     if menu_open:
#         stxt, scol = 'MENU OPEN', (0,220,80)
#     elif gesture_name == 'fist':
#         stxt, scol = 'FIST DETECTED', (0,180,255)
#     elif gesture_name:
#         stxt, scol = f'GESTURE: {gesture_name.upper()}', (0,200,120)
#     else:
#         stxt, scol = 'SHOW YOUR HAND...', (90,90,90)
#     cv.putText(frame, stxt, (w-560, 50),
#                cv.FONT_HERSHEY_SIMPLEX, 0.72, scol, 2)
#     if hold_progress > 0:
#         cv.rectangle(frame, (0,h-10), (w,h), (15,15,15), -1)
#         cv.rectangle(frame, (0,h-10), (int(w*hold_progress),h), (0,200,70), -1)


# def camera_loop():
#     global output_frame
#     options = HandLandmarkerOptions(
#         base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
#         running_mode=VisionRunningMode.VIDEO,
#         num_hands=1,
#         min_hand_detection_confidence=0.5,
#         min_hand_presence_confidence=0.5,
#         min_tracking_confidence=0.5,
#     )
#     pTime = 0
#     with HandLandmarker.create_from_options(options) as landmarker:
#         while cap.isOpened():
#             ok, frame = cap.read()
#             if not ok:
#                 time.sleep(0.01)
#                 continue

#             frame  = cv.flip(frame, 1)
#             rgb    = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
#             mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
#             ts_ms  = int(time.time() * 1000)
#             result = landmarker.detect_for_video(mp_img, ts_ms)

#             gesture_name  = None
#             hold_progress = 0.0
#             now           = time.time()

#             if result.hand_landmarks:
#                 for hand_lm in result.hand_landmarks:
#                     draw_landmarks(frame, hand_lm)
#                     gesture_name = classify_gesture(hand_lm)

#             if gesture_name:
#                 if gesture_hold['name'] != gesture_name:
#                     gesture_hold['name']  = gesture_name
#                     gesture_hold['start'] = now
#                 else:
#                     elapsed       = now - gesture_hold['start']
#                     hold_progress = min(elapsed / HOLD_TIME, 1.0)

#                     if hold_progress >= 1.0 and (now - state['last_emit_time']) > COOLDOWN:
#                         if gesture_name == 'fist':
#                             state['menu_open'] = not state['menu_open']
#                             socketio.emit('menu_toggle', {'open': state['menu_open']})
#                         elif state['menu_open'] and gesture_name in GESTURE_MAP:
#                             gid, key, label, url = GESTURE_MAP[gesture_name]
#                             socketio.emit('gesture_action', {
#                                 'gesture': gid, 'key': key,
#                                 'label': label, 'url': url,
#                             })
#                         state['last_emit_time'] = now
#                         gesture_hold['name']  = None
#                         gesture_hold['start'] = 0.0
#             else:
#                 gesture_hold['name']  = None
#                 gesture_hold['start'] = 0.0

#             # FPS
#             cTime = time.time()
#             fps   = 1 / (cTime - pTime) if cTime != pTime else 0
#             pTime = cTime
#             cv.putText(frame, f'FPS:{int(fps)}', (wCam-110, 65),
#                        cv.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,100), 2)

#             draw_hud(frame, gesture_name, hold_progress, state['menu_open'])

#             _, buf = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 75])
#             with frame_lock:
#                 output_frame = buf.tobytes()

#             time.sleep(0.01)


# def generate_frames():
#     global output_frame
#     while True:
#         with frame_lock:
#             if output_frame is None:
#                 time.sleep(0.05)
#                 continue
#             data = output_frame
#         yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + data + b'\r\n')
#         time.sleep(0.033)


# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')


# @app.route('/api/chats', methods=['GET'])
# def get_chats():
#     return jsonify(list_chats())

# @app.route('/api/chats/new', methods=['POST'])
# def create_chat():
#     return jsonify(new_chat())

# @app.route('/api/chats/<chat_id>', methods=['GET'])
# def get_chat(chat_id):
#     chat = load_chat(chat_id)
#     if not chat:
#         return jsonify({'error': 'Not found'}), 404
#     return jsonify(chat)

# @app.route('/api/chats/<chat_id>', methods=['DELETE'])
# def delete_chat(chat_id):
#     path = os.path.join(CHATS_DIR, f"{chat_id}.json")
#     if os.path.exists(path):
#         os.remove(path)
#     return jsonify({'ok': True})

# @app.route('/api/chats/<chat_id>/send', methods=['POST'])
# def send_message(chat_id):
#     chat = load_chat(chat_id)
#     if not chat:
#         return jsonify({'error': 'Chat not found'}), 404

#     user_text = request.json.get('message', '').strip()
#     if not user_text:
#         return jsonify({'error': 'Empty message'}), 400

#     chat['messages'].append({'role': 'user', 'content': user_text})
#     if len(chat['messages']) == 1:
#         chat['title'] = derive_title(chat['messages'])

#     system_message = {'role': 'system', 'content': 'Always answer in English. Be helpful and concise.'}
#     full_context   = [system_message] + chat['messages']

#     def generate():
#         full_response = ''
#         try:
#             response = client.chat.completions.create(
#                 model='stepfun/step-3.5-flash:free',
#                 messages=full_context,
#                 temperature=1.0,
#                 stream=True
#             )
#             for chunk in response:
#                 if chunk.choices[0].delta.content:
#                     token = chunk.choices[0].delta.content
#                     full_response += token
#                     yield f"data: {json.dumps({'token': token})}\n\n"
#         except Exception as e:
#             yield f"data: {json.dumps({'error': str(e)})}\n\n"

#         chat['messages'].append({'role': 'assistant', 'content': full_response})
#         save_chat(chat)
#         yield f"data: {json.dumps({'done': True, 'title': chat['title']})}\n\n"

#     return Response(stream_with_context(generate()), content_type='text/event-stream')


# @socketio.on('connect')
# def on_connect():
#     print('[+] Browser connected')
#     emit('menu_toggle', {'open': state['menu_open']})

# @socketio.on('debug_pinch')
# def on_debug_pinch():
#     state['menu_open'] = not state['menu_open']
#     emit('menu_toggle', {'open': state['menu_open']}, broadcast=True)

# @socketio.on('debug_gesture')
# def on_debug_gesture(data):
#     if state['menu_open']:
#         emit('gesture_action', data, broadcast=True)

# if __name__ == '__main__':
#     t = threading.Thread(target=camera_loop, daemon=True)
#     t.start()
#     print('\n🟢  Running  →  http://127.0.0.1:5000\n')
#     socketio.run(app, host='0.0.0.0', port=5000, debug=False)

import cv2 as cv
import mediapipe as mp
import time
import threading
import json
import os
import uuid
from datetime import datetime

from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO, emit
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Flask & SocketIO ─────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'gesture_control_2025'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ── OpenRouter client ────────────────────────────────────────────
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# ── Chat storage ─────────────────────────────────────────────────
CHATS_DIR = "chats"
os.makedirs(CHATS_DIR, exist_ok=True)

# ── MediaPipe setup ──────────────────────────────────────────────
BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

# ── Camera ───────────────────────────────────────────────────────
wCam, hCam = 1280, 720
cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# ── Hand connections ─────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]
tip_ids = [4, 8, 12, 16, 20]

# ── Gesture → action mapping ─────────────────────────────────────
# 'five' now opens the built-in chat widget (no external URL needed)
GESTURE_MAP = {
    'one':   ('one',   '1', 'Facebook',  'https://facebook.com'),
    'two':   ('two',   '2', 'Instagram', 'https://instagram.com'),
    'three': ('three', '3', 'WhatsApp',  'https://web.whatsapp.com'),
    'four':  ('four',  '4', 'Telegram',  'https://web.telegram.org'),
    'five':  ('five',  '5', 'Chat Bot',  '__chat__'),  # special: open widget
}

# ── CV State ─────────────────────────────────────────────────────
state = {
    'menu_open':      False,
    'last_emit_time': 0,
}
COOLDOWN  = 2.0
HOLD_TIME = 1.2

gesture_hold = {'name': None, 'start': 0.0}

# ── Shared MJPEG frame ───────────────────────────────────────────
output_frame = None
frame_lock   = threading.Lock()


# ════════════════════════════════════════════════════════════════
#  CHAT HELPERS
# ════════════════════════════════════════════════════════════════

def list_chats():
    chats = []
    for fname in sorted(os.listdir(CHATS_DIR), reverse=True):
        if fname.endswith(".json"):
            path = os.path.join(CHATS_DIR, fname)
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            chats.append({
                "id":         data.get("id"),
                "title":      data.get("title", "New Chat"),
                "created_at": data.get("created_at", ""),
            })
    return chats


def load_chat(chat_id):
    path = os.path.join(CHATS_DIR, f"{chat_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def save_chat(chat):
    path = os.path.join(CHATS_DIR, f"{chat['id']}.json")
    with open(path, "w", encoding='utf-8') as f:
        json.dump(chat, f, indent=2, ensure_ascii=False)


def new_chat():
    chat_id = str(uuid.uuid4())[:8]
    chat = {
        "id":         chat_id,
        "title":      "New Chat",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "messages":   []
    }
    save_chat(chat)
    return chat


def derive_title(messages):
    for m in messages:
        if m["role"] == "user":
            return m["content"][:40] + ("…" if len(m["content"]) > 40 else "")
    return "New Chat"


# ════════════════════════════════════════════════════════════════
#  GESTURE / CV HELPERS
# ════════════════════════════════════════════════════════════════

def get_fingers_up(landmarks):
    fingers = []
    fingers.append(1 if landmarks[tip_ids[0]].x < landmarks[tip_ids[0]-1].x else 0)
    for i in range(1, 5):
        fingers.append(1 if landmarks[tip_ids[i]].y < landmarks[tip_ids[i]-2].y else 0)
    return fingers

def is_fist(landmarks):
    return get_fingers_up(landmarks) == [0, 0, 0, 0, 0]

def classify_gesture(landmarks):
    if is_fist(landmarks):
        return 'fist'
    f = get_fingers_up(landmarks)
    patterns = {
        'one':   [0,1,0,0,0],
        'two':   [0,1,1,0,0],
        'three': [0,1,1,1,0],
        'four':  [0,1,1,1,1],
        'five':  [1,1,1,1,1],
    }
    for name, pat in patterns.items():
        if f == pat:
            return name
    return None

def draw_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape
    for s, e in HAND_CONNECTIONS:
        x1 = int(hand_landmarks[s].x * w); y1 = int(hand_landmarks[s].y * h)
        x2 = int(hand_landmarks[e].x * w); y2 = int(hand_landmarks[e].y * h)
        cv.line(frame, (x1,y1), (x2,y2), (0, 200, 255), 2)
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv.circle(frame, (cx,cy), 5, (0, 255, 120), cv.FILLED)
        cv.circle(frame, (cx,cy), 7, (255, 255, 255), 1)

def draw_hud(frame, gesture_name, hold_progress, menu_open):
    h, w, _ = frame.shape
    cv.rectangle(frame, (0,0), (w,75), (0,0,0), -1)
    cv.rectangle(frame, (0,73), (w,75), (0,150,60), -1)
    cv.putText(frame, 'CONTROL SYSTEM', (200,50),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,210,80), 2)
    if menu_open:
        stxt, scol = 'MENU OPEN', (0,220,80)
    elif gesture_name == 'fist':
        stxt, scol = 'FIST DETECTED', (0,180,255)
    elif gesture_name:
        stxt, scol = f'GESTURE: {gesture_name.upper()}', (0,200,120)
    else:
        stxt, scol = 'SHOW YOUR HAND...', (90,90,90)
    cv.putText(frame, stxt, (w-560, 50),
               cv.FONT_HERSHEY_SIMPLEX, 0.72, scol, 2)
    if hold_progress > 0:
        cv.rectangle(frame, (0,h-10), (w,h), (15,15,15), -1)
        cv.rectangle(frame, (0,h-10), (int(w*hold_progress),h), (0,200,70), -1)


# ════════════════════════════════════════════════════════════════
#  CAMERA LOOP (background thread)
# ════════════════════════════════════════════════════════════════

def camera_loop():
    global output_frame
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pTime = 0
    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            frame  = cv.flip(frame, 1)
            rgb    = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms  = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_img, ts_ms)

            gesture_name  = None
            hold_progress = 0.0
            now           = time.time()

            if result.hand_landmarks:
                for hand_lm in result.hand_landmarks:
                    draw_landmarks(frame, hand_lm)
                    gesture_name = classify_gesture(hand_lm)

            # ── Hold-to-fire ──────────────────────────────────────
            if gesture_name:
                if gesture_hold['name'] != gesture_name:
                    gesture_hold['name']  = gesture_name
                    gesture_hold['start'] = now
                else:
                    elapsed       = now - gesture_hold['start']
                    hold_progress = min(elapsed / HOLD_TIME, 1.0)

                    if hold_progress >= 1.0 and (now - state['last_emit_time']) > COOLDOWN:
                        if gesture_name == 'fist':
                            state['menu_open'] = not state['menu_open']
                            socketio.emit('menu_toggle', {'open': state['menu_open']})
                        elif state['menu_open'] and gesture_name in GESTURE_MAP:
                            gid, key, label, url = GESTURE_MAP[gesture_name]
                            socketio.emit('gesture_action', {
                                'gesture': gid, 'key': key,
                                'label': label, 'url': url,
                            })
                        state['last_emit_time'] = now
                        gesture_hold['name']  = None
                        gesture_hold['start'] = 0.0
            else:
                gesture_hold['name']  = None
                gesture_hold['start'] = 0.0

            # FPS
            cTime = time.time()
            fps   = 1 / (cTime - pTime) if cTime != pTime else 0
            pTime = cTime
            cv.putText(frame, f'FPS:{int(fps)}', (wCam-110, 65),
                       cv.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,100), 2)

            draw_hud(frame, gesture_name, hold_progress, state['menu_open'])

            _, buf = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 75])
            with frame_lock:
                output_frame = buf.tobytes()

            time.sleep(0.01)


def generate_frames():
    global output_frame
    while True:
        with frame_lock:
            if output_frame is None:
                time.sleep(0.05)
                continue
            data = output_frame
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + data + b'\r\n')
        time.sleep(0.033)


# ════════════════════════════════════════════════════════════════
#  ROUTES — CV
# ════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ════════════════════════════════════════════════════════════════
#  ROUTES — CHAT API
# ════════════════════════════════════════════════════════════════

@app.route('/api/chats', methods=['GET'])
def get_chats():
    return jsonify(list_chats())

@app.route('/api/chats/new', methods=['POST'])
def create_chat():
    return jsonify(new_chat())

@app.route('/api/chats/<chat_id>', methods=['GET'])
def get_chat(chat_id):
    chat = load_chat(chat_id)
    if not chat:
        return jsonify({'error': 'Not found'}), 404
    return jsonify(chat)

@app.route('/api/chats/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    path = os.path.join(CHATS_DIR, f"{chat_id}.json")
    if os.path.exists(path):
        os.remove(path)
    return jsonify({'ok': True})

@app.route('/api/chats/<chat_id>/send', methods=['POST'])
def send_message(chat_id):
    import queue as queue_module

    chat = load_chat(chat_id)
    if not chat:
        return jsonify({'error': 'Chat not found'}), 404

    user_text = request.json.get('message', '').strip()
    if not user_text:
        return jsonify({'error': 'Empty message'}), 400

    chat['messages'].append({'role': 'user', 'content': user_text})
    if len(chat['messages']) == 1:
        chat['title'] = derive_title(chat['messages'])

    system_message = {'role': 'system', 'content': 'Always answer in English. Be helpful and concise.'}
    full_context   = [system_message] + chat['messages']

    # Use a queue so the background thread feeds the stream safely
    q = queue_module.Queue()

    def llm_worker():
        full_response = ''
        try:
            response = client.chat.completions.create(
                model='stepfun/step-3.5-flash:free',
                messages=full_context,
                temperature=1.0,
                stream=True
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_response += token
                    q.put(f"data: {json.dumps({'token': token})}\n\n")
        except Exception as e:
            q.put(f"data: {json.dumps({'error': str(e)})}\n\n")

        chat['messages'].append({'role': 'assistant', 'content': full_response})
        save_chat(chat)
        q.put(f"data: {json.dumps({'done': True, 'title': chat['title']})}\n\n")
        q.put(None)  # sentinel — stream finished

    worker = threading.Thread(target=llm_worker, daemon=True)
    worker.start()

    def generate():
        while True:
            item = q.get()
            if item is None:
                break
            yield item

    resp = Response(generate(), content_type='text/event-stream')
    resp.headers['X-Accel-Buffering'] = 'no'
    resp.headers['Cache-Control'] = 'no-cache'
    return resp


# ════════════════════════════════════════════════════════════════
#  SOCKET.IO EVENTS
# ════════════════════════════════════════════════════════════════

@socketio.on('connect')
def on_connect():
    print('[+] Browser connected')
    emit('menu_toggle', {'open': state['menu_open']})

@socketio.on('debug_pinch')
def on_debug_pinch():
    state['menu_open'] = not state['menu_open']
    emit('menu_toggle', {'open': state['menu_open']}, broadcast=True)

@socketio.on('debug_gesture')
def on_debug_gesture(data):
    if state['menu_open']:
        emit('gesture_action', data, broadcast=True)


# ════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()
    print('\n🟢  Running  →  http://127.0.0.1:5000\n')
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)

# cd C:\Users\abdel\OneDrive\Desktop\python_project_oop
# python app.py