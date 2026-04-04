# Gesture Control System + AI Chatbot

A single-port Flask application that combines **real-time hand gesture recognition** with an **embedded AI chatbot**. Control your browser with hand movements captured by your webcam, and chat with an AI assistant— all from one page.


## Features

- 🤚 **Hand gesture detection** using MediaPipe — no clicks needed
- 📷 **Live camera feed** with landmark overlay streamed via MJPEG
- 🌐 **Open websites** (Facebook, Instagram, WhatsApp, Telegram) with gestures
- 💬 **Built-in AI chatbot** widget with real-time streaming responses
- 🗂️ **Chat history** — all conversations saved and reloadable
- ⚡ **Single port (5000)** — one server, one command


## Gesture Map

-------------------------------------------
| Fingers | Action                        |
|---------|-------------------------------|
|  Fist   | Open / close the gesture menu |
|  One    | Open **Facebook**             |
|  Two    | Open **Instagram**            |
|  Three  | Open **WhatsApp**             |
|  Four   | Open **Telegram**             |
|  Five   | Open the **Chat Bot** widget  |
-------------------------------------------


## ⚙️ Tech Stack

--------------------------------------------------------------------
| Layer           |                Technology                      |
|-----------------|------------------------------------------------|
| Backend         | Python 3.10, Flask, Flask-SocketIO             |
| Computer Vision | OpenCV, MediaPipe HandLandmarker               |
| AI / LLM        | OpenRouter API (`stepfun/step-3.5-flash:free`) |
| Streaming       | Server-Sent Events (SSE) + `queue.Queue`       |
| Real-time UI    | Socket.IO                                      |
| Frontend        | HTML5, CSS3, Vanilla JavaScript                |
| Storage         | JSON files (one per chat)                      |
--------------------------------------------------------------------


## 🔌 API Routes

----------------------------------------------------------------------------------
| Method | Route                  |              Description                     |
|--------|------------------------|----------------------------------------------|
| `GET`  | `/`                    | Main page                                    |
| `GET`  | `/video_feed`          | MJPEG camera stream                          |
| `GET`  | `/api/chats`           | List all saved chats                         |
| `POST` | `/api/chats/<id>/send` | Send a message (SSE stream)                  |
----------------------------------------------------------------------------------



-------------------------------------------- **Yasser** — **THE GOAT** ---------------------------------------------
