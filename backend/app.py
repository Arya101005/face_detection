import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from pymongo import MongoClient
from datetime import datetime
import base64
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
CAPTURED_FACES = "captured_faces"
MODELS_FOLDER = "models"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CAPTURED_FACES, exist_ok=True)

model = None
tracker = None
person_counter = 0
current_count = 0

mongo_client = None
persons_collection = None

mtcnn = None
resnet = None

person_embeddings_cache = {}
person_db_lookup = {}


def init_mongodb():
    global mongo_client, persons_collection
    try:
        mongo_client = MongoClient(
            "mongodb://localhost:27017/", serverSelectionTimeoutMS=5000
        )
        mongo_client.admin.command("ping")
        db = mongo_client["face_detection"]
        persons_collection = db["persons"]
        print("MongoDB connected successfully!")
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        mongo_client = None
        persons_collection = None


def load_model():
    global model, tracker, mtcnn, resnet
    model_path = os.path.join(MODELS_FOLDER, "yolov8n-face.pt")
    if os.path.exists(model_path):
        model = YOLO(model_path)
    else:
        model = YOLO("yolov8n-face.pt")
    tracker = DeepSort(max_age=60, n_init=3, nms_max_overlap=0.3)

    global mtcnn, resnet, person_embeddings_cache, person_db_lookup
    mtcnn = MTCNN(image_size=160, margin=0, device="cpu")
    resnet = InceptionResnetV1(pretrained="vggface2").eval()
    person_embeddings_cache = {}
    person_db_lookup = {}


def extract_face_embedding(face_crop):
    if face_crop.size == 0 or face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
        return None
    try:
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, (160, 160))
        face_tensor = torch.FloatTensor(face_rgb).permute(2, 0, 1).unsqueeze(0) / 255.0
        with torch.no_grad():
            embedding = resnet(face_tensor).numpy().flatten()
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        return None


def compare_embeddings(embedding1, embedding2, threshold=0.6):
    if embedding1 is None or embedding2 is None:
        return False
    similarity = np.dot(embedding1, embedding2)
    return similarity > threshold


def load_persons_to_cache():
    global person_embeddings_cache, person_db_lookup
    person_embeddings_cache = {}
    person_db_lookup = {}
    if persons_collection is not None:
        try:
            for person in persons_collection.find():
                if person.get("embedding") is not None:
                    name = person["name"]
                    person_embeddings_cache[name] = np.array(person["embedding"])
                    person_db_lookup[name] = person["_id"]
        except:
            pass


def find_matching_person(embedding):
    if embedding is None:
        return None

    best_match = None
    best_similarity = 0

    for name, stored_emb in person_embeddings_cache.items():
        similarity = np.dot(embedding, stored_emb)
        if similarity > 0.6 and similarity > best_similarity:
            best_similarity = similarity
            best_match = {
                "name": name,
                "embedding": stored_emb,
                "_id": person_db_lookup.get(name),
            }

    return best_match


def save_person(embedding, image_filename, person_name):
    if persons_collection is None:
        return None

    person_doc = {
        "name": person_name,
        "embedding": embedding.tolist() if embedding is not None else None,
        "image": image_filename,
        "created_at": datetime.now(),
        "detection_count": 1,
    }
    result = persons_collection.insert_one(person_doc)
    if embedding is not None:
        person_embeddings_cache[person_name] = embedding
        person_db_lookup[person_name] = result.inserted_id
    return result.inserted_id


def update_person_detection(person_id):
    if persons_collection is None:
        return
    try:
        persons_collection.update_one(
            {"_id": person_id}, {"$inc": {"detection_count": 1}}
        )
    except:
        pass


def is_likely_face(x1, y1, x2, y2, frame):
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return False
    if width < 30 or height < 30:
        return False
    return True


@app.route("/upload", methods=["POST"])
def upload_video():
    global person_counter, current_count, person_embeddings_cache, person_db_lookup
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    person_counter = 0
    current_count = 0
    person_embeddings_cache = {}
    person_db_lookup = {}

    video = request.files["video"]
    filename = video.filename
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    video.save(video_path)

    return jsonify(
        {
            "message": "Video uploaded successfully",
            "path": video_path,
            "filename": filename,
        }
    )


@app.route("/detect", methods=["GET"])
def detect_faces():
    filename = request.args.get("filename", "")
    global person_counter, current_count

    load_persons_to_cache()

    if persons_collection is not None:
        try:
            persons = list(persons_collection.find({}, {"name": 1}))
            max_num = 0
            for p in persons:
                try:
                    num = int(p["name"].split()[-1])
                    max_num = max(max_num, num)
                except:
                    pass
            person_counter = max_num
            current_count = max_num
        except:
            pass

    if model is None:
        load_model()

    video_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(video_path):
        return jsonify({"error": "Video not found"}), 404

    cap = cv2.VideoCapture(video_path)
    results_data = []
    frame_id = 0
    seen_face_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        results = model(frame, verbose=False)
        tracks_input = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                if confidence > 0.3:
                    tracks_input.append(
                        ([x1, y1, x2 - x1, y2 - y1], confidence, "face")
                    )

        if tracker is not None:
            tracks = tracker.update_tracks(tracks_input, frame=frame)

            frame_faces = []
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id

                if track_id in seen_face_ids:
                    continue
                seen_face_ids.add(track_id)

                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                bbox = [x1, y1, x2, y2]

                face_crop = frame[y1:y2, x1:x2]
                embedding = extract_face_embedding(face_crop)

                matched_person = find_matching_person(embedding)

                if matched_person:
                    person_name = matched_person["name"]
                    person_id = matched_person["_id"]
                    update_person_detection(person_id)
                    person_counter = max(
                        person_counter,
                        int(matched_person["name"].split()[-1])
                        if matched_person["name"]
                        else 0,
                    )
                else:
                    person_counter += 1
                    person_name = f"Person {person_counter}"
                    if embedding is not None:
                        save_person(
                            embedding,
                            f"Person_{person_counter}.jpg",
                            person_name,
                        )
                        if face_crop.size > 0:
                            face_filename = os.path.join(
                                CAPTURED_FACES, f"Person_{person_counter}.jpg"
                            )
                            cv2.imwrite(face_filename, face_crop)

                frame_faces.append({"id": track_id, "name": person_name, "bbox": bbox})
        else:
            frame_faces = []

        current_count = person_counter
        results_data.append(
            {"frame": frame_id, "count": current_count, "faces": frame_faces}
        )
        frame_id += 1

    cap.release()

    return jsonify(
        {
            "total_count": current_count,
            "frames": results_data,
            "persons": list(
                persons_collection.find({}, {"_id": 0, "name": 1, "image": 1}).limit(50)
            )
            if persons_collection
            else [],
        }
    )


def generate_video_feed(video_path, is_upload=False):
    global person_counter, current_count, model, tracker

    load_persons_to_cache()

    if not is_upload and persons_collection is not None:
        try:
            persons = list(persons_collection.find({}, {"name": 1}))
            max_num = 0
            for p in persons:
                try:
                    num = int(p["name"].split()[-1])
                    max_num = max(max_num, num)
                except:
                    pass
            person_counter = max_num
            current_count = max_num
        except:
            pass

    if model is None:
        load_model()

    if is_upload:
        person_counter = 0
        current_count = 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    seen_face_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        results = model(frame, verbose=False)
        tracks_input = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                if confidence > 0.3:
                    tracks_input.append(
                        ([x1, y1, x2 - x1, y2 - y1], confidence, "face")
                    )

        if tracker is not None:
            tracks = tracker.update_tracks(tracks_input, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id

                if track_id in seen_face_ids:
                    continue
                seen_face_ids.add(track_id)

                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)

                face_crop = frame[y1:y2, x1:x2]
                embedding = extract_face_embedding(face_crop)

                matched_person = find_matching_person(embedding)

                if matched_person:
                    person_name = matched_person["name"]
                    update_person_detection(matched_person["_id"])
                    try:
                        person_counter = max(
                            person_counter, int(matched_person["name"].split()[-1])
                        )
                    except:
                        pass
                else:
                    person_counter += 1
                    person_name = f"Person {person_counter}"
                    if embedding is not None:
                        save_person(
                            embedding,
                            f"Person_{person_counter}.jpg",
                            person_name,
                        )
                        if face_crop.size > 0:
                            face_filename = os.path.join(
                                CAPTURED_FACES, f"Person_{person_counter}.jpg"
                            )
                            cv2.imwrite(face_filename, face_crop)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.rectangle(frame, (x1, y1 - 35), (x2, y1), (0, 255, 0), -1)
                cv2.putText(
                    frame,
                    person_name,
                    (x1 + 5, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    2,
                )

        current_count = person_counter

        cv2.rectangle(frame, (10, 10), (230, 50), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"Unique: {current_count}",
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            break
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()


def generate_camera_feed():
    global person_counter, current_count, model, tracker

    load_persons_to_cache()

    if persons_collection is not None:
        try:
            persons = list(persons_collection.find({}, {"name": 1}))
            max_num = 0
            for p in persons:
                try:
                    num = int(p["name"].split()[-1])
                    max_num = max(max_num, num)
                except:
                    pass
            person_counter = max_num
            current_count = max_num
        except:
            pass

    if model is None:
        load_model()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    seen_face_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        results = model(frame, verbose=False)
        tracks_input = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                if confidence > 0.3:
                    tracks_input.append(
                        ([x1, y1, x2 - x1, y2 - y1], confidence, "face")
                    )

        if tracker is not None:
            tracks = tracker.update_tracks(tracks_input, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id

                if track_id in seen_face_ids:
                    continue
                seen_face_ids.add(track_id)

                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)

                face_crop = frame[y1:y2, x1:x2]
                embedding = extract_face_embedding(face_crop)

                matched_person = find_matching_person(embedding)

                if matched_person:
                    person_name = matched_person["name"]
                    update_person_detection(matched_person["_id"])
                    try:
                        person_counter = max(
                            person_counter, int(matched_person["name"].split()[-1])
                        )
                    except:
                        pass
                else:
                    person_counter += 1
                    person_name = f"Person {person_counter}"
                    if embedding is not None:
                        save_person(
                            embedding,
                            f"Person_{person_counter}.jpg",
                            person_name,
                        )

                    if face_crop.size > 0:
                        face_filename = os.path.join(
                            CAPTURED_FACES, f"Person_{person_counter}.jpg"
                        )
                        cv2.imwrite(face_filename, face_crop)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.rectangle(frame, (x1, y1 - 35), (x2, y1), (0, 255, 0), -1)
                cv2.putText(
                    frame,
                    person_name,
                    (x1 + 5, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    2,
                )

        current_count = person_counter

        cv2.rectangle(frame, (10, 10), (230, 50), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"Unique: {current_count}",
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


@app.route("/camera", methods=["GET"])
def camera_feed():
    return Response(
        generate_camera_feed(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/count", methods=["GET"])
def get_count():
    global model
    if model is None:
        try:
            load_model()
        except Exception as e:
            return jsonify({"count": 0, "error": str(e)})
    return jsonify({"count": current_count})


@app.route("/faces", methods=["GET"])
def get_captured_faces():
    faces = []
    for filename in os.listdir(CAPTURED_FACES):
        if filename.endswith(".jpg"):
            faces.append(filename)
    return jsonify({"faces": faces})


@app.route("/persons", methods=["GET"])
def get_persons():
    if persons_collection is None:
        return jsonify({"persons": []})
    persons = list(
        persons_collection.find(
            {}, {"_id": 0, "name": 1, "image": 1, "detection_count": 1}
        )
    )
    return jsonify({"persons": persons})


@app.route("/captured_faces/<path:filename>", methods=["GET"])
def serve_face(filename):
    return send_from_directory(CAPTURED_FACES, filename)


@app.route("/reset", methods=["POST"])
def reset():
    global person_counter, current_count, person_embeddings_cache, person_db_lookup
    person_counter = 0
    current_count = 0
    person_embeddings_cache = {}
    person_db_lookup = {}
    for filename in os.listdir(CAPTURED_FACES):
        if filename.endswith(".jpg"):
            os.remove(os.path.join(CAPTURED_FACES, filename))
    if persons_collection is not None:
        try:
            persons_collection.delete_many({})
        except:
            pass
    return jsonify({"message": "Reset successful"})


@app.route("/video_feed", methods=["GET"])
def video_feed():
    filename = request.args.get("filename", "")
    mode = request.args.get("mode", "upload")

    if not filename:
        return jsonify({"error": "No filename provided"}), 400
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(video_path):
        return jsonify({"error": "Video not found"}), 404

    if model is None:
        load_model()

    is_upload = mode == "upload"

    return Response(
        generate_video_feed(video_path, is_upload),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    init_mongodb()
    load_model()
    app.run(debug=True, port=5000)
