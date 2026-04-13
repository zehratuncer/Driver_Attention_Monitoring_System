import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np


# -----------------------------
# Landmark indeksleri (yüz mesh noktaları)
# -----------------------------

# EAR (Eye Aspect Ratio) hesabı için her gözden 6 nokta kullanıyoruz:
#   EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]


# (Burun ucu + sol/sağ yüz referansları + alın referansı) --> noktalar
NOSE_TIP_IDX = 1
LEFT_FACE_IDX = 234
RIGHT_FACE_IDX = 454
FOREHEAD_IDX = 10


# -----------------------------
# Veri yapıları
# -----------------------------
@dataclass
class DrowsinessState:
    eyes_closed: bool = False
    closed_since: float = 0.0
    drowsy: bool = False


def _euclidean(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


def calculate_EAR(eye_pts: np.ndarray) -> float:
    """
    6 göz landmark noktasından EAR (Eye Aspect Ratio) hesapla.

    eye_pts şekli: (6, 2) ve noktalar [p1, p2, p3, p4, p5, p6] sırasındadır.
    """
    p1, p2, p3, p4, p5, p6 = eye_pts
    vertical_1 = _euclidean(p2, p6)
    vertical_2 = _euclidean(p3, p5)
    horizontal = _euclidean(p1, p4)

    # Çok nadir de olsa yatay mesafe 0'a yakınsa bölme hatasını önledik.
    if horizontal < 1e-6:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


def detect_drowsiness(
    ear: float,
    state: DrowsinessState,
    ear_threshold: float,
    closed_duration_sec: float,
    now: float,
) -> DrowsinessState:
    """
    EAR eşiği ve gözün kapalı kalma süresine göre uyuklama durumunu güncelle.

    - EAR < threshold ise göz kapalı kabul edilir.
    - Gözler closed_duration_sec boyunca kesintisiz kapalı kalırsa drowsy=True olur.
    """
    eyes_closed = ear < ear_threshold

    if eyes_closed and not state.eyes_closed:
        # Geçiş: açık -> kapalı (zamanlayıcıyı başlat)
        state.closed_since = now
    elif not eyes_closed:
        # Göz açıldıysa süreyi ve drowsy durumunu sıfırla
        state.closed_since = 0.0
        state.drowsy = False

    state.eyes_closed = eyes_closed

    if state.eyes_closed and state.closed_since > 0.0:
        # Kapalı kalma süresi eşiği geçti mi?
        state.drowsy = (now - state.closed_since) >= closed_duration_sec

    return state


def estimate_head_position(
    landmarks_xy: np.ndarray,
    frame_w: int,
    frame_h: int,
    yaw_thresh: float = 0.12,
    down_thresh: float = 0.10,
) -> str:
    """
    Basit baş yönü etiketi tahmini: 'CENTER', 'LEFT', 'RIGHT', 'DOWN'.

    Kullanılan yöntem basit :
    - Burun konumunu sol/sağ yüz referanslarının orta noktasına göre kıyasla (yaw).
    - Burun konumunu alın referansına göre kıyasla (pitch/aşağı bakma).

    yaw_thresh ve down_thresh normalize eşiklerdir (görüntü uzayı [0,1]).
    """
    nose = landmarks_xy[NOSE_TIP_IDX]
    left_face = landmarks_xy[LEFT_FACE_IDX]
    right_face = landmarks_xy[RIGHT_FACE_IDX]
    forehead = landmarks_xy[FOREHEAD_IDX]

    # Çözünürlüğe bağımlılığı azaltmak için [0,1] normalize uzaya çevir.
    nose_n = np.array([nose[0] / frame_w, nose[1] / frame_h], dtype=np.float32)
    left_n = np.array([left_face[0] / frame_w, left_face[1] / frame_h], dtype=np.float32)
    right_n = np.array([right_face[0] / frame_w, right_face[1] / frame_h], dtype=np.float32)
    forehead_n = np.array([forehead[0] / frame_w, forehead[1] / frame_h], dtype=np.float32)

    mid_x = (left_n[0] + right_n[0]) / 2.0
    # Burun sağa kayarsa yaw pozitif olur (kameraya göre görüntü koordinatı).
    yaw = nose_n[0] - mid_x

    # "DOWN": burun, alın referansına göre belirgin şekilde aşağıdaysa.
    pitch_down = nose_n[1] - forehead_n[1]

    if pitch_down > down_thresh:
        return "DOWN"
    if yaw > yaw_thresh:
        return "RIGHT"
    if yaw < -yaw_thresh:
        return "LEFT"
    return "CENTER"


def compute_attention_score(
    ear: float,
    drowsy: bool,
    head_pos: str,
    ear_threshold: float,
) -> Tuple[int, str]:
    """
    Basit bir dikkat skoru (0-100) ve durum etiketi hesapla:
    - ATTENTIVE
    - DROWSY
    - DISTRACTED

    Kurallar:
    - DROWSY en öncelikli durumdur.
    - Drowsy değilse ve baş CENTER değilse -> DISTRACTED.
    - Diğer durumlarda -> ATTENTIVE.
    """
    score = 100

    # Göz temelli ceza (drowsy tetiklenmeden önce de küçük bir ceza uygular)
    if ear < ear_threshold:
        score -= 30

    # Baş yönüne göre cezalar
    if head_pos in {"LEFT", "RIGHT"}:
        score -= 25
    if head_pos == "DOWN":
        score -= 35

    if drowsy:
        score = min(score, 20)
        return max(0, score), "DROWSY"

    if head_pos != "CENTER":
        return max(0, score), "DISTRACTED"

    return max(0, score), "ATTENTIVE"


# -----------------------------
# MediaPipe yardımcıları
# -----------------------------
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def _ensure_face_landmarker_model(model_path: Path) -> Path:
    """
    Face Landmarker modelinin yerelde mevcut olmasını sağlar.

    MediaPipe "Tasks" API bir `.task` model dosyasına ihtiyaç duyar.
    MVP'nin kolay çalışması için dosya yoksa otomatik indiriyoruz.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        print(f"[INFO] Downloading Face Landmarker model to: {model_path}")
        urlretrieve(MODEL_URL, str(model_path))
    return model_path


def _extract_landmarks_xy_from_normalized(
    normalized_landmarks: List[object], frame_w: int, frame_h: int
) -> np.ndarray:
    """
    Normalize landmark'ları (x,y [0..1]) piksel koordinatlarına çevir.

    Dönüş: (N, 2) float32 dizi.
    """
    n = len(normalized_landmarks)
    pts = np.zeros((n, 2), dtype=np.float32)
    for i, lm in enumerate(normalized_landmarks):
        pts[i, 0] = float(lm.x) * frame_w
        pts[i, 1] = float(lm.y) * frame_h
    return pts


def _draw_face_landmarks_points(frame: np.ndarray, pts_xy: np.ndarray, step: int = 1) -> None:
    """Yüz landmark noktalarını küçük noktalar şeklinde çiz (hafif/hızlı)."""
    for i in range(0, pts_xy.shape[0], step):
        x, y = int(pts_xy[i, 0]), int(pts_xy[i, 1])
        cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)


def _draw_status_panel(
    frame: np.ndarray,
    label: str,
    score: int,
    ear: float,
    head_pos: str,
    drowsy: bool,
) -> None:
    """Sol üstte durum metnini ve uyarıları çiz."""
    color = (0, 255, 0)  # green
    if label == "DISTRACTED":
        color = (0, 165, 255)  # orange
    if label == "DROWSY":
        color = (0, 0, 255)  # red

    cv2.putText(frame, f"STATE: {label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(frame, f"SCORE: {score:3d}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"EAR: {ear:.3f}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"HEAD: {head_pos}", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if drowsy:
        cv2.putText(
            frame,
            "WARNING: DROWSINESS DETECTED!",
            (20, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            3,
        )


def run_driver_attention_monitor(
    camera_index: int = 0,
    ear_threshold: float = 0.23,
    closed_duration_sec: float = 1.2,
    draw_landmarks: bool = True,
) -> None:
    """
    Ana döngü:
    - OpenCV ile webcamin karelerini oku
    - MediaPipe ile yüz landmark noktalarını tespit et
    - EAR hesapla -> uyuklama kontrolü
    - Baş yönünü tahmin et
    - Dikkat skoru/durum etiketini hesapla
    - Ekran üstü (UI) bilgileri çiz
    """
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try a different camera_index.")

    # Bazı MediaPipe Python sürümlerinde `mp.solutions` yoktur; bunun yerine "Tasks" API bulunur.
    # Bu yüzden `mediapipe.tasks` altındaki FaceLandmarker kullanıyoruz.
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision

    state = DrowsinessState()

    model_path = _ensure_face_landmarker_model(
        Path(__file__).resolve().parent / "models" / "face_landmarker.task"
    )

    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Webcam kullanımında daha doğal olması için aynala.
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # MediaPipe RGB görüntü bekler (OpenCV BGR verir).
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            ear = 0.0
            head_pos = "CENTER"
            label = "ATTENTIVE"
            score = 100

            if results.face_landmarks:
                # Bulunan ilk yüzün landmark listesi (normalize koordinatlar)
                face_lms = results.face_landmarks[0]
                pts = _extract_landmarks_xy_from_normalized(face_lms, w, h)

                # Bazı modeller 478 landmark döndürebilir (iris dahil).
                # Bizim kullandığımız indeksler ilk 468 içinde olduğu için genelde uyumludur.
                if pts.shape[0] <= max(RIGHT_FACE_IDX, max(RIGHT_EYE_IDX)):
                    # Landmark sayısı yetersizse index hatası olmaması için "yüz yok" gibi davran.
                    score, label = 40, "DISTRACTED"
                    state.drowsy = False
                    state.eyes_closed = False
                    state.closed_since = 0.0
                    _draw_status_panel(frame, label, score, ear, head_pos, state.drowsy)
                    cv2.imshow("Driver Attention Monitoring (MVP)", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        break
                    continue

                # Daha stabil sonuç için iki gözün EAR değerinin ortalamasını al.
                left_eye = pts[LEFT_EYE_IDX]
                right_eye = pts[RIGHT_EYE_IDX]
                ear_left = calculate_EAR(left_eye)
                ear_right = calculate_EAR(right_eye)
                ear = (ear_left + ear_right) / 2.0

                now = time.time()
                state = detect_drowsiness(
                    ear=ear,
                    state=state,
                    ear_threshold=ear_threshold,
                    closed_duration_sec=closed_duration_sec,
                    now=now,
                )

                head_pos = estimate_head_position(pts, w, h)
                score, label = compute_attention_score(
                    ear=ear,
                    drowsy=state.drowsy,
                    head_pos=head_pos,
                    ear_threshold=ear_threshold,
                )

                if draw_landmarks:
                    _draw_face_landmarks_points(frame, pts, step=1)

            else:
                # Yüz bulunamadıysa "dalgın" kabul et.
                score, label = 40, "DISTRACTED"
                state.drowsy = False
                state.eyes_closed = False
                state.closed_since = 0.0

            _draw_status_panel(
                frame=frame,
                label=label,
                score=score,
                ear=ear,
                head_pos=head_pos,
                drowsy=state.drowsy,
            )

            cv2.imshow("Driver Attention Monitoring (MVP)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    # Kameraya, ışığa ve yüze olan mesafeye göre eşikleri ayarlaman gerekebilir.
    config: Dict[str, object] = {
        "camera_index": 0,
        "ear_threshold": 0.23,
        "closed_duration_sec": 1.2,
        "draw_landmarks": True,
    }
    run_driver_attention_monitor(**config)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()

