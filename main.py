import csv
import time
import os
import numpy as np
import cv2
import mediapipe as mp

from utils import extract_gaze_features, eye_openness
from calibration import GazeCalibrator, build_calibration_targets
from heatmap import GazeHeatmap

CAM_INDEX = 0
SCREEN_W, SCREEN_H = 1280, 720
CALIB_GRID = (3, 3)
MIN_EYE_OPENNESS = 0.15
LOG_PATH = "gaze_log.csv"
CALIB_SAVE_PATH = "calibration_model.pkl"
HEATMAP_SAVE_PATH = "session_heatmap.png"


def run_calibration(cap, face_mesh):
    targets = build_calibration_targets(SCREEN_W, SCREEN_H, grid=CALIB_GRID)
    features, collected_targets = [], []

    win = "Calibration - look at the dot, press SPACE"
    cv2.namedWindow(win)

    for (tx, ty) in targets:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
            cv2.circle(canvas, (tx, ty), 18, (0, 255, 0), -1)
            cv2.putText(canvas, "Look at the dot, press SPACE. ESC to cancel.",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(win, canvas)
            cv2.imshow("Webcam", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                cv2.destroyWindow(win)
                return None
            if key == ord(' '):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb)
                if not result.multi_face_landmarks:
                    print("No face detected, try again.")
                    continue
                lm = result.multi_face_landmarks[0]
                if eye_openness(lm, w, h) < MIN_EYE_OPENNESS:
                    print("Eye looked closed, try again.")
                    continue
                feat = extract_gaze_features(lm, w, h)
                if feat is None:
                    print("Could not extract features, try again.")
                    continue
                features.append(feat)
                collected_targets.append((tx, ty))
                break

    cv2.destroyWindow(win)
    calibrator = GazeCalibrator()
    calibrator.fit(features, collected_targets)
    calibrator.save(CALIB_SAVE_PATH)
    print(f"Calibration complete with {len(features)} points. Saved to {CALIB_SAVE_PATH}")
    return calibrator


def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check CAM_INDEX / permissions.")

    calibrator = None
    if os.path.exists(CALIB_SAVE_PATH):
        calibrator = GazeCalibrator.load(CALIB_SAVE_PATH)
        print("Loaded existing calibration. Press 'c' to redo it.")

    heatmap = GazeHeatmap(SCREEN_W, SCREEN_H, decay=0.985)
    show_heatmap = True

    log_file = open(LOG_PATH, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["timestamp", "screen_x", "screen_y"])

    print("Press 'c' to calibrate, 'h' to toggle heatmap, 's' to snapshot, 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            gaze_point = None
            if result.multi_face_landmarks:
                lm = result.multi_face_landmarks[0]
                if eye_openness(lm, w, h) >= MIN_EYE_OPENNESS:
                    feat = extract_gaze_features(lm, w, h)
                    if feat is not None and calibrator is not None:
                        gaze_point = calibrator.predict(feat)

            if gaze_point is not None:
                gx, gy = gaze_point
                gx = float(np.clip(gx, 0, SCREEN_W - 1))
                gy = float(np.clip(gy, 0, SCREEN_H - 1))
                heatmap.add_point(gx, gy)
                log_writer.writerow([time.time(), gx, gy])

            heatmap.step_decay()

            canvas = np.full((SCREEN_H, SCREEN_W, 3), 30, dtype=np.uint8)
            if show_heatmap:
                canvas = heatmap.render(background=canvas, alpha=0.6)
            if gaze_point is not None:
                cv2.circle(canvas, (int(gx), int(gy)), 10, (0, 255, 255), -1)

            status = "CALIBRATED" if calibrator is not None else "NOT CALIBRATED - press 'c'"
            cv2.putText(canvas, status, (20, SCREEN_H - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Webcam", frame)
            cv2.imshow("Gaze Screen", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                new_cal = run_calibration(cap, face_mesh)
                if new_cal is not None:
                    calibrator = new_cal
            elif key == ord('h'):
                show_heatmap = not show_heatmap
            elif key == ord('s'):
                heatmap.save(HEATMAP_SAVE_PATH)
                print(f"Saved heatmap snapshot to {HEATMAP_SAVE_PATH}")

    finally:
        heatmap.save(HEATMAP_SAVE_PATH)
        log_file.close()
        cap.release()
        cv2.destroyAllWindows()
        print(f"Session ended. Gaze log: {LOG_PATH}, heatmap: {HEATMAP_SAVE_PATH}")


if __name__ == "__main__":
    main()