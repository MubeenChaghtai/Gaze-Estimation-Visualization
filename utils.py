import numpy as np

LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)

LEFT_EYE_TOP_BOTTOM = (159, 145)
RIGHT_EYE_TOP_BOTTOM = (386, 374)

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473


def _landmark_xy(landmarks, idx, frame_w, frame_h):
    lm = landmarks[idx]
    return np.array([lm.x * frame_w, lm.y * frame_h])


def extract_gaze_features(face_landmarks, frame_w, frame_h):
    lm = face_landmarks.landmark

    def eye_ratio(iris_center_idx, h_corners, v_corners):
        iris = _landmark_xy(lm, iris_center_idx, frame_w, frame_h)
        outer = _landmark_xy(lm, h_corners[0], frame_w, frame_h)
        inner = _landmark_xy(lm, h_corners[1], frame_w, frame_h)
        top = _landmark_xy(lm, v_corners[0], frame_w, frame_h)
        bottom = _landmark_xy(lm, v_corners[1], frame_w, frame_h)

        h_span = np.linalg.norm(inner - outer)
        v_span = np.linalg.norm(bottom - top)
        if h_span < 1e-3 or v_span < 1e-3:
            return None

        h_vec = inner - outer
        h_ratio = np.dot(iris - outer, h_vec) / (h_span ** 2)

        v_vec = bottom - top
        v_ratio = np.dot(iris - top, v_vec) / (v_span ** 2)

        return h_ratio, v_ratio

    left = eye_ratio(LEFT_IRIS_CENTER, LEFT_EYE_CORNERS, LEFT_EYE_TOP_BOTTOM)
    right = eye_ratio(RIGHT_IRIS_CENTER, RIGHT_EYE_CORNERS, RIGHT_EYE_TOP_BOTTOM)

    if left is None or right is None:
        return None

    return np.array([left[0], left[1], right[0], right[1]], dtype=np.float64)


def eye_openness(face_landmarks, frame_w, frame_h):
    lm = face_landmarks.landmark
    top_l = _landmark_xy(lm, LEFT_EYE_TOP_BOTTOM[0], frame_w, frame_h)
    bot_l = _landmark_xy(lm, LEFT_EYE_TOP_BOTTOM[1], frame_w, frame_h)
    outer_l = _landmark_xy(lm, LEFT_EYE_CORNERS[0], frame_w, frame_h)
    inner_l = _landmark_xy(lm, LEFT_EYE_CORNERS[1], frame_w, frame_h)

    v = np.linalg.norm(top_l - bot_l)
    h = np.linalg.norm(inner_l - outer_l)
    if h < 1e-3:
        return 0.0
    return v / h