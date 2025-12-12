"""
Space Painting-ReXtro 2025
Features:
- Virtual Buttons: Colors, Brushes, Clear, Surprise.
- Gestures: Index (Draw), Open Hand (Hover UI).
- Interaction: Hover over a button for 2.0s to click.
- Snapshot: Perfect Heart Shape + Close Button.
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import os
import threading
from collections import deque

# ---------- Config ----------
WINDOW_NAME = 'Space Painting-ReXtro 2025'
CAP_INDEX = 0
CAP_WIDTH = 1280
CAP_HEIGHT = 720
HOVER_TIME = 2.0  # Time in seconds to trigger a click (Adjust this to 5.0 if you want)

# Colors (BGR)
COLORS = [
    (255, 255, 255), # White
    (0, 0, 0),       # Black
    (0, 0, 255),     # Red
    (0, 255, 0),     # Green
    (255, 0, 0),     # Blue
    (0, 255, 255)    # Yellow
]
COLOR_NAMES = ["White", "Black", "Red", "Green", "Blue", "Yellow"]

BRUSH_SIZES = [5, 15, 30]

# ---------- Global State ----------
state = {
    'color_idx': 2,
    'brush_idx': 1,
    'strokes': [],
    'current_stroke': None,
    'draw_cooldown': 0,
    'ai_image': None,
    'is_generating': False,
    'ui_active': False,

    # Hover State
    'hover_start_time': 0,
    'last_hovered_item': None, # Can be a button index or "CLOSE_X"
}

# Smoothing Buffer
smoother_buffer = deque(maxlen=5)

# ---------- MediaPipe Setup ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---------- UI Button Class ----------
class Button:
    def __init__(self, x, y, w, h, text, callback, param=None, color=(200,200,200)):
        self.rect = (x, y, w, h)
        self.text = text
        self.callback = callback
        self.param = param
        self.base_color = color

    def draw(self, img, is_hovering, progress=0.0):
        x, y, w, h = self.rect

        # Color change on hover
        if is_hovering:
            color = (255, 255, 255) # Highlight Border
            thick = 2
        else:
            color = self.base_color
            thick = 2

        # Draw box
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thick)

        # Draw Progress Bar (Bottom)
        if is_hovering and progress > 0:
            bar_w = int(w * progress)
            cv2.rectangle(img, (x, y + h - 5), (x + bar_w, y + h), (0, 255, 0), -1)

        # Draw Text/Icon
        font_scale = 0.6
        if self.param is not None and isinstance(self.param, int) and "Brush" in self.text:
             cv2.circle(img, (x + w//2, y + h//2), self.param, (255,255,255), -1)
        elif "Color" in self.text:
            c = COLORS[self.param]
            cv2.rectangle(img, (x+5, y+5), (x+w-5, y+h-5), c, -1)
        else:
            (tw, th), _ = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            tx = x + (w - tw) // 2
            ty = y + (h + th) // 2
            cv2.putText(img, self.text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), 2)

    def is_inside(self, x, y):
        rx, ry, rw, rh = self.rect
        return rx < x < rx + rw and ry < y < ry + rh

    def click(self):
        if self.callback:
            if self.param is not None:
                self.callback(self.param)
            else:
                self.callback()

# ---------- Callbacks ----------
def set_color(idx):
    state['color_idx'] = idx
    print(f"Color set to {COLOR_NAMES[idx]}")

def set_brush(idx):
    state['brush_idx'] = idx
    print(f"Brush size set to {BRUSH_SIZES[idx]}")

def clear_canvas():
    state['strokes'] = []
    state['current_stroke'] = None
    state['ai_image'] = None
    print("Canvas Cleared")

def trigger_surprise(user_frame):
    if state['is_generating']: return
    state['is_generating'] = True

    def worker(frame_snap):
        print("🎉 Surprise: Creating snapshot...")
        try:
            time.sleep(0.5)
            snapshot = frame_snap.copy()
            h, w, _ = snapshot.shape
            center_x, center_y = w // 2, h // 2

            # Heart Math
            scale = 15
            heart_pts = []
            for t in np.linspace(0, 2 * np.pi, 200):
                x = 16 * (np.sin(t) ** 3)
                y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
                px = int(center_x + x * scale)
                py = int(center_y - y * scale)
                heart_pts.append((px, py))

            pts_array = np.array(heart_pts, np.int32).reshape((-1, 1, 2))
            pink = (180, 105, 255)
            cv2.polylines(snapshot, [pts_array], True, pink, 8, cv2.LINE_AA)

            text = "Have a nice day!"
            font = cv2.FONT_HERSHEY_DUPLEX
            scale_font = 1.5
            thick = 3
            (tw, th), _ = cv2.getTextSize(text, font, scale_font, thick)
            text_x = center_x - tw // 2
            text_y = center_y + 220
            cv2.putText(snapshot, text, (text_x+2, text_y+2), font, scale_font, (0,0,0), thick)
            cv2.putText(snapshot, text, (text_x, text_y), font, scale_font, (255,255,255), thick)

            state['ai_image'] = snapshot
            print("✅ Surprise Ready!")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            state['is_generating'] = False

    t = threading.Thread(target=worker, args=(user_frame,))
    t.start()

# ---------- Setup Buttons ----------
buttons = []
start_x = 300
for i in range(6):
    buttons.append(Button(start_x + (i*70), 20, 60, 60, "Color", set_color, i))

buttons.append(Button(1150, 100, 80, 80, "Brush", set_brush, 0))
buttons.append(Button(1150, 200, 80, 80, "Brush", set_brush, 1))
buttons.append(Button(1150, 300, 80, 80, "Brush", set_brush, 2))

buttons.append(Button(50, 20, 150, 60, "CLEAR", clear_canvas, color=(0,0,255)))
buttons.append(Button(1100, 20, 150, 60, "Surprise", lambda: trigger_surprise(None), color=(255,0,255)))

# ---------- Helper Functions ----------
def get_finger_status(lm, h, w):
    fingers = []
    # Thumb
    if lm.landmark[4].x > lm.landmark[3].x: fingers.append(True)
    else: fingers.append(False)
    # Fingers
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for t, p in zip(tips, pips):
        if lm.landmark[t].y < lm.landmark[p].y: fingers.append(True)
        else: fingers.append(False)
    return fingers

def detect_gesture(fingers):
    up_count = fingers.count(True)
    # 4 or 5 fingers up = UI Mode
    if up_count >= 4: return "OPEN_HAND"
    # Index only = Draw Mode
    if fingers[1] and not fingers[2]: return "INDEX_POINT"
    return "UNKNOWN"

def main():
    cap = cv2.VideoCapture(CAP_INDEX)
    cap.set(3, CAP_WIDTH)
    cap.set(4, CAP_HEIGHT)

    canvas_layer = np.zeros((CAP_HEIGHT, CAP_WIDTH, 3), np.uint8)
    print("--- APP STARTED: ReXtro 2025 ---")
    print(f"Hover Mode: Hold hand over button for {HOVER_TIME}s to click.")

    while True:
        try:
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            ui_layer = np.zeros_like(frame)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            cursor_pos = None
            gesture = "UNKNOWN"

            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0]
                fingers_up = get_finger_status(lm, h, w)
                gesture = detect_gesture(fingers_up)

                cx, cy = int(lm.landmark[8].x * w), int(lm.landmark[8].y * h)
                smoother_buffer.append((cx, cy))
                avg_x = int(np.mean([p[0] for p in smoother_buffer]))
                avg_y = int(np.mean([p[1] for p in smoother_buffer]))
                cursor_pos = (avg_x, avg_y)

            # --- LOGIC ---

            # 1. UI MODE (OPEN HAND - HOVER LOGIC)
            if gesture == "OPEN_HAND":
                state['ui_active'] = True
                state['draw_cooldown'] = time.time() + 0.5

                # Check what we are hovering over
                hovered_something = False

                # A. Check Normal Buttons
                for i, btn in enumerate(buttons):
                    is_hover = btn.is_inside(*cursor_pos)
                    progress = 0.0

                    if is_hover:
                        hovered_something = True

                        # Logic: Is this a new hover or continuation?
                        if state['last_hovered_item'] == i:
                            elapsed = time.time() - state['hover_start_time']
                            progress = min(elapsed / HOVER_TIME, 1.0)

                            # TRIGGER CLICK
                            if elapsed >= HOVER_TIME:
                                print(f"Auto-Click: {btn.text}")
                                if btn.text == "Surprise":
                                    trigger_surprise(frame)
                                else:
                                    btn.click()
                                # Reset hover so it doesn't loop click
                                state['last_hovered_item'] = None
                                progress = 0.0
                        else:
                            # Start new hover
                            state['last_hovered_item'] = i
                            state['hover_start_time'] = time.time()

                    btn.draw(ui_layer, is_hover, progress)

                # B. Check X Button (If Snapshot Active)
                if state['ai_image'] is not None:
                    ai_h, ai_w = 250, 400
                    bx, by = w - 40, h - ai_h - 10
                    # Hitbox for X
                    if bx < cursor_pos[0] < bx+30 and by < cursor_pos[1] < by+30:
                        hovered_something = True

                        # Draw Hover Effect on X
                        cv2.rectangle(ui_layer, (bx-2, by-2), (bx+32, by+32), (0,255,0), 2)

                        if state['last_hovered_item'] == "CLOSE_X":
                            elapsed = time.time() - state['hover_start_time']
                            # Draw Loading bar under X
                            prog = min(elapsed / HOVER_TIME, 1.0)
                            bar_w = int(30 * prog)
                            cv2.rectangle(ui_layer, (bx, by+35), (bx+bar_w, by+40), (0,255,0), -1)

                            if elapsed >= HOVER_TIME:
                                print("Auto-Click: Close Snapshot")
                                state['ai_image'] = None
                                state['last_hovered_item'] = None
                        else:
                            state['last_hovered_item'] = "CLOSE_X"
                            state['hover_start_time'] = time.time()

                # If we aren't hovering anything, reset the timer
                if not hovered_something:
                    state['last_hovered_item'] = None
                    state['hover_start_time'] = 0

                # Draw Cursor
                cv2.circle(ui_layer, cursor_pos, 10, (255, 255, 255), 2)
                cv2.putText(ui_layer, "HOVER TO CLICK", (cursor_pos[0]+15, cursor_pos[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # 2. DRAW MODE (INDEX)
            elif gesture == "INDEX_POINT":
                state['last_hovered_item'] = None # Reset hover if drawing

                if time.time() > state['draw_cooldown']:
                    state['ui_active'] = False
                    if state['current_stroke'] is None:
                        state['current_stroke'] = {'pts': [], 'color': COLORS[state['color_idx']], 'size': BRUSH_SIZES[state['brush_idx']]}
                        smoother_buffer.clear()
                    state['current_stroke']['pts'].append(cursor_pos)
                    cv2.circle(frame, cursor_pos, 5, state['current_stroke']['color'], -1)
                else:
                    cv2.putText(ui_layer, "Wait...", (cursor_pos[0], cursor_pos[1]-20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

            else:
                # No valid gesture
                state['last_hovered_item'] = None
                if state['current_stroke']:
                    if len(state['current_stroke']['pts']) > 1:
                        state['strokes'].append(state['current_stroke'])
                    state['current_stroke'] = None

            # --- Rendering ---

            canvas_layer[:] = (0,0,0)
            for s in state['strokes']:
                if len(s['pts']) > 1:
                    pts = np.array(s['pts'], np.int32)
                    cv2.polylines(canvas_layer, [pts], False, s['color'], s['size'], cv2.LINE_AA)
            if state['current_stroke'] and len(state['current_stroke']['pts']) > 1:
                pts = np.array(state['current_stroke']['pts'], np.int32)
                cv2.polylines(canvas_layer, [pts], False, state['current_stroke']['color'], state['current_stroke']['size'], cv2.LINE_AA)

            # Snapshot Rendering
            if state['ai_image'] is not None:
                ai_h, ai_w = 250, 400
                small_ai = cv2.resize(state['ai_image'], (ai_w, ai_h))
                frame[h-ai_h-10:h-10, w-ai_w-10:w-10] = small_ai
                cv2.rectangle(frame, (w-ai_w-10, h-ai_h-10), (w-10, h-10), (255,255,255), 2)

                # Draw X Button
                bx, by = w - 40, h - ai_h - 10
                cv2.rectangle(frame, (bx, by), (bx+30, by+30), (0,0,200), -1)
                cv2.line(frame, (bx+5, by+5), (bx+25, by+25), (255,255,255), 2)
                cv2.line(frame, (bx+5, by+25), (bx+25, by+5), (255,255,255), 2)

            if state['is_generating']:
                cv2.putText(frame, "Preparing Surprise...", (w//2 - 150, h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            gray_canvas = cv2.cvtColor(canvas_layer, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)
            inv_mask = cv2.bitwise_not(mask)
            frame_bg = cv2.bitwise_and(frame, frame, mask=inv_mask)
            frame = cv2.add(frame_bg, canvas_layer)
            frame = cv2.addWeighted(frame, 1.0, ui_layer, 0.8, 0)

            cv2.putText(frame, "ReXtro-2025", (w - 200, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            info = f"Color: {COLOR_NAMES[state['color_idx']]} | Size: {state['brush_idx']+1}"
            cv2.putText(frame, info, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)

            cv2.imshow(WINDOW_NAME, frame)

            k = cv2.waitKey(1) & 0xFF
            if k == 27 or k == ord('q'): break

        except Exception as e:
            print(f"Error: {e}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
