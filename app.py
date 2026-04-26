from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import random
import time

# 🔥 NEW IMPORTS (ADDED ONLY)
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torch

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

latest_damage = {}
latest_paint = {}
report_history = []

# 🔥 LOAD AI MODEL (ADDED ONLY)
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16 if device=="cuda" else torch.float32
)

pipe = pipe.to(device)


# LOGIN
@app.route("/", methods=["GET","POST"])
def login():
    if request.method == "POST":
        if request.form["username"] == "pv8213" and request.form["password"] == "1234":
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error="Invalid Username or Password")
    return render_template("login.html")


# DASHBOARD
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html",
                           damage=latest_damage,
                           paint=latest_paint,
                           reports=report_history)


@app.route("/damage")
def damage():
    return render_template("damage_upload.html")


@app.route("/paint")
def paint():
    return render_template("paint_suggestion.html", colors=None)


# 🔥 DAMAGE ANALYSIS (UNCHANGED)
@app.route("/upload_damage", methods=["POST"])
def upload_damage():

    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    image = cv2.imread(filepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 80, 180)
    crack_pixels = np.sum(edges > 0)

    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    damp_pixels = np.sum(thresh > 0)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture = np.mean(np.abs(laplacian))

    brightness = np.mean(gray)
    variance = np.var(gray)

    total_pixels = gray.size

    crack_ratio = crack_pixels / total_pixels
    damp_ratio = damp_pixels / total_pixels

    damage_score = crack_ratio*0.4 + damp_ratio*0.3 + (texture/255)*0.2 + (variance/1000)*0.1
    damage_percent = min(int(damage_score*100), 95)

    if damage_percent < 20:
        severity = "Low"
    elif damage_percent < 50:
        severity = "Medium"
    else:
        severity = "High"

    if crack_ratio > damp_ratio:
        damage_type = "Structural Cracks"
    else:
        damage_type = "Water Leakage / Damp"

    building_health = 100 - damage_percent

    explanations = [
        "Edge concentration indicates structural cracks.",
        "Texture variation suggests surface degradation.",
        "Brightness inconsistency shows moisture damage.",
        "High-frequency edges indicate cracks formation.",
        "Surface irregularities confirm damage presence.",
        "Pixel variance reveals structural instability.",
        "Dark patches suggest damp accumulation.",
        "Contrast imbalance indicates material stress."
    ]

    explanation = " ".join(random.sample(explanations, 3))

    if severity == "Low":
        suggestion = "Minor cleaning and repaint recommended."
    elif severity == "Medium":
        suggestion = "Apply crack fillers and waterproof coating."
    else:
        suggestion = "Immediate structural repair required."

    priority = severity

    maintenance_actions = {
        "Low": ["Clean surface", "Repaint"],
        "Medium": ["Fill cracks", "Waterproofing", "Seal surface"],
        "High": ["Structural repair", "Reinforcement", "Full renovation"]
    }

    filename = f"processed_{int(time.time())}.jpg"
    cv2.imwrite(f"static/{filename}", image)

    result = {
        "damage": damage_type,
        "severity": severity,
        "health": building_health,
        "confidence": random.randint(92,99),
        "issues": int(damage_percent/8)+1,
        "image": filename,
        "suggestion": suggestion,
        "explanation": explanation,
        "priority": priority,
        "actions": maintenance_actions[severity]
    }

    global latest_damage
    latest_damage = result

    return render_template("damage_result.html", result=result)


# 🎨 🔥 REAL AI PAINT ANALYSIS (UPDATED ONLY)
@app.route("/paint_analysis", methods=["POST"])
def paint_analysis():

    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    image_pil = Image.open(filepath).convert("RGB")

    # 🔥 CREATE WALL MASK
    img_cv = cv2.imread(filepath)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    mask_pil = Image.fromarray(mask).convert("RGB")

    # 🔥 COLOR OPTIONS
    colors = [
        "pastel blue wall",
        "mint green wall",
        "modern grey wall",
        "warm white wall",
        "light peach wall"
    ]

    selected_color = random.choice(colors)

    prompt = f"realistic interior {selected_color}, smooth paint, high quality, photorealistic"

    try:
        # 🔥 AI INPAINT (REAL REPAINT)
        result = pipe(
            prompt=prompt,
            image=image_pil,
            mask_image=mask_pil
        ).images[0]

        filename = f"painted_{int(time.time())}.png"
        result.save(f"static/{filename}")

        global latest_paint
        latest_paint = {
            "primary": selected_color,
            "secondary": "AI Generated",
            "accent": "Realistic Finish",
            "reason": "Wall repainted using Stable Diffusion Inpainting (REAL AI)"
        }

    except Exception as e:
        print("AI failed → fallback:", e)

        # 🔥 FALLBACK (YOUR OLD METHOD)
        image = cv2.imread(filepath)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower = np.array([0, 0, 100])
        upper = np.array([180, 60, 255])

        mask = cv2.inRange(hsv, lower, upper)

        repaint = image.copy()
        color = (200,220,255)

        for i in range(3):
            repaint[:,:,i] = np.where(mask==255,
                                     repaint[:,:,i]*0.4 + color[i]*0.6,
                                     repaint[:,:,i])

        filename = f"painted_{int(time.time())}.jpg"
        cv2.imwrite(f"static/{filename}", repaint)

        latest_paint = {
            "primary": "Fallback Paint",
            "secondary": "Basic Mode",
            "accent": "Safe Mode",
            "reason": "AI failed → fallback repaint used"
        }

    return render_template("paint_suggestion.html",
                           colors=latest_paint,
                           original=file.filename,
                           painted=filename)


# REPORT
@app.route("/generate_report")
def generate_report():

    summary = "No data available"
    if latest_damage:
        summary = f"Building condition is {latest_damage.get('severity')} with {latest_damage.get('damage')} issues."

    report = {
        "damage": latest_damage,
        "paint": latest_paint,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": summary
    }

    report_history.append(report)

    return render_template("report.html",
                           damage=latest_damage,
                           paint=latest_paint,
                           history=report_history,
                           summary=summary)


# ADMIN
@app.route("/admin", methods=["GET","POST"])
def admin():

    files = os.listdir(UPLOAD_FOLDER)

    message = ""
    if request.method == "POST":
        file = request.files["dataset"]
        file.save(os.path.join(UPLOAD_FOLDER, file.filename))
        message = "Dataset uploaded successfully!"

    stats = {
        "total_datasets": len(files),
        "reports_generated": len(report_history),
        "last_upload": files[-1] if files else "None"
    }

    return render_template("admin.html",
                           message=message,
                           files=files,
                           stats=stats)


if __name__ == "__main__":
    app.run(debug=True)