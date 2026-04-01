from flask import Flask, render_template, request, send_from_directory, redirect, url_for, session
from datetime import timedelta, datetime
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import os
import csv
import numpy as np
import cv2
import sqlite3

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

from werkzeug.security import generate_password_hash, check_password_hash

# ---------------------------
# Initialize Flask app
# ---------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PDF_FOLDER'] = 'pdfs'
app.config['HEATMAP_FOLDER'] = 'heatmaps'
app.config['DB_FILE'] = 'doctors.db'   # ✅ database
app.secret_key = 'your_secret_key'
app.permanent_session_lifetime = timedelta(minutes=30)

# Create folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PDF_FOLDER'], exist_ok=True)
os.makedirs(app.config['HEATMAP_FOLDER'], exist_ok=True)

# Load model
model = load_model('model/brain_model.h5')

# Class labels
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']


# ---------------------------
# ✅ Database setup
# ---------------------------
def init_db():
    conn = sqlite3.connect(app.config['DB_FILE'])
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS doctors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fullname TEXT NOT NULL,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()


# ---------------------------
# ✅ Grad-CAM
# ---------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)

    return heatmap.numpy()


def save_gradcam_images(original_path, heatmap, output_base_name):
    img = cv2.imread(original_path)
    img = cv2.resize(img, (300, 300))

    heatmap_resized = cv2.resize(heatmap, (300, 300))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    heatmap_filename = f"{output_base_name}_heatmap.jpg"
    overlay_filename = f"{output_base_name}_overlay.jpg"

    heatmap_path = os.path.join(app.config['HEATMAP_FOLDER'], heatmap_filename)
    overlay_path = os.path.join(app.config['HEATMAP_FOLDER'], overlay_filename)

    cv2.imwrite(heatmap_path, heatmap_color)
    cv2.imwrite(overlay_path, overlay)

    return heatmap_filename, overlay_filename


# ---------------------------
# Routes
# ---------------------------
@app.route('/')
def index():
    return render_template('index.html')


# ✅ Signup (Doctor Register)
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        fullname = request.form.get('fullname', '').strip()
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if not fullname or not username or not password:
            return render_template('signup.html', error="All fields are required!")

        password_hash = generate_password_hash(password)

        try:
            conn = sqlite3.connect(app.config['DB_FILE'])
            cur = conn.cursor()
            cur.execute("INSERT INTO doctors (fullname, username, password_hash) VALUES (?, ?, ?)",
                        (fullname, username, password_hash))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return render_template('signup.html', error="Username already exists! Try another.")

    return render_template('signup.html')


# ✅ Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        conn = sqlite3.connect(app.config['DB_FILE'])
        cur = conn.cursor()
        cur.execute("SELECT fullname, password_hash FROM doctors WHERE username=?", (username,))
        doctor = cur.fetchone()
        conn.close()

        if doctor and check_password_hash(doctor[1], password):
            session['doctor'] = doctor[0]   # store doctor fullname
            session['username'] = username
            return redirect(url_for('admin'))
        else:
            return render_template('login.html', error="Invalid username or password!")

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# ✅ Predict
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files or request.files['image'].filename == '':
        return "No image uploaded!"

    timestamp = datetime.now().strftime("%d-%m-%Y %I:%M %p")

    name = request.form.get('name', 'Unknown')
    age = request.form.get('age', 'Unknown')
    gender = request.form.get('gender', 'Unknown')
    symptoms = request.form.get('symptoms', 'Not provided')

    img_file = request.files['image']
    filename = img_file.filename
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img_file.save(img_path)

    img = image.load_img(img_path, target_size=(300, 300))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)[0]
    pred_class = classes[np.argmax(prediction)]
    confidence = round(100 * np.max(prediction), 2)

    image_url = f"/uploads/{filename}"

    heatmap_filename = None
    overlay_filename = None

    try:
        last_conv_layer = "top_conv"
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)
        base_name = os.path.splitext(filename)[0]
        heatmap_filename, overlay_filename = save_gradcam_images(img_path, heatmap, base_name)
    except Exception as e:
        print("Grad-CAM Error:", e)

    heatmap_url = f"/heatmaps/{heatmap_filename}" if heatmap_filename else None
    overlay_url = f"/heatmaps/{overlay_filename}" if overlay_filename else None

    # Save CSV
    with open('patients.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, name, age, gender, symptoms, pred_class.title(), confidence, filename])

    # PDF
    pdf_filename = f"{name.replace(' ', '_')}_report.pdf"
    pdf_path = os.path.join(app.config['PDF_FOLDER'], pdf_filename)

    c = canvas.Canvas(pdf_path)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, 810, "Brain Tumor Classification Report")
    c.line(100, 805, 450, 805)

    c.setFont("Helvetica", 12)
    c.drawString(100, 785, f"Date/Time: {timestamp}")
    c.drawString(100, 765, f"Patient Name: {name}")
    c.drawString(100, 745, f"Age: {age}")
    c.drawString(100, 725, f"Gender: {gender}")
    c.drawString(100, 705, f"Symptoms: {symptoms}")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, 675, f"Predicted Tumor Type: {pred_class.title()}")
    c.drawString(100, 655, f"Confidence: {confidence}%")

    c.setFont("Helvetica", 11)
    c.drawString(100, 630, "Explainable AI: Grad-CAM overlay visualization is provided below.")

    if overlay_filename:
        overlay_path = os.path.join(app.config['HEATMAP_FOLDER'], overlay_filename)
        if os.path.exists(overlay_path):
            try:
                overlay_img = ImageReader(overlay_path)
                c.setFont("Helvetica-Bold", 12)
                c.drawString(100, 600, "Grad-CAM Overlay:")
                c.drawImage(overlay_img, 100, 340, width=320, height=240,
                            preserveAspectRatio=True, mask='auto')
            except Exception as e:
                print("PDF Overlay Insert Error:", e)

    c.save()

    return render_template(
        'result.html',
        name=name,
        age=age,
        gender=gender,
        symptoms=symptoms,
        result=pred_class.title(),
        confidence=confidence,
        image_url=image_url,
        pdf_url=f"/pdfs/{pdf_filename}",
        heatmap_url=heatmap_url,
        overlay_url=overlay_url
    )


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/heatmaps/<filename>')
def heatmap_file(filename):
    return send_from_directory(app.config['HEATMAP_FOLDER'], filename)


@app.route('/pdfs/<filename>')
def download_pdf(filename):
    return send_from_directory(app.config['PDF_FOLDER'], filename)


# ✅ Admin (only for logged doctor)
@app.route('/admin')
def admin():
    if 'doctor' not in session:
        return redirect(url_for('login'))

    patient_data = []
    if os.path.exists('patients.csv'):
        with open('patients.csv', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                patient_data.append(row)

    return render_template('admin.html', doctor=session['doctor'], patients=patient_data)


# ✅ Delete patient record
@app.route('/delete/<int:row_id>', methods=['POST'])
def delete_record(row_id):
    if 'doctor' not in session:
        return redirect(url_for('login'))

    rows = []
    if os.path.exists('patients.csv'):
        with open('patients.csv', newline='') as file:
            reader = csv.reader(file)
            rows = list(reader)

    if 0 <= row_id < len(rows):
        del rows[row_id]

    with open('patients.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

    return redirect(url_for('admin'))


if __name__ == '__main__':
    app.run(debug=True)
