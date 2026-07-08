from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
import cv2
import face_recognition
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pyttsx3
import threading
from scipy.spatial import distance as dist
import dlib
import os
import json
from datetime import datetime, timedelta

app = Flask(__name__)
app.secret_key = 'kunci_rahasia_ta_perpus_secure'

# Konfigurasi Database (Sesuaikan password dengan pgAdmin kamu)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:140203@localhost:5432/db_ai_advisor'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads_buku'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)

# =========================================================================
# MODEL DATABASE (POSTGRESQL)
# =========================================================================
class Users(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

class Mahasiswa(db.Model):
    __tablename__ = 'mahasiswa'
    id = db.Column(db.Integer, primary_key=True)
    nim = db.Column(db.String(50), unique=True, nullable=False)
    nama = db.Column(db.String(150), nullable=False)
    prodi = db.Column(db.String(100))
    foto_path = db.Column(db.String(255))

class SkripsiRepository(db.Model):
    __tablename__ = 'skripsi_repository'
    id = db.Column(db.Integer, primary_key=True)
    judul = db.Column(db.Text, nullable=False)
    abstrak = db.Column(db.Text)
    penulis = db.Column(db.String(150))
    tahun_lulus = db.Column(db.Integer)
    rak_lokasi = db.Column(db.String(50))

class Kunjungan(db.Model):
    __tablename__ = 'kunjungan'
    id = db.Column(db.Integer, primary_key=True)
    mahasiswa_id = db.Column(db.Integer, db.ForeignKey('mahasiswa.id', ondelete='CASCADE'))
    waktu_kunjungan = db.Column(db.DateTime, default=datetime.now)
    baca_buku_count = db.Column(db.Integer, default=0) 
    mahasiswa = db.relationship('Mahasiswa', backref='kunjungan_list')

class BukuDibacaLog(db.Model):
    __tablename__ = 'buku_dibaca_log'
    id = db.Column(db.Integer, primary_key=True)
    kunjungan_id = db.Column(db.Integer, db.ForeignKey('kunjungan.id', ondelete='CASCADE'))
    judul_buku = db.Column(db.String(255), nullable=False)
    waktu_input = db.Column(db.DateTime, default=datetime.now)

class BukuRepository(db.Model):
    __tablename__ = 'buku_repository'
    id = db.Column(db.Integer, primary_key=True)
    judul_buku = db.Column(db.String(255), nullable=False)
    penulis = db.Column(db.String(150))
    detail_rak = db.Column(db.String(50), nullable=False)
    file_path = db.Column(db.String(255))
    waktu_diupload = db.Column(db.DateTime, default=datetime.now)

with app.app_context():
    db.create_all()

# =========================================================================
# AI CORE (DETEKSI WAJAH & KEDIPAN ANTI-SPOOFING)
# =========================================================================
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 2
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except:
    print("File shape_predictor_68_face_landmarks.dat tidak ditemukan!")

(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

BLINK_COUNTER = 0
TOTAL_BLINKS = 0
known_face_encodings = []
known_face_names = []
detected_user = None  
has_greeted = False 

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def speak_text(text):
    def target():
        engine = pyttsx3.init()
        engine.setProperty('rate', 140)
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=target).start()

def reload_face_encodings():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    for mhs in Mahasiswa.query.all():
        try:
            img = face_recognition.load_image_file(mhs.foto_path)
            known_face_encodings.append(face_recognition.face_encodings(img)[0])
            known_face_names.append(mhs.nama)
        except: pass

with app.app_context():
    reload_face_encodings()

camera = cv2.VideoCapture(0)

def gen_frames():
    global detected_user, has_greeted, BLINK_COUNTER, TOTAL_BLINKS
    while True:
        success, frame = camera.read()
        if not success: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape_np = np.zeros((68, 2), dtype="int")
            for i in range(0, 68): shape_np[i] = (shape.part(i).x, shape.part(i).y)
            ear = (eye_aspect_ratio(shape_np[lStart:lEnd]) + eye_aspect_ratio(shape_np[rStart:rEnd])) / 2.0

            if ear < EYE_AR_THRESH: BLINK_COUNTER += 1
            else:
                if BLINK_COUNTER >= EYE_AR_CONSEC_FRAMES: TOTAL_BLINKS += 1
                BLINK_COUNTER = 0

        cv2.putText(frame, f"Kedipan: {TOTAL_BLINKS}/2", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if TOTAL_BLINKS >= 2:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb, face_recognition.face_locations(rgb))

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if len(distances) > 0:
                    idx = np.argmin(distances)
                    if matches[idx]:
                        nama_mhs = known_face_names[idx]
                        detected_user = nama_mhs
                        if not has_greeted:
                            speak_text(f"Verifikasi sukses. Selamat datang, {nama_mhs}.")
                            with app.app_context():
                                mhs_obj = Mahasiswa.query.filter_by(nama=nama_mhs).first()
                                if mhs_obj:
                                    log = Kunjungan(mahasiswa_id=mhs_obj.id, baca_buku_count=0)
                                    db.session.add(log)
                                    db.session.commit()
                            has_greeted = True
        else:
            cv2.putText(frame, "Mohon berkedip untuk verifikasi", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# =========================================================================
# LAYOUT UTAMA: FULL SCREEN CINEMATIC TRANSPARENT INTERACTION
# =========================================================================
def aesthetic_layout(title, content, robot_mode="standby", extra_head=""):
    three_js_cdn = '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>'
    
    return f"""
    <html>
        <head>
            <title>{title}</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
            {three_js_cdn}
            {extra_head}
            <style>
                /* FIX: Merubah overflow: hidden menjadi overflow-y: auto agar halaman dashboard bisa di-scroll ke bawah */
                body, html {{ margin: 0; padding: 0; width: 100%; height: 100%; overflow-y: auto; background: #060913; font-family: 'Poppins', sans-serif; }}
                
                #canvas3d {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 1; }}
                
                .ui-layer {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 2; display: flex; flex-direction: column; justify-content: space-between; padding: 40px; box-sizing: border-box; pointer-events: none; }}
                .ui-header {{ text-align: center; pointer-events: auto; margin-top: 10px; }}
                .ui-header h1 {{ color: #ffffff; font-size: 32px; font-weight: 600; letter-spacing: 2px; text-shadow: 0 4px 20px rgba(0,0,0,0.8); margin: 0; }}
                .ui-header p {{ color: #00d2ff; font-size: 14px; letter-spacing: 1px; margin: 8px 0 0 0; text-shadow: 0 2px 10px rgba(0,210,255,0.4); }}
                
                .ui-center-content {{ background: rgba(6, 9, 19, 0.45); border: 1px solid rgba(255,255,255,0.05); padding: 35px; border-radius: 24px; max-width: 700px; width: 100%; margin: auto; text-align: center; pointer-events: auto; box-shadow: 0 20px 60px rgba(0,0,0,0.5); box-sizing: border-box; color: #fff; }}
                
                .btn {{ display: inline-block; padding: 14px 35px; border-radius: 14px; font-weight: 600; text-decoration: none; transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275); font-size: 14px; cursor: pointer; border: none; margin: 10px; pointer-events: auto; }}
                .btn-primary {{ background: linear-gradient(135deg, #00d2ff, #0066ff); color: white; box-shadow: 0 4px 20px rgba(0,102,255,0.4); }}
                .btn-primary:hover {{ transform: translateY(-3px); box-shadow: 0 8px 25px rgba(0,102,255,0.6); }}
                .btn-secondary {{ background: rgba(255,255,255,0.06); color: #fff; border: 1px solid rgba(255,255,255,0.1); }}
                .btn-secondary:hover {{ background: rgba(255,255,255,0.15); transform: translateY(-2px); }}
                .btn-success {{ background: linear-gradient(135deg, #2ecc71, #27ae60); color: white; box-shadow: 0 4px 15px rgba(46,204,113,0.3); }}
                
                .input-field {{ width: 100%; padding: 14px; margin: 12px 0; border: 1px solid rgba(255,255,255,0.1); background: rgba(0,0,0,0.3); color: #fff; border-radius: 10px; font-family: 'Poppins', sans-serif; box-sizing: border-box; }}
                .input-field:focus {{ border-color: #00d2ff; outline: none; }}
                
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 13px; background: rgba(0,0,0,0.2); border-radius: 10px; overflow: hidden; }}
                th, td {{ padding: 14px; border-bottom: 1px solid rgba(255,255,255,0.05); text-align: left; }}
                th {{ background: rgba(0,210,255,0.12); color: #00d2ff; font-weight: 600; }}
                td {{ color: #e2e8f0; }}

                .status-node {{ font-size: 12px; font-weight: 600; letter-spacing: 1px; color: #00d2ff; margin-bottom: 15px; text-transform: uppercase; text-shadow: 0 0 10px rgba(0,210,255,0.5); }}
                .success-node {{ color: #2ecc71 !important; text-shadow: 0 0 10px rgba(46,204,113,0.5) !important; }}
            </style>
        </head>
        <body>
            <div id="canvas3d"></div>

            <div class="ui-layer">
                <div class="ui-header">
                    <h1>Sistem Cerdas Informasi Perpustakaan</h1>
                    <p>AI Academic Advisor & Biometric Security Gate</p>
                </div>

                <div class="ui-center-content">
                    <div class="status-node {'success-node' if robot_mode == 'success' else ''}">
                        {'✨ SYSTEM: VERIFIKASI SELESAI, WELCOME! ✨' if robot_mode == 'success' else '📡 SYSTEM: MENUNGGU DETEKSI WAJAH & KEDIPAN MATA... 📡'}
                    </div>
                    {content}
                </div>
                
                <div style="text-align: center; color: rgba(255,255,255,0.2); font-size: 11px; letter-spacing: 1px;">
                    UIN ANTASARI BANJARMASIN • TEKNOLOGI INFORMASI
                </div>
            </div>

            <script>
                const container = document.getElementById('canvas3d');
                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0x050811);
                scene.fog = new THREE.FogExp2(0x050811, 0.06);

                const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
                camera.position.set(0, 0, 10);

                const renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize(window.innerWidth, window.innerHeight);
                container.appendChild(renderer.domElement);

                const ambientLight = new THREE.AmbientLight(0x222a45, 1.8);
                scene.add(ambientLight);

                const blueLight = new THREE.DirectionalLight(0x00d2ff, 1.5);
                blueLight.position.set(-6, 6, 5);
                scene.add(blueLight);

                const orangeLight = new THREE.DirectionalLight(0xe67e22, 1.2);
                orangeLight.position.set(6, 6, 5);
                scene.add(orangeLight);

                const starsGeom = new THREE.BufferGeometry();
                const starsCount = 500;
                const starPositions = new Float32Array(starsCount * 3);
                for(let i=0; i<starsCount*3; i++) {{
                    starPositions[i] = (Math.random() - 0.5) * 25;
                }}
                starsGeom.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
                const starsMat = new THREE.PointsMaterial({{ color: 0xffffff, size: 0.035, transparent: true, opacity: 0.7 }});
                const starParticles = new THREE.Points(starsGeom, starsMat);
                scene.add(starParticles);

                // WALL-E MODEL
                const walle = new THREE.Group();
                const wBodyGeom = new THREE.BoxGeometry(1.6, 1.5, 1.5);
                const wBodyMat = new THREE.MeshStandardMaterial({{ color: 0xe67e22, roughness: 0.3, metalness: 0.3 }});
                const wBody = new THREE.Mesh(wBodyGeom, wBodyMat);
                walle.add(wBody);

                const wPlateGeom = new THREE.BoxGeometry(1.2, 1.0, 0.06);
                const wPlateMat = new THREE.MeshStandardMaterial({{ color: 0xbdc3c7, roughness: 0.4 }});
                const wPlate = new THREE.Mesh(wPlateGeom, wPlateMat);
                wPlate.position.set(0, 0, 0.76);
                walle.add(wPlate);

                const eyeGroup = new THREE.Group();
                const wEyeGeom = new THREE.CylinderGeometry(0.36, 0.28, 0.6, 32);
                const wEyeMat = new THREE.MeshStandardMaterial({{ color: 0x2c3e50, roughness: 0.2 }});
                const wLensMat = new THREE.MeshBasicMaterial({{ color: "{0x2ecc71 if robot_mode == 'success' else 0x050505}" }});

                const wEyeL = new THREE.Mesh(wEyeGeom, wEyeMat);
                wEyeL.rotation.x = Math.PI / 2;
                wEyeL.position.set(-0.4, 1.1, 0.1);
                eyeGroup.add(wEyeL);

                const wLensL = new THREE.Mesh(new THREE.CylinderGeometry(0.26, 0.26, 0.05, 32), wLensMat);
                wLensL.rotation.x = Math.PI / 2;
                wLensL.position.set(-0.4, 1.1, 0.4);
                eyeGroup.add(wLensL);

                const wEyeR = wEyeL.clone(); wEyeR.position.x = 0.4; eyeGroup.add(wEyeR);
                const wLensR = wLensL.clone(); wLensR.position.x = 0.4; eyeGroup.add(wLensR);
                walle.add(eyeGroup);

                const trackGeom = new THREE.BoxGeometry(1.9, 0.4, 1.6);
                const trackMat = new THREE.MeshStandardMaterial({{ color: 0x161d26, roughness: 0.8 }});
                const wTracks = new THREE.Mesh(trackGeom, trackMat);
                wTracks.position.y = -0.9;
                walle.add(wTracks);

                scene.add(walle);

                // EVE MODEL
                const eve = new THREE.Group();
                const eBodyGeom = new THREE.CylinderGeometry(0.65, 0.35, 1.8, 32);
                const eBodyMat = new THREE.MeshStandardMaterial({{ color: 0xffffff, roughness: 0.02, metalness: 0.15 }});
                const eveBody = new THREE.Mesh(eBodyGeom, eBodyMat);
                eve.add(eveBody);

                const eHeadGeom = new THREE.SphereGeometry(0.63, 32, 32);
                const eveHead = new THREE.Mesh(eHeadGeom, eBodyMat);
                eveHead.position.y = 1.05;
                eveHead.scale.set(1, 0.85, 1);
                eve.add(eveHead);

                const eScreenGeom = new THREE.SphereGeometry(0.48, 32, 16, 0, Math.PI*2, 0, Math.PI/2);
                const eScreenMat = new THREE.MeshBasicMaterial({{ color: 0x050505 }});
                const eveScreen = new THREE.Mesh(eScreenGeom, eScreenMat);
                eveScreen.position.set(0, 1.06, 0.23);
                eveScreen.rotation.x = Math.PI / 2.3;
                eve.add(eveScreen);

                const eEyeGeom = new THREE.SphereGeometry(0.08, 16, 16);
                const eEyeMat = new THREE.MeshBasicMaterial({{ color: "{0x2ecc71 if robot_mode == 'success' else 0x00d2ff}" }});
                
                const eveEyeL = new THREE.Mesh(eEyeGeom, eEyeMat);
                eveEyeL.position.set(-0.16, 1.15, 0.62);
                eve.add(eveEyeL);

                const eveEyeR = eveEyeL.clone(); eveEyeR.position.x = 0.16; eve.add(eveEyeR);

                scene.add(eve);

                let clock = 0;
                function animate() {{
                    requestAnimationFrame(animate);
                    clock += 0.02;

                    starParticles.rotation.y = clock * 0.015;

                    if ("{robot_mode}" === "success") {{
                        walle.position.x = Math.sin(clock * 2) * 2.5;
                        walle.position.y = Math.cos(clock * 2) * 1.5;
                        walle.rotation.y += 0.05;

                        eve.position.x = -Math.sin(clock * 2) * 2.5;
                        eve.position.y = -Math.cos(clock * 2) * 1.5;
                        eve.rotation.y -= 0.05;
                    }} else {{
                        walle.position.x = Math.sin(clock) * 5.5;
                        walle.position.y = Math.cos(clock * 1.5) * 2.2;
                        walle.position.z = Math.sin(clock * 0.5) * 1.5;
                        
                        walle.rotation.y = clock * 0.5;
                        walle.rotation.x = Math.sin(clock) * 0.2;

                        eve.position.x = -Math.sin(clock * 0.8) * 5.5; 
                        eve.position.y = Math.sin(clock * 1.2) * 2.2;
                        eve.position.z = Math.cos(clock * 0.6) * 1.5;
                        
                        eve.rotation.y = -clock * 0.6;
                        eve.rotation.z = Math.cos(clock) * 0.15;
                    }}

                    renderer.render(scene, camera);
                }}
                animate();

                window.addEventListener('resize', () => {{
                    camera.aspect = window.innerWidth / window.innerHeight;
                    camera.updateProjectionMatrix();
                    renderer.setSize(window.innerWidth, window.innerHeight);
                }});
            </script>
        </body>
    </html>
    """

# =========================================================================
# ROUTING PAGES
# =========================================================================
@app.route('/')
def index():
    global detected_user, has_greeted, TOTAL_BLINKS
    detected_user, has_greeted, TOTAL_BLINKS = None, False, 0
    content = """
    <p style="color:#e2e8f0;">Silakan klik opsi di bawah ini untuk memulai pencatatan sirkulasi kehadiran</p>
    <div style="margin-top: 25px;">
        <a href="/verifikasi_mahasiswa" class="btn btn-primary">Scan Biometrik Wajah</a>
        <a href="/admin_login" class="btn btn-secondary">Akses Staf Perpustakaan</a>
    </div>
    """
    return aesthetic_layout("Menu Utama Perpustakaan", content, robot_mode="standby")

@app.route('/verifikasi_mahasiswa')
def verifikasi_mahasiswa():
    content = """
    <h3 style="color:#fff; margin:0 0 5px 0;">Otentikasi Kamera Mengawasi Anda</h3>
    <p style="color: #ff4d4f; font-weight: 600; margin: 0 0 15px 0;">Sistem Deteksi Keaktifan: Silakan berkedip 2 kali.</p>
    <div style="margin: 15px 0;">
        <img src="/video_feed" style="border-radius: 15px; border:2px solid #00d2ff; box-shadow: 0 0 20px rgba(0,210,255,0.3); max-width: 100%; width:380px;">
    </div>
    <a href="/" class="btn btn-secondary" style="padding:8px 25px; font-size:12px;">Batalkan Presensi</a>
    <script>
        setInterval(function() {
            fetch('/check_auth').then(r => r.json()).then(data => {
                if(data.auth) { setTimeout(function() { window.location.href = '/dashboard'; }, 1000); }
            });
        }, 1500);
    </script>
    """
    return aesthetic_layout("Scan Face ID Robot", content, robot_mode="standby")

@app.route('/video_feed')
def video_feed(): return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check_auth')
def check_auth(): return jsonify({'auth': True if detected_user else False})

@app.route('/logout_mahasiswa')
def logout_mahasiswa():
    global detected_user, has_greeted, TOTAL_BLINKS
    detected_user, has_greeted, TOTAL_BLINKS = None, False, 0
    return redirect(url_for('index'))

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        admin_acc = Users.query.filter_by(username=username, password=password).first()
        if admin_acc:
            session['admin_logged_in'] = True
            return redirect(url_for('admin_dashboard'))
        else: error = "Username atau password admin salah!"
            
    content = f"""
    <h3 style="color:#fff; margin-bottom:5px;">Otentikasi Khusus Staf</h3>
    <form method="POST" style="max-width:320px; margin:15px auto; text-align:left;">
        <input type="text" name="username" class="input-field" placeholder="Username Admin" required>
        <input type="password" name="password" class="input-field" placeholder="Password Kredensial" required>
        {f'<p style="color:#ff4d4f; font-size:12px; text-align:center; margin:5px 0;">{error}</p>' if error else ''}
        <button type="submit" class="btn btn-primary" style="width:100%; margin-top:10px;">Login Panel Pusat</button>
    </form>
    <a href="/" style="font-size:12px; color:#8a99ad; text-decoration:none;">← Batalkan Akses</a>
    """
    return aesthetic_layout("Login Admin Secure", content, robot_mode="standby")

@app.route('/admin_logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('index'))

# =========================================================================
# DASHBOARD MAHASISWA (FIXED: JEDA DIUBAH MENJADI 10 DETIK REAL)
# =========================================================================
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    global detected_user
    if not detected_user: return redirect('/')
    
    mhs = Mahasiswa.query.filter_by(nama=detected_user).first()
    riwayat_pribadi = Kunjungan.query.filter_by(mahasiswa_id=mhs.id).order_by(Kunjungan.waktu_kunjungan.desc()).all()
    total_kunjungan = len(riwayat_pribadi)
    total_baca = db.session.query(db.func.sum(Kunjungan.baca_buku_count)).filter_by(mahasiswa_id=mhs.id).scalar() or 0

    grafik_kunjungan = Kunjungan.query.filter_by(mahasiswa_id=mhs.id).order_by(Kunjungan.waktu_kunjungan.asc()).limit(7).all()
    labels_pribadi = [k.waktu_kunjungan.strftime('%d/%m') for k in grafik_kunjungan]
    data_pribadi = [k.baca_buku_count for k in grafik_kunjungan]

    extra_head = "<script src='https://cdn.jsdelivr.net/npm/chart.js'></script>"
    content = f"""
    <div style="background: rgba(46, 204, 113, 0.2); border: 1px solid #2ecc71; padding: 10px; border-radius: 8px; margin-bottom: 25px; color: #2ecc71; font-size: 13px; font-weight: 600;">
        /* FIX: Menyesuaikan teks notifikasi agar selaras dengan waktu 10 detik */
        ⏱️ Transmisi Sukses! Layar kembali otomatis ke mode standby dalam <span id="countdown" style="font-size:16px; color:#ff4d4f; font-weight:bold;">10</span> detik...
    </div>
    <h3 style="color:#fff; margin:0;">Layanan Pintar Mahasiswa</h3>
    <p style="color:#a0aec0; margin:5px 0 20px 0;">Selamat datang, <b>{mhs.nama}</b> ({mhs.nim}) - {mhs.prodi if mhs.prodi else 'Umum'}</p>
    
    <div style="display: flex; gap: 20px; margin: 25px 0; flex-wrap: wrap;">
        <div style="flex:1; background:rgba(255,255,255,0.03); padding:20px; border-radius:12px; border:1px solid rgba(255,255,255,0.05);">
            <span style="font-size:11px; color:#a0aec0; letter-spacing:1px;">TOTAL KUNJUNGAN</span>
            <h2 style="margin:5px 0 0 0; color:#00d2ff;">{total_kunjungan} Kali</h2>
        </div>
        <div style="flex:1; background:rgba(255,255,255,0.03); padding:20px; border-radius:12px; border:1px solid rgba(255,255,255,0.05);">
            <span style="font-size:11px; color:#a0aec0; letter-spacing:1px;">JUDUL UNIK DIBACA</span>
            <h2 style="margin:5px 0 0 0; color:#2ecc71;">{total_baca} Judul</h2>
        </div>
    </div>

    <div style="background: rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.05); padding: 20px; border-radius: 12px; margin-bottom: 25px; text-align: left;">
        <h3 style="color:#fff; font-size:14px; margin:0 0 15px 0; letter-spacing:1px;">📈 Tren Grafik Aktivitas Belajar Anda</h3>
        <canvas id="studentLineChart" style="max-height: 160px; width: 100%;"></canvas>
    </div>
    <a href="/logout_mahasiswa" class="btn btn-secondary" style="width: 100%; box-sizing:border-box; margin:0;">Keluar Sekarang</a>

    <script>
        /* FIX: Mengganti variabel durasi hitung mundur dari 5 menjadi 10 detik */
        var seconds = 10; 
        var countdownElement = document.getElementById("countdown");
        var interval = setInterval(function() {{
            seconds--; countdownElement.textContent = seconds;
            if (seconds <= 0) {{ clearInterval(interval); window.location.href = "/logout_mahasiswa"; }}
        }}, 1000);

        const ctxLine = document.getElementById('studentLineChart').getContext('2d');
        new Chart(ctxLine, {{
            type: 'line',
            data: {{
                labels: {json.dumps(labels_pribadi)},
                datasets: [{{
                    data: {json.dumps(data_pribadi)},
                    borderColor: '#2ecc71',
                    backgroundColor: 'rgba(46, 204, 113, 0.05)',
                    borderWidth: 3, fill: true, tension: 0.3
                }}]
            }},
            options: {{ 
                responsive: true, 
                plugins: {{ legend: {{ display: false }} }},
                scales: {{ 
                    x: {{ grid: {{ color: 'rgba(255,255,255,0.03)' }}, ticks: {{ color: '#a0aec0' }} }},
                    y: {{ grid: {{ color: 'rgba(255,255,255,0.03)' }}, ticks: {{ color: '#a0aec0' }} }}
                }}
            }}
        }});
    </script>
    """
    return aesthetic_layout("Dashboard Mahasiswa", content, robot_mode="success", extra_head=extra_head)

# =========================================================================
# FEATURE: PORTAL DARI SCAN QR MEJA VIA HP MAHASISWA
# =========================================================================
@app.route('/scan_meja/<int:no_meja>', methods=['GET', 'POST'])
def scan_meja(no_meja):
    if request.method == 'POST':
        nim_input = request.form['nim']
        judul_baru = request.form['judul_buku'].strip()
        
        mhs = Mahasiswa.query.filter_by(nim=nim_input).first()
        if mhs:
            kunjungan_terakhir = Kunjungan.query.filter_by(mahasiswa_id=mhs.id).order_by(Kunjungan.waktu_kunjungan.desc()).first()
            if kunjungan_terakhir:
                sudah_ada = BukuDibacaLog.query.filter_by(kunjungan_id=kunjungan_terakhir.id).filter(BukuDibacaLog.judul_buku.ilike(judul_baru)).first()
                if not sudah_ada:
                    log_baru = BukuDibacaLog(kunjungan_id=kunjungan_terakhir.id, judul_buku=judul_baru)
                    db.session.add(log_baru)
                    kunjungan_terakhir.baca_buku_count += 1
                    db.session.commit()
                    msg = f"✔️ Judul Baru Terdeteksi! Berhasil menambahkan '{judul_baru}' ke dalam rekap."
                else: msg = f"ℹ️ Judul '{judul_baru}' sudah pernah Anda input hari ini. Total tidak bertambah."

                content_sukses = f"""
                <p style="text-align:left; color:#a0aec0;"><b>Validasi Identitas:</b><br>Nama: {mhs.nama}<br>NIM: {mhs.nim}<br>Jurusan: {mhs.prodi}</p>
                <p style="background:rgba(46, 204, 113, 0.15); padding:12px; border-radius:8px; border-left:4px solid #2ecc71; text-align:left; color:#2ecc71; font-size:13px;">{msg}</p>
                <p>Total Buku Terkalkulasi Hari Ini: <b>{kunjungan_terakhir.baca_buku_count} Judul Buku</b></p>
                """
                return aesthetic_layout("Validasi Sukses HP", content_sukses, robot_mode="success")
        return "<h3>NIM Salah atau Anda belum melakukan Face ID di Pintu Masuk!</h3>", 400

    content_hp = f"""
    <h3 style="color:#fff; margin:0;">Portal Pustaka Meja {no_meja:02d}</h3>
    <form method="POST" style="text-align: left; margin-top: 15px;">
        <label style="font-weight:600; font-size:12px; color:#00d2ff;">Konfirmasi NIM Anda:</label>
        <input type="text" name="nim" class="input-field" placeholder="Ketik NIM untuk pencarian indeks..." required>
        <label style="font-weight:600; font-size:12px; display:block; margin-top:15px; color:#00d2ff;">Judul Buku Fisik yang Dibaca:</label>
        <input type="text" name="judul_buku" class="input-field" placeholder="Ketik Judul Sampul Buku..." required>
        <button type="submit" class="btn btn-success" style="width:100%; margin-top:15px; font-weight:600;">Simpan Log Judul</button>
    </form>
    """
    return aesthetic_layout(f"Scan Meja {no_meja}", content_hp, robot_mode="standby")

# =========================================================================
# DASHBOARD ADMIN
# =========================================================================
@app.route('/admin_dashboard', methods=['GET', 'POST'])
def admin_dashboard():
    if not session.get('admin_logged_in'): return redirect(url_for('admin_login'))

    if request.method == 'POST' and 'tambah_buku' in request.form:
        buku_baru = BukuRepository(judul_buku=request.form['judul_buku'], penulis=request.form['penulis'], detail_rak=request.form['detail_rak'])
        db.session.add(buku_baru)
        db.session.commit()

    if request.method == 'POST' and 'edit_mahasiswa' in request.form:
        mhs_diubah = Mahasiswa.query.get(request.form['mhs_id'])
        if mhs_diubah:
            mhs_diubah.nama, mhs_diubah.nim, mhs_diubah.prodi = request.form['nama_baru'], request.form['nim_baru'], request.form['prodi_baru']
            db.session.commit()
            reload_face_encodings()

    filter_type = request.args.get('filter', 'all')
    query_kunjungan = Kunjungan.query
    list_kunjungan = query_kunjungan.order_by(Kunjungan.waktu_kunjungan.desc()).all()
    daftar_mahasiswa = Mahasiswa.query.all()

    rekap_fakultas = db.session.query(Mahasiswa.prodi, db.func.count(Kunjungan.id)).join(Kunjungan, Mahasiswa.id == Kunjungan.mahasiswa_id).group_by(Mahasiswa.prodi).all()
    labels_chart, data_chart = [str(f[0]) if f[0] else "Umum" for f in rekap_fakultas], [int(f[1]) for f in rekap_fakultas]

    extra_head = "<script src='https://cdn.jsdelivr.net/npm/chart.js'></script>"
    content = f"""
    <div style="text-align: left;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
            <h3 style="color:#fff; margin:0;">Panel Pemantauan Log Pusat</h3>
            <a href="/admin_logout" class="btn btn-secondary" style="padding:5px 15px; font-size:11px; margin:0;">Logout Staf</a>
        </div>
        <div class="charts-grid" style="margin-bottom:25px;">
            <div class="chart-container" style="background:rgba(0,0,0,0.2); border:1px solid rgba(255,255,255,0.05); padding:15px; border-radius:10px;"><h4 style="margin:0 0 10px 0; font-size:12px; color:#00d2ff;">Proporsi Distribusi Fakultas</h4><canvas id="pieChart" style="max-height:160px;"></canvas></div>
            <div class="chart-container" style="background:rgba(0,0,0,0.2); border:1px solid rgba(255,255,255,0.05); padding:15px; border-radius:10px;"><h4 style="margin:0 0 10px 0; font-size:12px; color:#00d2ff;">Total Frekuensi Kunjungan</h4><canvas id="barChart" style="max-height:160px;"></canvas></div>
        </div>
        <table id="rekapTable">
            <thead><tr><th>Nama Lengkap</th><th>NIM</th><th>Waktu Presensi</th><th>Buku Dibaca</th></tr></thead>
            <tbody>
                {"".join([f"<tr><td><b>{k.mahasiswa.nama}</b></td><td>{k.mahasiswa.nim}</td><td>{k.waktu_kunjungan.strftime('%d %b %Y, %H:%M')} WITA</td><td>{k.baca_buku_count} Judul Buku</td></tr>" for k in list_kunjungan]) if list_kunjungan else "<tr><td colspan='4'>Belum ada transmisi data.</td></tr>"}
            </tbody>
        </table>
    </div>

    <script>
        const colors = ['#00d2ff', '#2ecc71', '#e67e22', '#e74c3c'];
        const chartOpt = {{ plugins: {{ legend: {{ labels: {{ color: '#fff' }} }} }} }};
        new Chart(document.getElementById('pieChart'), {{ type: 'pie', data: {{ labels: {json.dumps(labels_chart)}, datasets: [{{ data: {json.dumps(data_chart)}, backgroundColor: colors }}] }}, options: chartOpt }});
        /* FIX: Menghapus tanda potong titik akhir pada baris inisialisasi bar chart admin */
        new Chart(document.getElementById('barChart'), {{ type: 'bar', data: {{ labels: {json.dumps(labels_chart)}, datasets: [{{ data: {json.dumps(data_chart)}, backgroundColor: '#00d2ff' }}] }}, options: {{ indexAxis: 'y', plugins:{{legend:{{display:false}}}}, scales:{{x:{{ticks:{{color:'#fff'}},grid:{{color:'rgba(255,255,255,0.03)'}}}},y:{{ticks:{{color:'#fff'}},grid:{{color:'rgba(255,255,255,0.03)'}}}}}} }} }});
    </script>
    """
    return aesthetic_layout("Dashboard Admin Control Center", content, robot_mode="success", extra_head=extra_head)

if __name__ == '__main__':
    app.run(debug=True, port=5000)