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
app.secret_key = 'kunci_rahasia_ta_perpus_secure' # Kunci session login admin

# Konfigurasi Database PostgreSQL (Sesuaikan password dengan pgAdmin kamu)
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
    baca_buku_count = db.Column(db.Integer, default=0) # Default awal 0, diisi via HP mahasiswa
    mahasiswa = db.relationship('Mahasiswa', backref='kunjungan_list')

class BukuRepository(db.Model):
    __tablename__ = 'buku_repository'
    id = db.Column(db.Integer, primary_key=True)
    judul_buku = db.Column(db.String(255), nullable=False)
    penulis = db.Column(db.String(150))
    detail_rak = db.Column(db.String(50), nullable=False)
    file_path = db.Column(db.String(255))
    waktu_diupload = db.Column(db.DateTime, default=datetime.now)

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
        engine.setProperty('rate', 150)
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
                                    # Set nilai awal 0, pengisian jumlah buku nyata dilakukan lewat HP mahasiswa via QR Code
                                    log = Kunjungan(mahasiswa_id=mhs_obj.id, baca_buku_count=0)
                                    db.session.add(log)
                                    db.session.commit()
                            has_greeted = True
        else:
            cv2.putText(frame, "Mohon berkedip untuk verifikasi", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# =========================================================================
# LAYOUT UTAMA: AESTHETIC WHITE WITH CARTOON HIGH-FIVE ANIMATION
# =========================================================================
def aesthetic_layout(title, content, show_success_animation=False, extra_head=""):
    success_banner = ""
    if show_success_animation:
        success_banner = """
        <div class="success-alert-box no-print">
            <div class="characters-container">
                <svg class="char char-left" viewBox="0 0 64 64">
                    <circle cx="32" cy="24" r="16" fill="#4a90e2"/>
                    <circle cx="26" cy="22" r="2" fill="#fff"/>
                    <circle cx="38" cy="22" r="2" fill="#fff"/>
                    <path d="M26 30s2 3 6 3 6-3 6-3" stroke="#fff" stroke-width="2" stroke-linecap="round" fill="none"/>
                    <path d="M44 24 L54 12" stroke="#4a90e2" stroke-width="4" stroke-linecap="round"/>
                </svg>
                <svg class="char char-right" viewBox="0 0 64 64">
                    <circle cx="32" cy="24" r="16" fill="#2ecc71"/>
                    <circle cx="26" cy="22" r="2" fill="#fff"/>
                    <circle cx="38" cy="22" r="2" fill="#fff"/>
                    <path d="M26 30s2 3 6 3 6-3 6-3" stroke="#fff" stroke-width="2" stroke-linecap="round" fill="none"/>
                    <path d="M20 24 L10 12" stroke="#2ecc71" stroke-width="4" stroke-linecap="round"/>
                </svg>
            </div>
            <div class="success-text">🎉 VERIFIKASI BERHASIL! 🎉</div>
        </div>
        """

    return f"""
    <html>
        <head>
            <title>{title}</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
            {extra_head}
            <style>
                body {{ 
                    font-family: 'Poppins', sans-serif; 
                    background-color: #fcfcfc; 
                    color: #2c3e50; 
                    margin: 0; 
                    padding: 0; 
                    display: block; 
                }}
                .app-card {{ 
                    background: white; 
                    padding: 40px; 
                    border-radius: 20px; 
                    box-shadow: 0 10px 30px rgba(0,0,0,0.05); 
                    width: 100%; 
                    max-width: 850px; 
                    text-align: center; 
                    position: relative; 
                    border: 1px solid #f0f0f0; 
                    margin: 40px auto; 
                    box-sizing: border-box;
                }}
                h1, h2, h3 {{ color: #1a1a1a; font-weight: 600; }}
                p {{ color: #7f8c8d; font-size: 14px; }}
                .btn {{ display: inline-block; padding: 12px 30px; border-radius: 10px; font-weight: 600; text-decoration: none; transition: all 0.3s; font-size: 14px; cursor: pointer; border: none; margin: 10px; }}
                .btn-primary {{ background: #4a90e2; color: white; }}
                .btn-primary:hover {{ background: #357abd; transform: translateY(-2px); }}
                .btn-danger {{ background: #e74c3c; color: white; }}
                .btn-secondary {{ background: #f5f6fa; color: #7f8c8d; }}
                .btn-success {{ background: #2ecc71; color: white; }}
                .btn-success:hover {{ background: #27ae60; transform: translateY(-2px); }}
                .input-field {{ width: 100%; padding: 12px; margin: 10px 0; border: 1px solid #e0e0e0; border-radius: 8px; font-family: 'Poppins', sans-serif; box-sizing: border-box; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 13px; }}
                th, td {{ padding: 12px; border-bottom: 1px solid #f0f0f0; text-align: left; }}
                th {{ background: #f9fbfd; color: #4a90e2; }}
                
                .success-alert-box {{ background: #e8f5e9; border: 1px solid #c8e6c9; padding: 15px; border-radius: 15px; margin-bottom: 25px; animation: popIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275); }}
                .characters-container {{ display: flex; justify-content: center; gap: 20px; margin-bottom: 5px; }}
                .char {{ width: 50px; height: 50px; }}
                .char-left {{ animation: jumpLeft 0.6s ease-in-out infinite alternate; }}
                .char-right {{ animation: jumpRight 0.6s ease-in-out infinite alternate; }}
                .success-text {{ font-weight: 600; color: #2e7d32; font-size: 15px; letter-spacing: 1px; }}
                
                @keyframes popIn {{ 0% {{ transform: scale(0.8); opacity: 0; }} 100% {{ transform: scale(1); opacity: 1; }} }}
                @keyframes jumpLeft {{ 0% {{ transform: translateY(0px) rotate(0deg); }} 100% {{ transform: translateY(-8px) rotate(5deg); }} }}
                @keyframes jumpRight {{ 0% {{ transform: translateY(0px) rotate(0deg); }} 100% {{ transform: translateY(-8px) rotate(-5deg); }} }}
                
                .logo-animation {{ width: 60px; height: 60px; margin: 0 auto 20px auto; animation: float 3s ease-in-out infinite; }}
                @keyframes float {{ 0% {{ transform: translateY(0px); }} 50% {{ transform: translateY(-6px); }} 100% {{ transform: translateY(0px); }} }}
                
                .charts-grid {{ display: flex; gap: 20px; margin-top: 25px; flex-wrap: wrap; }}
                .chart-container {{ flex: 1; min-width: 280px; background: #f9fbfd; padding: 15px; border-radius: 12px; border: 1px solid #e2e8f0; box-sizing: border-box; }}
                
                @media print {{
                    body {{ background: white; color: black; }}
                    .no-print {{ display: none !important; }}
                    .app-card {{ box-shadow: none; border: none; padding: 0; max-width: 100%; margin: 0; }}
                    th {{ background: #f5f5f5 !important; color: black !important; }}
                }}
            </style>
            <script>
                function downloadExcel() {{
                    var table = document.getElementById("rekapTable");
                    if (!table) return;
                    var csv = [];
                    for (var i = 0; i < table.rows.length; i++) {{
                        var row = [], cols = table.rows[i].cells;
                        for (var j = 0; j < cols.length; j++) 
                            row.push('"' + cols[j].innerText.trim() + '"');
                        csv.push(row.join(","));
                    }}
                    var csvContent = "data:text/csv;charset=utf-8,\\uFEFF" + csv.join("\\n");
                    var encodedUri = encodeURI(csvContent);
                    var link = document.createElement("a");
                    link.setAttribute("href", encodedUri);
                    link.setAttribute("download", "Rekap_Kunjungan_" + new Date().toISOString().slice(0,10) + ".csv");
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                }}
            </script>
        </head>
        <body>
            <div class="app-card">
                {success_banner}
                {"<div class='logo-animation'><svg viewBox='0 0 64 64' width='60' height='60' class='no-print'><path d='M12 56V12c0-2.2 1.8-4 4-4h32c2.2 0 4 1.8 4 4v44L32 44 12 56z' fill='#4a90e2'/><circle cx='24' cy='28' r='3' fill='#fff'/><circle cx='40' cy='28' r='3' fill='#fff'/><path d='M28 36s2 3 4 3 4-3 4-3' stroke='#fff' stroke-width='2' fill='none'/></svg></div>" if not show_success_animation else ""}
                {content}
            </div>
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
    <h2>Sistem Cerdas Informasi Perpustakaan</h2>
    <p>Selamat datang, silakan pilih opsi pintu masuk ruang akademik</p>
    <div style="margin-top: 30px;">
        <a href="/verifikasi_mahasiswa" class="btn btn-primary">Masuk Sebagai Mahasiswa</a>
        <a href="/admin_login" class="btn btn-secondary">Gerbang Admin</a>
    </div>
    """
    return aesthetic_layout("Menu Utama Perpustakaan", content)

@app.route('/verifikasi_mahasiswa')
def verifikasi_mahasiswa():
    content = """
    <h2>Verifikasi Kamera Biometrik</h2>
    <p style="color: #e74c3c; font-weight: 600;">Sistem Anti-Pemalsuan Aktif: Silakan berkedip 2 kali.</p>
    <div style="margin: 20px 0;">
        <img src="/video_feed" style="border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); max-width: 100%; width:440px;">
    </div>
    <a href="/" class="btn btn-secondary">Kembali</a>
    <script>
        setInterval(function() {
            fetch('/check_auth').then(r => r.json()).then(data => {
                if(data.auth) { setTimeout(function() { window.location.href = '/dashboard'; }, 1000); }
            });
        }, 1500);
    </script>
    """
    return aesthetic_layout("Scan Face ID", content)

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
        else:
            error = "Username atau password admin salah!"
            
    content = f"""
    <h2>Otentikasi Staf Admin</h2>
    <p>Silakan masukkan kode kredensial untuk mengelola data rahasia</p>
    <form method="POST" style="max-width:340px; margin:20px auto; text-align:left;">
        <input type="text" name="username" class="input-field" placeholder="Username Admin" required>
        <input type="password" name="password" class="input-field" placeholder="Password" required>
        {f'<p style="color:#e74c3c; font-size:12px; text-align:center;">{error}</p>' if error else ''}
        <button type="submit" class="btn btn-primary" style="width:100%; margin:15px 0 0 0;">Verifikasi Masuk</button>
    </form>
    <a href="/" style="font-size:12px; color:#7f8c8d; text-decoration:none;">← Kembali</a>
    """
    return aesthetic_layout("Login Admin Secure", content)

@app.route('/admin_logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('index'))

# =========================================================================
# DASHBOARD MAHASISWA (AUTO LOGOUT 5 DETIK UNTUK ANTRIAN KAMPUS)
# =========================================================================
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    global detected_user
    if not detected_user: return redirect('/')
    
    mhs = Mahasiswa.query.filter_by(nama=detected_user).first()
    
    riwayat_pribadi = Kunjungan.query.filter_by(mahasiswa_id=mhs.id).order_by(Kunjungan.waktu_kunjungan.desc()).all()
    total_kunjungan = len(riwayat_pribadi)
    total_baca = db.session.query(db.func.sum(Kunjungan.baca_buku_count)).filter_by(mahasiswa_id=mhs.id).scalar() or 0

    rekomendasi_buku = BukuRepository.query.filter(BukuRepository.detail_rak.ilike(f"%{mhs.prodi}%")).all()
    if not rekomendasi_buku:
        rekomendasi_buku = BukuRepository.query.order_by(BukuRepository.waktu_diupload.desc()).limit(3).all()

    grafik_kunjungan = Kunjungan.query.filter_by(mahasiswa_id=mhs.id).order_by(Kunjungan.waktu_kunjungan.asc()).suffix_with('').limit(7).all()
    labels_pribadi = [k.waktu_kunjungan.strftime('%d/%m') for k in grafik_kunjungan]
    data_pribadi = [k.baca_buku_count for k in grafik_kunjungan]

    hasil_html = ""
    if request.method == 'POST':
        ide_user = request.form['ide_ta']
        skripsi_db = SkripsiRepository.query.all()
        daftar_judul = [s.judul for s in skripsi_db]
        if daftar_judul and ide_user:
            vectorizer = TfidfVectorizer().fit_transform([ide_user] + daftar_judul)
            vectors = vectorizer.toarray()
            scores = cosine_similarity([vectors[0]], vectors[1:])[0]
            idx = np.argmin(scores) if len(scores) > 0 else 0
            persentase = round(scores[idx] * 100, 2) if len(scores) > 0 else 0
            skripsi_mirip = skripsi_db[idx] if skripsi_db else None
            
            hasil_html = f"""
            <div style="background: #fff5f5; padding: 20px; border-radius: 10px; margin-top: 20px; text-align: left; border-left: 5px solid #e74c3c;">
                <h4 style="color:#e74c3c; margin:0 0 10px 0;">Hasil Analisis Originalitas ({persentase}%)</h4>
                <p style="margin:5px 0;"><b>Judul Terdekat:</b> "{skripsi_mirip.judul if skripsi_mirip else 'N/A'}"</p>
                <p style="margin:5px 0;"><b>Rekomendasi Rak:</b> {skripsi_mirip.rak_lokasi if skripsi_mirip else 'N/A'}</p>
            </div>
            """

    extra_head = """
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    """

    content = f"""
    <div style="background: #fffae6; border: 1px solid #ffe58f; padding: 10px; border-radius: 8px; margin-bottom: 20px; color: #d46b08; font-size: 13px; font-weight: 600;">
        ⏱️ Pintu Masuk Terbuka! Halaman akan kembali ke layar utama dalam <span id="countdown" style="font-size:16px; color:#e74c3c;">5</span> detik...
    </div>

    <h2>Layanan Academic Advisor Mahasiswa</h2>
    <p>Selamat datang kembali, <b>{mhs.nama}</b> ({mhs.nim})</p>
    <p style="font-size:12px; margin-top:-5px; color:#34495e;">Fakultas / Program Studi: <span style="background:#e8f4fd; color:#4a90e2; padding:3px 8px; border-radius:5px; font-weight:600;">{mhs.prodi if mhs.prodi else 'Umum'}</span></p>
    
    <div style="display: flex; gap: 20px; margin: 25px 0; flex-wrap: wrap;">
        <div style="flex:1; min-width:180px; background:#f9fbfd; padding:20px; border-radius:12px; border:1px solid #e2e8f0;">
            <span style="font-size:12px; color:#7f8c8d;">TOTAL KUNJUNGAN PERPUS</span>
            <h2 style="margin:10px 0 0 0; color:#4a90e2;">{total_kunjungan} Kali</h2>
        </div>
        <div style="flex:1; min-width:180px; background:#f9fbfd; padding:20px; border-radius:12px; border:1px solid #e2e8f0;">
            <span style="font-size:12px; color:#7f8c8d;">ESTIMASI BUKU DIBACA</span>
            <h2 style="margin:10px 0 0 0; color:#2ecc71;">{total_baca} Buku</h2>
        </div>
        <div style="flex:1; min-width:180px; background:#fffdf9; padding:20px; border-radius:12px; border:1px solid #fbd5b5;">
            <span style="font-size:12px; color:#e67e22;">STATUS BEBAS PUSTAKA</span>
            <h4 style="margin:12px 0 0 0; color:#d35400; font-size:13px;">🟢 Memenuhi Syarat TA</h4>
        </div>
    </div>

    <div style="background: #fdfdfd; border: 1px solid #f0f0f0; padding: 20px; border-radius: 12px; margin-bottom: 25px; text-align: left;">
        <h3 style="color:#2c3e50; font-size:15px; margin:0 0 15px 0;">📈 Tren Intensitas Membaca Anda (7 Kunjungan Terakhir)</h3>
        <canvas id="studentLineChart" style="max-height: 180px; width: 100%;"></canvas>
    </div>

    <form method="POST" style="text-align: left; margin-top: 20px;">
        <label style="font-weight:600; font-size:13px;">Konsultasi Ide Judul TA (AI Advisor):</label>
        <textarea name="ide_ta" class="input-field" style="height:80px; resize:none;" placeholder="Ketik draf judul TA..." required></textarea>
        <button type="submit" class="btn btn-primary" style="margin:10px 0 0 0; width:100%;">Cek Duplikasi Judul</button>
    </form>
    {hasil_html}

    <div style="text-align: left; margin-top: 30px;">
        <h3 style="color:#4a90e2; font-size:15px; margin-bottom:10px;">🌟 Rekomendasi Koleksi Buku untuk Prodi Anda:</h3>
        <div style="display:flex; gap:15px; flex-wrap:wrap;">
            {"".join([f'''
            <div style="flex:1; min-width:200px; background:#fff; border:1px solid #e2e8f0; padding:15px; border-radius:10px; box-shadow: 0 2px 5px rgba(0,0,0,0.02);">
                <h4 style="margin:0 0 5px 0; color:#2c3e50; font-size:13px;">{b.judul_buku}</h4>
                <p style="margin:0; font-size:11px; color:#95a5a6;">Penulis: {b.penulis}</p>
                <span style="display:inline-block; margin-top:10px; font-size:10px; background:#f1f2f6; color:#57606f; padding:2px 6px; border-radius:4px; font-weight:600;">📍 {b.detail_rak}</span>
            </div>
            ''' for b in rekomendasi_buku]) if rekomendasi_buku else "<p style='font-size:12px; color:#bdc3c7;'>Belum tersedia buku khusus untuk prodi ini.</p>"}
        </div>
    </div>

    <div style="text-align: left; margin-top: 30px;">
        <h3 style="font-size:15px; color:#2c3e50;">📅 Riwayat Kehadiran Pribadi Anda</h3>
        <table id="rekapTable">
            <thead>
                <tr><th>Waktu Masuk Perpustakaan</th><th>Status Verifikasi</th><th>Jumlah Buku Dibaca</th></tr>
            </thead>
            <tbody>
                {"".join([f"<tr><td>{k.waktu_kunjungan.strftime('%d %B %Y, %H:%M')} WITA</td><td><span style='color:#2ecc71; font-weight:600;'>✔️ Face ID + Blink Success</span></td><td>{k.baca_buku_count} Buku</td></tr>" for k in riwayat_pribadi]) if riwayat_pribadi else "<tr><td colspan='3' style='text-align:center;'>Belum ada log riwayat presensi.</td></tr>"}
            </tbody>
        </table>
    </div>

    <br><br>
    <a href="/logout_mahasiswa" class="btn btn-secondary" style="width: 100%; box-sizing: border-box; margin:0;">Keluar Sekarang</a>

    <script>
        var seconds = 5;
        var countdownElement = document.getElementById("countdown");
        
        var interval = setInterval(function() {{
            seconds--;
            countdownElement.textContent = seconds;
            if (seconds <= 0) {{
                clearInterval(interval);
                window.location.href = "/logout_mahasiswa";
            }}
        }}, 1000);

        const ctxLine = document.getElementById('studentLineChart').getContext('2d');
        new Chart(ctxLine, {{
            type: 'line',
            data: {{
                labels: {json.dumps(labels_pribadi)},
                datasets: [{{
                    label: 'Jumlah Buku',
                    data: {json.dumps(data_pribadi)},
                    borderColor: '#4a90e2',
                    backgroundColor: 'rgba(74, 144, 226, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.3,
                    pointBackgroundColor: '#fff',
                    pointBorderColor: '#4a90e2',
                    pointRadius: 4
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    y: {{ beginAtZero: true, ticks: {{ stepSize: 1, font: {{ family: 'Poppins' }} }} }},
                    x: {{ ticks: {{ font: {{ family: 'Poppins' }} }} }}
                }}
            }}
        }});
    </script>
    """
    return aesthetic_layout("Dashboard Mahasiswa", content, show_success_animation=True, extra_head=extra_head)


# =========================================================================
# FEATURE BARU: ENDPOINT QR CODE SCAN MANDIRI VIA HP MAHASISWA (BYOD)
# =========================================================================
@app.route('/scan_meja/<int:no_meja>', methods=['GET', 'POST'])
def scan_meja(no_meja):
    # Jika mahasiswa mengirimkan form konfirmasi buku selesai baca dari HP-nya
    if request.method == 'POST':
        nim_input = request.form['nim']
        jumlah_buku = int(request.form['jumlah_buku'])
        
        # Cari data mahasiswa berdasarkan NIM
        mhs = Mahasiswa.query.filter_by(nim=nim_input).first()
        if mhs:
            # Cari baris kunjungan hari ini yang paling baru untuk mahasiswa tersebut
            kunjungan_hari_ini = Kunjungan.query.filter_by(mahasiswa_id=mhs.id).order_by(Kunjungan.waktu_kunjungan.desc()).first()
            if kunjungan_hari_ini:
                kunjungan_hari_ini.baca_buku_count = jumlah_buku
                db.session.commit()
                
                content_sukses = f"""
                <div style='color: #2ecc71; font-weight: 600; font-size: 16px; margin-bottom: 20px;'>✔️ RESPONS BERHASIL DISIMPAN!</div>
                <p>Terima kasih telah berkontribusi menjaga budaya baca mandiri di Meja {no_meja}.</p>
                <p>Data Anda telah otomatis terekam ke sistem pusat perpustakaan.</p>
                """
                return aesthetic_layout("Konfirmasi Sukses", content_sukses, show_success_animation=True)
        
        return "NIM tidak terdaftar atau Anda belum melakukan Face ID di pintu masuk!", 400

    # Layout Form Tampilan Web yang muncul di HP Mahasiswa saat scan QR meja
    katalog_buku = BukuRepository.query.all()
    content_hp = f"""
    <h3 style="color:#4a90e2;">Portal Baca Mandiri (Meja {no_meja:02d})</h3>
    <p>Silakan validasi aktivitas membaca Anda untuk rekap otomatis bebas pustaka Tugas Akhir.</p>
    
    <form method="POST" style="text-align: left; margin-top: 20px;">
        <label style="font-weight:600; font-size:12px;">Masukkan NIM Anda:</label>
        <input type="text" name="nim" class="input-field" placeholder="Contoh: 2101010099" required>
        
        <label style="font-weight:600; font-size:12px; display:block; margin-top:15px;">Berapa total buku yang Anda baca di meja ini?</label>
        <select name="jumlah_buku" class="input-field" required>
            <option value="1">1 Buku</option>
            <option value="2">2 Buku</option>
            <option value="3">3 Buku</option>
            <option value="4">4+ Buku</option>
        </select>
        
        <button type="submit" class="btn btn-success" style="width:100%; margin:20px 0 0 0;">Kirim Laporan Aktivitas</button>
    </form>
    """
    return aesthetic_layout(f"Scan Meja {no_meja}", content_hp)


# =========================================================================
# DASHBOARD ADMIN (CRUD EDIT MAHASISWA & ADMIN ANALYTICS)
# =========================================================================
@app.route('/admin_dashboard', methods=['GET', 'POST'])
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    if request.method == 'POST' and 'tambah_buku' in request.form:
        judul = request.form['judul_buku']
        penulis = request.form['penulis']
        rak = request.form['detail_rak']
        file = request.files['file_buku']
        
        file_path = None
        if file and file.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

        buku_baru = BukuRepository(judul_buku=judul, penulis=penulis, detail_rak=rak, file_path=file_path)
        db.session.add(buku_baru)
        db.session.commit()

    if request.method == 'POST' and 'edit_mahasiswa' in request.form:
        mhs_id = request.form['mhs_id']
        mhs_diubah = Mahasiswa.query.get(mhs_id)
        if mhs_diubah:
            mhs_diubah.nama = request.form['nama_baru']
            mhs_diubah.nim = request.form['nim_baru']
            mhs_diubah.prodi = request.form['prodi_baru']
            db.session.commit()
            reload_face_encodings()

    filter_type = request.args.get('filter', 'all')
    query_kunjungan = Kunjungan.query
    if filter_type == 'week':
        query_kunjungan = query_kunjungan.filter(Kunjungan.waktu_kunjungan >= datetime.now() - timedelta(days=7))
    elif filter_type == 'month':
        query_kunjungan = query_kunjungan.filter(Kunjungan.waktu_kunjungan >= datetime.now() - timedelta(days=30))
    elif filter_type == 'year':
        query_kunjungan = query_kunjungan.filter(Kunjungan.waktu_kunjungan >= datetime.now() - timedelta(days=365))
    
    list_kunjungan = query_kunjungan.order_by(Kunjungan.waktu_kunjungan.desc()).all()
    daftar_mahasiswa = Mahasiswa.query.all()

    rekap_fakultas = db.session.query(Mahasiswa.prodi, db.func.count(Kunjungan.id))\
                       .join(Kunjungan, Mahasiswa.id == Kunjungan.mahasiswa_id)\
                       .group_by(Mahasiswa.prodi).all()
    
    labels_chart = [str(f[0]) if f[0] else "Umum" for f in rekap_fakultas]
    data_chart = [int(f[1]) for f in rekap_fakultas]

    extra_head = """
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    """

    content = f"""
    <div style="text-align: left;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <h2>Panel Kontrol Data Perpustakaan</h2>
            <a href="/admin_logout" class="btn btn-danger no-print" style="padding:6px 15px; font-size:12px;">Logout</a>
        </div>
        <p>Akses Khusus Staf Pengelola Perpustakaan Pusat</p>
        <hr style="border:0; border-top:1px solid #f0f0f0; margin:20px 0;">
        
        <h3 class="no-print" style="color:#e67e22;">[📊] Analitik Tren Fakultas (Real-time Chart)</h3>
        <div class="charts-grid">
            <div class="chart-container">
                <h4 style="text-align:center; font-size:12px; margin-bottom:10px;">Persentase Kunjungan (Pie)</h4>
                <canvas id="pieChart" style="max-height: 230px;"></canvas>
            </div>
            <div class="chart-container">
                <h4 style="text-align:center; font-size:12px; margin-bottom:10px;">Perbandingan Jumlah (Bar Horizontal)</h4>
                <canvas id="barChart" style="max-height: 230px;"></canvas>
            </div>
        </div>

        <hr class="no-print" style="border:0; border-top:1px solid #f0f0f0; margin:30px 0;">

        <div class="no-print" style="margin-bottom:30px;">
            <h3 style="color:#e67e22;">[✏️] CRUD: Update Data Mahasiswa / Fakultas</h3>
            <form method="POST" style="background:#fffcf9; padding:20px; border-radius:12px; border:1px solid #fbd5b5;">
                <input type="hidden" name="edit_mahasiswa" value="1">
                <label style="font-size:12px; font-weight:600;">Pilih Mahasiswa Berdasarkan Foto ID:</label>
                <select name="mhs_id" class="input-field" required>
                    {"".join([f"<option value='{m.id}'>{m.nama} ({m.nim})</option>" for m in daftar_mahasiswa])}
                </select>
                <input type="text" name="nama_baru" class="input-field" placeholder="Ketik Nama Baru" required>
                <input type="text" name="nim_baru" class="input-field" placeholder="Ketik NIM Baru" required>
                <input type="text" name="prodi_baru" class="input-field" placeholder="Ketik Fakultas / Prodi Baru" required>
                <button type="submit" class="btn btn-primary" style="background:#e67e22; margin-top:10px; display:block;">Perbarui Data Mahasiswa</button>
            </form>
        </div>

        <div class="no-print">
            <h3 style="color:#4a90e2;">[+] CRUD: Input Koleksi Buku Baru</h3>
            <form method="POST" enctype="multipart/form-data" style="background:#f9fbfd; padding:20px; border-radius:12px; border:1px solid #e2e8f0; margin-bottom:30px;">
                <input type="hidden" name="tambah_buku" value="1">
                <input type="text" name="judul_buku" class="input-field" placeholder="Judul Buku / E-Book" required>
                <input type="text" name="penulis" class="input-field" placeholder="Nama Penulis" required>
                <input type="text" name="detail_rak" class="input-field" placeholder="Lokasi Fisik Kode Rak (Contoh: RAK-05B)" required>
                <label style="font-size:12px; color:#7f8c8d; display:block; margin:10px 0 5px 0;">Unggah Dokumen Buku Digital (Opsional):</label>
                <input type="file" name="file_buku">
                <button type="submit" class="btn btn-primary" style="margin-top:15px; display:block;">Simpan Koleksi Buku</button>
            </form>
        </div>

        <h3>[=] Analitik Rekapitulasi Kehadiran</h3>
        <div style="margin: 15px 0; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;" class="no-print">
            <div>
                <span style="font-size:12px; margin-right:10px;">Periode Filter:</span>
                <a href="/admin_dashboard?filter=all" class="btn btn-secondary" style="padding:4px 12px; font-size:11px;">Semua</a>
                <a href="/admin_dashboard?filter=week" class="btn btn-primary" style="padding:4px 12px; font-size:11px;">Mingguan</a>
                <a href="/admin_dashboard?filter=month" class="btn btn-primary" style="padding:4px 12px; font-size:11px;">Bulanan</a>
                <a href="/admin_dashboard?filter=year" class="btn btn-primary" style="padding:4px 12px; font-size:11px;">Tahunan</a>
            </div>
            <div>
                <button onclick="window.print()" class="btn btn-success" style="padding:6px 15px; font-size:11px; font-weight:600;">🖨️ Cetak PDF</button>
                <button onclick="downloadExcel()" class="btn btn-success" style="padding:6px 15px; font-size:11px; font-weight:600; background:#34495e;">📊 Unduh Excel</button>
            </div>
        </div>
        
        <table id="rekapTable">
            <thead>
                <tr><th>Nama Lengkap</th><th>NIM</th><th>Waktu Presensi</th><th>Buku Dibaca</th></tr>
            </thead>
            <tbody>
                {"".join([f"<tr><td><b>{k.mahasiswa.nama}</b></td><td>{k.mahasiswa.nim}</td><td>{k.waktu_kunjungan.strftime('%d %B %Y, %H:%M')} WITA</td><td>{k.baca_buku_count} Buku</td></tr>" for k in list_kunjungan]) if list_kunjungan else "<tr><td colspan='4' style='text-align:center;'>Belum ada riwayat rekap log kunjungan.</td></tr>"}
            </tbody>
        </table>
    </div>

    <script class="no-print">
        const ctxPie = document.getElementById('pieChart').getContext('2d');
        const ctxBar = document.getElementById('barChart').getContext('2d');
        
        const labelsData = {json.dumps(labels_chart)};
        const chartValues = {json.dumps(data_chart)};
        
        const colors = ['#4a90e2', '#2ecc71', '#e67e22', '#e74c3c', '#9b59b6', '#1abc9c', '#f1c40f'];

        new Chart(ctxPie, {{
            type: 'pie',
            data: {{
                labels: labelsData,
                datasets: [{{
                    data: chartValues,
                    backgroundColor: colors,
                    borderWidth: 2
                }}]
            }},
            options: {{ responsive: true, plugins: {{ legend: {{ labels: {{ font: {{ family: 'Poppins' }} }} }} }} }}
        }});

        new Chart(ctxBar, {{
            type: 'bar',
            data: {{
                labels: labelsData,
                datasets: [{{
                    label: 'Jumlah Kunjungan',
                    data: chartValues,
                    backgroundColor: '#4a90e2',
                    borderRadius: 6
                }}]
            }},
            options: {{
                indexAxis: 'y',
                responsive: true,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{ x: {{ ticks: {{ font: {{ family: 'Poppins' }} }} }}, y: {{ ticks: {{ font: {{ family: 'Poppins' }} }} }} }}
            }}
        }});
    </script>
    """
    return aesthetic_layout("Dashboard Admin Control Center", content, show_success_animation=True, extra_head=extra_head)

if __name__ == '__main__':
    app.run(debug=True, port=5000)