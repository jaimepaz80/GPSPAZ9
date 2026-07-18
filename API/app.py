import os
import math
import datetime
import urllib.request
import gzip
import shutil
import ssl
import json
import threading
import re
import functools
from flask import Flask, request, send_file, Response
import tempfile

app = Flask(__name__)

# --- RUTA DINÁMICA DE TRABAJO (EDICIÓN VERCEL SERVERLESS) ---
BASE_DIR = tempfile.gettempdir()

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'temp_rinex')
REPORT_FOLDER = os.path.join(BASE_DIR, 'informes')
STATE_FILE = os.path.join(UPLOAD_FOLDER, 'estado_proyecto.json')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

STATE_LOCK = threading.Lock()

# --- CONSTANTES GEODÉSICAS ---
C_LIGHT = 299792458.0
OMEGA_E = 7.2921151467e-5
MU = 3.986005e14
FREQ_L1 = 1575.42e6
FREQ_L5 = 1176.45e6
WAVE_L1 = C_LIGHT / FREQ_L1
WAVE_L5 = C_LIGHT / FREQ_L5

# --- FORMATEADOR DE ALTA PRECISIÓN ---
def f_14(val):
    if val is None: return "0.0"
    s = f"{val:.14f}"
    if '.' in s:
        s = s.rstrip('0')
        if s.endswith('.'): s += '0'
    return s

def safe_f(val, default=0.0):
    try: return float(val) if val and str(val).strip() != '' else default
    except: return default

def safe_i(val, default=19):
    try: return int(val) if val and str(val).strip() != '' else default
    except: return default

def guardar_estado(clave, valor):
    with STATE_LOCK:
        estado = {}
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r', encoding='utf-8') as f: estado = json.load(f)
            except: pass
        estado[clave] = valor
        try:
            with open(STATE_FILE, 'w', encoding='utf-8') as f: json.dump(estado, f)
        except: pass

def leer_estado(clave):
    with STATE_LOCK:
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r', encoding='utf-8') as f: return json.load(f).get(clave)
            except: pass
        return None

def gps_time_to_tow(year, month, day, hour, minute, second):
    sec_int, sec_frac = int(second), second - int(second)
    total = (datetime.datetime(year, month, day, hour, minute, sec_int) - datetime.datetime(1980, 1, 6)).total_seconds() + sec_frac
    return total - (int(total // 604800) * 604800)

# =====================================================================
# INTEGRACIÓN GOOGLE DRIVE
# =====================================================================
def descargar_desde_gdrive(url, filepath):
    match = re.search(r'/file/d/([a-zA-Z0-9_-]+)', url)
    if not match:
        match = re.search(r'id=([a-zA-Z0-9_-]+)', url)
    if not match:
        raise ValueError("URL de Google Drive no reconocida.")
    
    file_id = match.group(1)
    direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    req = urllib.request.Request(direct_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, context=ctx, timeout=120) as response, open(filepath, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    
    return True

# =====================================================================
# ÁLGEBRA LINEAL DE ESTADO SÓLIDO
# =====================================================================
def transpose_matrix(M):
    if not M or not M[0]: return []
    try: return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]
    except IndexError: return []

def matmul(A, B):
    if not A or not B or not A[0] or not B[0]: return []
    try:
        result = [[0.0 for _ in range(len(B[0]))] for _ in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B[0])):
                for k in range(len(B)):
                    result[i][j] += A[i][k] * B[k][j]
        return result
    except IndexError: return []

def matadd(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def matsub(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def matid(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

def cholesky_decompose(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            sum1 = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = A[i][i] - sum1
                if val <= 0: raise ValueError("Matriz no definida positiva")
                L[i][j] = math.sqrt(val)
            else:
                L[i][j] = (A[i][j] - sum1) / L[j][j]
    return L

def invert_lower_triangular(L):
    n = len(L)
    inv = [[0.0] * n for _ in range(n)]
    for i in range(n):
        inv[i][i] = 1.0 / L[i][i]
        for j in range(i):
            sum1 = sum(L[i][k] * inv[k][j] for k in range(j, i))
            inv[i][j] = -sum1 / L[i][i]
    return inv

def gauss_jordan_inverse(M):
    n = len(M)
    A = [[float(M[i][j]) for j in range(n)] for i in range(n)]
    I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for i in range(n):
        max_k = i
        for k in range(i + 1, n):
            if abs(A[k][i]) > abs(A[max_k][i]): max_k = k
        if max_k != i:
            A[i], A[max_k] = A[max_k], A[i]
            I[i], I[max_k] = I[max_k], I[i]
        pivot = A[i][i]
        if abs(pivot) < 1e-15: return None 
        for j in range(n):
            A[i][j] /= pivot
            I[i][j] /= pivot
        for k in range(n):
            if k == i: continue
            factor = A[k][i]
            for j in range(n):
                A[k][j] -= factor * A[i][j]
                I[k][j] -= factor * I[i][j]
    return I

def invert_matrix_nxn(M):
    if not M or not M[0]: return None
    try:
        L = cholesky_decompose(M)
        L_inv = invert_lower_triangular(L)
        return matmul(transpose_matrix(L_inv), L_inv)
    except:
        return gauss_jordan_inverse(M)

# =====================================================================
# PARSERS Y GESTIÓN DE ARCHIVOS (EXTRACCIÓN SMARTPHONE L1/L5)
# =====================================================================
def parse_rinex_obs_completo(path):
    obs = {}
    sys_idx = {}
    sys_tokens = {}
    last_sys_char = None
    
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        in_h = True
        tow = None
        for line in f:
            if in_h:
                if "SYS / # / OBS TYPES" in line:
                    sys_char = line[0].strip()
                    if sys_char: last_sys_char = sys_char
                    if last_sys_char:
                        tokens = [x.strip() for x in line[6:60].split() if x.strip()]
                        sys_tokens.setdefault(last_sys_char, []).extend(tokens)
                elif "END OF HEADER" in line: 
                    in_h = False
                    for sc, t in sys_tokens.items():
                        sys_idx[sc] = {
                            'C1': next((i for i, x in enumerate(t) if x.startswith('C1') or x.startswith('C2')), -1),
                            'L1': next((i for i, x in enumerate(t) if x.startswith('L1') or x.startswith('L2')), -1),
                            'C5': next((i for i, x in enumerate(t) if x.startswith('C5') or x.startswith('C7')), -1),
                            'L5': next((i for i, x in enumerate(t) if x.startswith('L5') or x.startswith('L7')), -1),
                            'S1': next((i for i, x in enumerate(t) if x.startswith('S1') or x.startswith('S2')), -1),
                            'S5': next((i for i, x in enumerate(t) if x.startswith('S5') or x.startswith('S7')), -1)
                        }
            elif line.startswith('>'):
                p = line[1:].split()
                if len(p) >= 6:
                    y, m, d, h, mn, sec = int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4]), float(p[5])
                    tow = round(gps_time_to_tow(y, m, d, h, mn, sec), 6)
                    obs[tow] = {'_meta': (y, m, d, h, mn, sec)}
            elif tow and len(line) > 3 and line[0] in 'GRECSJ':
                sys_char = line[0]
                idx_c1 = sys_idx.get(sys_char, {}).get('C1', -1)
                idx_l1 = sys_idx.get(sys_char, {}).get('L1', -1)
                idx_c5 = sys_idx.get(sys_char, {}).get('C5', -1)
                idx_l5 = sys_idx.get(sys_char, {}).get('L5', -1)
                idx_s1 = sys_idx.get(sys_char, {}).get('S1', -1)
                idx_s5 = sys_idx.get(sys_char, {}).get('S5', -1)
                
                data = {}
                if idx_c1 >= 0 and len(line) >= 17 + 16 * idx_c1:
                    v = line[3+16*idx_c1 : 17+16*idx_c1].strip()
                    if v: data['C1'] = float(v.replace('D', 'E').replace('d', 'e'))
                if idx_c5 >= 0 and len(line) >= 17 + 16 * idx_c5:
                    v = line[3+16*idx_c5 : 17+16*idx_c5].strip()
                    if v: data['C5'] = float(v.replace('D', 'E').replace('d', 'e'))
                if idx_l1 >= 0 and len(line) >= 17 + 16 * idx_l1:
                    v = line[3+16*idx_l1 : 17+16*idx_l1].strip()
                    if v: data['L1'] = float(v.replace('D', 'E').replace('d', 'e'))
                if idx_l5 >= 0 and len(line) >= 17 + 16 * idx_l5:
                    v = line[3+16*idx_l5 : 17+16*idx_l5].strip()
                    if v: data['L5'] = float(v.replace('D', 'E').replace('d', 'e'))
                if idx_s1 >= 0 and len(line) >= 17 + 16 * idx_s1:
                    v = line[3+16*idx_s1 : 17+16*idx_s1].strip()
                    if v: data['S1'] = float(v.replace('D', 'E').replace('d', 'e'))
                if idx_s5 >= 0 and len(line) >= 17 + 16 * idx_s5:
                    v = line[3+16*idx_s5 : 17+16*idx_s5].strip()
                    if v: data['S5'] = float(v.replace('D', 'E').replace('d', 'e'))
                
                valid_p = ('C1' in data and data['C1'] > 15000000.0) or ('C5' in data and data['C5'] > 15000000.0)
                if valid_p:
                    obs.setdefault(tow, {})[line[0:3].strip()] = data
    return obs

def interpolar_base_a_rover(obs_base, tr, max_gap=0.05):
    tiempos_base = sorted(list(obs_base.keys()))
    if not tiempos_base: return None
    idx = min(range(len(tiempos_base)), key=lambda i: abs(tiempos_base[i] - tr))
    if abs(tiempos_base[idx] - tr) <= max_gap: return obs_base[tiempos_base[idx]].copy()
    return None

def generar_rinex_sincronizado(raw_path, out_path, obs_dict):
    header_lines = []
    with open(raw_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if "SYS / # / OBS TYPES" in line: continue 
            header_lines.append(line)
            if "END OF HEADER" in line: break
    
    idx = next((i for i, l in enumerate(header_lines) if "END OF HEADER" in l), -1)
    if idx != -1:
        constelaciones_requeridas = ['G', 'E', 'C', 'R', 'S', 'J']
        offset = 0
        for c in constelaciones_requeridas:
            header_lines.insert(idx + offset, f"{c}    4 C1 L1 C5 L5                                       SYS / # / OBS TYPES\n")
            offset += 1
            
    with open(out_path, 'w', encoding='utf-8') as f_out:
        for line in header_lines: f_out.write(line)
        for tow in sorted(obs_dict.keys()):
            meta = obs_dict[tow].get('_meta')
            if not meta: continue
            y, m, d, h, mn, sec = meta
            sats = [k for k in obs_dict[tow].keys() if k != '_meta']
            f_out.write(f"> {y} {m:02d} {d:02d} {h:02d} {mn:02d} {sec:11.7f}  0 {len(sats):2d}\n")
            for sat in sats:
                c1, l1 = obs_dict[tow][sat].get('C1', 0.0), obs_dict[tow][sat].get('L1', 0.0)
                c5, l5 = obs_dict[tow][sat].get('C5', 0.0), obs_dict[tow][sat].get('L5', 0.0)
                c1_s = f"{c1:14.3f}" if c1 > 0 else "              "
                l1_s = f"{l1:14.3f}" if l1 > 0 else "              "
                c5_s = f"{c5:14.3f}" if c5 > 0 else "              "
                l5_s = f"{l5:14.3f}" if l5 > 0 else "              "
                f_out.write(f"{sat}{c1_s}  {l1_s}  {c5_s}  {l5_s}  \n")

def obtener_fecha_obs(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('>'):
                partes = line[1:].strip().split()
                if len(partes) >= 6: 
                    try:
                        y = int(partes[0])
                        return (y if y>100 else y+2000), int(partes[1]), int(partes[2]), int(partes[3]), int(partes[4]), float(partes[5])
                    except: pass
    return None

# =====================================================================
# PRODUCTOS IGS Y EFEMÉRIDES
# =====================================================================
SP3_CACHE = {}
SP3_CACHE_KEYS = []
MAX_CACHE_SIZE = 2048

def parse_sp3_preciso(path):
    sp3_data = {}
    if not path or not os.path.exists(path): return sp3_data
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        current_time = None
        for line in f:
            if line.startswith('* '):
                p = line.split()
                if len(p) >= 7:
                    try:
                        y, m, d, h, mn, s = int(p[1]), int(p[2]), int(p[3]), int(p[4]), int(p[5]), float(p[6])
                        current_time = gps_time_to_tow(y, m, d, h, mn, s)
                    except: pass
            elif line.startswith('P') and current_time:
                sys_char = line[1]
                if sys_char in 'GECR':
                    sat_id = line[1:4].strip()
                    try:
                        x = float(line[4:18]) * 1000.0
                        y = float(line[18:32]) * 1000.0
                        z = float(line[32:46]) * 1000.0
                        clk = float(line[46:60]) / 1e6 if len(line)>46 and line[46:60].strip() else 0.0
                        sp3_data.setdefault(sat_id, []).append((current_time, x, y, z, clk))
                    except: pass
    for sat in sp3_data: sp3_data[sat].sort(key=lambda item: item[0])
    return sp3_data

def lagrange_interpolate(x, x_pts, y_pts):
    n = len(x_pts); val = 0.0
    for i in range(n):
        p = 1.0
        for j in range(n):
            if i != j: p *= (x - x_pts[j]) / (x_pts[i] - x_pts[j])
        val += y_pts[i] * p
    return val

def interpolate_sp3(sp3_data, sat, t_emision, degree=9):
    global SP3_CACHE, SP3_CACHE_KEYS
    cache_key = f"{sat}_{t_emision}"
    
    if cache_key in SP3_CACHE: return SP3_CACHE[cache_key]

    if sat not in sp3_data: return None
    data = sp3_data[sat]
    if len(data) < degree + 1: return None
    
    idx = min(range(len(data)), key=lambda i: abs(data[i][0] - t_emision))
    half = degree // 2
    start = max(0, idx - half)
    end = min(len(data), start + degree + 1)
    if end - start < degree + 1: start = max(0, end - degree - 1)
    pts = data[start:end]
    
    t_pts = [p[0] for p in pts]; x_pts = [p[1] for p in pts]
    y_pts = [p[2] for p in pts]; z_pts = [p[3] for p in pts]
    clk_pts = [p[4] for p in pts]
    
    result = (
        lagrange_interpolate(t_emision, t_pts, x_pts),
        lagrange_interpolate(t_emision, t_pts, y_pts),
        lagrange_interpolate(t_emision, t_pts, z_pts),
        lagrange_interpolate(t_emision, t_pts, clk_pts)
    )
    
    if len(SP3_CACHE) >= MAX_CACHE_SIZE:
        oldest_key = SP3_CACHE_KEYS.pop(0)
        SP3_CACHE.pop(oldest_key, None)
        
    SP3_CACHE[cache_key] = result
    SP3_CACHE_KEYS.append(cache_key)
    return result

def parse_rinex_nav_real(path):
    ephemeris = {'_iono': {'alpha': [0]*4, 'beta': [0]*4}}
    if not path or not os.path.exists(path): return ephemeris
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        in_h, sat, data = True, None, []
        for line in f:
            if in_h:
                if "IONOSPHERIC CORR" in line:
                    sys_type = line[0:4].strip()
                    vals = []
                    for i in range(4):
                        try:
                            chunk = line[5+i*12 : 5+(i+1)*12].strip().replace('D', 'E').replace('d', 'e')
                            vals.append(float(chunk) if chunk else 0.0)
                        except: vals.append(0.0)
                    if sys_type == 'GPSA': ephemeris['_iono']['alpha'] = vals
                    elif sys_type == 'GPSB': ephemeris['_iono']['beta'] = vals
                elif "END OF HEADER" in line: in_h = False
                continue
            if len(line) > 8 and line[0] in 'GECSJ' and line[1:3].isdigit():
                if sat and len(data) >= 20: 
                    ephemeris.setdefault(sat, []).append({'af0':data[0],'af1':data[1],'af2':data[2],'Crs':data[4],'Delta_n':data[5],'M0':data[6],'Cuc':data[7],'e':data[8],'Cus':data[9],'sqrtA':data[10],'Toe':data[11],'Cic':data[12],'OMEGA':data[13],'Cis':data[14],'i0':data[15],'Crc':data[16],'omega':data[17],'OMEGA_DOT':data[18],'IDOT':data[19]})
                sat = line[0:3].strip()
                data = [float(line[23:42].replace('D','E').replace('d','e')), float(line[42:61].replace('D','E').replace('d','e')), float(line[61:80].replace('D','E').replace('d','e'))]
            elif sat and line.startswith('    '): 
                data.extend([float(line[i:i+19].replace('D','E').replace('d','e').strip()) for i in range(4, 80, 19) if line[i:i+19].strip()])
        if sat and len(data) >= 20: 
            ephemeris.setdefault(sat, []).append({'af0':data[0],'af1':data[1],'af2':data[2],'Crs':data[4],'Delta_n':data[5],'M0':data[6],'Cuc':data[7],'e':data[8],'Cus':data[9],'sqrtA':data[10],'Toe':data[11],'Cic':data[12],'OMEGA':data[13],'Cis':data[14],'i0':data[15],'Crc':data[16],'omega':data[17],'OMEGA_DOT':data[18],'IDOT':data[19]})
    return ephemeris

def seleccionar_efemeride_optima(eph_list, t_target):
    if not eph_list: return None
    return min(eph_list, key=lambda x: abs(x.get('Toe', 0) - t_target))

# =====================================================================
# GEODESIA ESPACIAL Y CORRECCIONES
# =====================================================================
def correccion_mareas_solidas(X, Y, Z, tow, year, month, day):
    try:
        h2, l2 = 0.609, 0.085
        Re = 6378137.0
        GM_earth, GM_sun, GM_moon = 3.986004418e14, 1.327124e20, 4.902801e12
        jd = 367 * year - (7 * (year + (month + 9) // 12)) // 4 + (275 * month) // 9 + day + 1721013.5
        t_jc = (jd - 2451545.0 + (tow / 86400.0)) / 36525.0
        
        mean_long_sun = 280.460 + 36000.771 * t_jc
        mean_anom_sun = 357.528 + 35999.050 * t_jc
        ecl_lon_sun = mean_long_sun + 1.915 * math.sin(math.radians(mean_anom_sun)) + 0.020 * math.sin(math.radians(2 * mean_anom_sun))
        dist_sun = 1.495978707e11 * (1.00014 - 0.01671 * math.cos(math.radians(mean_anom_sun)) - 0.00014 * math.cos(math.radians(2 * mean_anom_sun)))
        obliquity = 23.439 - 0.013 * t_jc
        
        xs_sun = dist_sun * math.cos(math.radians(ecl_lon_sun))
        ys_sun = dist_sun * math.cos(math.radians(obliquity)) * math.sin(math.radians(ecl_lon_sun))
        zs_sun = dist_sun * math.sin(math.radians(obliquity)) * math.sin(math.radians(ecl_lon_sun))
        
        mean_long_moon = 218.316 + 481267.881 * t_jc
        mean_anom_moon = 134.963 + 477198.867 * t_jc
        mean_dist_moon = 93.272 + 483202.017 * t_jc
        ecl_lon_moon = mean_long_moon + 6.289 * math.sin(math.radians(mean_anom_moon))
        ecl_lat_moon = 5.128 * math.sin(math.radians(mean_dist_moon))
        dist_moon = 385000000.0 - 20905000.0 * math.cos(math.radians(mean_anom_moon))
        
        xs_moon = dist_moon * math.cos(math.radians(ecl_lon_moon)) * math.cos(math.radians(ecl_lat_moon))
        ys_moon = dist_moon * (math.cos(math.radians(obliquity)) * math.sin(math.radians(ecl_lon_moon)) * math.cos(math.radians(ecl_lat_moon)) - math.sin(math.radians(obliquity)) * math.sin(math.radians(ecl_lat_moon)))
        zs_moon = dist_moon * (math.sin(math.radians(obliquity)) * math.sin(math.radians(ecl_lon_moon)) * math.cos(math.radians(ecl_lat_moon)) + math.cos(math.radians(obliquity)) * math.sin(math.radians(ecl_lat_moon)))
        
        r_sta = math.sqrt(X**2 + Y**2 + Z**2)
        if r_sta == 0: return 0.0, 0.0, 0.0
        rx, ry, rz = X/r_sta, Y/r_sta, Z/r_sta
        
        def deformacion_cuerpo(mass_ratio, R_body, xs, ys, zs):
            dist_body = math.sqrt(xs**2 + ys**2 + zs**2)
            if dist_body == 0: return 0.0, 0.0, 0.0
            ux, uy, uz = xs/dist_body, ys/dist_body, zs/dist_body
            cos_theta = rx*ux + ry*uy + rz*uz
            p2 = 1.5 * cos_theta**2 - 0.5
            p2_prime = 3.0 * cos_theta
            coef = (GM_earth / Re**2) * mass_ratio * (Re / dist_body)**3 * Re
            dr_radial = h2 * coef * p2
            dr_tangent = l2 * coef * p2_prime
            return dr_radial * rx + dr_tangent * (ux - cos_theta * rx), dr_radial * ry + dr_tangent * (uy - cos_theta * ry), dr_radial * rz + dr_tangent * (uz - cos_theta * rz)

        dx_sun, dy_sun, dz_sun = deformacion_cuerpo(GM_sun/GM_earth, dist_sun, xs_sun, ys_sun, zs_sun)
        dx_moon, dy_moon, dz_moon = deformacion_cuerpo(GM_moon/GM_earth, dist_moon, xs_moon, ys_moon, zs_moon)
        return dx_sun + dx_moon, dy_sun + dy_moon, dz_sun + dz_moon
    except: return 0.0, 0.0, 0.0 

def calcular_saastamoinen(lat_deg, alt, elev_deg):
    if elev_deg < 5.0: elev_deg = 5.0
    lat_rad, elev_rad = max(math.radians(lat_deg), -math.pi/2), math.radians(elev_deg)
    H = max(0.0, min(alt, 40000.0))
    P = 1013.25 * ((1.0 - 2.2557e-5 * H) ** 5.2568)
    T = 288.15 - 0.0065 * H
    e = 6.11 * 0.5 * (10.0 ** (7.5 * (T - 273.15) / (T - 273.15 + 237.3))) * ((1.0 - 2.2557e-5 * H) ** 5.2568)
    zhd = (0.0022768 * P) / (1.0 - 0.00266 * math.cos(2.0 * lat_rad) - 0.00028 * (H / 1000.0))
    zwd = 0.0022768 * ((1255.0 / T) + 0.05) * e
    return (zhd + zwd) * (1.0 / math.sin(elev_rad))

def geodesicas_a_ecef(lat_deg, lon_deg, alt):
    a, e2 = 6378137.0, 0.0066943799901413155
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    N = a / math.sqrt(1 - e2 * (math.sin(lat) ** 2))
    return (N + alt) * math.cos(lat) * math.cos(lon), (N + alt) * math.cos(lat) * math.sin(lon), (N * (1 - e2) + alt) * math.sin(lat)

def ecef_a_geodesicas(x, y, z):
    a, e2 = 6378137.0, 0.0066943799901413155
    b = math.sqrt(a**2 * (1 - e2)); ep2 = (a**2 - b**2) / b**2
    p = math.sqrt(x**2 + y**2); th = math.atan2(a * z, b * p)
    lat = math.atan2((z + ep2 * b * (math.sin(th) ** 3)), (p - e2 * a * (math.cos(th) ** 3)))
    N = a / math.sqrt(1 - e2 * (math.sin(lat) ** 2))
    return math.degrees(lat), math.degrees(math.atan2(y, x)), p / math.cos(lat) - N

def geodesicas_a_utm(lat, lon, force_zone=19):
    a, e2 = 6378137.0, 0.0066943799901413155
    lat_r, lon_r = math.radians(lat), math.radians(lon)
    LongOrig = math.radians((force_zone - 1) * 6 - 180 + 3)
    ep2 = e2 / (1 - e2)
    N = a / math.sqrt(1 - e2 * math.sin(lat_r)**2)
    T = math.tan(lat_r)**2; C = ep2 * math.cos(lat_r)**2; A = math.cos(lat_r) * (lon_r - LongOrig)
    M = a * ((1 - e2/4 - 3*e2**2/64 - 5*e2**3/256)*lat_r - (3*e2/8 + 3*e2**2/32 + 45*e2**3/1024)*math.sin(2*lat_r) + (15*e2**2/256 + 45*e2**3/1024)*math.sin(4*lat_r) - (35*e2**3/3072)*math.sin(6*lat_r))
    Easting = 0.9996 * N * (A + (1-T+C)*A**3/6 + (5-18*T+T**2+72*C-58*ep2)*A**5/120) + 500000.0
    Northing = 0.9996 * (M + N*math.tan(lat_r)*(A**2/2 + (5-T+9*C+4*C**2)*A**4/24 + (61-58*T+T**2+600*C-330*ep2)*A**6/720))
    return (Northing + 10000000.0 if lat < 0 else Northing), Easting

def utm_a_geodesicas(easting, northing, zone=19, hemisferio='N'):
    a, e2 = 6378137.0, 0.0066943799901413155
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    x, y = easting - 500000.0, northing if hemisferio.upper() == 'N' else northing - 10000000.0
    m = y / 0.9996; mu = m / (a * (1 - e2/4 - 3*e2**2/64 - 5*e2**3/256))
    phi1_rad = mu + (3*e1/2 - 27*e1**3/32)*math.sin(2*mu) + (21*e1**2/16 - 55*e1**4/32)*math.sin(4*mu)
    n1 = a / math.sqrt(1 - e2*math.sin(phi1_rad)**2)
    t1, c1 = math.tan(phi1_rad)**2, e2 / (1 - e2) * math.cos(phi1_rad)**2
    r1 = a * (1 - e2) / ((1 - e2*math.sin(phi1_rad)**2)**1.5)
    d = x / (n1 * 0.9996)
    lat_rad = phi1_rad - (n1*math.tan(phi1_rad)/r1) * (d**2/2 - (5 + 3*t1 + 10*c1)*d**4/24)
    lon_rad = (d - (1 + 2*t1 + c1)*d**3/6) / math.cos(phi1_rad)
    lon_origen = math.radians((zone - 1) * 6 - 180 + 3)
    return math.degrees(lat_rad), math.degrees(lon_rad + lon_origen), 0.0

def calcular_topocentricas(xs, ys, zs, X_usr, Y_usr, Z_usr):
    lat_val, lon_val, alt_val = ecef_a_geodesicas(X_usr, Y_usr, Z_usr)
    lat_r, lon_r = math.radians(lat_val), math.radians(lon_val)
    dx, dy, dz = xs - X_usr, ys - Y_usr, zs - Z_usr
    sin_lat, cos_lat = math.sin(lat_r), math.cos(lat_r)
    sin_lon, cos_lon = math.sin(lon_r), math.cos(lon_r)
    e = -sin_lon * dx + cos_lon * dy
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    dist = math.sqrt(dx**2 + dy**2 + dz**2)
    if dist < 1e-6: return 0.0, 0.0
    val_asin = max(-1.0, min(1.0, u / dist))
    el = math.degrees(math.asin(val_asin))
    az = math.degrees(math.atan2(e, n))
    if az < 0: az += 360.0
    return el, az

def calcular_klobuchar(lat_deg, lon_deg, el_deg, az_deg, tow, alpha, beta):
    if not any(alpha) and not any(beta): return 0.0
    phi_u, lam_u = lat_deg / 180.0, lon_deg / 180.0
    E, A = el_deg / 180.0, az_deg / 180.0
    psi = 0.0137 / (E + 0.11) - 0.022
    phi_i = phi_u + psi * math.cos(A * math.pi)
    if phi_i > 0.416: phi_i = 0.416
    elif phi_i < -0.416: phi_i = -0.416
    lam_i = lam_u + (psi * math.sin(A * math.pi)) / math.cos(phi_i * math.pi)
    phi_m = phi_i + 0.064 * math.cos((lam_i - 1.617) * math.pi)
    t = 43200.0 * lam_i + tow
    t = t % 86400.0
    if t < 0: t += 86400.0
    F = 1.0 + 16.0 * (0.53 - E) ** 3
    PER = beta[0] + beta[1]*phi_m + beta[2]*(phi_m**2) + beta[3]*(phi_m**3)
    if PER < 72000.0: PER = 72000.0
    AMP = alpha[0] + alpha[1]*phi_m + alpha[2]*(phi_m**2) + alpha[3]*(phi_m**3)
    if AMP < 0.0: AMP = 0.0
    x = (2.0 * math.pi * (t - 50400.0)) / PER
    if abs(x) < 1.5707963267948966:
        return F * (5e-9 + AMP * (1.0 - (x**2)/2.0 + (x**4)/24.0)) * C_LIGHT
    return F * 5e-9 * C_LIGHT

def calcular_posicion_satelite_wgs84(eph, t_emision, tau_vuelo, sys_char='G'):
    if not eph or eph['sqrtA'] <= 0.0: return None
    mu_sys = 3.986004418e14 if sys_char in 'EC' else MU
    omega_e_sys = 7.292115e-5 if sys_char == 'C' else OMEGA_E
    A = eph['sqrtA'] ** 2
    n0 = math.sqrt(mu_sys / (A ** 3))
    t_k = t_emision - eph['Toe']
    if sys_char == 'C': t_k -= 14.0
    if t_k > 302400: t_k -= 604800
    elif t_k < -302400: t_k += 604800
    M_k = eph['M0'] + (n0 + eph['Delta_n']) * t_k; E_k = M_k
    for _ in range(5): E_k = M_k + eph['e'] * math.sin(E_k)
    dt_sat = eph['af0'] + eph['af1'] * t_k + eph['af2'] * (t_k ** 2)
    nu_k = math.atan2((math.sqrt(1 - eph['e']**2) * math.sin(E_k)), (math.cos(E_k) - eph['e']))
    phi_k = nu_k + eph['omega']
    u_k = phi_k + eph['Cus'] * math.sin(2 * phi_k) + eph['Cuc'] * math.cos(2 * phi_k)
    r_k = A * (1 - eph['e'] * math.cos(E_k)) + eph['Crs'] * math.sin(2 * phi_k) + eph['Crc'] * math.cos(2 * phi_k)
    i_k = eph['i0'] + eph['Cic'] * math.cos(2 * phi_k) + eph['Cis'] * math.sin(2 * phi_k) + eph['IDOT'] * t_k
    x_k, y_k = r_k * math.cos(u_k), r_k * math.sin(u_k)
    omega_k = eph['OMEGA'] + (eph['OMEGA_DOT'] - omega_e_sys) * t_k - omega_e_sys * eph['Toe']
    xs = x_k * math.cos(omega_k) - y_k * math.cos(i_k) * math.sin(omega_k)
    ys = x_k * math.sin(omega_k) + y_k * math.cos(i_k) * math.cos(omega_k)
    zs = y_k * math.sin(i_k)
    theta = omega_e_sys * tau_vuelo
    return (xs * math.cos(theta) + ys * math.sin(theta), -xs * math.sin(theta) + ys * math.cos(theta), zs, dt_sat)

# =====================================================================
# MOTOR PPK HÍBRIDO DETERMINISTA (VERSIÓN 9 ARREGLADA)
# =====================================================================
def aislar_diferencias_simples_ppk(obs_b, obs_r):
    sd_suavizada = {}
    for tow in sorted(list(obs_r.keys())):
        if tow not in obs_b: continue
        sd_epoca = {'_meta': obs_r[tow]['_meta']}
        for s, d_r in obs_r[tow].items():
            if s == '_meta' or s not in obs_b[tow]: continue
            d_b = obs_b[tow][s]
            
            pr_b1 = d_b.get('C1'); pr_r1 = d_r.get('C1')
            cp_b1 = d_b.get('L1'); cp_r1 = d_r.get('L1')
            pr_b5 = d_b.get('C5'); pr_r5 = d_r.get('C5')
            cp_b5 = d_b.get('L5'); cp_r5 = d_r.get('L5')
            
            if not pr_b1 and not pr_b5: continue
            if not pr_r1 and not pr_r5: continue
            
            snr_b = d_b.get('S1') or d_b.get('S5', 30.0)
            snr_r = d_r.get('S1') or d_r.get('S5', 30.0)
            
            sd_epoca[s] = {
                'pr_b1': pr_b1, 'pr_r1': pr_r1, 'cp_b1': cp_b1, 'cp_r1': cp_r1,
                'pr_b5': pr_b5, 'pr_r5': pr_r5, 'cp_b5': cp_b5, 'cp_r5': cp_r5,
                'snr': min(snr_b, snr_r), 'sys': s[0]
            }
        if len(sd_epoca) > 1: sd_suavizada[tow] = sd_epoca
    return sd_suavizada

def decorrelacion_lambda_z(Q):
    n = len(Q)
    Z = matid(n)
    try: L = cholesky_decompose(Q)
    except: return Z, Q 
    for i in range(n - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            mu = round(L[i][j] / L[j][j])
            if mu != 0:
                for k in range(j + 1): L[i][k] -= mu * L[j][k]
                for k in range(n): Z[k][i] -= mu * Z[k][j]
    Z_T = transpose_matrix(Z)
    Q_z = matmul(matmul(Z_T, Q), Z)
    return Z, Q_z

def suavizador_rts_backward(forward_states):
    n = len(forward_states)
    if n == 0: return []
    smoothed_states = [None] * n
    smoothed_states[-1] = forward_states[-1]['X_post']
    for k in range(n-2, -1, -1):
        P_post_k = forward_states[k]['P_post']
        P_pri_k1 = forward_states[k+1]['P_pri']
        P_pri_inv = invert_matrix_nxn(P_pri_k1)
        if not P_pri_inv:
            smoothed_states[k] = forward_states[k]['X_post']
            continue
        C_k = matmul(P_post_k, P_pri_inv)
        X_smooth_k1 = smoothed_states[k+1]
        X_pri_k1 = forward_states[k+1]['X_pri']
        dx = [[X_smooth_k1[i][0] - X_pri_k1[i][0]] for i in range(3)]
        correction = matmul(C_k, dx)
        X_post_k = forward_states[k]['X_post']
        smoothed_states[k] = [[X_post_k[i][0] + correction[i][0]] for i in range(3)]
    return smoothed_states

def procesar_ekF_lambda(sd_epoca, nav, sp3, kf_estado, tr, mask_angle, snr_mask):
    try:
        X_pri = [[kf_estado['X'][0][0]], [kf_estado['X'][1][0]], [kf_estado['X'][2][0]]]
        P_pri = [row[:] for row in kf_estado['P']]
        h_r = kf_estado.get('h_r', 0.0)
        
        X_iter, Y_iter, Z_iter = X_pri[0][0], X_pri[1][0], X_pri[2][0]
        lat_r, lon_r, alt_r = ecef_a_geodesicas(X_iter, Y_iter, Z_iter)
        lat_rad, lon_rad = math.radians(lat_r), math.radians(lon_r)
        
        X_apc = X_iter + h_r * math.cos(lat_rad) * math.cos(lon_rad)
        Y_apc = Y_iter + h_r * math.cos(lat_rad) * math.sin(lon_rad)
        Z_apc = Z_iter + h_r * math.sin(lat_rad)
        
        alpha = nav.get('_iono', {}).get('alpha', [0]*4)
        beta = nav.get('_iono', {}).get('beta', [0]*4)
        
        y_m, m_m, d_m, h_m, mn_m, sec_m = sd_epoca['_meta']
        dx_tide, dy_tide, dz_tide = correccion_mareas_solidas(
            kf_estado['X_base'][0], kf_estado['X_base'][1], kf_estado['X_base'][2], 
            tr, y_m, m_m, d_m
        )
        
        X_base_corr = kf_estado['X_base'][0] + dx_tide
        Y_base_corr = kf_estado['X_base'][1] + dy_tide
        Z_base_corr = kf_estado['X_base'][2] + dz_tide
        lat_base, lon_base, alt_base = ecef_a_geodesicas(X_base_corr, Y_base_corr, Z_base_corr)
        
        # [VERSIÓN 9 ARREGLADA: ESCUDO 1 RELAJADO] Tolerancia a Android Clock Steering
        cs_state = kf_estado.get('cs_state', {})
        cycle_slips = {}
        for s, d in sd_epoca.items():
            if s == '_meta': continue
            cs = False
            if d['cp_r1'] is not None and d['cp_r5'] is not None:
                gf = (d['cp_r1'] * WAVE_L1) - (d['cp_r5'] * WAVE_L5)
                if s in cs_state:
                    # Umbral relajado a 2.5m para absorber el ajuste de reloj del smartphone
                    if abs(gf - cs_state[s]) > 2.5: cs = True 
                cs_state[s] = gf
            cycle_slips[s] = cs
        kf_estado['cs_state'] = cs_state
        
        sat_positions = {}
        for s, d in sd_epoca.items():
            if s == '_meta': continue 
            tau_r = (d['pr_r1'] if d['pr_r1'] else d['pr_r5']) / C_LIGHT
            tau_b = (d['pr_b1'] if d['pr_b1'] else d['pr_b5']) / C_LIGHT
            t_emision_r = tr - tau_r
            t_emision_b = tr - tau_b
            
            sp_r, sp_b = None, None
            if sp3 and s in sp3:
                sp3_res_r = interpolate_sp3(sp3, s, t_emision_r)
                sp3_res_b = interpolate_sp3(sp3, s, t_emision_b)
                if sp3_res_r and sp3_res_b:
                    theta_r = OMEGA_E * tau_r
                    xs_r = sp3_res_r[0] * math.cos(theta_r) + sp3_res_r[1] * math.sin(theta_r)
                    ys_r = -sp3_res_r[0] * math.sin(theta_r) + sp3_res_r[1] * math.cos(theta_r)
                    sp_r = (xs_r, ys_r, sp3_res_r[2], sp3_res_r[3]) 
                    theta_b = OMEGA_E * tau_b
                    xs_b = sp3_res_b[0] * math.cos(theta_b) + sp3_res_b[1] * math.sin(theta_b)
                    ys_b = -sp3_res_b[0] * math.sin(theta_b) + sp3_res_b[1] * math.cos(theta_b)
                    sp_b = (xs_b, ys_b, sp3_res_b[2], sp3_res_b[3])
            
            if not sp_r or not sp_b:
                sp_r = calcular_posicion_satelite_wgs84(seleccionar_efemeride_optima(nav.get(s), t_emision_r), t_emision_r, tau_r, s[0])
                sp_b = calcular_posicion_satelite_wgs84(seleccionar_efemeride_optima(nav.get(s), t_emision_b), t_emision_b, tau_b, s[0])
            
            if sp_r and sp_b:
                el_r, az_r = calcular_topocentricas(sp_r[0], sp_r[1], sp_r[2], X_apc, Y_apc, Z_apc)
                if el_r >= mask_angle and d['snr'] >= snr_mask:
                    sat_positions[s] = {'sp_r': sp_r, 'sp_b': sp_b, 'd': d, 'el_r': el_r, 'az_r': az_r}
        
        if len(sat_positions) < 4: return None, "FAILED", kf_estado, None
        
        def calc_components(s_info, X, Y, Z, lat_v, lon_v, alt_v):
            dist = math.sqrt((s_info['sp_r'][0]-X)**2 + (s_info['sp_r'][1]-Y)**2 + (s_info['sp_r'][2]-Z)**2)
            tropo = calcular_saastamoinen(lat_v, alt_v, s_info['el_r'])
            iono = calcular_klobuchar(lat_v, lon_v, s_info['el_r'], s_info['az_r'], tr, alpha, beta)
            return dist, tropo, iono
            
        sat_data_processed = {}
        for s, info in sat_positions.items():
            dist_r, tropo_r, iono_r = calc_components(info, X_apc, Y_apc, Z_apc, lat_r, lon_r, alt_r + h_r)
            dist_b = math.sqrt((info['sp_b'][0]-X_base_corr)**2 + (info['sp_b'][1]-Y_base_corr)**2 + (info['sp_b'][2]-Z_base_corr)**2)
            el_b, az_b = calcular_topocentricas(info['sp_b'][0], info['sp_b'][1], info['sp_b'][2], X_base_corr, Y_base_corr, Z_base_corr)
            tropo_b = calcular_saastamoinen(lat_base, alt_base, el_b)
            iono_b = calcular_klobuchar(lat_base, lon_base, el_b, az_b, tr, alpha, beta)
            
            d = info['d']
            SD_P_L1, SD_CP_L1 = None, None
            if d['pr_r1'] is not None and d['pr_b1'] is not None:
                SD_P_L1 = d['pr_r1'] - d['pr_b1']
                if d['cp_r1'] is not None and d['cp_b1'] is not None:
                    SD_CP_L1 = (d['cp_r1'] - d['cp_b1']) * WAVE_L1
                    
            SD_P_IF = None
            if d['pr_r1'] and d['pr_b1'] and d['pr_r5'] and d['pr_b5']:
                gamma = (FREQ_L1 / FREQ_L5)**2
                pr_r_if = (gamma * d['pr_r1'] - d['pr_r5']) / (gamma - 1.0)
                pr_b_if = (gamma * d['pr_b1'] - d['pr_b5']) / (gamma - 1.0)
                SD_P_IF = pr_r_if - pr_b_if
                
            sat_data_processed[s] = {
                'el_r': info['el_r'], 'dist_r': dist_r, 'sp_r': info['sp_r'],
                'SD_P_L1': SD_P_L1, 'SD_CP_L1': SD_CP_L1, 'SD_P_IF': SD_P_IF,
                'SD_P_calc_L1': (dist_r + tropo_r + iono_r) - (dist_b + tropo_b + iono_b),
                'SD_P_calc_IF': (dist_r + tropo_r) - (dist_b + tropo_b),
                'snr': d['snr'], 'cycle_slip': cycle_slips[s], 'sys': d['sys']
            }
        
        sat_list_full = list(sat_data_processed.keys())
        constellations = set([s[0] for s in sat_list_full])
        ref_sats = {}
        sat_list = []
        for c in constellations:
            c_sats = [s for s in sat_list_full if s[0] == c]
            if len(c_sats) >= 2:
                r_candidate = max(c_sats, key=lambda k: sat_data_processed[k]['el_r'])
                ref_sats[c] = r_candidate
                c_sats.remove(r_candidate)
                sat_list.extend(c_sats)
        
        if len(sat_list) < 3: return None, "FAILED", kf_estado, None
        
        H = []; L = []; R_diag = []
        for s in sat_list:
            c = s[0]
            data = sat_data_processed[s]
            rc = sat_data_processed[ref_sats[c]]
            
            use_IF = False
            if data['snr'] >= 32.0 and rc['snr'] >= 32.0 and not data['cycle_slip'] and data['SD_P_IF'] is not None and rc['SD_P_IF'] is not None:
                use_IF = True
                
            DD_P_obs, DD_P_calc = None, None
            var_multiplier = 9.0
            
            if use_IF:
                DD_P_obs = data['SD_P_IF'] - rc['SD_P_IF']
                DD_P_calc = data['SD_P_calc_IF'] - rc['SD_P_calc_IF']
                var_multiplier = 15.0 
            elif data['SD_P_L1'] is not None and rc['SD_P_L1'] is not None:
                DD_P_obs = data['SD_P_L1'] - rc['SD_P_L1']
                DD_P_calc = data['SD_P_calc_L1'] - rc['SD_P_calc_L1']
                if data['cycle_slip']: var_multiplier = 100.0
                
            if DD_P_obs is None: continue
            
            v = DD_P_obs - DD_P_calc
            dx_geom = [
                -(data['sp_r'][0] - X_apc) / data['dist_r'] - (-(rc['sp_r'][0] - X_apc) / rc['dist_r']),
                -(data['sp_r'][1] - Y_apc) / data['dist_r'] - (-(rc['sp_r'][1] - Y_apc) / rc['dist_r']),
                -(data['sp_r'][2] - Z_apc) / data['dist_r'] - (-(rc['sp_r'][2] - Z_apc) / rc['dist_r'])
            ]
            
            r_val = ((10.0 ** (-data['snr'] / 10.0)) * 100.0) * var_multiplier
            
            # [VERSIÓN 9 ARREGLADA: ESCUDO 3] Test de Innovación Chi-Cuadrado (4 Sigma) con CASTIGO
            S_ii = r_val
            for r_idx in range(3):
                for c_idx in range(3): S_ii += dx_geom[r_idx] * P_pri[r_idx][c_idx] * dx_geom[c_idx]
            
            if (v**2 / max(1e-6, S_ii)) > 16.0: 
                r_val *= 100.0 # Castigo de peso estadístico en matriz R, evitando expulsión geométrica
                
            L.append([v]); H.append(dx_geom); R_diag.append(r_val)
            
            if not use_IF and not data['cycle_slip'] and data['SD_CP_L1'] is not None and rc['SD_CP_L1'] is not None:
                DD_CP_obs = data['SD_CP_L1'] - rc['SD_CP_L1']
                v_cp = DD_CP_obs - (data['SD_P_calc_L1'] - rc['SD_P_calc_L1'])
                
                var_base_cp = (10.0 ** (-data['snr'] / 10.0)) * 100.0
                S_ii_cp = var_base_cp * 0.0001
                for r_idx in range(3):
                    for c_idx in range(3): S_ii_cp += dx_geom[r_idx] * P_pri[r_idx][c_idx] * dx_geom[c_idx]
                        
                if (v_cp**2 / max(1e-6, S_ii_cp)) < 9.0: 
                    var_amb = [[var_base_cp * 0.0001]]
                    Z_trans, Q_z = decorrelacion_lambda_z(var_amb)
                    ambiguity_float = v_cp / WAVE_L1
                    amb_z = ambiguity_float * Z_trans[0][0]
                    amb_restored = round(amb_z) / Z_trans[0][0]
                    
                    if abs(ambiguity_float - amb_restored) < 0.20:
                        v_fixed = (DD_CP_obs - amb_restored * WAVE_L1) - (data['SD_P_calc_L1'] - rc['SD_P_calc_L1'])
                        L.append([v_fixed]); H.append(dx_geom); R_diag.append(var_base_cp * 0.0001)
                        kf_estado['fix_flags'] += 1

        if not H: return None, "FAILED", kf_estado, None
        
        H_T = transpose_matrix(H)
        R_inv = matid(len(R_diag))
        for i in range(len(R_diag)): R_inv[i][i] = 1.0 / max(1e-6, R_diag[i])
        
        P_inv = invert_matrix_nxn(P_pri)
        if not P_inv: return None, "FAILED", kf_estado, None
        
        H_T_R_inv = matmul(H_T, R_inv)
        N_mat = matadd(matmul(H_T_R_inv, H), P_inv)
        U_vec = matmul(H_T_R_inv, L)
        
        Q_cov = invert_matrix_nxn(N_mat)
        if not Q_cov: return None, "FAILED", kf_estado, None
        Delta_X = matmul(Q_cov, U_vec)
        
        X_post = [[X_pri[0][0] + Delta_X[0][0]], [X_pri[1][0] + Delta_X[1][0]], [X_pri[2][0] + Delta_X[2][0]]]
        
        kf_estado['X'] = X_post
        kf_estado['P'] = Q_cov
        status = "FIXED (PPK)" if kf_estado['fix_flags'] > 4 else "FLOAT (DGPS)"
        kf_estado['fix_flags'] = 0 
        
        state_dict = {'tow': tr, 'X_pri': X_pri, 'P_pri': P_pri, 'X_post': X_post, 'P_post': Q_cov}
        return (X_post[0][0], X_post[1][0], X_post[2][0]), status, kf_estado, state_dict

    except Exception as e: return None, f"FAILED_EXCEPTION:_{str(e)}", kf_estado, None

# =====================================================================
# ESTADÍSTICAS Y FILTRADO VINCULANTE
# =====================================================================
def estadistica_desacoplada(coordenadas, conf_plani, conf_alti, err_hor_max, err_ver_max):
    if not coordenadas: return None, None, None, 0, 0, 0, 0, 0.0
    N_list = [c[0] for c in coordenadas]; E_list = [c[1] for c in coordenadas]; Z_list = [c[2] for c in coordenadas]
    def get_median(lst):
        s = sorted(lst); n = len(s)
        if n == 0: return 0
        return s[n//2] if n % 2 == 1 else (s[n//2 - 1] + s[n//2]) / 2.0
    med_N = get_median(N_list); med_E = get_median(E_list); med_Z = get_median(Z_list)
    valid_coords = []
    for c in coordenadas:
        dh = math.hypot(c[0] - med_N, c[1] - med_E)
        dv = abs(c[2] - med_Z)
        if (err_hor_max > 0.0 and dh > err_hor_max) or (err_ver_max > 0.0 and dv > err_ver_max): continue
        valid_coords.append(c)
    if not valid_coords: return None, None, None, 0, 0, 0, 0, 0.0
    def calc_mean_std(arr):
        n = len(arr); m = sum(arr) / max(1, n)
        return m, (math.sqrt(sum((x - m)**2 for x in arr) / n) if n > 1 else 0.0)
    N_v = [c[0] for c in valid_coords]; E_v = [c[1] for c in valid_coords]; Z_v = [c[2] for c in valid_coords]
    N_m, N_s = calc_mean_std(N_v); E_m, E_s = calc_mean_std(E_v); Z_m, Z_s = calc_mean_std(Z_v)
    final_coords = []
    for c in valid_coords:
        if N_s > 0 and abs(c[0] - N_m) > conf_plani * N_s: continue
        if E_s > 0 and abs(c[1] - E_m) > conf_plani * E_s: continue
        if Z_s > 0 and abs(c[2] - Z_m) > conf_alti * Z_s: continue
        final_coords.append(c)
    if not final_coords: return None, None, None, 0, 0, 0, 0, 0.0
    N_f = [c[0] for c in final_coords]; E_f = [c[1] for c in final_coords]; Z_f = [c[2] for c in final_coords]
    f_v = [c[3] for c in final_coords if len(c) > 3 and "FIXED" in c[3]]
    fix_ratio = (len(f_v) / len(final_coords)) * 100 if final_coords else 0.0
    return get_median(N_f), get_median(E_f), get_median(Z_f), N_s, E_s, Z_s, len(final_coords), fix_ratio

# =====================================================================
# GENERADORES DE INFORMES
# =====================================================================
def generar_informe_homogeneizacion_detallado(base_name, rover_name, base_raw, rover_raw, rover_sinc):
    def get_stats(obs):
        c = {'G':0, 'E':0, 'C':0, 'R':0, 'S':0, 'J':0}
        tiempos = sorted(list(obs.keys()))
        if not tiempos: return c, 0, None, None, 0.0, 0
        epocas = len(obs)
        t_ini, t_fin = obs[tiempos[0]]['_meta'], obs[tiempos[-1]]['_meta']
        intervalos = [tiempos[i] - tiempos[i-1] for i in range(1, epocas)]
        tasa_muestreo = sum(intervalos)/len(intervalos) if intervalos else 0.0
        gaps = sum(1 for i in intervalos if i > tasa_muestreo * 1.5)
        for t in tiempos:
            for s in obs[t]:
                if s != '_meta' and s[0] in c: c[s[0]] += 1
        return {k: v/epocas for k, v in c.items()}, epocas, t_ini, t_fin, tasa_muestreo, gaps
    
    cb, eb, b_ini, b_fin, tr_b, g_b = get_stats(base_raw)
    cr, er, r_ini, r_fin, tr_r, g_r = get_stats(rover_raw)
    cs, es, s_ini, s_fin, tr_s, _ = get_stats(rover_sinc)
    t_exito = (es / er * 100) if er > 0 else 0.0
    
    b_ini_str = f"{b_ini[3]:02d}:{b_ini[4]:02d}:{b_ini[5]}" if b_ini else "N/A"
    b_fin_str = f"{b_fin[3]:02d}:{b_fin[4]:02d}:{b_fin[5]}" if b_fin else "N/A"
    r_ini_str = f"{r_ini[3]:02d}:{r_ini[4]:02d}:{r_ini[5]}" if r_ini else "N/A"
    r_fin_str = f"{r_fin[3]:02d}:{r_fin[4]:02d}:{r_fin[5]}" if r_fin else "N/A"
    
    informe = f"""
========================================================================
    AUDITORÍA FORENSE DE EMPAREJAMIENTO DE ÉPOCAS
========================================================================
[1] PARÁMETROS DE CONTROL (BASE) : {base_name}
  [-] Épocas Crudas Registradas : {eb}
  [-] Ventana de Observación    : {b_ini_str} - {b_fin_str}

[2] PARÁMETROS DEL MÓVIL (ROVER) : {rover_name}
  [-] Épocas Crudas Registradas : {er}
  [-] Ventana de Observación    : {r_ini_str} - {r_fin_str}

[3] MATRIZ RESULTANTE (ESTRICTA, SIN INTERPOLACIÓN)
  [-] Épocas Útiles Sincronizadas: {es}
  [-] Tasa de Éxito sobre Rover  : {f_14(t_exito)}%
========================================================================
"""
    return informe

def generar_informe_ascii(tipo, p_dict):
    estado_sol = f"HÍBRIDO PPK/EKF ({p_dict['fix_r']:.1f}% FIXED)" if p_dict['fix_r'] > 0 else 'FLOAT (EKF)'
    err_h_str = f"± {f_14(p_dict['err_h'])} m (Vinculante)" if p_dict['err_h'] > 0 else 'Inactiva'
    err_v_str = f"± {f_14(p_dict['err_v'])} m (Vinculante)" if p_dict['err_v'] > 0 else 'Inactiva'
    sp3_str = p_dict['sp3_file'] if p_dict.get('sp3_file') else "No provisto (Fallback a Broadcast NAV)"
    nav_str = p_dict.get('nav_file', "auto_nav.nav")
    
    informe = f"""
========================================================================
             INFORME DE PROCESAMIENTO GNSSJP PRO 
========================================================================

[*] RESULTADO DE MEDICIÓN ABSOLUTA ({estado_sol})
------------------------------------------------------------------------
  [-] Tolerancia Horizontal  : {err_h_str}
  [-] Tolerancia Vertical    : {err_v_str}
  [-] Máscara Elevación      : {f_14(p_dict['mask'])}°
  [-] Filtro Planimétrico    : {f_14(p_dict['cp'])} Sigma
  [-] Filtro Altimétrico     : {f_14(p_dict['ca'])} Sigma
  [-] Tolerancia Sync        : {f_14(p_dict.get('max_gap', 0.5))} s
  [-] Máscara SNR            : {f_14(p_dict.get('snr', 25.0))} dBHz
  [-] Tasa Ambiguity FIX     : {p_dict['fix_r']:.2f}% (Resolución Entera)
  [-] Épocas Útiles Retenidas: {p_dict['ret']} ({(p_dict['ret']/max(1, p_dict['total']))*100:.2f}% del total)

[1] TRAZABILIDAD DEL PROYECTO Y ARCHIVOS
------------------------------------------------------------------------
  [-] Archivo Control (Base) : {p_dict['base_file']}
  [-] Archivo Móvil (Rover)  : {p_dict['rover_file']}
  [-] Archivo Efemérides NAV : {nav_str}
  [-] Archivo Preciso SP3    : {sp3_str}

[2] ESTRATEGIA MATEMÁTICA Y ESTADÍSTICA
------------------------------------------------------------------------
  [-] Motor Algorítmico      : Filtro Kalman Extendido (EKF PPK) + RTS Smoother
  [-] Observables Inyectadas : L1/L5 Adaptativo Iono-Free + C1/C5
  [-] Control de Ruido       : Geometry-Free Cycle Slip + Chi-Cuadrado Soft Penalization
  [-] Correcciones Geofísicas: Mareas Sólidas Terrestres
  [-] Órbitas Satelitales    : SP3 Interpolación Lagrange 9°

[3] CALIDAD GEOMÉTRICA (QA / QC)
------------------------------------------------------------------------
  [-] Error Horizontal (RMS) : ± {f_14(math.hypot(p_dict['std_n'], p_dict['std_e']))} m
  [-] Error Espacial (3D RMS): ± {f_14(math.sqrt(p_dict['std_n']**2 + p_dict['std_e']**2 + p_dict['std_z']**2))} m

[4] RESULTADOS VECTORIALES FINALES
------------------------------------------------------------------------
  * COORDENADA DE CONTROL (BASE FIJA):
      Norte : {f_14(p_dict['b_n'])} m
      Este  : {f_14(p_dict['b_e'])} m
      Cota  : {f_14(p_dict['b_z'])} m

  * COORDENADA CALCULADA (AJUSTE KALMAN {estado_sol}):
      Norte : {f_14(p_dict['r_n_calc'])} m
      Este  : {f_14(p_dict['r_e_calc'])} m
      Cota  : {f_14(p_dict['r_z_calc'])} m
========================================================================
"""
    return informe

# =====================================================================
# RUTAS FLASK
# =====================================================================
@app.route('/')
def index():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_path = os.path.join(base_dir, 'index.html')
    return send_file(index_path)

@app.route('/API/tab1_homogenizar', methods=['POST'])
def tab1_homogenizar():
    with STATE_LOCK:
        if os.path.exists(STATE_FILE):
            try: os.remove(STATE_FILE)
            except: pass
    
    url_base = request.form.get('url_base')
    url_rover = request.form.get('url_rover')
    if not url_base or not url_rover: return Response("> [ERROR CRÍTICO] Enlaces faltantes.\n", mimetype='text/plain')
    p_b_raw = os.path.join(UPLOAD_FOLDER, 'base_raw.obs')
    p_r_raw = os.path.join(UPLOAD_FOLDER, 'rover_calibracion_raw.obs')

    def procesar():
        try:
            yield "> [RED] Descargando RINEX Base...\n"
            descargar_desde_gdrive(url_base, p_b_raw)
            yield "> [RED] Descargando RINEX Rover...\n"
            descargar_desde_gdrive(url_rover, p_r_raw)
            yield f"\n> [SISTEMA] Iniciando Emparejamiento...\n"
            base_raw_dict = parse_rinex_obs_completo(p_b_raw)
            rover_raw_dict = parse_rinex_obs_completo(p_r_raw)
            base_sinc, rover_sinc = {}, {}
            total_epochs = len(rover_raw_dict)
            c = 0
            for tr in sorted(list(rover_raw_dict.keys())):
                c += 1
                if total_epochs > 0 and c % max(1, total_epochs // 10) == 0: yield f"[PROGRESO] Cotejando épocas... {int((c / total_epochs) * 100)}%\n"
                base_interp = interpolar_base_a_rover(base_raw_dict, tr)
                if base_interp:
                    base_sinc[tr] = base_interp
                    base_sinc[tr]['_meta'] = rover_raw_dict[tr]['_meta']
                    rover_sinc[tr] = rover_raw_dict[tr]
            
            if not base_sinc: yield "\n> [ERROR FATAL] Cero épocas en común."; return
            p_b_h = os.path.join(UPLOAD_FOLDER, 'base_calib_homo.obs')
            p_r_h = os.path.join(UPLOAD_FOLDER, 'rover_calib_homo.obs')
            generar_rinex_sincronizado(p_b_raw, p_b_h, base_sinc)
            generar_rinex_sincronizado(p_r_raw, p_r_h, rover_sinc)
            
            guardar_estado('base_raw', p_b_raw)
            guardar_estado('base_calib_homo', p_b_h)
            guardar_estado('rover_calib_homo', p_r_h)
            name_base = "Drive_Base_Pivote.obs"
            name_rover = "Drive_Rover_Calib.obs"
            guardar_estado('name_base_raw', name_base)
            guardar_estado('name_rover_calib_raw', name_rover)
            yield generar_informe_homogeneizacion_detallado(name_base, name_rover, base_raw_dict, rover_raw_dict, rover_sinc)
            yield "\n[SUCCESS]"
        except Exception as e: yield f"\n> [ERROR] Falla estructural: {str(e)}"
    return Response(procesar(), mimetype='text/plain')

@app.route('/API/tab2_efemerides', methods=['POST'])
def tab2_efemerides():
    f_sp3 = request.files.get('file_sp3')
    sp3_path = None
    if f_sp3 and f_sp3.filename != '':
        sp3_path = os.path.join(UPLOAD_FOLDER, 'manual_sp3.sp3')
        f_sp3.save(sp3_path)
        guardar_estado('sp3_path', sp3_path)
        guardar_estado('name_sp3_file', f_sp3.filename)
    else:
        guardar_estado('sp3_path', None)
        guardar_estado('name_sp3_file', None)

    def procesar():
        try:
            yield "> [SISTEMA] Iniciando Inyección Híbrida de Efemérides...\n"
            bp = leer_estado('base_raw')
            if not bp or not os.path.exists(bp): yield "> [ERROR FATAL] Falta RINEX Base.\n"; return
            ft = obtener_fecha_obs(bp)
            if not ft: yield "> [ERROR FATAL] Imposible extraer fecha.\n"; return
            
            year, month, day = ft[0], ft[1], ft[2]
            dt = datetime.datetime(year, month, day)
            doy = dt.timetuple().tm_yday
            nav_gz = os.path.join(UPLOAD_FOLDER, f"auto_nav_{year}_{doy:03d}.nav.gz")
            nav_path = os.path.join(UPLOAD_FOLDER, f"auto_nav_{year}_{doy:03d}.nav")
            
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            if not os.path.exists(nav_path):
                urls_to_try = [
                    f"https://igs.bkg.bund.de/root_ftp/IGS/BRDC/{year}/{doy:03d}/BRDC00IGS_R_{year}{doy:03d}0000_01D_MN.rnx.gz",
                    f"https://igs.bkg.bund.de/root_ftp/IGS/BRDC/{year}/{doy:03d}/BRDM00DLR_S_{year}{doy:03d}0000_01D_MN.rnx.gz"
                ]
                descargado = False
                for url_nav in urls_to_try:
                    try:
                        req = urllib.request.Request(url_nav, headers={'User-Agent': 'Mozilla/5.0'})
                        with urllib.request.urlopen(req, context=ctx, timeout=8) as res:
                            with open(nav_gz, 'wb') as f: f.write(res.read())
                        descargado = True; break 
                    except: continue
                if not descargado: raise Exception("HTTP 404: NAV no publicado.")
                with gzip.open(nav_gz, 'rb') as f_in, open(nav_path, 'wb') as f_out: shutil.copyfileobj(f_in, f_out)
            
            guardar_estado('nav_path', nav_path)
            guardar_estado('name_nav_file', os.path.basename(nav_path))
            yield f"  [-] Archivo NAV listo.\n\n[SUCCESS]"
        except Exception as e: yield f"\n> [ERROR FATAL] {str(e)}\n"
    return Response(procesar(), mimetype='text/plain')

@app.route('/API/tab3_calibrar', methods=['POST'])
def tab3_calibrar():
    utm_n, utm_e, utm_c = safe_f(request.form.get('utm_norte')), safe_f(request.form.get('utm_este')), safe_f(request.form.get('utm_cota'))
    utm_n_r, utm_e_r, utm_c_r = safe_f(request.form.get('utm_norte_r')), safe_f(request.form.get('utm_este_r')), safe_f(request.form.get('utm_cota_r'))
    utm_h, utm_hem = safe_i(request.form.get('utm_huso'), 19), request.form.get('utm_hemisferio', 'N')
    h_b, h_r = safe_f(request.form.get('altura_base')), safe_f(request.form.get('altura_rover'))
    p_max_gap, p_snr = safe_f(request.form.get('param_max_gap'), 0.5), safe_f(request.form.get('param_snr'), 25.0)
    p_iter = max(1, safe_i(request.form.get('param_iter'), 6))

    def procesar():
        try:
            yield f"> [SISTEMA] Búsqueda Determinista V9 Arreglada ({p_iter} Iteraciones)...\n"
            if utm_e == 0.0 or utm_n == 0.0: yield "> [ERROR] Coordenadas Base requeridas.\n"; return
            
            nav_path, sp3_path = leer_estado('nav_path'), leer_estado('sp3_path')
            p_b_h, p_r_h = leer_estado('base_calib_homo'), leer_estado('rover_calib_homo')
            if not nav_path or not p_b_h: yield "> [ERROR] Faltan archivos.\n"; return

            obs_b_raw, obs_r_raw = parse_rinex_obs_completo(p_b_h), parse_rinex_obs_completo(p_r_h)
            nav, sp3 = parse_rinex_nav_real(nav_path), parse_sp3_preciso(sp3_path) if sp3_path else {}
            
            sd_suavizada = aislar_diferencias_simples_ppk(obs_b_raw, obs_r_raw)
            if not sd_suavizada: yield "> [ERROR] No hay épocas sincronizadas válidas.\n"; return

            lat_b, lon_b, _ = utm_a_geodesicas(utm_e, utm_n, utm_h, utm_hem)
            X_b, Y_b, Z_b = geodesicas_a_ecef(lat_b, lon_b, utm_c + h_b)
            X_bg, Y_bg, Z_bg = geodesicas_a_ecef(lat_b, lon_b, utm_c)

            P_init = matid(3)
            for i in range(3): P_init[i][i] = 100.0
            
            kf_estado_raw = {'X': [[X_bg], [Y_bg], [Z_bg]], 'P': P_init, 'X_base': (X_b, Y_b, Z_b), 'fix_flags': 0, 'h_r': h_r, 'cs_state': {}}
            coords_raw = []
            for t in list(sd_suavizada.keys()):
                sem, status, kf_estado_raw, _ = procesar_ekF_lambda(sd_suavizada[t], nav, sp3, kf_estado_raw, t, 10.0, p_snr)
                if sem:
                    la, lo, al = ecef_a_geodesicas(sem[0], sem[1], sem[2])
                    nt, et = geodesicas_a_utm(la, lo, utm_h)
                    coords_raw.append((nt, et, al, status))
            
            if not coords_raw: yield "> [ERROR] Kalman colapsado. Verifica RINEX.\n"; return
                
            deltas_h = [math.hypot(c[0] - utm_n_r, c[1] - utm_e_r) for c in coords_raw]
            deltas_v = [abs(c[2] - utm_c_r) for c in coords_raw]
            deltas_h.sort(); deltas_v.sort()
            
            def get_mad(data):
                if not data: return 0.0, 0.0
                med = data[len(data)//2]
                return med, sorted([abs(x - med) for x in data])[len(data)//2]

            med_h, mad_h = get_mad(deltas_h)
            med_v, mad_v = get_mad(deltas_v)
            best_eh, best_ev = max(0.01, med_h + 3.0 * mad_h), max(0.01, med_v + 3.0 * mad_v)
            
            global_best_score, best_rmse, best_params = float('inf'), float('inf'), {}
            m_center, m_span = 10.0, 5.0
            cp_center, cp_span = 2.0, 1.5
            ca_center, ca_span = 2.0, 1.5
            snr_center, snr_span = p_snr, 5.0
            gap_center, gap_span = p_max_gap, 0.2
            
            p_b_raw, p_r_raw = leer_estado('base_raw'), os.path.join(UPLOAD_FOLDER, 'rover_calibracion_raw.obs')
            obs_b_full = parse_rinex_obs_completo(p_b_raw) if p_b_raw and os.path.exists(p_b_raw) else obs_b_raw
            obs_r_full = parse_rinex_obs_completo(p_r_raw) if os.path.exists(p_r_raw) else obs_r_raw
            rover_tows_full, base_tows_full = sorted(list(obs_r_full.keys())), sorted(list(obs_b_full.keys()))
            
            for nivel in range(p_iter):
                yield f"  [+] Refinando espacio de búsqueda (Zoom {nivel+1}/{p_iter})...\n"
                m_grid = [max(1.0, min(25.0, x)) for x in [m_center - m_span, m_center, m_center + m_span]]
                cp_grid = [max(0.1, min(5.0, x)) for x in [cp_center - cp_span, cp_center, cp_center + cp_span]]
                ca_grid = [max(0.1, min(5.0, x)) for x in [ca_center - ca_span, ca_center, ca_center + ca_span]]
                snr_grid = [max(25.0, min(45.0, x)) for x in [snr_center - snr_span, snr_center, snr_center + snr_span]]
                gap_grid = [max(0.01, min(2.0, x)) for x in [gap_center - gap_span, gap_center, gap_center + gap_span]]
                
                for gap in set(gap_grid):
                    obs_b_sync = {}
                    for tr in rover_tows_full:
                        if not base_tows_full: continue
                        idx = min(range(len(base_tows_full)), key=lambda i: abs(base_tows_full[i] - tr))
                        if abs(base_tows_full[idx] - tr) <= gap:
                            obs_b_sync[tr] = obs_b_full[base_tows_full[idx]].copy()
                            obs_b_sync[tr]['_meta'] = obs_r_full[tr]['_meta']
                    
                    sd_suav = aislar_diferencias_simples_ppk(obs_b_sync, obs_r_full)
                    if not sd_suav: continue
                    
                    for m in set(m_grid):
                        for snr in set(snr_grid):
                            kf_est = {'X': [[X_bg], [Y_bg], [Z_bg]], 'P': P_init, 'X_base': (X_b, Y_b, Z_b), 'fix_flags': 0, 'h_r': h_r, 'cs_state': {}}
                            coords = []
                            for t in list(sd_suav.keys()):
                                sem, status, kf_est, _ = procesar_ekF_lambda(sd_suav[t], nav, sp3, kf_est, t, m, snr)
                                if sem:
                                    la, lo, al = ecef_a_geodesicas(sem[0], sem[1], sem[2])
                                    nt, et = geodesicas_a_utm(la, lo, utm_h)
                                    coords.append((nt, et, al, status))
                            
                            if not coords: continue
                            for cp in set(cp_grid):
                                for ca in set(ca_grid):
                                    res = estadistica_desacoplada(coords, cp, ca, best_eh, best_ev)
                                    if res[0] is None: continue
                                    nf, ef, zf, std_n, std_e, std_z, ret, fix_ratio = res
                                    if ret < max(15, int(len(coords) * 0.05)): continue
                                    
                                    rmse_3d = math.sqrt((nf - utm_n_r)**2 + (ef - utm_e_r)**2 + (zf - utm_c_r)**2)
                                    score = (rmse_3d ** 3) * (1.0 + gap * 0.05) * (1.0 + (1.0 - (fix_ratio/100.0)) * 2.0)
                                    
                                    if score < global_best_score:
                                        global_best_score = score
                                        best_rmse = rmse_3d
                                        best_params = {
                                            'mask': m, 'cp': cp, 'ca': ca, 'eh': best_eh, 'ev': best_ev,
                                            'max_gap': gap, 'snr': snr, 'rmse': rmse_3d, 'ret': ret,
                                            'dn': nf - utm_n_r, 'de': ef - utm_e_r, 'dz': zf - utm_c_r
                                        }
                
                if global_best_score != float('inf'):
                    m_center, m_span = best_params['mask'], m_span / 2.0
                    cp_center, cp_span = best_params['cp'], cp_span / 2.0
                    ca_center, ca_span = best_params['ca'], ca_span / 2.0
                    snr_center, snr_span = best_params['snr'], snr_span / 2.0
                    gap_center, gap_span = best_params['max_gap'], gap_span / 2.0
                else: m_span /= 2.0; cp_span /= 2.0; ca_span /= 2.0; snr_span /= 2.0; gap_span /= 2.0
            
            # [VERSIÓN 9 ARREGLADA]: Restauración de la caja negra detallada
            if global_best_score != float('inf'):
                yield "\n========================================================\n"
                yield "      [INFORME] PARÁMETROS ÓPTIMOS (CALIBRACIÓN EKF/PPK)\n"
                yield "========================================================\n"
                yield f"  [-] Tolerancia Sync (max_gap): {f_14(best_params['max_gap'])}\n"
                yield f"  [-] Máscara SNR (dBHz): {f_14(best_params['snr'])}\n"
                yield f"  [-] Máscara Elevación (°): {f_14(best_params['mask'])}\n"
                yield f"  [-] Filtro Sigma Plan (cp): {f_14(best_params['cp'])}\n"
                yield f"  [-] Filtro Sigma Alt (ca): {f_14(best_params['ca'])}\n"
                yield f"  [-] Error Permitido Horizontal (m): {f_14(best_params['eh'])}\n"
                yield f"  [-] Error Permitido Vertical (m): {f_14(best_params['ev'])}\n"
                yield "--------------------------------------------------------\n"
                yield f"  [*] RMSE Global 3D al Punto: {f_14(best_params['rmse'])} m\n"
                yield f"  [*] Deltas Residuales -> N: {f_14(best_params['dn'])}m, E: {f_14(best_params['de'])}m, Z: {f_14(best_params['dz'])}m\n"
                yield f"  [*] Épocas Retenidas EKF: {best_params['ret']}\n"
                yield "========================================================\n"
                yield "\n[SUCCESS]"
            else: yield "\n> [ERROR] El modelo Kalman no convergió.\n"
        except Exception as e: yield f"\n> [ERROR FATAL] {str(e)}"
    return Response(procesar(), mimetype='text/plain')

@app.route('/API/tab4_procesar', methods=['POST'])
def tab4_procesar():
    utm_n, utm_e, utm_c = safe_f(request.form.get('utm_norte')), safe_f(request.form.get('utm_este')), safe_f(request.form.get('utm_cota'))
    utm_h, utm_hem = safe_i(request.form.get('utm_huso'), 19), request.form.get('utm_hemisferio', 'N')
    h_b, h_r = safe_f(request.form.get('altura_base')), safe_f(request.form.get('altura_rover'))
    p_mask, p_snr = safe_f(request.form.get('param_mask'), 10.0), safe_f(request.form.get('param_snr'), 25.0)
    p_cp, p_ca = safe_f(request.form.get('param_cp'), 2.5), safe_f(request.form.get('param_ca'), 1.5)
    err_hor_max, err_ver_max = safe_f(request.form.get('err_hor_max'), 0.5), safe_f(request.form.get('err_ver_max'), 0.5)
    p_max_gap = safe_f(request.form.get('param_max_gap'), 0.5)
    url_rover_nuevo = request.form.get('url_rover_nuevo')
    if not url_rover_nuevo: return Response("> [ERROR] Falta RINEX Rover.\n", mimetype='text/plain')

    p_r_nuevo = os.path.join(UPLOAD_FOLDER, 'rover_nuevo_raw.obs')

    def procesar():
        try:
            yield "> [RED] Descargando Nuevo RINEX Rover...\n"
            descargar_desde_gdrive(url_rover_nuevo, p_r_nuevo)
            yield f"\n> [SISTEMA] Iniciando Procesamiento (EKF + RTS Smoother)...\n"
            
            nav_path, sp3_path, p_b_raw = leer_estado('nav_path'), leer_estado('sp3_path'), leer_estado('base_raw') 
            if not nav_path or not p_b_raw: yield "> [ERROR FATAL] Faltan archivos Base/Efemerides.\n"; return

            obs_b_raw, obs_r_raw = parse_rinex_obs_completo(p_b_raw), parse_rinex_obs_completo(p_r_nuevo) 
            nav, sp3 = parse_rinex_nav_real(nav_path), parse_sp3_preciso(sp3_path) if sp3_path else {}
            
            rover_tows, base_tows = sorted(list(obs_r_raw.keys())), sorted(list(obs_b_raw.keys()))
            obs_b_sync = {}
            for tr in rover_tows:
                if not base_tows: continue
                idx = min(range(len(base_tows)), key=lambda i: abs(base_tows[i] - tr))
                if abs(base_tows[idx] - tr) <= p_max_gap:
                    obs_b_sync[tr] = obs_b_raw[base_tows[idx]].copy()
                    obs_b_sync[tr]['_meta'] = obs_r_raw[tr]['_meta']
            
            sd_suavizada = aislar_diferencias_simples_ppk(obs_b_sync, obs_r_raw)
            if not sd_suavizada: yield "\n> [ERROR] No hay épocas sincronizadas válidas.\n"; return

            lat_b, lon_b, _ = utm_a_geodesicas(utm_e, utm_n, utm_h, utm_hem)
            X_b, Y_b, Z_b = geodesicas_a_ecef(lat_b, lon_b, utm_c + h_b)
            X_bg, Y_bg, Z_bg = geodesicas_a_ecef(lat_b, lon_b, utm_c)

            P_init = matid(3)
            for i in range(3): P_init[i][i] = 100.0
            kf_est = {'X': [[X_bg], [Y_bg], [Z_bg]], 'P': P_init, 'X_base': (X_b, Y_b, Z_b), 'fix_flags': 0, 'h_r': h_r, 'cs_state': {}}
            fwd_states = []
            
            yield "[PROGRESO] Fase 1: Pasada Forward EKF + Klobuchar/IF Adaptativo...\n"
            for t in sd_suavizada:
                sem, status, kf_est, st_dict = procesar_ekF_lambda(sd_suavizada[t], nav, sp3, kf_est, t, p_mask, p_snr)
                if sem and st_dict:
                    st_dict['status'] = status
                    fwd_states.append(st_dict)

            if not fwd_states: yield "\n> [ERROR] Colapso total del Filtro Kalman.\n"; return
            
            yield "[PROGRESO] Fase 2: Suavizador RTS Bidireccional...\n"
            sm_states = suavizador_rts_backward(fwd_states)
            coords = []
            for i in range(len(sm_states)):
                la, lo, al = ecef_a_geodesicas(sm_states[i][0][0], sm_states[i][1][0], sm_states[i][2][0])
                nt, et = geodesicas_a_utm(la, lo, utm_h)
                coords.append((nt, et, al, fwd_states[i]['status']))

            res_estadistica = estadistica_desacoplada(coords, p_cp, p_ca, err_hor_max, err_ver_max)
            if res_estadistica[0] is None: yield "\n> [ERROR] 100% de épocas superan Error Máximo.\n"; return
                
            nf, ef, zf, std_n, std_e, std_z, ret, fix_ratio = res_estadistica
            p_dict = {
                'mask': p_mask, 'cp': p_cp, 'ca': p_ca, 'max_gap': p_max_gap, 'snr': p_snr,
                'err_h': err_hor_max, 'err_v': err_ver_max, 'nf': nf, 'ef': ef, 'zf': zf, 
                'ret': ret, 'total': len(coords), 'std_n': std_n, 'std_e': std_e, 'std_z': std_z,
                'fix_r': fix_ratio, 'base_file': leer_estado('name_base_raw') or "Base.obs",
                'rover_file': "Drive_Nuevo_Rover.obs", 'nav_file': leer_estado('name_nav_file') or "auto.nav",
                'sp3_file': leer_estado('name_sp3_file'), 'b_n': utm_n, 'b_e': utm_e, 'b_z': utm_c,
                'r_n_calc': nf, 'r_e_calc': ef, 'r_z_calc': zf
            }
            yield generar_informe_ascii("MEDICION", p_dict)
            yield "\n[SUCCESS]"
        except Exception as e: yield f"\n> [ERROR FATAL] {str(e)}"
    return Response(procesar(), mimetype='text/plain')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000, debug=True)

