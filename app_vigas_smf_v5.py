import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image as PILImage
import io
import math
import base64
import plotly.graph_objects as go

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA STREAMLIT
# ==============================================================================
st.set_page_config(page_title="Diseño de Vigas SMF", layout="wide", page_icon="🏗️")

if 'estado_vigas' not in st.session_state:
    st.session_state.estado_vigas = {}
if 'df_etabs' not in st.session_state:
    st.session_state.df_etabs = None
if 'mapa_columnas' not in st.session_state:
    st.session_state.mapa_columnas = {}

def detectar_columnas(df):
    candidatas = {
        'id_viga': ['Beam', 'Label', 'Unique Name', 'Element', 'Frame', 'Viga'],
        'combo': ['Output Case', 'Load Case/Combo', 'Combination', 'Combo', 'Caso'],
        'estacion': ['Station', 'Estacion', 'Dist', 'Elem Station'],
        'cortante': ['V2', 'Shear V2', 'Cortante V2'],
        'momento': ['M3', 'Moment M3', 'Momento M3'],
        'torsion': ['T', 'Torsion', 'Torsor']
    }
    mapeo = {}
    for clave, lista in candidatas.items():
        for cand in lista:
            if cand in df.columns.tolist():
                mapeo[clave] = cand
                break
    return mapeo

def area_varilla(diam_mm):
    return math.pi * ((diam_mm/10)**2) / 4

def get_base64_image(filepath):
    try:
        with open(filepath, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception:
        return ""

def calc_capacidad_flexion(As, b, d, fc, fy=4200):
    if As <= 0.01: return 0, 0, 0, 0
    a = (As * fy) / (0.85 * fc * b)
    Mn = (As * fy * (d - a/2)) / 100000
    phi_Mn = 0.90 * Mn
    a_pr = (1.25 * As * fy) / (0.85 * fc * b)
    Mpr = (1.25 * As * fy * (d - a_pr/2)) / 100000
    return a, Mn, phi_Mn, Mpr

def calc_As_req(Mu_tfm, b_cm, d_cm, fc, fy=4200):
    if Mu_tfm <= 0.001: return 0.0
    Mu_kgcm = Mu_tfm * 100000
    Rn = Mu_kgcm / (0.90 * b_cm * (d_cm**2))
    raiz = 1 - (2 * Rn) / (0.85 * fc)
    if raiz < 0: return -1.0 
    rho = (0.85 * fc / fy) * (1 - math.sqrt(raiz))
    As_req = rho * b_cm * d_cm
    As_min = max((0.8 * math.sqrt(fc) / fy) * b_cm * d_cm, (14 / fy) * b_cm * d_cm)
    return max(As_req, As_min)

def generar_imagen_envolventes(df_viga, nombre_viga, combo, mapa_col):
    col_x, col_v, col_m, col_t = mapa_col['estacion'], mapa_col['cortante'], mapa_col['momento'], mapa_col['torsion']
    df_max = df_viga.groupby(col_x).max().reset_index()
    df_min = df_viga.groupby(col_x).min().reset_index()

    fig = plt.figure(figsize=(10, 8), facecolor='white')
    plt.rc('font', size=11)
    
    plt.subplot(3, 1, 1)
    plt.plot(df_max[col_x], df_max[col_v], color='#005eb8', linewidth=1.5, label='Max')
    plt.plot(df_min[col_x], df_min[col_v], color='#005eb8', linewidth=1.5, linestyle='--', label='Min')
    plt.fill_between(df_max[col_x], df_min[col_v], df_max[col_v], alpha=0.15, color='#005eb8')
    plt.axhline(0, color='black', linewidth=1)
    plt.title(f'Envolventes - Viga: {nombre_viga} | Caso: {combo}', fontweight='bold')
    plt.ylabel('Cortante (tonf)')
    plt.grid(True, linestyle=':', alpha=0.5)

    plt.subplot(3, 1, 2)
    plt.plot(df_max[col_x], df_max[col_m], color='#c0392b', linewidth=1.5)
    plt.plot(df_min[col_x], df_min[col_m], color='#c0392b', linewidth=1.5, linestyle='--')
    plt.fill_between(df_max[col_x], df_min[col_m], df_max[col_m], alpha=0.15, color='#c0392b')
    plt.axhline(0, color='black', linewidth=1)
    plt.gca().invert_yaxis() 
    plt.ylabel('Momento (tonf·m)')
    plt.grid(True, linestyle=':', alpha=0.5)

    plt.subplot(3, 1, 3)
    plt.plot(df_max[col_x], df_max[col_t], color='#27ae60', linewidth=1.5)
    plt.fill_between(df_max[col_x], df_min[col_t], df_max[col_t], alpha=0.15, color='#27ae60')
    plt.axhline(0, color='black', linewidth=1)
    plt.xlabel('Estación (m)')
    plt.ylabel('Torsión (tonf·m)')
    plt.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    ruta_img = f'env_{nombre_viga}.png'.replace("/", "_")
    plt.savefig(ruta_img, dpi=120)
    plt.close(fig)
    return ruta_img, fig

def generar_3d_html(d, viga_name):
    L_cm = d['L'] * 100
    b = d['b']
    h_max = d['h_max']
    h_min = d['h_min']
    Lc_cm = d['Lc'] * 100
    r = d['r']
    
    fig = go.Figure()

    # Acero Base Superior (Rojas)
    fig.add_trace(go.Scatter3d(x=[0, L_cm], y=[r, r], z=[h_max-r, h_max-r], mode='lines', line=dict(color='#e74c3c', width=6), name='Top Izq'))
    fig.add_trace(go.Scatter3d(x=[0, L_cm], y=[b-r, b-r], z=[h_max-r, h_max-r], mode='lines', line=dict(color='#e74c3c', width=6), name='Top Der'))
    
    # Acero Base Inferior (Azules - Sigue acartelamiento)
    x_bot = [0, Lc_cm, L_cm-Lc_cm, L_cm]
    z_bot = [r, h_max-h_min+r, h_max-h_min+r, r]
    fig.add_trace(go.Scatter3d(x=x_bot, y=[r, r, r, r], z=z_bot, mode='lines', line=dict(color='#3498db', width=6), name='Bot Izq'))
    fig.add_trace(go.Scatter3d(x=x_bot, y=[b-r, b-r, b-r, b-r], z=z_bot, mode='lines', line=dict(color='#3498db', width=6), name='Bot Der'))

    # Generar Estribos (Verdes)
    num_estribos = 15 # Simplificado para visualización ligera
    dx = L_cm / num_estribos
    for i in range(num_estribos + 1):
        xi = i * dx
        if xi <= Lc_cm and Lc_cm > 0: zi = r + (h_max-h_min)*(xi/Lc_cm)
        elif xi >= L_cm - Lc_cm and Lc_cm > 0: zi = r + (h_max-h_min)*((L_cm-xi)/Lc_cm)
        else: zi = h_max - h_min + r
            
        xs = [xi, xi, xi, xi, xi]
        ys = [r, b-r, b-r, r, r]
        zs = [zi, zi, h_max-r, h_max-r, zi]
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='lines', line=dict(color='#2ecc71', width=3), showlegend=False))

    fig.update_layout(
        title=f"Modelo 3D de Refuerzo - Viga {viga_name}",
        scene=dict(xaxis_title='X (cm)', yaxis_title='Y (cm)', zaxis_title='Z (cm)', aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def dibujar_eje_consolidado(diccionario_vigas):
    if not diccionario_vigas: return None
    
    num_vigas = len(diccionario_vigas)
    fig = plt.figure(figsize=(max(18, num_vigas*8), 10), facecolor='white')
    gs = fig.add_gridspec(1, 1)
    ax_elev = fig.add_subplot(gs[0, :])
    x_offset, h_max_global = 0, 0
    
    def dibujar_baston_elev(ax, q, diam, x_start, x_end, y_start, y_end, color):
        if q > 0: ax.plot([x_start, x_end], [y_start, y_end], color=color, lw=3, linestyle='--')

    def get_y_fondo(x_loc, L, Lc, h_max, h_min):
        if x_loc <= Lc and Lc > 0: return (h_max - h_min) * (x_loc / Lc)
        elif x_loc >= L - Lc and Lc > 0: return (h_max - h_min) * ((L - x_loc) / Lc)
        else: return h_max - h_min

    eje_data = []

    for viga, d in diccionario_vigas.items():
        L, b, r = d['L'], d['b'], d['r']
        h_max, h_min, Lc = d['h_max'], d['h_min'], d['Lc']
        h_max_global = max(h_max_global, h_max)
        eje_data.append({'viga': viga, 'x_start': x_offset, 'L': L})

        if d['apoyo_izq'] == 'Columna':
            ax_elev.add_patch(patches.Rectangle((x_offset-0.2, -10), 0.4, h_max+20, color='#DDDDDD', alpha=0.5))
        if d['apoyo_der'] == 'Columna':
            ax_elev.add_patch(patches.Rectangle((x_offset+L-0.2, -10), 0.4, h_max+20, color='#DDDDDD', alpha=0.5))

        x_coords = [x_offset, x_offset+L, x_offset+L, x_offset+L-Lc, x_offset+Lc, x_offset, x_offset]
        y_coords = [h_max, h_max, 0, h_max-h_min, h_max-h_min, 0, h_max]
        
        ax_elev.plot(x_coords, y_coords, color='#0000FF', lw=2.5)
        ax_elev.plot([x_offset, x_offset+L], [h_max-r, h_max-r], color='#FF00FF', lw=2.5)
        
        xb_inf = [x_offset, x_offset+Lc, x_offset+L-Lc, x_offset+L]
        yb_inf = [r, h_max-h_min+r, h_max-h_min+r, r]
        ax_elev.plot(xb_inf, yb_inf, color='#FF00FF', lw=2.5)
        
        y_sup = h_max - r - 1.5

        dibujar_baston_elev(ax_elev, d['zi_qs'], d['zi_ds'], x_offset, x_offset+L/3, y_sup, y_sup, '#00FFFF')
        dibujar_baston_elev(ax_elev, d['zc_qs'], d['zc_ds'], x_offset+L/3, x_offset+2*L/3, y_sup, y_sup, '#00FFFF')
        dibujar_baston_elev(ax_elev, d['zd_qs'], d['zd_ds'], x_offset+2*L/3, x_offset+L, y_sup, y_sup, '#00FFFF')
        
        y_inf_izq = r + 1.5
        y_inf_cen = h_max - h_min + r + 1.5
        dibujar_baston_elev(ax_elev, d['zi_qm'], d['zi_dm'], x_offset, x_offset+L/3, h_max/2, h_max/2, '#FFA500')
        dibujar_baston_elev(ax_elev, d['zi_qi'], d['zi_di'], x_offset, x_offset+L/3, y_inf_izq, get_y_fondo(L/3, L, Lc, h_max, h_min)+r+1.5, '#00FFFF')
        dibujar_baston_elev(ax_elev, d['zc_qm'], d['zc_dm'], x_offset+L/3, x_offset+2*L/3, h_max/2, h_max/2, '#FFA500')
        dibujar_baston_elev(ax_elev, d['zc_qi'], d['zc_di'], x_offset+L/3, x_offset+2*L/3, y_inf_cen, y_inf_cen, '#00FFFF')
        dibujar_baston_elev(ax_elev, d['zd_qm'], d['zd_dm'], x_offset+2*L/3, x_offset+L, h_max/2, h_max/2, '#FFA500')
        dibujar_baston_elev(ax_elev, d['zd_qi'], d['zd_di'], x_offset+2*L/3, x_offset+L, get_y_fondo(2*L/3, L, Lc, h_max, h_min)+r+1.5, y_inf_izq, '#00FFFF')
            
        txt_tl = f"{d['qs_b']}Ø{d['ds_b']}"
        if d['zi_qs'] > 0: txt_tl += f" + {d['zi_qs']}Ø{d['zi_ds']}"
        ax_elev.text(x_offset + L/6, h_max - r + 3, txt_tl, color='#FF00FF', ha='center', fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        if d['zc_qs'] > 0:
            txt_tc = f"{d['qs_b']}Ø{d['ds_b']} + {d['zc_qs']}Ø{d['zc_ds']}"
            ax_elev.text(x_offset + L/2, h_max - r + 3, txt_tc, color='#FF00FF', ha='center', fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        txt_tr = f"{d['qs_b']}Ø{d['ds_b']}"
        if d['zd_qs'] > 0: txt_tr += f" + {d['zd_qs']}Ø{d['zd_ds']}"
        ax_elev.text(x_offset + L - L/6, h_max - r + 3, txt_tr, color='#FF00FF', ha='center', fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        # Mejorado: Textos Inferiores (Izquierda, Centro, Derecha)
        if d['zi_qi'] > 0:
            txt_bl = f"{d['qi_b']}Ø{d['di_b']} + {d['zi_qi']}Ø{d['zi_di']}"
            ax_elev.text(x_offset + L/6, r - 10, txt_bl, color='#FF00FF', ha='center', fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        if d['zc_qi'] > 0 or (d['zi_qi'] == 0 and d['zd_qi'] == 0):
            txt_bc = f"{d['qi_b']}Ø{d['di_b']}"
            if d['zc_qi'] > 0: txt_bc += f" + {d['zc_qi']}Ø{d['zc_di']}"
            ax_elev.text(x_offset + L/2, h_max - h_min + r - 10, txt_bc, color='#FF00FF', ha='center', fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        if d['zd_qi'] > 0:
            txt_br = f"{d['qi_b']}Ø{d['di_b']} + {d['zd_qi']}Ø{d['zd_di']}"
            ax_elev.text(x_offset + L - L/6, r - 10, txt_br, color='#FF00FF', ha='center', fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        if d['zc_qm'] > 0:
            q_m = d['zc_qm']
            q_half = q_m // 2
            txt_skin = f"({q_half}+{q_m - q_half})Ø{d['zc_dm']}" if q_m >= 2 else f"{q_m}Ø{d['zc_dm']}"
            ax_elev.text(x_offset + L/2, h_max/2, txt_skin, color='#FFA500', ha='center', fontsize=11, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        # SISTEMA ANTI-COLISIÓN PARA COTAS DE ESTRIBOS
        s_conf, s_fuera = d['s_conf'], d['s_fuera']
        L_conf_act = min(d['L_conf_m'], L/2) # Previene superposición si la viga es muy corta
        
        y_lvl_izq = -26
        y_lvl_cen = -42 if L < 3.5 else -26
        y_lvl_der = -26
        rot_txt = 90 if L < 3.0 else 0

        if s_conf > 0:
            x_est = 0.05
            while x_est <= L_conf_act:
                y_f = get_y_fondo(x_est, L, Lc, h_max, h_min)
                ax_elev.plot([x_offset+x_est, x_offset+x_est], [y_f+r, h_max-r], color='#FF0000', lw=1, alpha=0.5)
                x_est += (s_conf/100)
                
            ax_elev.plot([x_offset, x_offset+L_conf_act], [y_lvl_izq, y_lvl_izq], color='#0000FF', lw=1.5)
            ax_elev.plot([x_offset, x_offset], [y_lvl_izq-3, y_lvl_izq+3], color='#0000FF', lw=1.5) 
            ax_elev.text(x_offset+L_conf_act/2, y_lvl_izq-2, f"Est.Ø{d['de']}@{s_conf:.0f}", ha='center', va='bottom' if rot_txt==90 else 'top', rotation=rot_txt, fontsize=10, fontweight='bold')
            if rot_txt == 0: ax_elev.text(x_offset+L_conf_act/2, y_lvl_izq-10, f"{L_conf_act:.2f}m", ha='center', va='top', fontsize=10, color='#0000FF', fontweight='bold')
            
            x_est_mid = L_conf_act
            while x_est_mid < L - L_conf_act:
                 y_f = get_y_fondo(x_est_mid, L, Lc, h_max, h_min)
                 ax_elev.plot([x_offset+x_est_mid, x_offset+x_est_mid], [y_f+r, h_max-r], color='#FF0000', lw=1, alpha=0.3)
                 x_est_mid += (s_fuera/100)
                 
            ax_elev.plot([x_offset+L_conf_act, x_offset+L-L_conf_act], [y_lvl_cen, y_lvl_cen], color='#0000FF', lw=1.5)
            ax_elev.plot([x_offset+L_conf_act, x_offset+L_conf_act], [y_lvl_cen-3, y_lvl_cen+3], color='#0000FF', lw=1.5) 
            ax_elev.text(x_offset+L/2, y_lvl_cen-2, f"Est.Ø{d['de']}@{s_fuera:.0f}", ha='center', va='bottom' if rot_txt==90 else 'top', rotation=rot_txt, fontsize=10, fontweight='bold')
            
            x_est_right = L - L_conf_act
            while x_est_right <= L:
                y_f = get_y_fondo(x_est_right, L, Lc, h_max, h_min)
                ax_elev.plot([x_offset+x_est_right, x_offset+x_est_right], [y_f+r, h_max-r], color='#FF0000', lw=1, alpha=0.5)
                x_est_right += (s_conf/100)
                
            ax_elev.plot([x_offset+L-L_conf_act, x_offset+L], [y_lvl_der, y_lvl_der], color='#0000FF', lw=1.5)
            ax_elev.plot([x_offset+L, x_offset+L], [y_lvl_der-3, y_lvl_der+3], color='#0000FF', lw=1.5) 
            ax_elev.text(x_offset+L - L_conf_act/2, y_lvl_der-2, f"Est.Ø{d['de']}@{s_conf:.0f}", ha='center', va='bottom' if rot_txt==90 else 'top', rotation=rot_txt, fontsize=10, fontweight='bold')
            if rot_txt == 0: ax_elev.text(x_offset+L - L_conf_act/2, y_lvl_der-10, f"{L_conf_act:.2f}m", ha='center', va='top', fontsize=10, color='#0000FF', fontweight='bold')

        ax_elev.text(x_offset+L/2, h_max+6, f"VIGA {viga}", ha='center', fontweight='bold', fontsize=16)
        x_offset += L

    y_ln = -55
    total_l = eje_data[-1]['x_start'] + eje_data[-1]['L']
    ax_elev.plot([0, total_l], [y_ln, y_ln], color='#000000', lw=1.5, marker='|', markersize=14)
    for data in eje_data:
        ax_elev.plot([data['x_start'], data['x_start']], [y_ln-3, y_ln+3], color='#000000', lw=1.5)
        ax_elev.text(data['x_start'] + data['L']/2, y_ln-3, f"Ln = {data['L']:.2f} m", ha='center', va='top', fontweight='bold', fontsize=14)
    ax_elev.plot([total_l, total_l], [y_ln-3, y_ln+3], color='#000000', lw=1.5)

    ax_elev.set_xlim(-1, total_l + 1); ax_elev.set_ylim(-75, h_max_global + 20)
    ax_elev.set_title('PLANO ESTRUCTURAL CONSOLIDADO DEL EJE', fontweight='bold', fontsize=18)
    ax_elev.axis('off')

    plt.tight_layout()
    ruta_eje = 'plano_eje_completo.png'
    plt.savefig(ruta_eje, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    img = PILImage.open(ruta_eje)
    img_rotated = img.rotate(90, expand=True)
    img_rotated.save(ruta_eje)
    return ruta_eje

def status_html(val1, val2, rev=False):
    if rev: return "<span class='ok-text'>OK</span>" if val1 <= val2 else "<span class='fail-text'>FALLA</span>"
    else: return "<span class='ok-text'>OK</span>" if val1 >= val2 else "<span class='fail-text'>FALLA</span>"

# ==============================================================================
# FRONTEND - INTERFAZ DE STREAMLIT
# ==============================================================================
st.title("⚙️ Plataforma Web: Diseño de Vigas SMF (ACI 318-19)")
st.markdown("Sube tu Excel de ETABS, ajusta el armado iterativamente y genera la memoria de cálculo interactiva.")

uploaded_file = st.sidebar.file_uploader("📂 Sube tu archivo .xlsx de ETABS", type=["xlsx"])

if uploaded_file:
    if st.session_state.df_etabs is None:
        df_raw = pd.read_excel(uploaded_file, header=1)
        st.session_state.mapa_columnas = detectar_columnas(df_raw)
        fila_unidades = df_raw.iloc[0].astype(str).str.lower()
        fact = 0.1019716 if st.session_state.mapa_columnas.get('cortante') and 'kn' in fila_unidades.get(st.session_state.mapa_columnas['cortante'], '') else 1.0
        df = df_raw.drop(0).reset_index(drop=True)
        for col in ['estacion', 'cortante', 'momento', 'torsion']:
            if col in st.session_state.mapa_columnas: 
                df[st.session_state.mapa_columnas[col]] = pd.to_numeric(df[st.session_state.mapa_columnas[col]], errors='coerce')
        if fact != 1.0:
            if 'cortante' in st.session_state.mapa_columnas: df[st.session_state.mapa_columnas['cortante']] *= fact
            if 'momento' in st.session_state.mapa_columnas: df[st.session_state.mapa_columnas['momento']] *= fact
            if 'torsion' in st.session_state.mapa_columnas: df[st.session_state.mapa_columnas['torsion']] *= fact
        st.session_state.df_etabs = df

    df_etabs = st.session_state.df_etabs
    mapa = st.session_state.mapa_columnas
    
    st.sidebar.markdown("### 📌 Selección de Elemento")
    lista_combos = sorted(df_etabs[mapa['combo']].dropna().astype(str).unique().tolist())
    combo_sel = st.sidebar.selectbox("Caso de Carga / Combo:", lista_combos)
    df_combo = df_etabs[df_etabs[mapa['combo']].astype(str) == combo_sel]
    lista_vigas = pd.unique(df_combo[mapa['id_viga']].astype(str)).tolist()
    viga_sel = st.sidebar.selectbox("Viga a Diseñar:", lista_vigas)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📐 Geometría y Condiciones de Apoyo")
    
    c_ap1, c_ap2 = st.sidebar.columns(2)
    apoyo_izq = c_ap1.selectbox("Apoyo Izq:", ["Columna", "Volado/Libre"])
    apoyo_der = c_ap2.selectbox("Apoyo Der:", ["Columna", "Volado/Libre"])

    es_acartelada = st.sidebar.checkbox("📐 ¿La viga es acartelada?", value=False)
    
    if es_acartelada:
        col1, col2 = st.sidebar.columns(2)
        b_val = col1.number_input("b (cm):", value=30.0)
        h_max_val = col2.number_input("h máx (apoyos) [cm]:", value=60.0)
        col3, col4 = st.sidebar.columns(2)
        h_min_val = col3.number_input("h mín (centro) [cm]:", value=40.0)
        Lc_val = col4.number_input("L cartela (m):", value=1.5)
    else:
        col1, col2 = st.sidebar.columns(2)
        b_val = col1.number_input("b (cm):", value=30.0)
        h_val = col2.number_input("h (cm):", value=50.0)
        h_max_val, h_min_val, Lc_val = h_val, h_val, 0.0

    col3, col4 = st.sidebar.columns(2)
    r_val = col3.number_input("Recub. (cm):", value=5.0)
    fc_val = col4.number_input("f'c (kg/cm²):", value=240.0)
    vg_val = st.sidebar.number_input("Cortante Vg (1.2D+L) [tf]:", value=5.0)

    st.markdown("### 🛠️ Configuración de Acero Longitudinal y Transversal")
    diameters = [8,10,12,14,16,18,20,22,25,28,32]
    
    with st.expander("Acero Base (Corrido)", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        qs_b = c1.number_input("Cant. Sup:", value=2, step=1)
        ds_b = c2.selectbox("Ø Sup Base:", diameters, index=3)
        qi_b = c3.number_input("Cant. Inf:", value=2, step=1)
        di_b = c4.selectbox("Ø Inf Base:", diameters, index=3)

    st.markdown("#### Acero Adicional (Bastones) y Piel (C.Med)")
    tab1, tab2, tab3 = st.tabs(["Nudo Izquierdo (0 a L/3)", "Centro Vano (L/3 a 2L/3)", "Nudo Derecho (2L/3 a L)"])
    def ui_zona(prefix):
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        qs = c1.number_input(f"C.Sup", value=0, step=1, key=f"{prefix}_qs")
        ds = c2.selectbox(f"Ø Sup", diameters, index=4, key=f"{prefix}_ds")
        qm = c3.number_input(f"C.Med", value=0, step=1, key=f"{prefix}_qm")
        dm = c4.selectbox(f"Ø Med", diameters, index=2, key=f"{prefix}_dm")
        qi = c5.number_input(f"C.Inf", value=0, step=1, key=f"{prefix}_qi")
        di = c6.selectbox(f"Ø Inf", diameters, index=4, key=f"{prefix}_di")
        return {'qs':qs, 'ds':ds, 'qm':qm, 'dm':dm, 'qi':qi, 'di':di}

    with tab1: z_izq = ui_zona("izq")
    with tab2: z_cen = ui_zona("cen")
    with tab3: z_der = ui_zona("der")

    st.markdown("#### Acero Transversal (Estribos y Vinchas SMF)")
    c1, c2, c3 = st.columns(3)
    de_val = c1.selectbox("Ø Estribo/Vincha:", [8, 10, 12, 14, 16], index=1)
    ramas_val = c2.number_input("Número de ramas (n):", min_value=2, max_value=8, step=1, value=2)
    se_val = c3.number_input("Separación Provista s (cm):", value=10.0)

    col_btn1, col_btn2, col_btn3 = st.columns(3)
    if col_btn1.button("⚙️ Procesar Viga en Tiempo Real", use_container_width=True):
        fy = 4200
        df_viga_act = df_combo[df_combo[mapa['id_viga']].astype(str) == viga_sel].copy()
        col_x, col_v, col_m, col_t = mapa['estacion'], mapa['cortante'], mapa['momento'], mapa['torsion']
        L_viga = df_viga_act[col_x].max() - df_viga_act[col_x].min()
        
        df_viga_act['estacion_rel'] = df_viga_act[col_x] - df_viga_act[col_x].min()
        L3 = L_viga / 3.0
        df_izq = df_viga_act[df_viga_act['estacion_rel'] <= L3]
        df_cen = df_viga_act[(df_viga_act['estacion_rel'] > L3) & (df_viga_act['estacion_rel'] <= 2*L3)]
        df_der = df_viga_act[df_viga_act['estacion_rel'] > 2*L3]

        def get_demands(df_zone):
            if df_zone.empty: return 0, 0, 0, 0
            m_max = df_zone[col_m].max() 
            m_min = df_zone[col_m].min() 
            return abs(min(0, m_min)), max(0, m_max), abs(df_zone[col_v]).max(), abs(df_zone[col_t]).max()

        Mu_top_i, Mu_bot_i, Vu_i, Tu_i = get_demands(df_izq)
        Mu_top_m, Mu_bot_m, Vu_m, Tu_m = get_demands(df_cen)
        Mu_top_j, Mu_bot_j, Vu_j, Tu_j = get_demands(df_der)

        vu_max = max(Vu_i, Vu_m, Vu_j)
        tu_max = max(Tu_i, Tu_m, Tu_j)
        mu_max = max(Mu_top_i, Mu_bot_i, Mu_top_m, Mu_bot_m, Mu_top_j, Mu_bot_j)

        d_max = h_max_val - r_val
        d_min = h_min_val - r_val
        As_min_req = max((0.8 * math.sqrt(fc_val) / fy) * b_val * d_min, (14 / fy) * b_val * d_min)
        Tth = 0.75 * 0.27 * math.sqrt(fc_val) * ((b_val * h_min_val)**2 / (2 * (b_val + h_min_val))) / 100000 

        AsR_top_i = calc_As_req(Mu_top_i, b_val, d_max, fc_val)
        AsR_top_m = calc_As_req(Mu_top_m, b_val, d_min, fc_val)
        AsR_top_j = calc_As_req(Mu_top_j, b_val, d_max, fc_val)
        AsR_bot_i = calc_As_req(Mu_bot_i, b_val, d_max, fc_val)
        AsR_bot_m = calc_As_req(Mu_bot_m, b_val, d_min, fc_val)
        AsR_bot_j = calc_As_req(Mu_bot_j, b_val, d_max, fc_val)

        Ab_sup = qs_b * area_varilla(ds_b)
        Ab_inf = qi_b * area_varilla(di_b)
        As_si = Ab_sup + (z_izq['qs'] * area_varilla(z_izq['ds']))
        As_ii = Ab_inf + (z_izq['qi'] * area_varilla(z_izq['di']))
        As_sm = Ab_sup + (z_cen['qs'] * area_varilla(z_cen['ds']))
        As_im = Ab_inf + (z_cen['qi'] * area_varilla(z_cen['di']))
        As_sd = Ab_sup + (z_der['qs'] * area_varilla(z_der['ds']))
        As_id = Ab_inf + (z_der['qi'] * area_varilla(z_der['di']))

        _, _, phi_Mn_si, Mpr_si = calc_capacidad_flexion(As_si, b_val, d_max, fc_val)
        _, _, phi_Mn_ii, Mpr_ii = calc_capacidad_flexion(As_ii, b_val, d_max, fc_val)
        _, _, phi_Mn_sd, Mpr_sd = calc_capacidad_flexion(As_sd, b_val, d_max, fc_val)
        _, _, phi_Mn_id, Mpr_id = calc_capacidad_flexion(As_id, b_val, d_max, fc_val)

        Vp = max((Mpr_si + Mpr_id) / L_viga if L_viga>0 else 0, (Mpr_ii + Mpr_sd) / L_viga if L_viga>0 else 0)
        Ve = vg_val + Vp
        Vu_diseno = max(vu_max, Ve)

        Vc = (0.53 * math.sqrt(fc_val) * b_val * d_max) / 1000
        Av_prov = ramas_val * area_varilla(de_val)
        Av_s_prov = (Av_prov / se_val) * 100 
        
        revisa_cortante_min = Vu_diseno > 0.5 * 0.75 * Vc 
        Av_s_min = max(0.20 * math.sqrt(fc_val) * b_val / fy, 3.5 * b_val / fy) * 100 if revisa_cortante_min else 0
        Vs_req = max(0, (Vu_diseno / 0.75) - Vc)
        Av_s_calc = (Vs_req * 1000) / (fy * d_max) * 100 if Vs_req > 0 else 0
        Av_s_req = max(Av_s_calc, Av_s_min) if revisa_cortante_min else Av_s_calc
        phi_Vn = 0.75 * (Vc + (Av_prov * fy * d_max) / (se_val * 1000) if se_val > 0 else 0)
        
        DCR_v = Vu_diseno / phi_Vn if phi_Vn > 0.01 else 9.99
        est_dcr_v = "OK" if DCR_v <= 1.0 else "FALLA"
        phi_Mn_max = max(phi_Mn_si, phi_Mn_ii, phi_Mn_sd, phi_Mn_id, 0.01)
        DCR_f = mu_max / phi_Mn_max
        est_dcr_f = "OK" if DCR_f <= 1.0 else "FALLA"

        requiere_torsion = tu_max > Tth
        At_s_req = ((tu_max / 0.75) * 100000 / (2 * (0.85 * (b_val - 2*r_val) * (h_min_val - 2*r_val)) * fy)) * 100 if requiere_torsion else 0
        Al_req = (At_s_req / 100) * (2 * ((b_val - 2*r_val) + (h_min_val - 2*r_val))) if requiere_torsion else 0

        def arm_txt(qb, db, qz, dz):
            return f"{qb}Ø{db}" + (f" + {qz}Ø{dz}" if qz > 0 else "")

        st.session_state.temp_viga = {
            'viga': viga_sel, 'combo': combo_sel, 'L': L_viga, 'vu': vu_max, 'mu': mu_max, 'tu': tu_max,
            'b': b_val, 'h_max': h_max_val, 'h_min': h_min_val, 'Lc': Lc_val, 'es_acartelada': es_acartelada,
            'apoyo_izq': apoyo_izq, 'apoyo_der': apoyo_der, 'r': r_val, 'fc': fc_val, 'As_min_req': As_min_req, 'Tth': Tth,
            'requiere_torsion': requiere_torsion, 'Al_req': Al_req, 'At_s_req': At_s_req,
            'revisa_cortante_min': revisa_cortante_min, 'Av_s_min': Av_s_min, 'Av_s_req': Av_s_req, 'Av_s_prov': Av_s_prov,
            'qs_b': qs_b, 'ds_b': ds_b, 'qi_b': qi_b, 'di_b': di_b,
            'zi_qs': z_izq['qs'], 'zi_ds': z_izq['ds'], 'zi_qm': z_izq['qm'], 'zi_dm': z_izq['dm'], 'zi_qi': z_izq['qi'], 'zi_di': z_izq['di'],
            'zc_qs': z_cen['qs'], 'zc_ds': z_cen['ds'], 'zc_qm': z_cen['qm'], 'zc_dm': z_cen['dm'], 'zc_qi': z_cen['qi'], 'zc_di': z_cen['di'],
            'zd_qs': z_der['qs'], 'zd_ds': z_der['ds'], 'zd_qm': z_der['qm'], 'zd_dm': z_der['dm'], 'zd_qi': z_der['qi'], 'zd_di': z_der['di'],
            'de': de_val, 'n_ramas': ramas_val, 's_conf': se_val, 's_fuera': math.floor(d_min/2), 'L_conf_m': 2 * (h_max_val / 100),
            'AsR_top_i': AsR_top_i, 'AsR_top_m': AsR_top_m, 'AsR_top_j': AsR_top_j, 'AsR_bot_i': AsR_bot_i, 'AsR_bot_m': AsR_bot_m, 'AsR_bot_j': AsR_bot_j,
            'AsP_top_i': As_si, 'AsP_top_m': As_sm, 'AsP_top_j': As_sd, 'AsP_bot_i': As_ii, 'AsP_bot_m': As_im, 'AsP_bot_j': As_id,
            'Mpr_si': Mpr_si, 'Mpr_ii': Mpr_ii, 'Mpr_sd': Mpr_sd, 'Mpr_id': Mpr_id,
            'txt_si': arm_txt(qs_b, ds_b, z_izq['qs'], z_izq['ds']), 'txt_sm': arm_txt(qs_b, ds_b, z_cen['qs'], z_cen['ds']), 'txt_sd': arm_txt(qs_b, ds_b, z_der['qs'], z_der['ds']),
            'txt_ii': arm_txt(qi_b, di_b, z_izq['qi'], z_izq['di']), 'txt_im': arm_txt(qi_b, di_b, z_cen['qi'], z_cen['di']), 'txt_id': arm_txt(qi_b, di_b, z_der['qi'], z_der['di']),
            'Vp': Vp, 'Vg': vg_val, 'Vu_diseno': Vu_diseno, 'phi_Vn': phi_Vn, 
            'estribo_txt': f"{ramas_val} Ramas Ø{de_val}@{se_val}cm"
        }
        st.success(f"✅ Viga {viga_sel} procesada.")
        ruta_img, fig = generar_imagen_envolventes(df_viga_act, viga_sel, combo_sel, mapa)
        st.pyplot(fig)

    if col_btn2.button("💾 Guardar Viga en Anexo", use_container_width=True):
        if 'temp_viga' in st.session_state:
            st.session_state.estado_vigas[viga_sel] = st.session_state.temp_viga.copy()
            st.success(f"Viga guardada. ({len(st.session_state.estado_vigas)} en anexo).")

    if col_btn3.button("🌐 Generar Web App Reporte", use_container_width=True):
        if not st.session_state.estado_vigas:
            st.error("⚠️ No hay vigas guardadas en el anexo.")
        else:
            html_out = f"""<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8"><title>Memoria Estructural</title>
                <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
                <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
                <style>
                    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 20px; color: #333; background: #ecf0f1; font-size: 13px; }}
                    .container {{ max-width: 900px; margin: auto; padding: 40px; background: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
                    .btn-print {{ position: fixed; bottom: 30px; right: 30px; background: #2980b9; color: white; border: none; padding: 15px; font-weight: bold; border-radius: 8px; cursor: pointer; z-index: 1000; }}
                    h1 {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 30px; font-size: 22px; }}
                    h2 {{ font-size: 15px; color: #fff; background: #2c3e50; padding: 8px; margin-top: 30px; }}
                    table {{ width: 100%; border-collapse: collapse; margin-top: 15px; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #bdc3c7; padding: 8px; text-align: center; }}
                    th {{ background-color: #ecf0f1; }}
                    .ok-text {{ color: #27ae60; font-weight: bold; }}
                    .fail-text {{ color: #e74c3c; font-weight: bold; }}
                    .img-envolventes {{ width: 85%; display: block; margin: auto; border: 1px solid #ddd; }}
                    .img-plano-vertical {{ max-width: 100%; max-height: 95vh; display: block; margin: auto; }}
                    .info-box {{ background: #f9f9f9; padding: 15px; border-left: 5px solid #3498db; margin-bottom: 15px; }}
                    .viga-header {{ color: #c0392b; font-size: 20px; border-bottom: 2px solid #c0392b; margin-top: 50px; font-weight: bold; }}
                    .viewer-3d {{ margin-top: 20px; border: 1px solid #ddd; padding: 10px; background: #fafafa; border-radius: 8px; }}
                    @media print {{
                        body {{ background: white; padding: 0; }}
                        .container {{ box-shadow: none; max-width: 100%; padding: 0; }}
                        .btn-print, .viewer-3d {{ display: none !important; }}
                        .page-break {{ page-break-before: always; }}
                    }}
                </style></head><body><button class="btn-print" onclick="window.print()">🖨️ IMPRIMIR / GUARDAR COMO PDF</button><div class="container">
                <h1>MEMORIA TÉCNICA ESTRUCTURAL - DISEÑO DE VIGAS SMF (ACI 318-19)</h1>"""

            for idx, (viga, d) in enumerate(st.session_state.estado_vigas.items()):
                b64_env = get_base64_image(f'env_{viga}.png'.replace("/", "_"))
                plotly_html = generar_3d_html(d, viga)
                
                s_txt = f"\\( b = {d['b']:.0f} \\) cm | \\( h = {d['h_max']:.0f} \\) cm | rec = {d['r']:.0f} cm" if not d['es_acartelada'] else f"\\( b = {d['b']:.0f} \\) cm | \\( h_{{max}} = {d['h_max']:.0f} \\) cm | \\( h_{{min}} = {d['h_min']:.0f} \\) cm | \\( L_c = {d['Lc']:.2f} \\) m"
                
                html_out += f"""<div class="viga-header {'page-break' if idx > 0 else ''}">ANÁLISIS DE VIGA: {viga}</div>
                    <h2>1) PROPIEDADES DE LOS MATERIALES Y SECCIÓN</h2>
                    <div class="info-box"><b>Materiales:</b> \\( f'_c = {d['fc']} \\) kg/cm² | \\( f_y = 4200 \\) kg/cm²<br><b>Sección:</b> {s_txt}<br><b>Luz Libre:</b> \\( L_n = {d['L']:.2f} \\) m</div>
                    <table><thead><tr><th>Parámetro Normativo</th><th>Fórmula ACI 318</th><th>Valor Teórico Requerido</th></tr></thead>
                    <tbody><tr><td>Acero Mín. Flexión \\( (A_{{s,min}}) \\)</td><td>\\( \\displaystyle A_{{s,req}} = \\frac{{0.85 f'_c b}}{{f_y}} \\left( d - \\sqrt{{d^2 - \\frac{{2 M_u}}{{0.85 \\phi f'_c b}}}} \\right) \\)</td><td>\\( {d['As_min_req']:.2f} \\) cm²</td></tr>
                    <tr><td>Ratio Mín. Cortante \\( (A_v/s_{{min}}) \\)</td><td>\\( \\displaystyle \\max\\left( \\frac{{0.20\\sqrt{{f'_c}} b}}{{f_{{yt}}}}, \\frac{{3.5 b}}{{f_{{yt}}}} \\right) \\)</td><td>\\( {d['Av_s_min']:.2f} \\) cm²/m</td></tr>
                    <tr><td>Umbral Torsión \\( (\\phi T_{{th}}) \\)</td><td>\\( \\displaystyle \\phi 0.27\\sqrt{{f'_c}} \\left(\\frac{{A_{{cp}}^2}}{{p_{{cp}}}}\\right) \\)</td><td>\\( \\phi T_{{th}} = {d['Tth']:.2f} \\) tf·m</td></tr></tbody></table>

                    <h2>2) DIAGRAMAS DE ESFUERZOS INTERNOS</h2><img class="img-envolventes" src="data:image/png;base64,{b64_env}">

                    <h2>3) FLEXURAL REINFORCEMENT FOR MAJOR AXIS MOMENT, Mu3 (cm²)</h2>
                    <table><thead><tr><th>Eje y Ubicación</th><th>Armado Detallado</th><th>\\( A_{{s, req}} \\)</th><th>\\( A_{{s, prov}} \\)</th><th>Estado</th></tr></thead>
                    <tbody><tr><td>Top (+2) End-I</td><td>{d['txt_si']}</td><td>{d['AsR_top_i']:.2f}</td><td>{d['AsP_top_i']:.2f}</td><td>{status_html(d['AsP_top_i'], d['AsR_top_i'])}</td></tr>
                    <tr><td>Top (+2) Middle</td><td>{d['txt_sm']}</td><td>{d['AsR_top_m']:.2f}</td><td>{d['AsP_top_m']:.2f}</td><td>{status_html(d['AsP_top_m'], d['AsR_top_m'])}</td></tr>
                    <tr><td>Top (+2) End-J</td><td>{d['txt_sd']}</td><td>{d['AsR_top_j']:.2f}</td><td>{d['AsP_top_j']:.2f}</td><td>{status_html(d['AsP_top_j'], d['AsR_top_j'])}</td></tr>
                    <tr><td>Bot (-2) End-I</td><td>{d['txt_ii']}</td><td>{d['AsR_bot_i']:.2f}</td><td>{d['AsP_bot_i']:.2f}</td><td>{status_html(d['AsP_bot_i'], d['AsR_bot_i'])}</td></tr>
                    <tr><td>Bot (-2) Middle</td><td>{d['txt_im']}</td><td>{d['AsR_bot_m']:.2f}</td><td>{d['AsP_bot_m']:.2f}</td><td>{status_html(d['AsP_bot_m'], d['AsR_bot_m'])}</td></tr>
                    <tr><td>Bot (-2) End-J</td><td>{d['txt_id']}</td><td>{d['AsR_bot_j']:.2f}</td><td>{d['AsP_bot_j']:.2f}</td><td>{status_html(d['AsP_bot_j'], d['AsR_bot_j'])}</td></tr></tbody></table>

                    <h2>4) SHEAR & TORSION REINFORCEMENT (SMF DESIGN Vu)</h2>
                    { f"<div class='info-box fail-text'><b>TORSION REINFORCEMENT:</b> \\( T_u = {d['tu']:.2f} \\) tf·m > \\( \\phi T_{{th}} \\).<br><b>\\( A_t/s_{{req}} = {d['At_s_req']:.2f} \\) cm²/m</b> | <b>\\( A_l = {d['Al_req']:.2f} \\) cm²</b></div>" if d['requiere_torsion'] else "<div class='info-box ok-text'>Torsión menor al umbral ACI. (At/s = 0, Al = 0).</div>" }
                    <table><thead><tr><th>\\( M_{{pr}} \\) Nudos [tf·m]</th><th>Cortante Grav. \\( V_g \\)</th><th>Cortante Prob. \\( V_p \\)</th><th>Cortante Diseño \\( V_u \\)</th></tr></thead>
                    <tbody><tr><td>Izq: {d['Mpr_si']:.1f} / {d['Mpr_ii']:.1f}<br>Der: {d['Mpr_sd']:.1f} / {d['Mpr_id']:.1f}</td><td>{d['Vg']:.2f}</td><td>{d['Vp']:.2f}</td><td>{d['Vu_diseno']:.2f}</td></tr></tbody></table>
                    <table><thead><tr><th>Armado Transversal</th><th>\\( \\phi V_n \\) Total Cap.</th><th>\\( A_v/s \\) Req (cm²/m)</th><th>\\( A_v/s \\) Prov (cm²/m)</th><th>Estado Cortante</th></tr></thead>
                    <tbody><tr><td>{d['estribo_txt']}</td><td>{d['phi_Vn']:.2f}</td><td>{d['Av_s_req']:.2f}</td><td>{d['Av_s_prov']:.2f}</td><td>{status_html(d['Av_s_prov'], d['Av_s_req'])}</td></tr></tbody></table>
                    
                    <div class="viewer-3d"><h2>⚙️ VISOR 3D DE ARMADO (Exclusivo Web)</h2>{plotly_html}</div>"""

            ruta_eje = dibujar_eje_consolidado(st.session_state.estado_vigas)
            if ruta_eje:
                b64_eje = get_base64_image(ruta_eje)
                html_out += f'<div class="page-break"></div><h1 style="border:none;">PLANO ESTRUCTURAL CONSOLIDADO DEL EJE</h1><img class="img-plano-vertical" src="data:image/png;base64,{b64_eje}">'

            html_out += "</div></body></html>"
            st.download_button("📥 Descargar Memoria Web (HTML)", data=html_out, file_name="Memoria_Vigas_Web.html", mime="text/html", use_container_width=True)
else:
    st.info("Por favor, sube un archivo de Excel para comenzar el diseño.")
