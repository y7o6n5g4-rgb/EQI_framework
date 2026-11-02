import json
import glob
import numpy as np
import os

# ── 1. EQI 핵심 상수 (Session 19/22에서 록온) ──────────
PHI = (1.0 + np.sqrt(5.0)) / 2.0
GAP = np.log(25.0) / 100.0
D2 = PHI ** 1.5

# RLC 감쇠 상수 및 고유 주파수 (이론 유도에 사용)
GAMMA = GAP / (2 * D2)          # ≈ 0.01605 rad/s
OMEGA_D = np.sqrt(1.0 - GAMMA**2)  # ≈ 0.99987 rad/s

# 청크 시간 (fs=4096, chunk_size=2048)
CHUNK_TIME = 2048 / 4096        # 0.5 초

def compute_bose_fermi_kappa(delta_H_sigma, delta_L_sigma, delta_H_etopo, delta_L_etopo, dt_obs):
    """v10.0 최종 통합: 보손-페르미온 통계적 상전이 계수(κ_Lu) 계산"""
    dh_sig = max(delta_H_sigma, 1e-300)
    dl_sig = max(delta_L_sigma, 1e-300)
    dh_eto = max(delta_H_etopo, 1e-300)
    dl_eto = max(delta_L_etopo, 1e-300)
    
    # 보손 응집도 변화 (D1: 보스-아인슈타인 통계)
    bose_evo = np.log(dh_sig / dh_eto)  
    
    # 페르미온 개별성 변화 (D2: 페르미-디랙 통계)
    fermi_evo = np.log(dl_sig / dl_eto)  
    
    # κ = (보손 변화 - 페르미온 변화) / (지연 시간 - 반주기)
    denominator = dt_obs - (np.pi / OMEGA_D)
    if abs(denominator) < 1e-12:
        return np.inf, bose_evo, fermi_evo  # 순수 보손 응축 (BEC_PURE)
    
    kappa_lu = (bose_evo - fermi_evo) / denominator
    return kappa_lu, bose_evo, fermi_evo

def calculate_lag_metrics(json_path):
    """단일 JSON 파일을 읽어 통계적 상전이 지표를 계산"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    event_name = data.get("event_name", "UNKNOWN")
    targets = data.get("v91_targets", {})
    chunks = data.get("chunk_results", [])
    
    if not chunks:
        return None
    
    # 1. 실측 청크 인덱스 추출
    idx_sigma = targets.get("sigma_ell_min_chunk", -1)  # D1 기하 정렬 (수축 최소점)
    idx_etopo = targets.get("E_topo_min_chunk", -1)     # 위상 에너지 정점
    
    if idx_sigma == -1 or idx_etopo == -1:
        return None
        
    delta_H_sigma = chunks[idx_sigma]["delta_H"]
    delta_L_sigma = chunks[idx_sigma]["delta_L"]
    delta_H_etopo = chunks[idx_etopo]["delta_H"]
    delta_L_etopo = chunks[idx_etopo]["delta_L"]
    
    # 2. 실측 지연 시간 (초)
    dt_obs = (idx_etopo - idx_sigma) * CHUNK_TIME
    
    # 3. 보손-페르미온 혼성 비율 계산 (여씨 상전이 계수 κ_Lu)
    kappa_lu, bose_evo, fermi_evo = compute_bose_fermi_kappa(
        delta_H_sigma, delta_L_sigma, delta_H_etopo, delta_L_etopo, dt_obs
    )
    
    # 4. 통계 역학적 상태 판별
    if np.isinf(kappa_lu):
        stat_status = "순수 보손 응축 (BEC_PURE) / 가역적 초유체"
    elif kappa_lu > 0:
        stat_status = "보손 응축 → 페르미온 기체 붕괴 (블랙홀 증발/측정)"
    elif kappa_lu < 0:
        stat_status = "페르미온 기체 → 보손 응축 재결합 (블랙홀 응집/제트)"
    else:
        stat_status = "보손-페르미온 완벽 대칭"

    return {
        "event": event_name,
        "idx_sigma": idx_sigma,
        "idx_etopo": idx_etopo,
        "dt_obs_sec": dt_obs,
        "bose_evo": bose_evo,
        "fermi_evo": fermi_evo,
        "kappa_lu": kappa_lu,
        "status": stat_status
    }

def main():
    json_files = glob.glob("eqi_monster_v91_ellipse_*.json")
    if not json_files:
        print("❌ 현재 폴더에 분석 결과 JSON 파일이 없습니다.")
        return
    
    print("=" * 90)
    print("  🌌 EQI v10.0 통계적 타원 기하학 검증 (Bose-Fermi Statistical Geometry)")
    print("=" * 90)
    print(f"   🔬 이론 감쇠 상수 γ = {GAMMA:.6f} rad/s")
    print(f"   🔬 고유 주파수 ω_d = {OMEGA_D:.6f} rad/s")
    print(f"   ⏱️  청크당 시간 = {CHUNK_TIME:.3f} 초")
    print("-" * 90)
    
    results = []
    for fpath in sorted(json_files):
        res = calculate_lag_metrics(fpath)
        if res:
            results.append(res)
            print(f"\n📄 {os.path.basename(fpath)}")
            print(f"   이벤트명: {res['event']}")
            print(f"   🎯 기하정렬(Chunk {res['idx_sigma']:3d}) → 에너지정점(Chunk {res['idx_etopo']:3d})")
            print(f"   ⏱️  실측 지연 (dt_obs): {res['dt_obs_sec']:.3f} 초")
            print(f"   🧬 통계적 변화율 | 보손(D1): {res['bose_evo']:.4f} | 페르미온(D2): {res['fermi_evo']:.4f}")
            
            kappa_str = "∞ (BEC_PURE)" if np.isinf(res['kappa_lu']) else f"{res['kappa_lu']:.6f}"
            print(f"   🀄 여씨 상전이 계수 (κ_Lu): {kappa_str}")
            print(f"   ✅ 통계역학 해석: {res['status']}")
    
    print("\n" + "=" * 90)
    print("  📊 v10.0 보손-페르미온 상전이 온도(Crossover) 통합 리포트")
    print("-" * 90)
    print(f"  {'이벤트명':<12} | {'dt_obs':<8} | {'κ_Lu (상전이 계수)':<18} | {'통계역학적 해석'}")
    print("-" * 90)
    for res in results:
        kappa_str = "∞ (BEC_PURE)" if np.isinf(res['kappa_lu']) else f"{res['kappa_lu']:>10.6f}"
        print(f"  {res['event']:<12} | {res['dt_obs_sec']:>6.2f}s | {kappa_str:<18} | {res['status']}")
    print("=" * 90)

if __name__ == "__main__":
    main()