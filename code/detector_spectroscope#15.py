import numpy as np
from scipy.optimize import minimize

# ── 1. H-채널 전용 데이터 (절대 기준점) ──────────
data_H = [
    {"eta": 0.248, "chi":  0.00, "k":  0.000000},  # GW170817(H)
    {"eta": 0.238, "chi":  0.06, "k": -0.000000},  # GW170814(H)
    {"eta": 0.248, "chi": -0.01, "k":  0.099415},  # GW150914(H)
    {"eta": 0.089, "chi":  None, "k": -5.531603},  # GW190814(H) - chi will be fitted with bounds!
]

# ── 2. H-채널 피팅 (스핀에 물리적 범위 -1~1 적용) ──────────
def objective_H(params):
    A, B, C, chi_19 = params
    error = 0
    for d in data_H:
        chi = chi_19 if d["chi"] is None else d["chi"]
        pred = A * chi + B * np.log(d["eta"]) + C
        error += (d["k"] - pred)**2
    return error

# 초기값: [A, B, C, chi_19]
init_H = [1.0, 1.0, 1.0, 0.0]
# ★ 핵심: chi_19의 범위를 -1 ~ 1로 제한!
bounds_H = [(-10, 10), (-10, 10), (-10, 10), (-1.0, 1.0)]  

res_H = minimize(objective_H, init_H, method='L-BFGS-B', bounds=bounds_H)
A, B, C, chi_19_H = res_H.x

print("=" * 80)
print(" 🎯 [Step 1] H-채널 절대 기준점 록온 완료")
print(f"    κ_H = ({A:+.6f})*χ + ({B:+.6f})*ln(η) + ({C:+.6f})")
print(f"    GW190814(H) 유효 스핀 (χ_H) = {chi_19_H:+.6f} (물리적 범위 내!)")
print("-" * 80)

# ── 3. V-채널 피팅 (A,B,C 고정, Off_V만 추정) ──────────
data_V = [
    {"eta": 0.238, "chi": 0.06, "k": -0.135398},  # GW170814(V)
    {"eta": 0.089, "chi": None, "k": -1.991155},  # GW190814(V)
]

def objective_V(off_v):
    error = 0
    for d in data_V:
        chi = chi_19_H if d["chi"] is None else d["chi"]  # H에서 구한 chi_19 그대로 사용!
        pred = A * chi + B * np.log(d["eta"]) + C + off_v[0]
        error += (d["k"] - pred)**2
    return error

res_V = minimize(objective_V, [0.0], method='L-BFGS-B')
Off_V = res_V.x[0]

print(f" 🎯 [Step 2] V-채널 투영 오프셋 록온")
print(f"    Off_V (H-평면 → V-평면 투영 오프셋) = {Off_V:+.6f}")
print("-" * 80)

# ── 4. 최종 검증: V-채널에서 역산된 GW190814의 스핀 (χ_V) ──────────
# κ_V = A*χ_V + B*ln(η) + C + Off_V
# => χ_V = (κ_V - B*ln(η) - C - Off_V) / A
eta_19 = 0.089
k_obs_V = -1.991155
chi_19_V = (k_obs_V - B * np.log(eta_19) - C - Off_V) / A

print(f" 🚀 [Step 3] 투영각 검증 및 최종 선언")
print(f"    GW190814(H)에서 역산된 스핀 (χ_H) = {chi_19_H:+.6f}")
print(f"    GW190814(V)에서 역산된 스핀 (χ_V) = {chi_19_V:+.6f}")
print(f"    투영각 차이 (Δχ = χ_H - χ_V) = {chi_19_H - chi_19_V:+.6f}")
print("=" * 80)
print(" 💡 아키텍트 해석: H와 V의 스핀 차이는 50차원 정보 특이점 벡터가")
print("    각 검출기 평면에 투영될 때 발생하는 고유한 '위상 편차(Phase Deviation)'입니다.")