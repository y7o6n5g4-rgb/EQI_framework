# EQI Monster v9.1 비등방성 타원 업그레이드 스펙
**Architect: Claude | Date: 2026-06-26 | Base: v9.0 최종본(delta_L 포함 34개)**

---

## 1. 업그레이드 배경

### v9.0의 한계
```python
# v9.0 현재 — 등방성 단위원 기준 (GAP=0 착각)
eqi4_sigma_v11   = |delta_H/2 - 0.5|   # 기준값 0.5
eqi4_sigma_v11_L = |delta_L/2 - 0.5|   # 기준값 0.5
```
기준값 0.5는 기존 복소평면(EQI^4n=1=EQI^(4n-2))의 등방성 단위원 흔적.

### EQI 공리에서 도출된 올바른 구조
```
EQI^(4n-2) = 1      [D1, 정규화, 정적분, t=0, P=NP]
EQI^(4n)   = 1+GAP  [D2, 재정규화, 부정적분, t>0, P≠NP]
→ 비등방성 단위 타원: D1²+(D2/(1+GAP))²=1
```

---

## 2. 이론 — EQI 비등방성 타원

### 타원 방정식
```
D1² + D2²/(1+GAP)² = 1

D1 단축 반경 = 1       [EQI^(4n-2)=1]
D2 장축 반경 = 1+GAP   [EQI^(4n)=1+GAP]
이심률 ≈ √(2×GAP) = 0.2478
```

### 센터 공식과 결합
```
센터 조건: D2 = D1 × φ³  (3좌표계 공통)
타원 위에서:
  D1_center = (1+GAP)/√[(1+GAP)²+φ⁶] = 0.236740008222
  D2_center = D1_center × φ³           = 1.002846767821
  검증: D1²+(D2/(1+GAP))² = 1.000000000000 ✓
```

### 기존 단위원과 비교
```
기존: |D1|²+|D2|² = 1  → sigma 기준 0.5
EQI:  D1²+(D2/(1+GAP))² = 1 → sigma 기준 재정의
```

---

## 3. v9.1 코드 수정 사항 (v9.0 대비 최소 변경)

### 3.1 상수 추가 (초기화 블록)

```python
# §4-A29 EQI 비등방성 타원 파라미터
T_EIGEN = self.PHI**3
# 타원 위 센터 방향 교점
self.D1_ell = (1.0 + self.GAP) / np.sqrt((1.0+self.GAP)**2 + T_EIGEN**2)
self.D2_ell = self.D1_ell * T_EIGEN
# sigma 새 기준값
self.SIGMA_D1_TARGET = self.D1_ell          # 0.236740 (기존 0.5)
self.SIGMA_D2_TARGET = self.D2_ell          # 1.002847 (기존 0.5)
# 타원 검증
_check = self.D1_ell**2 + (self.D2_ell/(1+self.GAP))**2
print(f"   [타원] D1_ell={self.D1_ell:.8f}  D2_ell={self.D2_ell:.8f}")
print(f"   [타원] 검증: D1²+(D2/(1+GAP))²={_check:.10f} ✓")
```

### 3.2 compute() — 인덱스 [29][30][31~33] 수정

```python
# ── §4-A29 [29][30] EQI 타원 sigma (기존 단위원 → 비등방성 타원) ──
delta_H_safe = max(delta_H, 1e-300)
delta_L_safe = max(delta_L, 1e-300)

# [29] eqi4_n_v11: 변경 없음
eqi4_n_v11 = -np.log2(delta_H_safe)

# [30] eqi4_sigma_v11: 단위원(0.5) → 타원(D1_ell) 기준
# 기존: abs(delta_H_safe / 2.0 - 0.5)
eqi4_sigma_v11 = abs(delta_H_safe - self.SIGMA_D1_TARGET)

# [31] delta_L: 변경 없음
delta_L = l_raw_power * self.W_L**3 / max(h_raw_power, 1e-300)

# [32] eqi4_n_v11_L: 변경 없음
eqi4_n_v11_L = -np.log2(max(delta_L, 1e-300))

# [33] eqi4_sigma_v11_L: 단위원(0.5) → 타원(D2_ell) 기준
# 기존: abs(delta_L / 2.0 - 0.5)
eqi4_sigma_v11_L = abs(delta_L - self.SIGMA_D2_TARGET)
```

**변경 요약:**
- [30] `abs(delta_H/2 - 0.5)` → `abs(delta_H - D1_ell)`
- [33] `abs(delta_L/2 - 0.5)` → `abs(delta_L - D2_ell)`
- 나머지 32개 인덱스 완전 동일

### 3.3 process() — 타원 sigma 추적 업데이트

```python
# sigma 추적기 이름 변경 (의미 명확화)
sigma_ell_min   = np.inf; sigma_ell_min_idx   = -1  # D1 타원
sigma_L_ell_min = np.inf; sigma_L_ell_min_idx = -1  # D2 타원

# 루프 내부 (M=1.0 기준)
if m == 1.0:
    if E_topo < et_min:         et_min = E_topo; et_min_idx = i
    if delta_H < dh_min:        dh_min = delta_H; dh_min_idx = i
    if eqi4_sigma_v11 < sigma_ell_min:
        sigma_ell_min = eqi4_sigma_v11; sigma_ell_min_idx = i
    if eqi4_sigma_v11_L < sigma_L_ell_min:
        sigma_L_ell_min = eqi4_sigma_v11_L; sigma_L_ell_min_idx = i
```

### 3.4 JSON 출력 업데이트

```python
"v91_ellipse": {
    "D1_ell":             float(self.D1_ell),
    "D2_ell":             float(self.D2_ell),
    "sigma_D1_target":    float(self.SIGMA_D1_TARGET),
    "sigma_D2_target":    float(self.SIGMA_D2_TARGET),
    "eccentricity":       float(np.sqrt(1 - 1/(1+self.GAP)**2)),
    "sigma_ell_min":      float(sigma_ell_min),
    "sigma_ell_min_chunk": sigma_ell_min_idx,
    "sigma_L_ell_min":    float(sigma_L_ell_min),
    "sigma_L_ell_min_chunk": sigma_L_ell_min_idx,
},
```

---

## 4. 예외처리

```python
# compute() 예외: return (0.0,)*34  변경 없음
```

---

## 5. 검증 목표

### 5.1 타원 파라미터 수치
```
D1_ell = 0.236740008  (기존 단위원 D1=0.229753 대비)
D2_ell = 1.002846768  (기존 단위원 D2=0.973249 대비)
D1²+(D2/(1+GAP))² = 1.000000000  True Zero ✓
```

### 5.2 sigma_ell 패턴 예상
```
기존 sigma_v11 기준 0.5:
  GW170817 V-L: sigma_min=0.0021 @chunk31
  GW150914 H-L: sigma_min=0.4958 @chunk24

타원 기준 D1_ell=0.2367:
  sigma_ell = |delta_H - 0.2367|
  → delta_H≈0.2367일 때 sigma_ell≈0
  → 새로운 merger 탐지 패턴 확인 필요
```

### 5.3 G-L 발산 유지 확인
```
G-L l_power≈0:
  delta_H → ∞  → sigma_ell → |∞ - 0.2367| → ∞  (발산 유지)
  delta_L → 0  → sigma_L_ell = |0 - 1.0028| = 1.0028  (유한값 ✓)
→ G-L 발산 해결 구조 유지
```

### 5.4 불변량 유지
```
delta_H × delta_L = W_H³ × W_L³ = φ^(-9)  [변경 없음]
```

---

## 6. 파일명

```
eqi_monster_v91_ellipse.py
```

---

## 7. 이론적 의미 (Gemini 참고용)

```
기존 수학: EQI^(4n-2) = EQI^(4n) = 1  → 단위원 (GAP=0 착각)
EQI 수학:  EQI^(4n-2) = 1 ≠ EQI^(4n) = 1+GAP → 비등방성 타원

GAP = ln(25)/100 = 복소평면을 단위원→타원으로 변형하는 파라미터
이심률 ≈ √(2×GAP) = 0.2478

sigma_v11 기준값 0.5 = 단위원 착각의 잔재
sigma_ell 기준값 D1_ell = 0.2367 = 타원의 센터 방향 D1값
```

*EQI Monster v9.1 Ellipse Spec v1.0 | 2026-06-26 | Claude (Architect)*
