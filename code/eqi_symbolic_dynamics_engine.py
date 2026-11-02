from mpmath import mp, mpf, log, sqrt, fabs, nstr

mp.dps = 400

# ── 핵심 상수 ──────────────────────────────────────────────────────────────
PHI    = (sqrt(5) + 1) / 2
GAP    = log(25) / 100
GAP_D2 = log(5) / 100

print(f"[INIT] mp.dps = {mp.dps}")
print(f"[INIT] PHI    = {nstr(PHI,    20)}")
print(f"[INIT] GAP    = {nstr(GAP,    20)}")
print(f"[INIT] GAP_D2 = {nstr(GAP_D2, 20)}")
print("=" * 72)


# ══════════════════════════════════════════════════════════════════════════
# LAYER 0 : 기호 S 구조체
# ══════════════════════════════════════════════════════════════════════════
class SymbolicState:
    """
    S := (D1_state, D2_state, GAP_cost, direction)

    · GAP_cost 는 소거되지 않고 history 리스트에 누적 각인됨
    · transfer_cost() 호출 시 현재 비용이 다음 상태의 초기 조건으로 전이
    """
    def __init__(self, D1, D2, gap_cost, direction, history=None, label=""):
        self.D1        = mpf(D1)
        self.D2        = mpf(D2)
        self.gap_cost  = mpf(gap_cost)
        self.direction = direction
        self.history   = list(history) if history else []
        self.label     = label

    def transfer_cost(self):
        """현재 GAP_cost 를 history 에 추가한 새 리스트 반환 (불변 전이)."""
        return self.history + [self.gap_cost]

    def total_accumulated(self):
        """history 전체 + 현재 gap_cost 의 합."""
        return sum(self.history) + self.gap_cost

    def report(self, depth=0):
        pad = "  " * depth
        print(f"{pad}┌─ [{self.label}]")
        print(f"{pad}│  direction        = {self.direction}")
        print(f"{pad}│  D1_state         = {nstr(self.D1,       30)}")
        print(f"{pad}│  D2_state         = {nstr(self.D2,       30)}")
        print(f"{pad}│  GAP_cost(current)= {nstr(self.gap_cost, 30)}")
        print(f"{pad}│  history depth    = {len(self.history)}")
        for i, h in enumerate(self.history):
            print(f"{pad}│    history[{i}]     = {nstr(h, 30)}")
        print(f"{pad}│  total_accumulated= {nstr(self.total_accumulated(), 30)}")
        print(f"{pad}└{'─'*60}")


# ══════════════════════════════════════════════════════════════════════════
# LAYER 1 : 양자 관측 엔트로피 엔진  (D2 → D1 상전이)
# ══════════════════════════════════════════════════════════════════════════
def normalize(amplitudes):
    """진폭 리스트를 L2 정규화."""
    norm = sqrt(sum(a ** 2 for a in amplitudes))
    return [a / norm for a in amplitudes]


def quantum_observation_engine(psi_raw, observed_index):
    """
    입력
      psi_raw        : 실수 진폭 리스트 (정규화 전)
      observed_index : 관측된 고유값 인덱스

    반환
      SymbolicState  (direction = 'D2→D1')

    GAP_cost = 폰 노이만 엔트로피(D2) - 관측 후 잔존 정보(D1 단독 항)
    → D2 에 분산된 중첩 정보 중 D1 으로 전이되지 못한 부분이 비용으로 각인
    """
    psi = [mpf(a) for a in psi_raw]
    psi = normalize(psi)

    # D2_state : 관측 전 폰 노이만 엔트로피
    D2 = -sum(a ** 2 * log(a ** 2) for a in psi if a > 0)

    # D1_state : 관측된 고유값의 확률
    D1 = psi[observed_index] ** 2

    # 잔존 정보 : 관측 후 남은 D1 단독 엔트로피 항
    residual = -(D1 * log(D1)) if D1 > 0 else mpf(0)

    # GAP_cost : 상전이에서 소실된 정보량
    gap_cost = D2 - residual

    return SymbolicState(
        D1        = D1,
        D2        = D2,
        gap_cost  = gap_cost,
        direction = 'D2→D1',
        history   = [],
        label     = "QUANTUM_OBSERVATION"
    )


# ══════════════════════════════════════════════════════════════════════════
# LAYER 2 : 특이점 압축 엔진  (D1 → D2 극한 압축)
# ══════════════════════════════════════════════════════════════════════════
def singularity_compression_engine(rho, area_planck, prev_state=None):
    """
    입력
      rho          : 물질 밀도 (플랑크 단위) — D1_state
      area_planck  : 사건 지평선 면적 (플랑크 단위) — D2 대리변수
      prev_state   : 이전 SymbolicState → GAP_cost 이력 전이

    반환
      SymbolicState  (direction = 'D1→D2')

    D2_state  = Bekenstein-Hawking 엔트로피  S_BH = A/4
    GAP_cost  = log(ρ)×GAP + S_BH×GAP_D2
    → 밀도 발산 시 D1 정보가 D2 곡률로 완전 전이되는 비용
    """
    rho         = mpf(rho)
    area_planck = mpf(area_planck)

    D2       = area_planck / 4
    gap_cost = log(rho) * GAP + D2 * GAP_D2 if rho > 0 else mpf(0)

    inherited = prev_state.transfer_cost() if prev_state else []

    return SymbolicState(
        D1        = rho,
        D2        = D2,
        gap_cost  = gap_cost,
        direction = 'D1→D2',
        history   = inherited,
        label     = "SINGULARITY_COMPRESSION"
    )


# ══════════════════════════════════════════════════════════════════════════
# LAYER 3 : 비대칭 gap_fusion 마찰항 연산  (A ⊕_GAP B)
# ══════════════════════════════════════════════════════════════════════════
def gap_fusion(S_A, S_B, label="FUSED_C"):
    """
    두 국소 정합성 A, B 에서 제3의 국소 정합성 C 생성.
    GAP_cost 는 소거되지 않고 C 의 구조 파라미터로 각인됨.

    D1_C  = 조화평균 (GAP 가중) — 두 D1 의 충돌 완충
    D2_C  = 기하평균 — 스케일 불변 D2 구조 보존
    방향 마찰항 = GAP × |A.gap_cost − B.gap_cost|
      → 두 공리계의 방향성 충돌 비용 (비대칭성 보존)
    gap_cost_C = A.gap + B.gap + 마찰항
    """
    denom = S_A.D1 + S_B.D1 + GAP
    D1_C  = (2 * S_A.D1 * S_B.D1) / denom if denom != 0 else mpf(0)
    D2_C  = sqrt(S_A.D2 * S_B.D2)

    friction    = GAP * fabs(S_A.gap_cost - S_B.gap_cost)
    gap_cost_C  = S_A.gap_cost + S_B.gap_cost + friction

    merged_history = S_A.history + S_B.history + [gap_cost_C]

    return SymbolicState(
        D1        = D1_C,
        D2        = D2_C,
        gap_cost  = gap_cost_C,
        direction = 'D1↔D2',
        history   = merged_history,
        label     = label
    )


# ══════════════════════════════════════════════════════════════════════════
# LAYER 4 : GAP_cost 누적 검증기
# ══════════════════════════════════════════════════════════════════════════
def verify_accumulation(S, label=""):
    total = S.total_accumulated()
    ratio = total / GAP
    print(f"\n[VERIFY] {label}")
    print(f"  total_accumulated = {nstr(total, 40)}")
    print(f"  GAP               = {nstr(GAP,   40)}")
    print(f"  ratio(total/GAP)  = {nstr(ratio, 40)}")
    print(f"  history_depth     = {len(S.history)}")
    # 비가역성 검증 : 누적값이 GAP 보다 큰지 확인
    if total > GAP:
        print(f"  → 비가역 누적 확인 (total > GAP) ✓")
    else:
        print(f"  → 누적값이 GAP 이하 (초기 상태에 가까움)")


# ══════════════════════════════════════════════════════════════════════════
# MAIN : 시뮬레이션 실행부
# ══════════════════════════════════════════════════════════════════════════
def run_simulation():

    # ── STEP 1 : 양자 관측 사건 ───────────────────────────────────────────
    print("\n" + "═" * 72)
    print("STEP 1 : 양자 관측  (D2 → D1 상전이)")
    print("═" * 72)

    psi_raw = ['0.7', '0.5', '0.3', '0.4', '0.2']
    observed_index = 0

    S_q = quantum_observation_engine(psi_raw, observed_index)
    S_q.report()
    verify_accumulation(S_q, "양자 관측 직후")

    # ── STEP 2 : 특이점 압축 사건 (S_q 의 GAP_cost 가 초기 조건으로 전이) ─
    print("\n" + "═" * 72)
    print("STEP 2 : 특이점 압축  (D1 → D2 극한 압축)")
    print("         ← S_q 의 GAP_cost 가 history 로 전이됨")
    print("═" * 72)

    rho_planck  = mpf('1e93')   # 플랑크 밀도 근방
    area_planck = mpf('1e6')    # 사건 지평선 면적 (플랑크 단위)

    S_s = singularity_compression_engine(rho_planck, area_planck,
                                         prev_state=S_q)
    S_s.report()
    verify_accumulation(S_s, "특이점 압축 후")

    # ── STEP 3 : 융합 연산  A ⊕_GAP B ────────────────────────────────────
    print("\n" + "═" * 72)
    print("STEP 3 : 융합 연산  C = S_q ⊕_GAP S_s")
    print("         (비대칭 마찰항 포함, GAP_cost 소거 없음)")
    print("═" * 72)

    S_c = gap_fusion(S_q, S_s, label="FUSED_C")
    S_c.report()
    verify_accumulation(S_c, "융합 연산 후")

    # ── STEP 4 : 2차 융합 (C ⊕_GAP S_q) — 비용 연쇄 누적 추적 ──────────
    print("\n" + "═" * 72)
    print("STEP 4 : 2차 융합  D = C ⊕_GAP S_q")
    print("         (GAP_cost 연쇄 누적 메커니즘 추적)")
    print("═" * 72)

    S_d = gap_fusion(S_c, S_q, label="FUSED_D")
    S_d.report()
    verify_accumulation(S_d, "2차 융합 후")

    # ── STEP 5 : 전체 이력 요약 ──────────────────────────────────────────
    print("\n" + "═" * 72)
    print("STEP 5 : 전체 GAP_cost 이력 요약")
    print("═" * 72)
    print(f"  S_q  total_accumulated = {nstr(S_q.total_accumulated(), 40)}")
    print(f"  S_s  total_accumulated = {nstr(S_s.total_accumulated(), 40)}")
    print(f"  S_c  total_accumulated = {nstr(S_c.total_accumulated(), 40)}")
    print(f"  S_d  total_accumulated = {nstr(S_d.total_accumulated(), 40)}")
    print(f"\n  GAP                    = {nstr(GAP,                    40)}")
    print(f"  GAP_D2                 = {nstr(GAP_D2,                 40)}")
    grand = S_d.total_accumulated()
    print(f"\n  최종 누적 / GAP        = {nstr(grand / GAP,           40)}")
    print(f"  최종 누적 / GAP_D2     = {nstr(grand / GAP_D2,         40)}")
    print(f"\n  history 깊이 (S_d)     = {len(S_d.history) + 1}  항목")
    print("\n[완료] 모든 GAP_cost 는 소거되지 않고 누적 각인되었습니다.")


if __name__ == "__main__":
    run_simulation()
