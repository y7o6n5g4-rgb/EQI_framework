import mpmath as mp
import itertools

# 1. 초기 설정 (mp.dps=400)
mp.dps = 400
PHI = (mp.sqrt(5) + 1) / 2
GAP = mp.log(25) / 100
GAP_D2 = mp.log(5) / 100
PI = mp.pi
LN_C = mp.log(PHI) / 2
CENTER = mp.power(PHI, 1.5)
sigma = mp.mpf('0.5')
SQRT5 = mp.sqrt(5)

# 리만 첫 영점 근사치 (S36 이후 증명된 우주 지배 스케일)
t1 = mp.mpf('14.13472514173469379045725198356247027078')

def scan_taylor_residual():
    print("="*70)
    print(" EQI Session 52 - Taylor Series Residual Scanner")
    print("="*70)
    
    # 1. 타겟 및 S48 기하급수 본체 계산
    T_OB_D1 = CENTER - mp.power(PHI, -4.5)
    target_residual = T_OB_D1 * mp.power(sigma, 3) - LN_C
    s48_base = mp.power(PHI, -4.5) * GAP / (PHI - GAP)
    
    # 2. 미세 텐션 (Delta)
    delta_abs = target_residual - s48_base
    delta_ratio = (target_residual / s48_base) - mp.mpf('1.0')
    
    print(f"[Target Residual] = {target_residual}")
    print(f"[S48 Base (Geom)] = {s48_base}")
    print(f"★ Absolute Delta = {delta_abs}")
    print(f"★ Ratio Delta    = {delta_ratio}")
    print("-" * 70)
    print("[Scanning Qubit Algebra Taylor Terms (Factorials & Powers)...]")
    
    # 3. 테일러 기저 (Taylor Bases)
    # 기호주의를 타파하고 무한급수의 각 항(x^n / n!)으로 분해
    bases = {
        'GAP': GAP,
        'GAP_D2': GAP_D2,
        'LN_C': LN_C,
        'PHI': PHI,
        'PI': PI,
        '14': mp.mpf('14'),
        't1': t1,
        'Re(ψ5)^2': mp.power(PHI, 3) / 4,
        'Im(ψ5)^2': SQRT5 / 4
    }
    
    # 테일러 차수 (1차 ~ 5차)
    degrees = [1, 2, 3, 4, 5]
    
    # 테일러 항 생성기 (x^n / n!)
    taylor_terms = {}
    for name, val in bases.items():
        for n in degrees:
            # D1/D2 구분을 위한 팩토리얼(n!) 적용
            # D1(짝수/결맞음), D2(홀수/비결맞음)
            term_val = mp.power(val, n) / mp.fac(n)
            term_name = f"({name}^{n} / {n}!)"
            taylor_terms[term_name] = term_val
            
            # 팩토리얼 없는 단순 거듭제곱 (기하급수 항)
            geom_name = f"{name}^{n}"
            taylor_terms[geom_name] = mp.power(val, n)

    threshold = 0.05 # 0.05% 오차 허용

    # 단일 테일러 항 스캔 (Ratio Delta 타겟)
    print("\n[Phase 1: Single Taylor Term vs Ratio Delta]")
    found = False
    for name, val in taylor_terms.items():
        # 직접 매칭
        err = abs(val - delta_ratio) / delta_ratio * 100
        if err < threshold:
            print(f"  [Match] Ratio Delta ≈ {name}")
            print(f"    Value = {val}, Error = {float(err):.6f}%")
            found = True
            
        # 곱셈 조합 (테일러 항 * 14 또는 t1)
        err_14 = abs(val * 14 - delta_ratio) / delta_ratio * 100
        if err_14 < threshold:
            print(f"  [Match] Ratio Delta ≈ 14 * {name}")
            print(f"    Value = {val * 14}, Error = {float(err_14):.6f}%")
            found = True
            
        err_t1 = abs(val * t1 - delta_ratio) / delta_ratio * 100
        if err_t1 < threshold:
            print(f"  [Match] Ratio Delta ≈ t1 * {name}")
            print(f"    Value = {val * t1}, Error = {float(err_t1):.6f}%")
            found = True

    if not found:
        print("  (No direct single term matches found under 0.05%)")

    # 2개 테일러 항 덧셈/곱셈 스캔 (Absolute Delta 타겟)
    print("\n[Phase 2: Two Taylor Terms Combine vs Absolute Delta]")
    term_names = list(taylor_terms.keys())
    
    for i in range(len(term_names)):
        for j in range(i, len(term_names)):
            n1, n2 = term_names[i], term_names[j]
            v1, v2 = taylor_terms[n1], taylor_terms[n2]
            
            # 곱셈 매칭
            candidate_mul = v1 * v2
            err_mul = abs(candidate_mul - delta_abs) / delta_abs * 100
            if err_mul < threshold:
                print(f"  [Match MUL] Abs Delta ≈ {n1} * {n2}")
                print(f"    Error = {float(err_mul):.6f}%")
                
            # 덧셈/뺄셈 매칭 (테일러 급수 본질)
            candidate_add = v1 + v2
            err_add = abs(candidate_add - delta_abs) / delta_abs * 100
            if err_add < threshold:
                print(f"  [Match ADD] Abs Delta ≈ {n1} + {n2}")
                print(f"    Error = {float(err_add):.6f}%")
                
            candidate_sub = abs(v1 - v2)
            if candidate_sub > 1e-15:
                err_sub = abs(candidate_sub - delta_abs) / delta_abs * 100
                if err_sub < threshold:
                    print(f"  [Match SUB] Abs Delta ≈ |{n1} - {n2}|")
                    print(f"    Error = {float(err_sub):.6f}%")
                    
    print("="*70)

if __name__ == "__main__":
    scan_taylor_residual()