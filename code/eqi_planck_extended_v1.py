"""
EQI S117 — 플랑크 양자 에너지 확장 번역 v1
E/(hν) = 1  →  E/(hν) = Z(n),  n = n_re + i·n_im ∈ ℂ (우로보로스 순환 사이클)
  선형형 (원칙Y, CM-22):  Z_lin = 1 + n·GAP
  극형   (6-3 A, CM-23):  Z_pol = (1 + n_re·GAP)·(1+GAP)^(i·n_im)   (|Z| 페르마 D1, arg Z 파인만 D2)
대입: 이론적 기반 S24 세트 h·ν_p = φ^(−3/2) (ν_p = 1/T_p)
공통 모듈 v1.5 연동: phonon_order, p_tunnel, N_C, IG, euler_turn, SHUNYA_*, E_TRIGGER, X7_BLOCKED
"""
import mpmath
from mpmath import mpf, mpc, log, sqrt, nstr
import eqi_maxwell_layers as M

mpmath.mp.dps = 400
PHI, GAP, GD, IG = M.PHI, M.GAP, M.GD, M.IG
HNU = PHI ** (mpf(-3) / 2)            # h·ν_p (S24)
E_P = PHI ** (mpf(5) / 4)             # M_p·c²
WE = PHI ** (mpf(11) / 4)
CENTER = M.CENTER


def Z_lin(n):
    return 1 + mpc(n) * GAP


def Z_pol(n_re, n_im):
    return (1 + mpf(n_re) * GAP) * (1 + GAP) ** (mpc(0, 1) * mpf(n_im))


def state(n_re, n_im=0):
    """확장 번역 한 점의 장부: 에너지 (선형·극형), 포논 껍질, 방출, 터널링, 오일러 위상."""
    zl, zp = Z_lin(mpc(n_re, n_im)), Z_pol(n_re, n_im)
    k, resid = M.phonon_order(n_re)
    emit = max(mpf(n_re) - M.N_C, mpf(0)) * GAP          # (n − n_c)⁺·GAP
    theta = mpf(n_im) * IG                                # 누적 위상
    return dict(E_lin=HNU * zl, E_pol=HNU * zp, k=k, resid=resid,
                P_tun=(M.p_tunnel(k) if k >= 1 else None),
                E_emit=HNU * emit, theta_over_pi=theta / mpmath.pi)


def d(a, b):
    return mpmath.fabs(a - b)


if __name__ == "__main__":
    print("=" * 78)
    print("EQI 플랑크 양자 에너지 확장 번역 v1 (mp.dps = 400)")
    print("=" * 78)
    rows = []
    def line(tag, label, val, target=None):
        s = f"{tag:6s} {label:46s} {nstr(val, 16)}"
        if target is not None:
            s += f"   차 {nstr(d(val, target), 3)}"
        print(s)
    # ── 실수 n (페르마 D1) 특수점
    line("X-1", "n=0  E/hν_p (기존 물리, 보른 접점)", state(0)["E_lin"] / HNU, 1)
    line("X-2", "n=1  E/hν_p = 1+GAP (FEYNMAN)", state(1)["E_lin"] / HNU, M.FEYNMAN)
    st = state(M.N_C)
    line("X-3", "n=n_c  E = hν_p·φ^1.5 = 1 (항등원)", mpmath.re(st["E_lin"]), 1)
    line("X-4", "n=n_c  ln Z / ln φ³ = σ", log(1 + M.N_C * GAP) / log(PHI ** 3), mpf(1) / 2)
    line("X-5", "일함수 W = E(n_c) − E(0) = 1 − hν_p", 1 - HNU, HNU * M.TIP)
    line("X-6", "광전 K_max(n) = E(n) − E(n_c) [n=40]", mpmath.re(state(40)["E_lin"]) - 1, state(40)["E_emit"])
    for k, name, nk in M.PHONON_LADDER:
        s = state(nk)
        print(f"X-7.{k}  껍질 k={k} {name:26s} n_k={nstr(nk, 10):>12s}  E/hν_p={nstr(1 + nk * GAP, 12):>14s}"
              f"  P_tun={nstr(M.p_tunnel(k), 9)}")
    print(f"X-7.6  껍질 k=6 13차: X7 차단 = {M.X7_BLOCKED}")
    n_we = (WE - 1) / GAP
    line("X-8", "Z = We 되는 n (질량 계열 E_p 도달, 선형)", n_we)
    line("X-8b", "  확인 hν_p·(1+n·GAP) = E_p", HNU * (1 + n_we * GAP), E_P)
    line("X-8c", "  곱셈 사이클 ln We / IG", log(WE) / IG)
    line("X-9", "Z_lin = −1 되는 실수 n = −2/GAP", -2 / GAP)
    # ── 허수 n (파인만 D2)
    zi = Z_lin(mpc(0, PHI ** 3 / GAP))
    line("X-10", "n = iφ³/GAP: |E/hν_p|² = 1+φ⁶ (CM-22)", mpmath.fabs(zi) ** 2, 1 + PHI ** 6)
    zp = Z_pol(0, mpmath.pi / IG)
    line("X-11", "극형 n_im = π/IG: Re E/hν_p = −1 (CM-23)", mpmath.re(zp), -1)
    zq = Z_pol(0, M.PI_EQI / IG)
    line("X-12", "극형 n_im = π_EQI/IG: Im E/hν_p (슈냐 이월)", mpmath.im(zq), M.SHUNYA_IM)
    line("X-12b", "  같은 점 1 + Re (슈냐 실수부)", 1 + mpmath.re(zq), M.SHUNYA_RE)
    line("X-12c", "  이월 에너지 hν_p·SHUNYA_IM", HNU * M.SHUNYA_IM)
    # ── 가모프 복소 에너지 ↔ RLC 감쇠 (v89)
    zeta = GD * PHI ** (mpf(-3) / 2)                      # γ/ω₀, ω₀ = 1/√(LC) = 1
    n_im_rlc = -zeta / GAP
    line("X-13", "RLC 감쇠 ↔ n_im = −ζ/GAP", n_im_rlc, -PHI ** (mpf(-3) / 2) / 2)
    line("X-13b", "  Q = −1/(2·n_im·GAP) = CENTER/GAP", -1 / (2 * n_im_rlc * GAP), CENTER / GAP)
    # ── 선형형 vs 극형 1차 일치
    e = mpf("1e-20")
    line("X-14", "작은 n_im: Im(Z_lin − Z_pol)/n_im (=GAP−IG)", mpmath.im(Z_lin(mpc(0, e)) - Z_pol(0, e)) / e, GAP - IG)
    line("X-15", "E_p/(hν_p) = We (번역 v4 §2)", E_P / HNU, WE)
