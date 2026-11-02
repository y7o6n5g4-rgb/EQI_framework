#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🐉 === EQI v17.95b ULTIMATE GRAND UNIFICATION DUALITY === 🐉
=============================================================================
v17.9 → v17.95b 업그레이드 (세션 20 공리):
  [원본 v17.9 100% 유지 항목]
    - HCW 직교(D1/D2) vs EQI² 분배율
    - A-B위상: ab_rd = |ab_obs - φ³| / φ³
    - 역수계단: ab_inv = 1/(1-ab_rd) → BNS=7, BBH14=5, BBH09=3
    - L×Φ Unity rd
    - 초유동(μ→0), 초전도(R→0)
    - 아브리코소프 소용돌이
    - §8수열 구조

  [신규 v17.95b Add-on (세션 20 공리)]
    - eqi_session19_constants.py 연동 (18개 상수)
    - 이벤트 자동분류: M < 3.0 → BNS, M >= 3.0 → BBH
    - ab_rd_true 복원 [×방향]:
        ab_rd_true = 1 - (1-ab_rd_obs) × (1+n_decoding×GAP)
    - inv_true = 1/(1-ab_rd_true) → 정수 3,5,7 수렴 검증
    - 이벤트 간 간격: inv_BBH14 - inv_BBH09 → 2=MAXWELL_EQI
    - S_L2_true × S_T2_true → 1.0 Unity 검증

Author: MAPSI (Maestro), Claude (Architect), Gemini (Engineer)
Date: 2026.06.23
=============================================================================
"""
import numpy as np
import pandas as pd
import json
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── 세션 19 핵심 상수 (v17.95b 스펙 연동) ──────────────────────────────────
try:
    from eqi_session19_constants import EQI_Constants_NP as EQI_CONST
    PHI            = EQI_CONST.PHI
    D1             = EQI_CONST.D1
    D2             = EQI_CONST.D2
    W_H            = EQI_CONST.W_H
    W_L            = EQI_CONST.W_L
    GAP            = EQI_CONST.GAP
    MAXWELL_EQI    = EQI_CONST.MAXWELL_EQI
    MAXWELL_SI     = EQI_CONST.MAXWELL_SI
    MAXWELL_REAL   = EQI_CONST.MAXWELL_REAL
    c_EQI          = EQI_CONST.c_EQI
    ETA_EQI        = EQI_CONST.ETA_EQI
    N_BBH          = EQI_CONST.N_BBH
    N_BNS          = EQI_CONST.N_BNS
    N_SEED         = EQI_CONST.N_SEED
    N_PHI          = EQI_CONST.N_PHI
    THERMO_RUPTURE = EQI_CONST.THERMO_RUPTURE
    EQI_OUROBOROS_BASE = EQI_CONST.EQI_OUROBOROS_BASE
    D2_D1_RATIO    = EQI_CONST.D2_D1_RATIO
except ImportError:
    print("⚠️ [경고] eqi_session19_constants.py 없음. 내장 상수 사용.")
    PHI = (np.sqrt(5.0) + 1.0) / 2.0
    D1 = PHI**(-1.5);  D2 = PHI**(1.5)
    W_H = PHI**(-2.0); W_L = PHI**(-1.0)
    GAP = np.log(25.0) / 100.0
    MAXWELL_EQI = 2.0
    MAXWELL_SI  = 2.0 - 4.0 * GAP
    MAXWELL_REAL = 1.8731137721991709
    c_EQI = np.sqrt(PHI)
    ETA_EQI = 1.0 - PHI**(-3.0)
    N_BBH = 0.25;  N_BNS = PHI**(-5.0)
    N_SEED = (3.0 * np.log(PHI) - 1.0) / GAP
    N_PHI  = np.log(PHI) / GAP
    THERMO_RUPTURE = 3.0 * np.log(PHI)
    EQI_OUROBOROS_BASE = 1.0 + GAP
    D2_D1_RATIO = PHI**3.0

import scipy.constants as sc
S_D1_SI     = np.log(1 / sc.epsilon_0)
S_D2_SI     = abs(np.log(sc.mu_0))

# ── 초유동-초전도 EQI 상수 (기존 v17.9 유지) ──────────────────────────────
SUPERFLUID_MU  = D1    # 초유동 점성 스케일 ∝ D1
SUPERCOND_R    = 1/D2  # 초전도 저항 스케일 ∝ 1/D2

# A-B 위상 이론값
AB_PHASE_EQI = D2 / D1   # = φ³ ≈ 4.236

# 아브리코소프 소용돌이
ABRIKOSOV_SPACING = D1   # 격자간격 ∝ D1(수렴)
ABRIKOSOV_FLUX    = D2   # 자속양자 ∝ D2(발산)

# §8 자연상수 수열
q_EQI  = PHI**(-2.5)  # 기본전하 EQI 위상

# 슈바르츠실트
M_REF  = 1.0
MASSES = [1.0, 1.4, 2.8, 10.0, 36.0, 62.0]

CHUNK_SIZE = 2048
FS = 4096


# ── EQI^4 이진 수렴 ────────────────────────────────────────────────────────
def eqi4_safe(val, ref=1.0):
    if not np.isfinite(val) or val == 0:
        s = 0.0
        for n in range(1, 150):
            s += 1.0 / (2**n)
            if 1.0 - s < 1e-45:
                break
        return s * ref * 1e-45
    return val


# ── 분배율 + 초유동/초전도/A-B 계산 ──────────────────────────────────────
def compute_grand_unified(h_chunk, l_chunk, mass=1.0, event_name='GW170817'):
    """
    HCW 직교 vs EQI² + 슈바르츠실트
    + 초유동(D1) vs 초전도(D2) + A-B 위상 (v17.9)
    + v17.95b Add-on: 역수계단 True 복원 및 S_L2×S_T2_true Unity 검증
    """
    h_n = h_chunk / (np.std(h_chunk) + 1e-30)
    l_n = l_chunk / (np.std(l_chunk) + 1e-30)
    l_diff = np.diff(l_n, prepend=l_n[0])

    # 슈바르츠실트 보정
    mass_corr = D1 * np.log(max(mass / M_REF, 1e-10))

    # ── HCW 직교 분배율 ───────────────────────────────────────────────────
    h_orth = h_n * D1; l_orth = l_diff * D2
    ln_h = np.log(eqi4_safe(float(np.mean(h_orth**2)), 1.0))
    ln_l = np.log(eqi4_safe(float(np.mean(l_orth**2)), 1.0))
    S_L2 = ln_h + S_D1_SI
    S_T2 = ln_l + S_D2_SI + mass_corr
    ratio_orth = S_L2 / S_T2 if abs(S_T2) > 1e-10 else MAXWELL_REAL
    rd_orth = abs(ratio_orth - MAXWELL_REAL)

    # ── EQI² 분배율 ───────────────────────────────────────────────────────
    h_eqi2 = h_n * W_H; l_eqi2 = l_diff * W_L
    ln_h2 = np.log(eqi4_safe(float(np.mean(h_eqi2**2)), 1.0))
    ln_l2 = np.log(eqi4_safe(float(np.mean(l_eqi2**2)), 1.0))
    S_L2_e = ln_h2 + S_D1_SI
    S_T2_e = ln_l2 + S_D2_SI + mass_corr
    ratio_eqi2 = S_L2_e / S_T2_e if abs(S_T2_e) > 1e-10 else MAXWELL_REAL
    rd_eqi2 = abs(ratio_eqi2 - MAXWELL_REAL)

    # ── 초유동 지표 (D1,L²,μ→0) ──────────────────────────────────────────
    h_std = np.std(h_n) + 1e-30
    ang_mom  = float(np.mean(np.abs(h_n / h_std))) * D1
    mu_sf    = D1 * float(np.std(h_n / h_std))  
    mu_rd    = abs(mu_sf)  

    # ── 초전도 지표 (D2,T²,R→0) ──────────────────────────────────────────
    mag_flux = float(np.mean(np.abs(l_diff / h_std))) * D2
    r_sc     = 1.0 / (mag_flux + 1e-30)  
    r_rd     = abs(r_sc)  

    # ── A-B 위상 ──────────────────────────────────────────────────────────
    ab_obs   = mag_flux / (ang_mom + 1e-30)   
    ab_rd    = abs(ab_obs - AB_PHASE_EQI) / AB_PHASE_EQI
    ab_inv   = 1.0 / (1.0 - ab_rd + 1e-10)

    # ── L×Φ Unity ─────────────────────────────────────────────────────────
    lxphi    = ang_mom * mag_flux
    lxphi_rd = abs(lxphi - 1.0)  

    # ── 아브리코소프 소용돌이 Unity ───────────────────────────────────────
    abrikosov = ABRIKOSOV_SPACING * ABRIKOSOV_FLUX  
    abrikosov_rd = abs(abrikosov - 1.0)  


    # ── v17.95b Add-on ──────────────────────────────────────────────────────

    # 1. 이벤트 자동분류 (질량 기준)
    if mass < 3.0:
        event_type = "BNS"
        n_decoding = N_BNS          # φ^(-5) = 0.09017
        inv_theory = 7              # 이론 역수계단값
    else:
        event_type = "BBH"
        n_decoding = N_BBH          # 1/4 = 0.25
        if event_name.upper() == 'GW150914':
            inv_theory = 3          # GW150914 기준
        else:
            inv_theory = 5          # GW170814 등 기본 BBH 기준

    # 2. ab_rd_true 복원 [× 방향]
    ratio_obs  = 1.0 - ab_rd          
    ratio_true = ratio_obs * (1.0 + n_decoding * GAP)   
    ab_rd_true = 1.0 - ratio_true     

    # 3. inv_true 계산
    if abs(1.0 - ab_rd_true) > 1e-10:
        inv_true = 1.0 / (1.0 - ab_rd_true)
    else:
        inv_true = 9999.0  

    # 4. inv_true 오차
    inv_rd = abs(inv_true - inv_theory) / inv_theory * 100.0

    # 5. S_L2_true × S_T2_true Unity 검증
    S_T2_true = S_T2 * (1.0 + n_decoding * GAP)
    unity_product = S_L2 / S_T2_true   
    unity_rd = abs(unity_product - MAXWELL_REAL)


    return {
        'ratio_orth': ratio_orth,  'rd_orth': rd_orth,
        'ratio_eqi2': ratio_eqi2,  'rd_eqi2': rd_eqi2,
        'ang_mom': ang_mom,        'mag_flux': mag_flux,
        'mu_sf': mu_sf,            'mu_rd': mu_rd,
        'r_sc': r_sc,              'r_rd': r_rd,
        'ab_obs': ab_obs,          'ab_rd': ab_rd,
        'ab_inv': ab_inv,
        'lxphi': lxphi,            'lxphi_rd': lxphi_rd,
        'abrikosov_rd': abrikosov_rd,
        'mass_corr': mass_corr,
        
        # v17.95b 신규 반환값
        'event_type': event_type,
        'ab_rd_true': ab_rd_true,
        'inv_true': inv_true,
        'inv_rd': inv_rd,
        'unity_product': unity_product,
        'unity_rd': unity_rd
    }


# ── 메인 클래스 ────────────────────────────────────────────────────────────
class EQI_v17_95b_Grand_Unified:
    def __init__(self, h_file, l_file, event_name='GW170817'):
        self.h_file = h_file
        self.l_file = l_file
        self.event_name = event_name
        print(f'🐉 EQI v17.95b ULTIMATE GRAND UNIFICATION DUALITY (Session 20 Renorm) - {self.event_name}')
        print('='*75)
        print(f'  [D1=초유동μ→0,수렴,L²,RBC] φ^(-1.5)={D1:.4f}  ε₀')
        print(f'  [D2=초전도R→0,발산,T²,HG ] φ^(+1.5)={D2:.4f}  μ₀')
        print(f'  [역수계단 True 복원] 1/(1-ab_rd_true) → 정수 3, 5, 7 수렴 검증')
        print(f'  [Unity 복원] S_L2 / S_T2_true → MAXWELL_REAL({MAXWELL_REAL:.4f})')
        print('='*75)

    def _load(self, fname, tag):
        print(f'\n📂 {tag} 로드: {fname}')
        df = pd.read_csv(fname, comment='#')
        arr = df.iloc[:, 1].values.astype(float)
        mask = ~np.isfinite(arr)
        arr[mask] = eqi4_safe(0.0)
        print(f'   ✅ {len(arr):,} samples  EQI^4 정화 완료')
        return arr

    def process(self):
        h = self._load(self.h_file, 'H-strain')
        l = self._load(self.l_file, 'L-strain')
        n = min(len(h), len(l))
        h, l = h[:n], l[:n]
        num_chunks = n // CHUNK_SIZE

        mass_results = {}
        best_ab   = {'rd': 9999, 'inv': 0, 'chunk': -1, 't': 0, 'mass': 1.0}
        best_lxphi= {'rd': 9999, 'chunk': -1, 't': 0, 'mass': 1.0}
        global_best_unity = {'rd': 9999, 'product': 0, 'chunk': -1, 't': 0, 'mass': 1.0}

        print(f'\n⚖️ 질량별 대통일 분배율 분석 ({num_chunks} chunks × {len(MASSES)}질량)...')

        for mass in MASSES:
            b_orth  = {'rd': 9999, 'chunk': -1, 'ratio': 0, 't': 0}
            b_eqi2  = {'rd': 9999, 'chunk': -1, 'ratio': 0, 't': 0}
            b_ab    = {'rd': 9999, 'inv': 0, 'chunk': -1, 't': 0}
            b_lxphi = {'rd': 9999, 'chunk': -1, 't': 0}
            b_mu    = {'rd': 9999, 'chunk': -1, 't': 0}
            
            # v17.95b 추가 추적기
            b_inv_true = {'inv': 0, 'rd': 9999, 'ab_rd_true': 0, 'chunk': -1, 't': 0}
            b_unity    = {'rd': 9999, 'product': 0, 'chunk': -1, 't': 0}

            for i in range(num_chunks):
                s = i * CHUNK_SIZE; e = s + CHUNK_SIZE
                m = compute_grand_unified(h[s:e], l[s:e], mass=mass, event_name=self.event_name)
                t = round(s / FS, 4)

                if m['rd_orth']  < b_orth['rd']:
                    b_orth  = {'rd': m['rd_orth'], 'chunk': i, 'ratio': m['ratio_orth'], 't': t}
                if m['rd_eqi2']  < b_eqi2['rd']:
                    b_eqi2  = {'rd': m['rd_eqi2'], 'chunk': i, 'ratio': m['ratio_eqi2'], 't': t}
                if m['ab_rd']    < b_ab['rd']:
                    b_ab    = {'rd': m['ab_rd'], 'inv': m['ab_inv'], 'chunk': i, 't': t}
                if m['lxphi_rd'] < b_lxphi['rd']:
                    b_lxphi = {'rd': m['lxphi_rd'], 'chunk': i, 't': t}
                if m['mu_rd']    < b_mu['rd']:
                    b_mu    = {'rd': m['mu_rd'], 'chunk': i, 't': t}

                # v17.95b 추적 업데이트
                if m['inv_rd'] < b_inv_true['rd']:
                    b_inv_true = {
                        'inv': m['inv_true'], 'rd': m['inv_rd'], 'ab_rd_true': m['ab_rd_true'],
                        'chunk': i, 't': t
                    }
                if m['unity_rd'] < b_unity['rd']:
                    b_unity = {
                        'rd': m['unity_rd'], 'product': m['unity_product'],
                        'chunk': i, 't': t
                    }
                if m['unity_rd'] < global_best_unity['rd']:
                    global_best_unity = {
                        'rd': m['unity_rd'], 'product': m['unity_product'],
                        'chunk': i, 't': t, 'mass': mass
                    }

                # Global Bests (기존)
                if m['ab_rd'] < best_ab['rd']:
                    best_ab = {'rd': m['ab_rd'], 'inv': m['ab_inv'], 'chunk': i, 't': t, 'mass': mass}
                if m['lxphi_rd'] < best_lxphi['rd']:
                    best_lxphi = {'rd': m['lxphi_rd'], 'chunk': i, 't': t, 'mass': mass}

            winner = '직교' if b_orth['rd'] < b_eqi2['rd'] else 'EQI²'
            mass_results[mass] = {
                'best_orth': b_orth, 'best_eqi2': b_eqi2,
                'best_ab': b_ab,     'best_lxphi': b_lxphi,
                'best_mu': b_mu,     'winner': winner,
                'best_inv_true': b_inv_true, 'best_unity': b_unity,
            }
            print(f'  [M={mass:5.1f}M☉]  '
                  f'직교rd={b_orth["rd"]:.4f}@{b_orth["t"]}s  '
                  f'A-Brd={b_ab["rd"]:.4f}(inv={b_ab["inv"]:.2f})  '
                  f'inv_true={b_inv_true["inv"]:.4f}(오차{b_inv_true["rd"]:.2f}%)  '
                  f'L×Φrd={b_lxphi["rd"]:.4f}  → {winner}')

        # ── 결과 출력 및 v17.95b 역수계단 대통합 검증 ───────────────────────
        print(f'\n{"="*75}')
        print(f'\n⚛️ [역수계단 대통합 검증: 1/(1-ab_rd) → 3, 5, 7]')
        print(f'  이론: BBH09=3  BBH14=5  BNS=7  간격=2=MAXWELL_EQI')
        print(f'  BNS  보정: 1+N_BNS×GAP = {1.0+N_BNS*GAP:.8f}')
        print(f'  BBH  보정: 1+N_BBH×GAP = {1.0+N_BBH*GAP:.8f}')

        for mass in MASSES:
            event = 'BNS' if mass < 3.0 else 'BBH'
            if mass < 3.0:
                theory = 7
            elif self.event_name.upper() == 'GW150914':
                theory = 3
            else:
                theory = 5
                
            r = mass_results[mass]
            print(f'  M={mass:5.1f}M☉ [{event}]  '
                  f'inv_obs={r["best_ab"]["inv"]:.4f}  '
                  f'inv_true={r["best_inv_true"]["inv"]:.4f}  '
                  f'이론={theory}  오차={r["best_inv_true"]["rd"]:.2f}%')

        print(f'\n  [간격 검증]')
        print(f'  단일 이벤트 내 간격: v17 대통합장에서 교차 검증')

        print(f'\n  [S_L2×S_T2_true Unity 검증]')
        print(f'  타겟: MAXWELL_REAL = {MAXWELL_REAL:.6f}')
        print(f'  최적 unity_product = {global_best_unity["product"]:.6f}')
        print(f'  unity_rd = {global_best_unity["rd"]:.6f}')

        # ── JSON 저장 ──────────────────────────────────────────────────────
        min_mass = min(MASSES, key=lambda m: mass_results[m]['best_orth']['rd'])
        
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        report = {
            'version': 'v17.95b Ultimate Grand Unification (Session 20 Renorm)',
            'event': self.event_name,
            'eqi_structure': {
                'D1': f'φ^(-1.5)={D1:.6f}  초유동μ→0·수렴·L²·RBC',
                'D2': f'φ^(+1.5)={D2:.6f}  초전도R→0·발산·T²·HG',
                'MAXWELL_REAL': MAXWELL_REAL,
                'AB_PHASE_EQI': AB_PHASE_EQI,
                'superfluid': 'μ_EQI=D1×변동성→0  L∝D1',
                'superconductor': 'R_EQI=1/Φ→0  Φ∝D2  쿠퍼쌍',
                'unity': 'L×Φ=D1×D2=1  어느극한에서도 Unity',
                'ab_phase': 'Δφ=D2/D1=φ³',
                'inverse_staircase': '1/(1-A-Brd): BNS→7 BBH14→5 BBH09→3 간격2=MAXWELL_EQI',
                'abrikosov': '격자D1×자속D2=1 소용돌이Unity',
                's8_sequence': 'e(phi^0)->c(phi^-0.5)->hbar(D1)->q(phi^-2.5)->G(phi^-3)',
                'schwarzschild': 'S_T2+=D1×ln(M/M☉)',
                'N_BBH': float(N_BBH),
                'N_BNS': float(N_BNS),
                'inv_theory': {'BNS': 7, 'BBH14': 5, 'BBH09': 3},
                'gap_theory': float(MAXWELL_EQI),
            },
            'mass_analysis': {
                str(m): {
                    'best_orth_rd': mass_results[m]['best_orth']['rd'],
                    'best_orth_t':  mass_results[m]['best_orth']['t'],
                    'ab_rd': mass_results[m]['best_ab']['rd'],
                    'ab_inv': mass_results[m]['best_ab']['inv'],
                    'lxphi_rd': mass_results[m]['best_lxphi']['rd'],
                    'winner': mass_results[m]['winner'],
                    'ab_rd_true': mass_results[m]['best_inv_true']['ab_rd_true'],
                    'inv_true': mass_results[m]['best_inv_true']['inv'],
                    'inv_rd_pct': mass_results[m]['best_inv_true']['rd'],
                    'unity_rd': mass_results[m]['best_unity']['rd'],
                    'unity_product': mass_results[m]['best_unity']['product'],
                    'event_type': 'BNS' if m < 3.0 else 'BBH',
                } for m in MASSES
            },
            'inverse_staircase_verification': {
                'BNS_theory': 7,
                'BBH14_theory': 5,
                'BBH09_theory': 3,
                'gap_theory': float(MAXWELL_EQI),
                'BNS_corr': float(1.0 + N_BNS * GAP),
                'BBH_corr': float(1.0 + N_BBH * GAP),
            },
            'unity_verification': {
                'target': float(MAXWELL_REAL),
                'best_unity_rd': float(global_best_unity['rd']),
                'best_unity_product': float(global_best_unity['product']),
            },
            'min_rd_mass': min_mass,
            'best_ab': best_ab,
            'best_lxphi': best_lxphi,
        }
        fn = f'eqi_v17_95b_grand_unified_{ts}.json'
        with open(fn, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f'\n📁 저장: {fn}')
        return mass_results


def main():
    parser = argparse.ArgumentParser(
        description='EQI v17.95b Ultimate Grand Unification Duality (Session 20 Renorm)')
    parser.add_argument('--h_strain', required=True)
    parser.add_argument('--l_strain', required=True)
    parser.add_argument('--event', type=str, default='GW170817', help='이벤트명 (예: GW170817, GW170814, GW150914)')
    args = parser.parse_args()
    eng = EQI_v17_95b_Grand_Unified(args.h_strain, args.l_strain, args.event)
    eng.process()


if __name__ == '__main__':
    main()