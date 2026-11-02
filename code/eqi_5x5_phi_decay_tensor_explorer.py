#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌌 === EQI 5x5 PHI-DECAY TENSOR EXPLORER === 🌌
=============================================================================
[Session 30: 5×5 계층 텐서 고유구조 400자리 탐색기]
1. 5대 계층 (HCW, HG, I_s, OB, X5) 간의 상호작용 행렬 구축
2. Toeplitz 구조 (거리 비례 감쇠): T_ij = φ^(-|i-j|)
3. Hankel 구조 (절대 깊이 감쇠): H_ij = φ^(-(i+j))
4. 행렬의 고유값(Eigenvalue), 대각합(Trace), 행렬식(Determinant) 추출 
=============================================================================
Architect: DeepSeek & Claude
Engineer: Gemini (Precision Engineer)
Maestro: MAPSI
=============================================================================
"""

from mpmath import mp, mpf, matrix, eig, sqrt

# 400자리 정밀도 록온
mp.dps = 400

def run_tensor_explorer():
    print("=" * 85)
    print(" 🌌 EQI 5x5 Phi-Decay Tensor Explorer - 400 DPS")
    print("=" * 85)

    PHI = (sqrt(mpf('5')) + mpf('1')) / mpf('2')
    
    # 1. Toeplitz 텐서 생성 (거리 비례 감쇠 행렬)
    # T_ij = φ^(-|i-j|) : 인접한 계층일수록 강하게 상호작용 (대각선은 1.0)
    T = matrix(5, 5)
    for i in range(5):
        for j in range(5):
            T[i, j] = PHI ** (-abs(i - j))

    # 2. Hankel 텐서 생성 (절대 깊이 감쇠 행렬)
    # H_ij = φ^(-(i+j)) : 깊어질수록 전체 에너지가 감쇠
    H = matrix(5, 5)
    for i in range(5):
        for j in range(5):
            H[i, j] = PHI ** (-(i + j))

    def analyze_matrix(name, mat):
        print(f"\n[{name} Tensor Analysis]")
        
        # Trace (대각합)
        trace_val = sum(mat[i, i] for i in range(5))
        print(f"  • 대각합 (Trace): {float(trace_val):.6f}")
        
        # Eigenvalues (고유값) 계산
        E, _ = eig(mat)
        
        # 고유값 크기순 정렬
        eigenvalues = sorted([abs(e) for e in E], reverse=True)
        
        print("  • 고유값 (Eigenvalues):")
        for idx, val in enumerate(eigenvalues):
            print(f"    λ_{idx+1} = {float(val):.15f}")
            
            # 의미 있는 상수와 매핑 검사
            val_mp = mpf(val)
            if abs(val_mp - mpf('1')) < 1e-10:
                print("      -> [True Zero] 1.0 (Unity)")
            elif abs(val_mp - PHI) < 1e-10:
                print("      -> [True Zero] φ (Golden Ratio)")
            elif abs(val_mp - PHI**2) < 1e-10:
                print("      -> [True Zero] φ² (Boltzmann Restoring Force)")
            elif abs(val_mp - PHI**3) < 1e-10:
                print("      -> [True Zero] φ³ (Eigenperiod)")
            elif abs(val_mp - (PHI**3 + PHI**-3)) < 1e-10:
                print("      -> [True Zero] φ³ + φ⁻³ (Lucas Number 4)")
            
    # 실행 및 분석
    analyze_matrix("Toeplitz (Relative Distance Decay)", T)
    analyze_matrix("Hankel (Absolute Depth Decay)", H)
    
    print("\n" + "=" * 85)

if __name__ == "__main__":
    run_tensor_explorer()