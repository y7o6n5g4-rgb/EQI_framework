#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
검증 결과의 기계 판독 범위 기록 (보존 검증기 rupert_local_region_verify.py 무수정 재사용)
- 보존 검증기를 runpy 로 실행 (증명서 경로 인자 → 출력 파일 미기록). SystemExit = FAIL/UNKNOWN.
- 검증기가 스스로 재계산한 값 (전역 변수) 만 사용해 '검증한 정리' 레코드를 만든다.
- 생성기 요약값은 일치 여부 비교에만 쓰고, 판정 필드와 분리된 annotations 에 둔다.
- 요청 정리 평가: 요청 θ_max 가 검증 범위 [0, 1/20] 안일 때만 VERIFIED, 아니면 NOT_VERIFIED (PASS 미발행).
출력: code/rupert_local_region_scope.json, code/rupert_local_region_scope_output.txt
실행: python3 code/rupert_local_region_scope.py
"""

import contextlib
import hashlib
import io
import json
import runpy
import sys
from fractions import Fraction as Fr

CERT = "code/rupert_local_region_cert.json"
VERIFIER = "code/rupert_local_region_verify.py"
OUT_JSON = "code/rupert_local_region_scope.json"
OUT_TXT = "code/rupert_local_region_scope_output.txt"
THETA_VERIFIED = Fr(1, 20)
out = []


def log(s=""):
    print(s, flush=True)
    out.append(s)


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def evaluate_request(scope, theta_max):
    """요청 정리 (θ ∈ [0, theta_max]) 가 검증 범위 안인지. 범위 밖이면 PASS 를 내지 않는다."""
    th = Fr(theta_max)
    lo, hi = (Fr(x) for x in scope["verified_theorem"]["theta_range"])
    if scope["verified_theorem"]["verifier_status"] != "PASS":
        return "NOT_VERIFIED", "검증기 PASS 아님"
    if th < 0:
        return "NOT_VERIFIED", "θ_max < 0"
    if th <= hi:
        return "VERIFIED", f"요청 [0, {th}] ⊆ 검증 범위 [{lo}, {hi}]"
    return "NOT_VERIFIED", f"요청 θ_max = {th} > 검증 범위 상한 {hi} (범위 밖; 이 요청에 대한 PASS 없음)"


def main():
    cert_hash = sha256(CERT)
    verifier_hash = sha256(VERIFIER)
    buf = io.StringIO()
    argv0 = sys.argv
    sys.argv = [VERIFIER, CERT]
    status, G = "PASS", None
    try:
        with contextlib.redirect_stdout(buf):
            G = runpy.run_path(VERIFIER, run_name="__main__")
    except SystemExit as e:
        status = "FAIL_OR_UNKNOWN" if e.code else "PASS"
    finally:
        sys.argv = argv0
    vout = buf.getvalue()
    if status == "PASS" and (G is None or "PASS — 독립 재검증 완료" not in vout):
        status = "FAIL_OR_UNKNOWN"
    C = json.load(open(CERT))
    log("=" * 100)
    log("검증 결과 범위 기록 (보존 검증기 재실행)")
    log("=" * 100)
    log(f"  증명서 SHA-256 {cert_hash}")
    log(f"  검증기 SHA-256 {verifier_hash}")
    log(f"  검증기 상태: {status}")
    verified = {
        "verifier_status": status,
        "object": {"family": "Zeng_Pt", "definition": "P_t = conv{p_i, -t p_i}", "P0": C["object"]["P0"],
                   "t_shape": "11/20", "lambda": "1"},
        "Q": {"q1": C["Q"]["q1"], "q2": C["Q"]["q2"]},
        "rotation_convention": {"R_out(q)": "Ry(-q1) @ Rz(-q2)", "R_in": "R_out(q) @ expm(theta*[u]x)",
                                "Ry(a)": C["rotation"]["Ry(a)"], "Rz(a)": C["rotation"]["Rz(a)"],
                                "shadow": "first two coordinates of R v", "relative_rotation": "Delta = R_out^T R_in, angle(Delta) = theta"},
        "theta_range": ["0", str(THETA_VERIFIED)],
        "axes": "all unit u",
        "claim": "P_{11/20}, q in Q, all relative axes, 0 <= theta <= 1/20: strict projected containment impossible "
                 "(theta>0: B<0; theta=0: same-polygon lemma)",
        "certificate_sha256": cert_hash,
        "verifier_sha256": verifier_hash,
        "analytic_assumptions": [
            "theta in (0,1]: sin(theta) >= theta - theta^3/6 >= 5 theta/6",
            "1 - cos(theta) <= theta^2/2",
            "unit margin m <= sum_j w_j g_j / s', s' = sum_j w_j |n_j| (weak duality of translation LP)",
            "fixed contact vertex inner product <= support function h_I (support switching only lowers B)",
            "Rodrigues identity for exp(theta [u]x)",
            "theta=0 lemma: compact O with nonempty interior admits no tau with O + tau in int(O)",
            "cube-sphere: 6 faces x [-1,1]^2 normalized covers S^2",
            "cos/sin monotone on (0, 3/2) subset (0, pi/2) (pi > 3)",
        ],
    }
    if status == "PASS":
        dmin, amin = G["dmin"], G["amin"]
        gen = C["results"]
        structure = {
            "dual_vertices_edges": [list(J) for J in sorted(G["DV"])] == [d["edges"] for d in C["balance"]["dual_vertices"]],
            "n_certificates": len(G["CERTS"]) == len(C["certificates"]["list"]) == 24,
            "n_cells": len(C["sphere"]["cells"]) == 96,
        }
        gen_cmp = {
            "verifier_a_min_lower_floor15": f"0.{int(amin * 10 ** 15):015d}",
            "verifier_delta_lower_floor13": f"0.{int(dmin * 10 ** 13):013d}" if dmin < 1 else "1",
            "generator_a_Q_lower": gen["a_Q_lower"],
            "generator_delta_cellwise_lower": gen["delta_Q_cellwise_lower"],
            "both_delta_ge_verified_theta": bool(Fr(gen["delta_Q_cellwise_lower"]) >= THETA_VERIFIED and dmin >= THETA_VERIFIED),
            "generator_le_verifier_a": bool(Fr(gen["a_Q_lower"]) <= amin),
            "generator_le_verifier_delta": bool(Fr(gen["delta_Q_cellwise_lower"]) <= dmin),
            "structure_match": structure,
        }
        verified["generator_summary_consistent"] = bool(all(structure.values()) and gen_cmp["both_delta_ge_verified_theta"])
        verified["generator_comparison"] = gen_cmp
        log(f"  생성기 요약 일치: {verified['generator_summary_consistent']}  {json.dumps(gen_cmp, ensure_ascii=False)}")
    else:
        verified["generator_summary_consistent"] = None
    scope = {
        "format": "rupert_local_region_scope v1",
        "verified_theorem": verified,
        "annotations_ignored_for_decision": {
            "generator_delta_Q_cellwise": C["results"]["delta_Q_cellwise_lower"],
            "generator_delta_Q_uniform": C["results"]["delta_Q_uniform_lower"],
            "earlier_report_delta_Q": "0.0573932756328 (ed8c701, preserved; not part of verified theorem)",
            "note": "이 필드는 판정에 쓰지 않음. 판정은 verified_theorem 만 사용.",
        },
    }
    log(f"  검증한 정리: θ ∈ [0, 1/20], Q = q1 {C['Q']['q1']} × q2 {C['Q']['q2']}, t_shape 11/20, 모든 축")
    for th in ("9/10", "1/20", "3/50"):
        st, why = evaluate_request(scope, th)
        log(f"  요청 θ_max = {th}: {st} — {why}")
    with open(OUT_JSON, "w") as f:
        json.dump(scope, f, ensure_ascii=False, indent=1)
    with open(OUT_TXT, "w") as f:
        f.write("\n".join(out) + "\n")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
