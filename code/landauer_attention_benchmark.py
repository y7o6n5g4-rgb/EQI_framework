import torch
import triton
import triton.language as tl
import pynvml
import time
import math

class LandauerAttentionBenchmark:
    def __init__(self, device_id=0):
        self.device_id = device_id
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        self.k_B = 1.380649e-23  # Boltzmann constant (J/K)
        self.ln2 = 0.69314718056
        
        device_name = pynvml.nvmlDeviceGetName(self.handle)
        print(f"[EQI-BENCH] Initialized on GPU: {device_name}")

    def get_hardware_state(self):
        temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
        power_mW = pynvml.nvmlDeviceGetPowerUsage(self.handle)
        energy_mJ = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        return temp + 273.15, power_mW / 1000.0, energy_mJ / 1000.0  # Kelvin, Watt, Joule

    def run_benchmark(self, batch_size=8, n_heads=32, seq_len=4096, d_model=128, iterations=100):
        print(f"\n[EQI-BENCH] Running Configuration:")
        print(f"  Batch: {batch_size}, Heads: {n_heads}, SeqLen: {seq_len}, Dim: {d_model}")
        print(f"  Iterations: {iterations} (Warmup: 20)")

        # 텐서 할당 (FP16)
        dtype = torch.float16
        b_prec = 16.0
        q = torch.randn((batch_size, n_heads, seq_len, d_model), dtype=dtype, device='cuda')
        k = torch.randn((batch_size, n_heads, seq_len, d_model), dtype=dtype, device='cuda')
        v = torch.randn((batch_size, n_heads, seq_len, d_model), dtype=dtype, device='cuda')
        out = torch.empty_like(q)
        entropy_out = torch.empty((batch_size * n_heads, seq_len), dtype=torch.float32, device='cuda')

        sm_scale = 1.0 / math.sqrt(d_model)
        BLOCK_M = 64
        BLOCK_N = 64
        grid = (triton.cdiv(seq_len, BLOCK_M), batch_size * n_heads)

        # Warmup
        for _ in range(20):
            _fused_attention_entropy_kernel[grid](
                q, k, v, out, entropy_out,
                sm_scale,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                batch_size, n_heads, seq_len,
                BLOCK_M=BLOCK_M, BLOCK_DMODEL=d_model, BLOCK_N=BLOCK_N
            )
        torch.cuda.synchronize()

        # 계측 시작
        T_start, p_start, e_start = self.get_hardware_state()
        start_time = time.perf_counter()

        for _ in range(iterations):
            _fused_attention_entropy_kernel[grid](
                q, k, v, out, entropy_out,
                sm_scale,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                batch_size, n_heads, seq_len,
                BLOCK_M=BLOCK_M, BLOCK_DMODEL=d_model, BLOCK_N=BLOCK_N
            )
        torch.cuda.synchronize()

        elapsed_time = time.perf_counter() - start_time
        T_end, p_end, e_end = self.get_hardware_state()

        # 에너지 및 엔트로피 분석
        actual_energy_joules = e_end - e_start
        mean_temperature = (T_start + T_end) / 2.0

        # 정보 소거량 계산
        # H_max = log2(seq_len)
        max_entropy_per_row = math.log2(seq_len)
        mean_observed_entropy = entropy_out.mean().item()
        
        # 행당 비가역 정보 소거량 (bits)
        erased_bits_per_row = (max_entropy_per_row - mean_observed_entropy) * b_prec
        total_erased_bits = erased_bits_per_row * (batch_size * n_heads * seq_len) * iterations

        # 란다우 한계 이론 최소 에너지
        landauer_min_joules = total_erased_bits * (self.k_B * mean_temperature * self.ln2)

        # 열역학적 낭비율 산출
        waste_factor = actual_energy_joules / max(landauer_min_joules, 1e-30)

        # 결과 출력
        print("\n" + "=" * 80)
        print(" [EQI-BENCH] 열역학적 란다우 한계 실측 보고서")
        print("=" * 80)
        print(f"  ▶ 측정 시간 (Elapsed Time)     : {elapsed_time:.4f} sec")
        print(f"  ▶ 평균 소자 온도 (Temperature) : {mean_temperature:.2f} K ({mean_temperature-273.15:.1f} °C)")
        print(f"  ▶ 실제 하드웨어 소비 에너지    : {actual_energy_joules:.6f} J")
        print(f"  ▶ 누적 비가역 소거 비트 수    : {total_erased_bits:.4e} bits")
        print(f"  ▶ 이론적 란다우 소거 한계 에너지: {landauer_min_joules:.6e} J")
        print("-" * 80)
        print(f"  ★ 열역학적 에너지 낭비율 (η_waste) : {waste_factor:.4e} 배")
        print(f"  ★ 현행 CMOS 연산 효율성          : {(1.0 / waste_factor) * 100:.8e} %")
        print("=" * 80)
        
        pynvml.nvmlShutdown()
        return {
            "waste_factor": waste_factor,
            "actual_joules": actual_energy_joules,
            "landauer_joules": landauer_min_joules,
            "erased_bits": total_erased_bits
        }

if __name__ == "__main__":
    bench = LandauerAttentionBenchmark(device_id=0)
    bench.run_benchmark(batch_size=4, n_heads=32, seq_len=4096, d_model=128, iterations=50)