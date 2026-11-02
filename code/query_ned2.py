from astroquery.ipac.ned import Ned  # 업데이트된 경로
import time

targets = [
    ("NGC 4993",  "GW170817"),
    ("NGC 1600",  "GW150914"),
    ("NGC 5765b", "GW170814"),  # GW170814 호스트
]

for name, event in targets:
    try:
        result = Ned.query_object(name)
        vel = result["Velocity"][0]
        z = result["Redshift"][0]
        print(f"[{event}] {name}: v={vel:.1f} km/s, z={z:.6f}")
        time.sleep(3)  # 타임아웃 방지
    except Exception as e:
        print(f"[{event}] {name} 실패: {e}")