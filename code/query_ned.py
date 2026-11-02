from astroquery.ned import Ned
import astropy.units as u

targets = [
    ("NGC 4993",  "GW170817"),   # BNS
    ("NGC 7619",  "GW190814"),   # NSBH 근처 참고
    ("NGC 1600",  "GW150914"),   # BBH 참고
]

for name, event in targets:
    try:
        result = Ned.query_object(name)
        print(f"\n[{event}] {name}")
        print(result["Object Name", "Velocity", "Redshift"])
        
        # 속도 분산 쿼리
        phot = Ned.get_table(name, table='diameters')
        print(phot[:3])
    except Exception as e:
        print(f"[{event}] {name} 실패: {e}")