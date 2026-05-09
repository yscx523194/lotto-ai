import requests
import json
import os
import time

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lotto_cache.json")
API_URL = "https://www.dhlottery.co.kr/lt645/selectPstLt645InfoNew.do"


def create_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    })
    s.get("https://www.dhlottery.co.kr/", timeout=10)
    return s


def parse_item(item):
    ymd = item["ltRflYmd"]
    date_str = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"
    return {
        "회차": item["ltEpsd"],
        "추첨일": date_str,
        "당첨번호": sorted([
            item["tm1WnNo"], item["tm2WnNo"], item["tm3WnNo"],
            item["tm4WnNo"], item["tm5WnNo"], item["tm6WnNo"]
        ]),
        "보너스번호": item["bnsWnNo"],
        "1등 당첨금": item["rnk1WnAmt"],
        "1등 당첨자수": item["rnk1WnNope"],
    }


def fetch_batch(session, direction, epsd):
    if direction == "center":
        params = {"srchDir": "center", "srchLtEpsd": str(epsd)}
    else:
        params = {"srchDir": direction, "srchCursorLtEpsd": str(epsd)}

    r = session.get(API_URL, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    lst = data.get("data", {}).get("list", [])
    return [parse_item(item) for item in lst]


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def fetch_all():
    cache = load_cache()
    print(f"기존 캐시: {len(cache)}회차")

    session = create_session()

    # 최신 회차 가져오기: center로 큰 회차 조회
    latest_batch = fetch_batch(session, "center", "1300")
    if not latest_batch:
        # fallback
        latest_batch = fetch_batch(session, "center", "1222")
    if not latest_batch:
        print("최신 회차를 가져올 수 없습니다.")
        return

    latest = max(item["회차"] for item in latest_batch)
    for item in latest_batch:
        cache[str(item["회차"])] = item

    print(f"최신 회차: {latest}회")

    # 이미 모두 캐시되어 있는지 확인
    cached_rounds = set(int(k) for k in cache.keys())
    missing = [i for i in range(1, latest + 1) if i not in cached_rounds]

    if not missing:
        print(f"전체 {latest}회차 모두 캐시 완료!")
        save_cache(cache)
        return

    print(f"미캐싱 {len(missing)}개 → 다운로드 시작...")

    # center 방식으로 10개씩 가져오기 (중간값 기준 ±5 반환)
    fetched = 0
    epsd = latest

    while epsd >= 1:
        batch = fetch_batch(session, "center", str(epsd))
        if not batch:
            epsd -= 10
            continue

        for item in batch:
            key = str(item["회차"])
            if key not in cache:
                cache[key] = item
                fetched += 1

        oldest = min(item["회차"] for item in batch)
        pct = (latest - oldest + 1) / latest * 100
        print(f"  {oldest:4d}회차까지 수집... {pct:5.1f}%  [{fetched}개 추가]", end="\r")

        # 다음 조회 위치: 이번 배치의 가장 오래된 것보다 5 앞
        epsd = oldest - 5
        if oldest <= 1:
            break

        # 중간 저장
        if fetched % 100 < 10:
            save_cache(cache)

        time.sleep(0.15)

    save_cache(cache)
    print(f"\n완료! 총 {len(cache)}회차 캐시됨 (신규 {fetched}개)")

    # 누락 확인
    cached_rounds = set(int(k) for k in cache.keys())
    still_missing = [i for i in range(1, latest + 1) if i not in cached_rounds]
    if still_missing:
        print(f"⚠ 누락 {len(still_missing)}개 → 개별 보완 중...")
        for m in still_missing:
            batch = fetch_batch(session, "center", str(m))
            if batch:
                for item in batch:
                    cache[str(item["회차"])] = item
                    fetched += 1
            time.sleep(0.15)
        save_cache(cache)
        cached_rounds = set(int(k) for k in cache.keys())
        final_missing = [i for i in range(1, latest + 1) if i not in cached_rounds]
        if final_missing:
            print(f"⚠ 최종 누락: {final_missing}")
        else:
            print(f"✅ 1~{latest}회차 모두 캐싱 완료!")
    else:
        print(f"✅ 1~{latest}회차 모두 캐싱 완료!")


if __name__ == "__main__":
    fetch_all()
