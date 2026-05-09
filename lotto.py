import requests
import json
import sys
import os
import time

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lotto_cache.json")
API_URL = "https://www.dhlottery.co.kr/lt645/selectPstLt645InfoNew.do"


def create_session():
    """세션을 생성하고 쿠키를 획득합니다."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    })
    s.get("https://www.dhlottery.co.kr/", timeout=10)
    return s


def parse_item(item):
    """API 응답 항목을 정리된 딕셔너리로 변환합니다."""
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
        "1등 당첨금": f'{item["rnk1WnAmt"]:,}원',
        "1등 당첨자수": item["rnk1WnNope"],
    }


def fetch_batch(session, direction, epsd):
    """API에서 10개 단위로 회차 데이터를 가져옵니다."""
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
    """캐시 파일에서 데이터를 불러옵니다."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    """캐시 데이터를 파일에 저장합니다."""
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def get_lotto_number(round_no, cache):
    """캐시에서 조회합니다."""
    key = str(round_no)
    if key in cache:
        return cache[key]
    return None


def get_latest_round_no(session):
    """결과 페이지 HTML에서 최신 회차 번호를 추출합니다."""
    import re
    r = session.get("https://www.dhlottery.co.kr/lt645/result", timeout=10)
    m = re.search(r'회차별 당첨번호\s+(\d+)회', r.text)
    if m:
        return int(m.group(1))
    return None


def fetch_all_rounds(cache):
    """전체 회차를 API에서 가져와 캐싱합니다."""
    session = create_session()

    # 최신 회차 확인
    latest = get_latest_round_no(session)
    if not latest:
        print("  ❌ 최신 회차를 가져올 수 없습니다.")
        return 0

    # 이미 캐시된 회차 확인
    cached_rounds = set(int(k) for k in cache.keys())
    total = latest
    missing_count = total - len(cached_rounds)

    if missing_count <= 0:
        print(f"  ✅ 전체 {latest}회차 모두 캐시에 있습니다.")
        return latest

    print(f"  📥 총 {latest}회차 중 {missing_count}개 미캐싱 → 다운로드 시작...")

    # 최신 회차부터 older 방향으로 탐색
    fetched = 0
    cursor = latest + 1
    while True:
        batch = fetch_batch(session, "older", cursor)
        if not batch:
            break

        for item in batch:
            key = str(item["회차"])
            if key not in cache:
                cache[key] = item
                fetched += 1

        oldest_in_batch = min(item["회차"] for item in batch)
        cursor = oldest_in_batch

        # 진행률 표시
        progress = (latest - oldest_in_batch + 1) / total * 100
        print(f"    {oldest_in_batch}회차까지 수집... ({progress:.0f}%) [{fetched}개 추가]", end="\r")

        # 1회차까지 도달하면 종료
        if oldest_in_batch <= 1:
            break

        # 중간 저장 (100개마다)
        if fetched % 100 == 0:
            save_cache(cache)

        time.sleep(0.2)

    save_cache(cache)
    print(f"\n  ✅ {fetched}개 다운로드 완료! (총 {len(cache)}회차 캐시됨)")
    return latest


def main():
    print("=" * 60)
    print("         🎱 한국 로또 6/45 당첨번호 조회기")
    print("=" * 60)

    cache = load_cache()
    print(f"\n캐시 로드 완료 (기존 {len(cache)}회차 저장됨)")

    print("\n전체 회차 데이터 동기화 중...")
    latest = fetch_all_rounds(cache)
    save_cache(cache)
    print(f"현재 최신 회차: {latest}회\n")

    while True:
        print("-" * 60)
        print("옵션을 선택하세요:")
        print("  1. 특정 회차 조회")
        print("  2. 최근 N회차 조회")
        print("  3. 범위 조회 (시작~끝)")
        print("  4. 최신 회차 조회")
        print("  5. 캐시 갱신 (새 회차 추가)")
        print("  0. 종료")
        print("-" * 60)

        choice = input("선택: ").strip()

        if choice == "0":
            print("종료합니다.")
            break

        elif choice == "1":
            try:
                no = int(input(f"회차 번호 입력 (1~{latest}): "))
                result = get_lotto_number(no, cache)
                if result:
                    print_result(result)
                else:
                    print("해당 회차 정보가 없습니다.")
            except ValueError:
                print("숫자를 입력해주세요.")

        elif choice == "2":
            try:
                n = int(input("최근 몇 회차? "))
                start = max(1, latest - n + 1)
                for i in range(latest, start - 1, -1):
                    result = get_lotto_number(i, cache)
                    if result:
                        print_result(result)
            except ValueError:
                print("숫자를 입력해주세요.")

        elif choice == "3":
            try:
                s = int(input("시작 회차: "))
                e = int(input("끝 회차: "))
                for i in range(s, e + 1):
                    result = get_lotto_number(i, cache)
                    if result:
                        print_result(result)
                    else:
                        print(f"  {i}회차: 정보 없음")
            except ValueError:
                print("숫자를 입력해주세요.")

        elif choice == "4":
            result = get_lotto_number(latest, cache)
            if result:
                print_result(result)

        elif choice == "5":
            print("새 회차 확인 중...")
            latest = fetch_all_rounds(cache)
            save_cache(cache)
            print(f"최신 회차: {latest}회")

        else:
            print("올바른 옵션을 선택해주세요.")

        print()


def print_result(r):
    numbers = " ".join(f"[{n:2d}]" for n in r["당첨번호"])
    print(f"\n  📅 {r['회차']}회 ({r['추첨일']})")
    print(f"  🎱 당첨번호: {numbers} + 보너스 [{r['보너스번호']:2d}]")
    print(f"  💰 1등 당첨금: {r['1등 당첨금']} ({r['1등 당첨자수']}명)")


if __name__ == "__main__":
    main()
