"""
Smart 예측 번호 10게임 생성 + 자동 구매
5게임씩 2회 나누어 구매 (카트 최대 5게임)
"""
import sys
import os
import json
import time
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "lotto", "src"))

from lotto_smart import (
    load_data, build_structural_df, detect_regimes,
    StructuralPredictor, NumberScorer,
    train_transformer, predict_transformer, smart_select,
)
import xgboost as xgb


def generate_5_games():
    """Smart v2로 5게임 예측 생성"""
    print("=" * 60)
    print("  5게임 예측 번호 생성 중...")
    print("=" * 60)

    df = load_data()
    total = len(df)
    binary_matrix = np.zeros((total, 45), dtype=np.float32)
    for i, row in df.iterrows():
        for col in ["n1", "n2", "n3", "n4", "n5", "n6"]:
            binary_matrix[i, row[col] - 1] = 1.0

    struct_df = build_structural_df(df)
    regimes = detect_regimes(struct_df)
    current_regime = max(r for r in regimes if r < total)

    # XGBoost
    print("  XGBoost 학습...")
    regime_data = binary_matrix[current_regime:total]
    X_xgb, y_xgb = [], []
    for i in range(30, len(regime_data)):
        X_xgb.append(regime_data[i - 20:i].flatten())
        y_xgb.append(regime_data[i])
    X_xgb = np.array(X_xgb)
    y_xgb = np.array(y_xgb)

    xgb_models = []
    for num in range(45):
        m = xgb.XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.05,
            scale_pos_weight=39/6, verbosity=0, random_state=42
        )
        m.fit(X_xgb, y_xgb[:, num])
        xgb_models.append(m)

    seq = binary_matrix[total - 20:total].flatten().reshape(1, -1)
    xgb_probs = np.array([m.predict_proba(seq)[0, 1] for m in xgb_models])

    # Transformer
    print("  Transformer 학습...")
    tf_model = train_transformer(binary_matrix, total, seq_len=30, epochs=50)
    tf_probs = predict_transformer(tf_model, binary_matrix, total, seq_len=30)

    # Number Scorer
    scorer = NumberScorer()
    scorer_probs = scorer.score_numbers(df, total, current_regime)

    # Structural Predictor
    struct_predictor = StructuralPredictor()
    struct_predictor.train(struct_df, total)
    struct_pred = struct_predictor.predict(struct_df, total)

    # 앙상블
    combined = 0.30 * xgb_probs + 0.30 * tf_probs + 0.20 * scorer_probs
    combined = combined / combined.sum()

    # 5게임 선택
    top_combos = smart_select(combined, struct_pred, n_candidates=50000, top_n=5,
                              df=df, target_idx=total)

    last_round = int(df.iloc[-1]["round"])
    print(f"\n  {last_round + 1}회차 예측 5게임:")
    print(f"  {'─' * 50}")

    games = []
    for i, (combo, final, num_s, struct_s) in enumerate(top_combos):
        nums = [int(n) for n in combo]
        games.append(nums)
        total_sum = sum(nums)
        odd = sum(1 for n in nums if n % 2 == 1)
        print(f"  게임 {i+1}: {nums}  합={total_sum} 홀={odd} 짝={6-odd}")

    return games, last_round + 1


def buy_games(games):
    """Playwright로 5게임 구매"""
    from playwright.sync_api import sync_playwright
    from login import (
        login, is_logged_in, setup_dialog_handler, dismiss_popups,
        click_first_available, DEFAULT_USER_AGENT, DEFAULT_VIEWPORT,
        DEFAULT_HEADERS, GLOBAL_TIMEOUT, SESSION_PATH,
    )
    from pathlib import Path

    GAME_URL = "https://ol.dhlottery.co.kr/olotto/game_mobile/game645.do"
    HEADLESS = os.environ.get("HEADLESS", "true").lower() == "true"

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=HEADLESS, slow_mo=0 if HEADLESS else 500)

        storage_state = SESSION_PATH if Path(SESSION_PATH).exists() else None
        context = browser.new_context(
            storage_state=storage_state,
            user_agent=DEFAULT_USER_AGENT,
            viewport=DEFAULT_VIEWPORT,
            is_mobile=True, has_touch=True,
            extra_http_headers=DEFAULT_HEADERS,
        )

        try:
            page = context.new_page()
            setup_dialog_handler(page)

            # 로그인 확인
            if not is_logged_in(page):
                print("  로그인 중...")
                login(page)
            else:
                print("  세션 유효")

            # 구매 제한 확인 (토/일)
            try:
                resp = page.request.get("https://m.dhlottery.co.kr/selectMobPrchsCheck.do", timeout=GLOBAL_TIMEOUT)
                if resp.ok:
                    payload = resp.json()
                    mob = payload.get("data", {}).get("result", {}).get("mobPrchs", "")
                    if str(mob) == "1":
                        now_day = payload.get("data", {}).get("result", {}).get("nowDay", "")
                        print(f"\n  ⚠ 모바일 구매 제한: {now_day}")
                        print("  → PC 웹에서 직접 구매하거나 평일에 다시 시도하세요")
                        return 0
            except Exception:
                pass

            # 구매 페이지 이동
            print(f"  구매 페이지 이동...")
            page.goto(GAME_URL, timeout=GLOBAL_TIMEOUT, wait_until="domcontentloaded")

            if "/login" in page.url or "method=login" in page.url:
                login(page)
                page.goto(GAME_URL, timeout=GLOBAL_TIMEOUT, wait_until="domcontentloaded")

            time.sleep(2)
            dismiss_popups(page)

            # 5게임 수동 번호 입력
            for game_idx, numbers in enumerate(games):
                print(f"  게임 {game_idx + 1}: {numbers} 입력 중...")
                dismiss_popups(page)

                # '번호 선택하기' 클릭
                click_first_available(
                    page,
                    ["button:has-text('번호 선택하기')", "text='번호 선택하기'"],
                    "번호 선택 버튼",
                )
                page.wait_for_selector("#popupSelectNum", state="visible", timeout=GLOBAL_TIMEOUT)
                time.sleep(0.8)

                # 초기화
                reset_btn = page.locator("#btnInit, #popupSelectNum button:has-text('초기화')").first
                if reset_btn.is_visible(timeout=2000):
                    reset_btn.click(timeout=1500, force=True)
                    time.sleep(0.5)

                # 번호 선택
                for number in numbers:
                    num_el = page.locator(f"xpath=//div[contains(@class, 'lt-num') and text()='{number}']").first
                    if num_el.is_visible(timeout=2000):
                        num_el.click()
                        time.sleep(0.05)
                    else:
                        print(f"    번호 {number} 선택 실패!")

                # 선택완료
                cart_before = page.locator(".myNum-box:visible").count()
                select_done = page.locator("#btnSelectNum, #popupSelectNum button:has-text('선택완료')").first
                if select_done.is_visible(timeout=2000):
                    select_done.click(timeout=1500, force=True)

                # 알림 처리
                alert = page.locator("#popupLayerAlert:visible")
                if alert.is_visible(timeout=2000):
                    msg = alert.inner_text()
                    print(f"    알림: {msg}")
                    alert.locator("button:has-text('확인')").click()
                    time.sleep(0.5)

                # 추가 확인
                for _ in range(15):
                    time.sleep(0.3)
                    if page.locator(".myNum-box:visible").count() > cart_before:
                        break

                # 팝업 닫기
                try:
                    page.wait_for_selector("#popupSelectNum", state="hidden", timeout=3000)
                except:
                    close_btn = page.locator("#popupSelectNum .btn-pop-close").first
                    if close_btn.is_visible(timeout=1000):
                        close_btn.click()
                        time.sleep(0.5)

                time.sleep(0.5)
                print(f"    ✓ 추가 완료 (카트: {page.locator('.myNum-box:visible').count()}게임)")

            cart_count = page.locator(".myNum-box:visible").count()
            print(f"\n  카트 {cart_count}게임 → 구매 진행 ({cart_count * 1000:,}원)")

            if cart_count == 0:
                print("  카트 비어있음!")
                page.screenshot(path=os.path.join(BASE_DIR, "debug_empty_cart.png"))
                return 0

            # 구매하기 클릭
            dismiss_popups(page)
            click_first_available(
                page,
                ["#btnBuy", "button:has-text('구매하기')"],
                "구매하기 버튼",
            )

            # 확인 팝업
            confirm_btn = page.locator("#popupLayerConfirm .buttonOk, #popupLayerConfirm button:has-text('확인')").first
            try:
                confirm_btn.wait_for(state="visible", timeout=5000)
                confirm_btn.click(timeout=1500, force=True)
                print("  구매 확인 클릭")
            except Exception as e:
                print(f"  확인 버튼 실패: {e}")

            # 결과 확인
            try:
                page.wait_for_function(
                    """() => {
                        const text = document.body ? document.body.innerText : "";
                        const markers = ["구매가 완료되었습니다", "구매를 완료하였습니다",
                                         "예치금이 부족합니다", "구매한도", "주문번호"];
                        return markers.some(m => text.includes(m)) ||
                               document.querySelector("#report")?.offsetParent !== null;
                    }""",
                    timeout=30000,
                )

                body_text = page.locator("body").inner_text()
                if any(m in body_text for m in ["구매가 완료되었습니다", "구매를 완료하였습니다", "주문번호"]):
                    print(f"\n  ✅ 구매 성공! ({cart_count}게임 / {cart_count * 1000:,}원)")
                    page.screenshot(path=os.path.join(BASE_DIR, "purchase_success.png"))
                    context.storage_state(path=SESSION_PATH)
                    return cart_count
                elif "예치금이 부족" in body_text:
                    print("  ❌ 예치금 부족!")
                elif "구매한도" in body_text:
                    print("  ❌ 주간 구매한도 초과!")
                else:
                    print("  ⚠ 결과 불확실")
                    page.screenshot(path=os.path.join(BASE_DIR, "purchase_ambiguous.png"))

            except Exception as e:
                print(f"  결과 확인 타임아웃: {e}")
                page.screenshot(path=os.path.join(BASE_DIR, "purchase_timeout.png"))

            context.storage_state(path=SESSION_PATH)
            return cart_count

        except Exception as e:
            print(f"  오류: {e}")
            import traceback
            traceback.print_exc()
            try:
                page.screenshot(path=os.path.join(BASE_DIR, "purchase_error.png"))
            except:
                pass
            return 0
        finally:
            context.close()
            browser.close()


if __name__ == "__main__":
    # 1. 5게임 예측
    games, target_round = generate_5_games()

    # 저장
    with open(os.path.join(BASE_DIR, "models_v2", "purchase_games.json"), "w", encoding="utf-8") as f:
        json.dump({
            "target_round": target_round,
            "games": games,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  구매 진행 (5게임 / 5,000원)")
    print(f"{'=' * 60}")

    # 2. 구매
    purchased = buy_games(games)
    print(f"\n{'=' * 60}")
    print(f"  완료! {purchased}게임 구매 ({purchased * 1000:,}원)")
    print(f"{'=' * 60}")
