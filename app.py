import datetime
import io
import json
import os
import re
import time
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

import v17_4_1 as engine

# ==========================================
# ページ設定 & スマホ向けCSS
# ==========================================
st.set_page_config(
    page_title="競艇予測 v17.4 モバイル",
    page_icon="🚤",
    layout="centered",
    initial_sidebar_state="collapsed",
)

BOAT_COLORS = {
    1: ("#FFFFFF", "#111827", "#CBD5E1"),
    2: ("#1F2937", "#FFFFFF", "#1F2937"),
    3: ("#DC2626", "#FFFFFF", "#DC2626"),
    4: ("#2563EB", "#FFFFFF", "#2563EB"),
    5: ("#FACC15", "#111827", "#EAB308"),
    6: ("#16A34A", "#FFFFFF", "#16A34A"),
}

STADIUM_LIST = [engine.STADIUM_CODE_MAP[i] for i in range(1, 25)]

# 画像読み取りの速度と安定性を優先したモデル一覧（上から推奨順）
DEFAULT_GEMINI_MODELS = [
    "gemini-4-hydrogen",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-4-carbon",
    "gemini-3.1-pro-preview",
]

# 自動フォールバック順（選択モデルが 429 / 404 / タイムアウト時に順次試行）
FALLBACK_MODEL_CHAIN = [
    "gemini-4-hydrogen",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-pro",
]

MAX_IMAGE_SIDE = 1600
JPEG_QUALITY = 85
API_TIMEOUT_MS = 60_000

st.markdown(
    """
<style>
.block-container {
    padding-top: 1.0rem;
    padding-bottom: 2.5rem;
    max-width: 760px;
}
.boat-badge {
    display: inline-block;
    width: 26px;
    height: 26px;
    line-height: 24px;
    text-align: center;
    border-radius: 6px;
    font-weight: 700;
    font-size: 14px;
    margin-right: 6px;
}
.rank-banner {
    padding: 14px 16px;
    border-radius: 12px;
    margin-bottom: 12px;
    border: 1px solid #E2E8F0;
}
.rank-main {
    background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
    border-color: #34D399;
}
.rank-ana {
    background: linear-gradient(135deg, #FFF7ED 0%, #FFEDD5 100%);
    border-color: #FB923C;
}
.rank-skip {
    background: linear-gradient(135deg, #F8FAFC 0%, #E2E8F0 100%);
    border-color: #94A3B8;
}
.bet-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-left: 5px solid #0F766E;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
}
.bet-card-ana {
    border-left-color: #EA580C;
}
.bet-comb {
    font-size: 20px;
    font-weight: 800;
    letter-spacing: 1px;
    color: #0F172A;
}
.bet-meta {
    font-size: 13px;
    color: #475569;
    margin-top: 2px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# Gemini Structured Output スキーマ定義
# ==========================================
class BoatExtractedData(BaseModel):
    boat_no: int = Field(..., ge=1, le=6, description="枠番 (1〜6)")
    reg_no: Optional[str] = Field(None, description="4桁の選手登録番号 (例: '4320')。画像に無ければnull")
    racer_name: Optional[str] = Field(None, description="選手名 (スペース除去、例: '峰竜太')。画像に無ければnull")
    motor_rate: Optional[float] = Field(None, description="モーター2連対率(%) (例: 38.5)")
    exhibition_st: Optional[str] = Field(None, description="展示ST表記そのまま (例: '.14', 'F.02', '0.18')")
    tilt: Optional[float] = Field(0.0, description="チルト角度 (例: -0.5, 0.0, 0.5)。未記載は0.0")
    straight_time: Optional[float] = Field(None, description="直線タイム (例: 6.72)。徳山など直線タイムが無い場合は展示タイムを使用")
    turn_time: Optional[float] = Field(None, description="一周タイム (例: 37.15)。未計測(--.--)の場合はnull")
    inspect_time: Optional[float] = Field(None, description="まわり足タイム (例: 5.45)。未計測(--.--)の場合はnull")
    target_course: Optional[int] = Field(None, ge=1, le=6, description="進入コース(1〜6)。枠なり・未記載ならboat_noと同じ")
    is_flying: Optional[bool] = Field(False, description="今期フライング持ち(F1/F2等)があればtrue、空欄ならfalse")
    win_rate_local: Optional[float] = Field(None, description="当地勝率 (0.00や未記載ならnull)")
    rate_local: Optional[float] = Field(None, description="当地2連対率(%) (0.0や未記載ならnull)")


class RaceExtractionResult(BaseModel):
    stadium: Optional[str] = Field(None, description="開催会場名 (例: '戸田', '多摩川', '大村') または場番号('02'等)")
    race_no: Optional[int] = Field(None, ge=1, le=12, description="レース番号 (1〜12)")
    race_date: Optional[str] = Field(None, description="開催日 YYYYMMDD (例: '20261127')。画像に無ければnull")
    is_womens_race: Optional[bool] = Field(False, description="女子限定戦(オールレディース・ヴィーナス等)のみtrue。男女混合戦はfalse")
    wind_speed: Optional[float] = Field(0.0, description="風速(m)。未記載は0.0")
    wind_dir: Optional[str] = Field("normal", description="風向: 追い風='tail', 向かい風='head', 横風/無風/不明='normal'")
    boats: List[BoatExtractedData] = Field(..., description="1号艇〜6号艇の抽出データ一覧")
    notes: Optional[str] = Field(None, description="読み取り時の補足・不鮮明箇所・欠損項目があれば簡潔に記載")


GEMINI_EXTRACTION_PROMPT = """
あなたは日本のボートレース（競艇）の出走表・直前情報・展示タイム画面のスクリーンショットや写真を正確に読み取る専門アシスタントです。
添付された1枚または複数枚の画像から、レース条件と1号艇〜6号艇のデータを統合してJSONで抽出してください。

【読み取りルール】
1. 複数枚の画像（例: 出走表の画面 + 展示タイム/直前情報の画面）が与えられた場合、1〜6号艇ごとに情報を統合してください。
2. 登録番号（4桁）が読み取れる場合は必ず `reg_no` に設定してください。選手名はスペースを除いて `racer_name` に設定してください。
3. 展示STは `.14` や `F.02` のように画像上の表記をそのまま `exhibition_st` に入れてください（フライングは必ず先頭に `F` を付ける）。
4. チルトが未記載の場合は `0.0` にしてください。
5. 徳山など直線タイムが表示されない場では、展示タイムを `straight_time` に入れてください。
6. 一周タイムやまわり足タイムが `--.--` や未計測・未掲載の場合は無理に推測せず `null` にしてください。
7. 今期F持ち（F1, F2 等）の記載があれば `is_flying: true`、空欄なら `false` にしてください。
8. 当地勝率が `0.00` や当地2連率が `0.0`（または未掲載）の場合は `null` にしてください。
9. `is_womens_race` はオールレディース・ヴィーナスシリーズ等の「女子限定戦」のときだけ `true` にし、男女混合戦や一般戦は `false` にしてください。
10. 風向は「追い風」なら `"tail"`、「向かい風」なら `"head"`、それ以外（横風・無風・不明）は `"normal"` にしてください。
11. 画像に存在しない数値は推測せず `null` とし、`notes` に不足している項目を簡潔に書いてください。
"""


# ==========================================
# データ初期化・画像最適化・APIヘルパー
# ==========================================
@st.cache_resource
def init_engine_resources():
    meta = engine.load_model_params()
    db = engine.load_racer_db()
    return meta, db


def reload_engine_resources():
    init_engine_resources.clear()
    return init_engine_resources()


def get_default_boats_rows():
    return [
        {
            "枠": b,
            "登番/選手名": "",
            "モーター2連率": 35.0,
            "展示ST": ".15",
            "チルト": 0.0,
            "直線": 6.80,
            "一周": 37.00,
            "まわり足": 5.50,
            "進入": b,
            "F持ち": False,
            "当地勝率": None,
            "当地2連率": None,
        }
        for b in range(1, 7)
    ]


def ensure_session_state():
    today_str = datetime.datetime.now().strftime("%Y%m%d")
    defaults = {
        "stadium": "戸田",
        "race_no": 1,
        "race_date": today_str,
        "budget": 1500,
        "is_womens_race": False,
        "wind_speed": 0.0,
        "wind_dir": "normal",
        "use_real_odds": True,
        "rel_filter_enabled": bool(engine.RELIABILITY_FILTER_ENABLED),
        "boats_rows": get_default_boats_rows(),
        "ai_notes": "",
        "ai_meta_msg": "",
        "last_prediction": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_gemini_api_key():
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
        if secret_key:
            return secret_key
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "")


def optimize_image_bytes(raw_bytes: bytes, max_side: int = MAX_IMAGE_SIDE, quality: int = JPEG_QUALITY) -> Tuple[bytes, str]:
    """スマホの高解像度スクショ・写真をEXIF回転補正＆リサイズ圧縮してAPI転送を高速化する。"""
    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            elif img.mode == "L":
                img = img.convert("RGB")

            w, h = img.size
            longest = max(w, h)
            if longest > max_side:
                scale = max_side / float(longest)
                new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            return buf.getvalue(), "image/jpeg"
    except Exception:
        return raw_bytes, "image/jpeg"


def explain_api_error(exc: Exception) -> str:
    msg = str(exc)
    low = msg.lower()
    if "api_key" in low or "api key" in low or "401" in low or "403" in low or "permission" in low:
        return f"APIキーが無効、または権限がありません。Streamlit Secrets の GEMINI_API_KEY を確認してください。（詳細: {msg[:180]}）"
    if "429" in low or "quota" in low or "resource_exhausted" in low or "rate" in low:
        return f"利用上限（レート制限 / 429）に達しました。モデルを `gemini-3.5-flash` や `gemini-3.5-flash-lite` に切り替えるか、少し時間を置いて再試行してください。（詳細: {msg[:180]}）"
    if "404" in low or "not found" in low:
        return f"指定したモデル名が現在のAPIキーでは利用できません。サイドバーから別のモデルを選択してください。（詳細: {msg[:180]}）"
    if "timeout" in low or "deadline" in low:
        return f"応答がタイムアウトしました。画像を2枚以下に分けるか、より高速な `gemini-3.5-flash` を試してください。（詳細: {msg[:180]}）"
    return f"読み取りエラー: {msg[:240]}"


def test_gemini_connection(api_key: str, model_name: str) -> Tuple[bool, str]:
    from google import genai
    from google.genai import types

    t0 = time.time()
    try:
        client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=20_000))
        resp = client.models.generate_content(
            model=model_name,
            contents="OKとだけ返してください。",
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=16),
        )
        elapsed = time.time() - t0
        txt = (resp.text or "").strip()
        return True, f"接続成功（{model_name} / {elapsed:.1f}秒 / 応答: {txt or 'OK'}）"
    except Exception as e:
        return False, explain_api_error(e)


def _call_single_model_extraction(client, model_name: str, contents: list) -> RaceExtractionResult:
    from google.genai import types

    # 1st try: Structured Output (Pydantic schema)
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RaceExtractionResult,
                temperature=0.1,
            ),
        )
        if getattr(response, "parsed", None) is not None:
            return response.parsed
        raw_text = (response.text or "").strip()
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)
        return RaceExtractionResult.model_validate_json(raw_text)
    except Exception as first_err:
        # Schema非対応系エラーのみ JSON モードへフォールバック（認証/429/404は上位へ投げる）
        low = str(first_err).lower()
        if any(tok in low for tok in ("401", "403", "404", "429", "quota", "api_key", "api key", "not found", "timeout", "deadline")):
            raise
        schema_hint = (
            "\n出力は必ず次のキーを持つJSONオブジェクトのみ返してください: "
            "stadium, race_no, race_date, is_womens_race, wind_speed, wind_dir, "
            "boats([{boat_no, reg_no, racer_name, motor_rate, exhibition_st, tilt, "
            "straight_time, turn_time, inspect_time, target_course, is_flying, win_rate_local, rate_local}]), notes"
        )
        fallback_contents = [contents[0] + schema_hint] + contents[1:]
        response = client.models.generate_content(
            model=model_name,
            contents=fallback_contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        raw_text = (response.text or "").strip()
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)
        return RaceExtractionResult.model_validate_json(raw_text)


def extract_race_from_images(
    image_files,
    api_key: str,
    model_name: str,
    auto_fallback: bool = True,
) -> Tuple[RaceExtractionResult, str, float, str]:
    """画像をリサイズ圧縮してGemini APIへ送信。失敗時は自動で安定モデルへフォールバックする。"""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=API_TIMEOUT_MS))
    contents = [GEMINI_EXTRACTION_PROMPT]

    orig_kb = 0.0
    comp_kb = 0.0
    for uf in image_files:
        raw_bytes = uf.getvalue()
        orig_kb += len(raw_bytes) / 1024.0
        opt_bytes, mime_type = optimize_image_bytes(raw_bytes)
        comp_kb += len(opt_bytes) / 1024.0
        contents.append(types.Part.from_bytes(data=opt_bytes, mime_type=mime_type))

    candidates = [model_name]
    if auto_fallback:
        for fb in FALLBACK_MODEL_CHAIN:
            if fb not in candidates:
                candidates.append(fb)

    errors = []
    t0 = time.time()
    for cand in candidates:
        try:
            result = _call_single_model_extraction(client, cand, contents)
            elapsed = time.time() - t0
            size_info = f"{orig_kb:.0f}KB → {comp_kb:.0f}KB"
            return result, cand, elapsed, size_info
        except Exception as e:
            errors.append(f"{cand}: {e}")
            low = str(e).lower()
            # APIキー自体が無効なら他モデルを試しても無駄なので即終了
            if "api_key" in low or "api key" in low or "401" in low or "403" in low:
                raise RuntimeError(explain_api_error(e)) from e

    last_summary = " / ".join(errors[:3])
    raise RuntimeError(f"試行した全モデルで失敗しました。（{last_summary[:320]}）")


def apply_extracted_result_to_state(extracted: RaceExtractionResult, db: dict, meta_msg: str = ""):
    if extracted.stadium:
        norm_st = engine.normalize_stadium_name(extracted.stadium)
        if norm_st in STADIUM_LIST:
            st.session_state["stadium"] = norm_st

    if extracted.race_no and 1 <= extracted.race_no <= 12:
        st.session_state["race_no"] = int(extracted.race_no)

    if extracted.race_date and re.fullmatch(r"\d{8}", extracted.race_date.strip()):
        st.session_state["race_date"] = extracted.race_date.strip()

    st.session_state["is_womens_race"] = bool(extracted.is_womens_race)
    st.session_state["wind_speed"] = float(extracted.wind_speed or 0.0)
    wdir = str(extracted.wind_dir or "normal").lower()
    st.session_state["wind_dir"] = wdir if wdir in ("normal", "tail", "head") else "normal"

    by_boat = {b.boat_no: b for b in extracted.boats if 1 <= b.boat_no <= 6}

    valid_straight = [b.straight_time for b in by_boat.values() if b.straight_time is not None]
    valid_turn = [b.turn_time for b in by_boat.values() if b.turn_time is not None]
    valid_inspect = [b.inspect_time for b in by_boat.values() if b.inspect_time is not None]

    fallback_straight = round(sum(valid_straight) / len(valid_straight), 2) if valid_straight else 6.80
    fallback_turn = round(sum(valid_turn) / len(valid_turn), 2) if valid_turn else 37.00
    fallback_inspect = round(sum(valid_inspect) / len(valid_inspect), 2) if valid_inspect else 5.50

    rows = []
    for b_no in range(1, 7):
        item = by_boat.get(b_no)
        if item is None:
            rows.append(get_default_boats_rows()[b_no - 1])
            continue

        racer_key = ""
        if item.reg_no and item.reg_no.strip().isdigit():
            racer_key = item.reg_no.strip()
        elif item.racer_name:
            racer_key = re.sub(r"[\s　]+", "", item.racer_name)

        r_info = engine.find_racer_in_db(db, racer_key) if racer_key else None
        if r_info is None and item.racer_name:
            r_info = engine.find_racer_in_db(db, item.racer_name)
            if r_info:
                racer_key = r_info.get("reg_no") or r_info.get("name")

        is_f = bool(item.is_flying)
        if r_info and not is_f:
            is_f = bool(r_info.get("is_flying", False))

        win_loc = item.win_rate_local if (item.win_rate_local is not None and item.win_rate_local > 0) else None
        rate_loc = item.rate_local if (item.rate_local is not None and item.rate_local > 0) else None

        rows.append(
            {
                "枠": b_no,
                "登番/選手名": racer_key,
                "モーター2連率": float(item.motor_rate) if item.motor_rate is not None else 35.0,
                "展示ST": str(item.exhibition_st).strip() if item.exhibition_st else ".15",
                "チルト": float(item.tilt) if item.tilt is not None else 0.0,
                "直線": float(item.straight_time) if item.straight_time is not None else fallback_straight,
                "一周": float(item.turn_time) if item.turn_time is not None else fallback_turn,
                "まわり足": float(item.inspect_time) if item.inspect_time is not None else fallback_inspect,
                "進入": int(item.target_course) if item.target_course in (1, 2, 3, 4, 5, 6) else b_no,
                "F持ち": is_f,
                "当地勝率": win_loc,
                "当地2連率": rate_loc,
            }
        )

    st.session_state["boats_rows"] = rows
    st.session_state["ai_notes"] = extracted.notes or ""
    st.session_state["ai_meta_msg"] = meta_msg


def parse_quick_text_lines(raw_text: str, stadium_name: str, db: dict):
    """v17_4_1.py の7項目テキスト形式（1〜6号艇分）を一括パースしてフォームに反映する。"""
    lines = [ln.strip() for ln in raw_text.strip().splitlines() if ln.strip()]
    if len(lines) != 6:
        raise ValueError(f"6艇分の6行を入力してください（現在: {len(lines)}行）")

    rows = []
    for idx, line in enumerate(lines, start=1):
        tokens = re.split(r"\s+", line)
        if len(tokens) < 7:
            raise ValueError(f"{idx}号艇の項目数が不足しています（最低7項目必要）")

        racer_key = tokens[0]
        r_info = engine.find_racer_in_db(db, racer_key)
        motor = float(tokens[1])
        st_str = tokens[2]
        _ = engine.parse_st_value(st_str)
        tilt = float(tokens[3])
        straight_t = float(tokens[4])
        turn_t = float(tokens[5])
        inspect_t = float(tokens[6])

        target_course = idx
        is_flying = bool(r_info.get("is_flying", False)) if r_info else False
        win_loc = None
        rate_loc = None

        for opt in tokens[7:]:
            opt_low = opt.lower()
            if opt_low in ["f", "f1", "true"]:
                is_flying = True
            elif opt_low == "f0":
                is_flying = False
            elif opt_low.startswith("c") and opt_low[1:].isdigit() and 1 <= int(opt_low[1:]) <= 6:
                target_course = int(opt_low[1:])
            elif opt_low.startswith("l") and "," in opt_low[1:]:
                p0, p1 = opt_low[1:].split(",", 1)
                win_loc = float(p0)
                rate_loc = float(p1)

        rows.append(
            {
                "枠": idx,
                "登番/選手名": racer_key,
                "モーター2連率": motor,
                "展示ST": st_str,
                "チルト": tilt,
                "直線": straight_t,
                "一周": turn_t,
                "まわり足": inspect_t,
                "進入": target_course,
                "F持ち": is_flying,
                "当地勝率": win_loc,
                "当地2連率": rate_loc,
            }
        )
    return rows


def build_race_dataframe(rows: list, stadium_name: str, db: dict):
    records = []
    warnings = []

    for idx, row in enumerate(rows, start=1):
        racer_key = str(row.get("登番/選手名") or "").strip()
        racer_info = engine.find_racer_in_db(db, racer_key) if racer_key else None
        if racer_info is None:
            warnings.append(f"{idx}号艇 ({racer_key or '未入力'}) はDB未登録のためB2標準値(勝率2.50)で計算します。")
            racer_info = {
                "reg_no": racer_key if racer_key.isdigit() else "0000",
                "name": f"未登録({racer_key or idx})",
                "class_type": "B2",
                "win_nat": 2.50,
                "2rate_nat": 8.0,
                "is_flying": False,
                "local": {},
                "courses": engine.DEFAULT_COURSE_STATS,
            }

        st_val = engine.parse_st_value(row.get("展示ST", ".15"))
        target_course = int(row.get("進入") or idx)
        if not (1 <= target_course <= 6):
            target_course = idx

        local_stats = racer_info.get("local", {}).get(stadium_name, {})
        win_loc_in = row.get("当地勝率")
        rate_loc_in = row.get("当地2連率")
        win_loc = float(win_loc_in) if (win_loc_in is not None and not pd.isna(win_loc_in) and float(win_loc_in) > 0) else local_stats.get("win_loc", racer_info["win_nat"])
        rate_loc = float(rate_loc_in) if (rate_loc_in is not None and not pd.isna(rate_loc_in) and float(rate_loc_in) > 0) else local_stats.get("2rate_loc", racer_info["2rate_nat"])

        c_stats = racer_info.get("courses", {}).get(str(target_course), engine.DEFAULT_COURSE_STATS[str(target_course)])

        records.append(
            {
                "boat_no": idx,
                "racer_name": racer_info.get("name", racer_key),
                "reg_no": racer_info.get("reg_no", ""),
                "class_type": racer_info["class_type"],
                "win_rate_national": float(racer_info["win_nat"]),
                "2rate_national": float(racer_info["2rate_nat"]),
                "win_rate_local": float(win_loc),
                "2rate_local": float(rate_loc),
                "motor_rate": float(row.get("モーター2連率") or 35.0),
                "exhibition_st": float(st_val),
                "tilt": float(row.get("チルト") or 0.0),
                "straight_time": float(row.get("直線") or 6.80),
                "turn_time": float(row.get("一周") or 37.00),
                "inspect_time": float(row.get("まわり足") or 5.50),
                "target_course": target_course,
                "course_1st_rate": float(c_stats["1st"]),
                "course_2nd_rate": float(c_stats["2nd"]),
                "course_3rd_rate": float(c_stats["3rd"]),
                "makuri_rate": float(c_stats["makuri"]),
                "sashi_rate": float(c_stats["sashi"]),
                "is_flying": bool(row.get("F持ち", False)),
            }
        )

    df_res = pd.DataFrame(records)
    if df_res["target_course"].nunique() != 6:
        warnings.append("進入コースに重複があるため枠なり（1〜6コース）として計算します。")
        df_res["target_course"] = df_res["boat_no"]
        for i, r in df_res.iterrows():
            c_str = str(int(r["target_course"]))
            r_info = engine.find_racer_in_db(db, r["reg_no"] or r["racer_name"])
            if r_info:
                c_st = r_info.get("courses", {}).get(c_str, engine.DEFAULT_COURSE_STATS[c_str])
                df_res.at[i, "course_1st_rate"] = c_st["1st"]
                df_res.at[i, "course_2nd_rate"] = c_st["2nd"]
                df_res.at[i, "course_3rd_rate"] = c_st["3rd"]
                df_res.at[i, "makuri_rate"] = c_st["makuri"]
                df_res.at[i, "sashi_rate"] = c_st["sashi"]

    return df_res, warnings


def run_web_prediction(
    race_df: pd.DataFrame,
    stadium: str,
    race_id: str,
    budget: int,
    is_womens_race: bool,
    wind_speed: float,
    wind_dir: str,
    use_real_odds: bool,
    rel_filter_enabled: bool,
    db: dict,
    save_log: bool = True,
):
    stadium = engine.normalize_stadium_name(stadium)
    cond_prof = engine.get_condition_profile(
        is_womens_race=is_womens_race,
        wind_speed=wind_speed,
        wind_dir=wind_dir,
    )
    res = engine.predict_race_v15(
        race_df,
        stadium,
        engine.STADIUM_PROFILES,
        is_womens_race=is_womens_race,
        wind_speed=wind_speed,
        wind_dir=wind_dir,
    )
    in_boats = res.loc[res["target_course"] == 1, "boat_no"].values
    in_boat = int(in_boats[0]) if len(in_boats) > 0 else 1
    confidence_score = float(res.set_index("boat_no")["pred_1st_prob"].get(in_boat, 0.0))

    if confidence_score >= 0.75:
        conf_rank = "A"
    elif confidence_score >= 0.60:
        conf_rank = "B"
    elif confidence_score >= 0.50:
        conf_rank = "C+"
    elif confidence_score >= 0.40:
        conf_rank = "C"
    elif confidence_score >= 0.30:
        conf_rank = "C-"
    elif confidence_score >= 0.18:
        conf_rank = "D"
    else:
        conf_rank = "E"

    reliability = engine.calculate_prediction_reliability(res, stadium, db, cond_prof)
    rel_blocked = bool(rel_filter_enabled and reliability["score"] < engine.RELIABILITY_MIN_SCORE)

    odds_status_msg = ""
    if rel_blocked:
        main_bets_df = pd.DataFrame()
        report_data = []
        ana_bets_df = pd.DataFrame()
    else:
        fetched_odds = None
        ana_eligible = (conf_rank in engine.ANA_TARGET_RANKS) and (
            not engine.ANA_STRICT_RACE_FILTER
            or engine.is_strict_ana_target_race(
                res,
                conf_rank=conf_rank,
                is_womens_race=is_womens_race,
                wind_speed=wind_speed,
                wind_dir=wind_dir,
            )
        )
        if ana_eligible and use_real_odds:
            hd, jcd, rno = engine.parse_race_id_for_odds(race_id, stadium)
            if hd and jcd and rno:
                fetched_odds = engine.fetch_real_odds3t(hd, jcd, rno)
                if fetched_odds:
                    w_cur, _ = engine.get_active_weights(stadium)
                    ev_th = w_cur.get("ana_min_real_ev", 1.00)
                    odds_status_msg = f"公式実オッズ {len(fetched_odds)}通り取得済（実EV >= {ev_th:.2f} 適用）"
                else:
                    odds_status_msg = "公式オッズ未取得（発売前/通信エラー）→ 厳選仮想オッズで算出"

        main_bets_df, report_data = engine.generate_custom_bets_v15(res, conf_rank, budget=budget)
        ana_bets_df = engine.generate_ana_bets(
            res,
            conf_rank,
            budget=cond_prof["max_bets"] * 100,
            stadium_name=stadium,
            real_odds_map=fetched_odds,
            is_womens_race=is_womens_race,
            wind_speed=wind_speed,
            wind_dir=wind_dir,
        )

    if save_log:
        engine.save_prediction_log(race_id, stadium, res, conf_rank, ana_bets_df, main_bets_df=main_bets_df)

    _, w_label = engine.get_active_weights(stadium)
    return {
        "race_id": race_id,
        "stadium": stadium,
        "w_label": w_label,
        "cond_prof": cond_prof,
        "res": res,
        "in_boat": in_boat,
        "confidence_score": confidence_score,
        "conf_rank": conf_rank,
        "reliability": reliability,
        "rel_blocked": rel_blocked,
        "main_bets_df": main_bets_df,
        "ana_bets_df": ana_bets_df,
        "report_data": report_data,
        "budget": budget,
        "odds_status_msg": odds_status_msg,
        "is_womens_race": is_womens_race,
        "wind_speed": wind_speed,
        "wind_dir": wind_dir,
    }


def boat_badge_html(boat_no: int) -> str:
    bg, fg, border = BOAT_COLORS.get(int(boat_no), ("#E2E8F0", "#111827", "#CBD5E1"))
    return f'<span class="boat-badge" style="background:{bg};color:{fg};border:1px solid {border};">{boat_no}</span>'


def render_prediction_result(pred: dict):
    conf_rank = pred["conf_rank"]
    rel = pred["reliability"]
    rel_blocked = pred["rel_blocked"]
    main_bets_df = pred["main_bets_df"]
    ana_bets_df = pred["ana_bets_df"]
    res = pred["res"]

    if rel_blocked:
        banner_cls = "rank-skip"
        mode_tag = "見送り（信頼度フィルタ）"
    elif conf_rank in engine.MAIN_TARGET_RANKS:
        banner_cls = "rank-main"
        mode_tag = f"本線予想（{len(main_bets_df)}点・予算{pred['budget']}円）"
    else:
        banner_cls = "rank-ana"
        mode_tag = "穴予想対象（最大4点）" if not ana_bets_df.empty else "見送り（穴条件外）"

    st.markdown(
        f"""
        <div class="rank-banner {banner_cls}">
            <div style="font-size:13px;color:#475569;">🏟️ {pred['stadium']}（{pred['race_id']}） / {pred['w_label']}</div>
            <div style="font-size:22px;font-weight:800;margin:4px 0;color:#0F172A;">
                ランク {conf_rank} ｜ {mode_tag}
            </div>
            <div style="font-size:13px;color:#334155;">
                イン{pred['in_boat']}号艇 1着信頼度: <b>{pred['confidence_score']*100:.1f}%</b> ｜ 条件: <b>{pred['cond_prof']['label']}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("予測信頼度", f"{rel['score']}点", rel["grade"])
    c2.metric("明確さ", f"{rel['clarity']}/30")
    c3.metric("データ", f"{rel['data']}/10")
    c4.metric("安定性", f"{rel['stability']}/60")

    if pred["odds_status_msg"]:
        st.caption(f"🌐 {pred['odds_status_msg']}")

    st.subheader("🎯 推奨買い目")
    if rel_blocked:
        st.warning(
            f"予測信頼度 {rel['score']}点（{rel['grade']}）が基準（{engine.RELIABILITY_MIN_SCORE:.0f}点以上）未満のため自動見送りです。"
            " 買い目を確認したい場合は「信頼度70点フィルタ」をOFFにして再実行してください。"
        )
    elif conf_rank in engine.MAIN_TARGET_RANKS:
        copy_lines = []
        for _, b_row in main_bets_df.iterrows():
            comb = b_row["買い目"]
            amt = int(b_row["推奨金額"])
            copy_lines.append(f"{comb} {amt}円")
            st.markdown(
                f"""
                <div class="bet-card">
                    <div class="bet-comb">{comb} <span style="float:right;color:#0F766E;">{amt:,}円</span></div>
                    <div class="bet-meta">確率 {b_row['確率']*100:.2f}% ｜ 分岐オッズ {b_row['分岐オッズ']:.1f}倍〜 ｜ 妙味度 {b_row['妙味度']:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.code("\n".join(copy_lines), language="text")
    else:
        if ana_bets_df is None or ana_bets_df.empty:
            if engine.ANA_STRICT_RACE_FILTER and not engine.is_strict_ana_target_race(
                res,
                conf_rank=conf_rank,
                is_womens_race=pred["is_womens_race"],
                wind_speed=pred["wind_speed"],
                wind_dir=pred["wind_dir"],
            ):
                st.info(f"⏭️ 見送り（ケン）：ランク {conf_rank} は1枠A級または全国勝率5.0以上のため穴狙い対象外です。")
            else:
                st.info(f"⏭️ 見送り（ケン）：ランク {conf_rank} で期待値・確率条件を満たす穴買い目がありません。")
        else:
            is_real_mode = ("オッズ種別" in ana_bets_df.columns) and (ana_bets_df["オッズ種別"].iloc[0] == "実オッズ")
            copy_lines = []
            for _, b_row in ana_bets_df.iterrows():
                comb = b_row["買い目"]
                amt = int(b_row["推奨金額"])
                copy_lines.append(f"{comb} {amt}円")
                if is_real_mode:
                    meta_str = f"[{b_row['穴区分']}] 確率 {b_row['確率']*100:.2f}% ｜ 実オッズ {b_row['実オッズ']:.1f}倍 ｜ 実EV {b_row['実EV']:.2f}"
                else:
                    meta_str = f"[{b_row['穴区分']}] 確率 {b_row['確率']*100:.2f}% ｜ 概算 {b_row['概算オッズ']:.1f}倍 ｜ 妙味 {b_row['妙味度']:.2f}"
                st.markdown(
                    f"""
                    <div class="bet-card bet-card-ana">
                        <div class="bet-comb">{comb} <span style="float:right;color:#EA580C;">{amt:,}円</span></div>
                        <div class="bet-meta">{meta_str}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.code("\n".join(copy_lines), language="text")

        if conf_rank in ("D", "E") and pred["report_data"]:
            with st.expander("🔥 イン飛び展開参考（決まり手特性からの軸候補）"):
                for idx, rd in enumerate(pred["report_data"], start=1):
                    st.write(
                        f"**第{idx}候補：{rd['killer_boat']}号艇**（{rd['killer_course']}C / 1着率 {rd['killer_prob']*100:.1f}%） "
                        f"→ {rd['tactics']} / 相手軸: {rd['target1']}・{rd['target2']}号艇"
                    )

    st.subheader("📊 各艇パワー＆展示評価")
    disp_rows = []
    for _, r in res.sort_values("target_course").iterrows():
        disp_rows.append(
            {
                "枠": int(r["boat_no"]),
                "進入": f"{int(r['target_course'])}C",
                "選手名": f"{r['racer_name']} ({r['class_type']})",
                "1着率": f"{r['pred_1st_prob']*100:.1f}%",
                "展示ST": f"{r['exhibition_st']:+.2f}" + ("(F)" if r["is_flying"] else ""),
                "展示指数": round(float(r["ex_score"]), 1),
                "総合": round(float(r["pure_diff_score"]), 1),
                "C1着率": f"{r['course_1st_rate']:.1f}%",
            }
        )
    st.dataframe(pd.DataFrame(disp_rows), use_container_width=True, hide_index=True)


# ==========================================
# メインアプリ
# ==========================================
ensure_session_state()
meta, db = init_engine_resources()

with st.sidebar:
    st.header("⚙️ 設定")
    default_api_key = get_gemini_api_key()
    api_key_input = st.text_input(
        "Gemini APIキー",
        value=default_api_key,
        type="password",
        help="Streamlit Cloud の Secrets に GEMINI_API_KEY を設定すると自動入力されます。",
    )
    selected_model = st.selectbox(
        "Geminiモデル（推奨: gemini-4-hydrogen / gemini-3.5-flash）",
        DEFAULT_GEMINI_MODELS,
        index=0,
    )
    custom_model = st.text_input("カスタムモデル名（任意）", value="", placeholder="例: gemini-4-hydrogen")
    active_model = custom_model.strip() or selected_model
    auto_fallback = st.toggle(
        "エラー時 自動フォールバック",
        value=True,
        help="選択モデルが429/404/タイムアウト時に安定モデルへ自動切替して再試行します。",
    )

    if st.button("🔌 APIキー＆モデル疎通テスト", use_container_width=True):
        if not api_key_input:
            st.error("Gemini APIキーを入力してください。")
        else:
            with st.spinner(f"{active_model} に接続確認中..."):
                ok, msg = test_gemini_connection(api_key_input, active_model)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    st.divider()
    st.caption(f"選手DB: {len(db):,}名")
    venue_models = ", ".join(f"{k}({v['trained_races']}R)" for k, v in engine.VENUE_WEIGHTS.items()) or "全場共通のみ"
    st.caption(f"場別モデル: {venue_models}")

st.title("🚤 競艇予測 v17.4")
st.caption(f"Gemini 画像読み取り（高速リサイズ＆自動切替対応）｜既定: {active_model}")

tab_predict, tab_logs, tab_db = st.tabs(["🚤 レース予測", "📝 結果・収支", "🔍 選手・DB"])

# ==========================================
# タブ1: レース予測
# ==========================================
with tab_predict:
    with st.container(border=True):
        st.markdown("#### 📷 1. 写真・スクショから自動入力")
        input_mode = st.radio(
            "入力ソース",
            ["📁 スクショ・写真を選択（複数枚OK）", "📷 カメラで撮影", "📋 テキスト一括貼付"],
            horizontal=True,
            label_visibility="collapsed",
        )

        if input_mode == "📁 スクショ・写真を選択（複数枚OK）":
            uploaded_imgs = st.file_uploader(
                "出走表・直前情報・展示タイムのスクショを選択してください（送信前に自動圧縮します）",
                type=["png", "jpg", "jpeg", "webp"],
                accept_multiple_files=True,
            )
            if uploaded_imgs:
                st.caption(f"{len(uploaded_imgs)}枚の画像を選択中")
                if st.button("✨ Gemini で画像を読み取る", type="primary", use_container_width=True):
                    if not api_key_input:
                        st.error("左上のサイドバー（＞）または Streamlit Secrets に `GEMINI_API_KEY` を設定してください。")
                    else:
                        with st.spinner(f"{active_model} で画像を最適化＆解析中..."):
                            try:
                                extracted, used_model, elapsed, size_info = extract_race_from_images(
                                    uploaded_imgs,
                                    api_key_input,
                                    active_model,
                                    auto_fallback=auto_fallback,
                                )
                                meta_msg = f"使用モデル: {used_model} ｜ 処理時間: {elapsed:.1f}秒 ｜ 画像最適化: {size_info}"
                                apply_extracted_result_to_state(extracted, db, meta_msg=meta_msg)
                                st.success("読み取り完了！下のフォームで数値を確認して予測ボタンを押してください。")
                                st.rerun()
                            except Exception as e:
                                st.error(explain_api_error(e))

        elif input_mode == "📷 カメラで撮影":
            cam_img = st.camera_input("出走表・直前情報を撮影")
            if cam_img is not None:
                if st.button("✨ 撮影した写真を読み取る", type="primary", use_container_width=True):
                    if not api_key_input:
                        st.error("左上のサイドバー（＞）または Streamlit Secrets に `GEMINI_API_KEY` を設定してください。")
                    else:
                        with st.spinner(f"{active_model} で写真を最適化＆解析中..."):
                            try:
                                extracted, used_model, elapsed, size_info = extract_race_from_images(
                                    [cam_img],
                                    api_key_input,
                                    active_model,
                                    auto_fallback=auto_fallback,
                                )
                                meta_msg = f"使用モデル: {used_model} ｜ 処理時間: {elapsed:.1f}秒 ｜ 画像最適化: {size_info}"
                                apply_extracted_result_to_state(extracted, db, meta_msg=meta_msg)
                                st.success("読み取り完了！下のフォームを確認してください。")
                                st.rerun()
                            except Exception as e:
                                st.error(explain_api_error(e))

        else:
            st.caption("形式: `[登番/選手名] [モーター] [ST] [チルト] [直線] [一周] [まわり足] (任意: c進入 F/F0 L当地勝率,当地2連率)` × 6行")
            raw_lines = st.text_area("6艇分のテキストを貼り付け", height=160, placeholder="4320 42.5 .12 0.0 6.72 36.85 5.42\n...")
            if st.button("📥 テキストをフォームに反映", use_container_width=True):
                try:
                    parsed_rows = parse_quick_text_lines(raw_lines, st.session_state["stadium"], db)
                    st.session_state["boats_rows"] = parsed_rows
                    st.session_state["ai_notes"] = ""
                    st.session_state["ai_meta_msg"] = "テキスト一括貼付から反映"
                    st.success("6艇分のテキストを反映しました。")
                    st.rerun()
                except Exception as e:
                    st.error(f"テキスト解析エラー: {e}")

        if st.session_state.get("ai_meta_msg"):
            st.caption(f"✅ {st.session_state['ai_meta_msg']}")
        if st.session_state.get("ai_notes"):
            st.info(f"🤖 AI読み取りメモ: {st.session_state['ai_notes']}")

    # 2. レース条件フォーム
    with st.container(border=True):
        st.markdown("#### 🏟️ 2. レース条件の確認・修正")
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            st.session_state["stadium"] = st.selectbox(
                "会場",
                STADIUM_LIST,
                index=STADIUM_LIST.index(st.session_state["stadium"]) if st.session_state["stadium"] in STADIUM_LIST else 1,
            )
        with col2:
            st.session_state["race_no"] = st.number_input("R番号", min_value=1, max_value=12, value=int(st.session_state["race_no"]), step=1)
        with col3:
            st.session_state["race_date"] = st.text_input("開催日(YYYYMMDD)", value=st.session_state["race_date"])

        col4, col5, col6 = st.columns([1.2, 1.5, 1.3])
        with col4:
            st.session_state["wind_speed"] = st.number_input("風速(m)", min_value=0.0, max_value=20.0, value=float(st.session_state["wind_speed"]), step=1.0)
        with col5:
            wdir_labels = {"normal": "横風/無風/不明", "tail": "追い風", "head": "向かい風"}
            wdir_keys = list(wdir_labels.keys())
            cur_idx = wdir_keys.index(st.session_state["wind_dir"]) if st.session_state["wind_dir"] in wdir_keys else 0
            st.session_state["wind_dir"] = st.selectbox("風向", wdir_keys, index=cur_idx, format_func=lambda k: wdir_labels[k])
        with col6:
            st.session_state["budget"] = st.number_input("本線予算(円)", min_value=400, max_value=50000, value=int(st.session_state["budget"]), step=100)

        t1, t2, t3 = st.columns(3)
        with t1:
            st.session_state["is_womens_race"] = st.toggle("女子限定戦", value=bool(st.session_state["is_womens_race"]), help="オールレディース・ヴィーナス等のみON（男女混合はOFF）")
        with t2:
            st.session_state["use_real_odds"] = st.toggle("実オッズ自動取得", value=bool(st.session_state["use_real_odds"]), help="ランクC+以下の穴予想で公式3連単オッズを取得")
        with t3:
            st.session_state["rel_filter_enabled"] = st.toggle("信頼度70点フィルタ", value=bool(st.session_state["rel_filter_enabled"]), help="予測信頼度70点(高S)未満を自動見送り")

    # 3. 6艇データ確認・編集
    with st.container(border=True):
        st.markdown("#### ⛵ 3. 6艇データ（タップして微修正可能）")
        edit_df = pd.DataFrame(st.session_state["boats_rows"])
        edited_df = st.data_editor(
            edit_df,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_config={
                "枠": st.column_config.NumberColumn("枠", disabled=True, width="small"),
                "登番/選手名": st.column_config.TextColumn("登番/選手名", width="medium"),
                "モーター2連率": st.column_config.NumberColumn("モーター%", format="%.1f", width="small"),
                "展示ST": st.column_config.TextColumn("展示ST", width="small"),
                "チルト": st.column_config.NumberColumn("チルト", format="%.1f", step=0.5, width="small"),
                "直線": st.column_config.NumberColumn("直線", format="%.2f", width="small"),
                "一周": st.column_config.NumberColumn("一周", format="%.2f", width="small"),
                "まわり足": st.column_config.NumberColumn("まわり足", format="%.2f", width="small"),
                "進入": st.column_config.NumberColumn("進入", min_value=1, max_value=6, step=1, width="small"),
                "F持ち": st.column_config.CheckboxColumn("F", width="small"),
                "当地勝率": st.column_config.NumberColumn("当地勝率", format="%.2f", width="small"),
                "当地2連率": st.column_config.NumberColumn("当地2連%", format="%.1f", width="small"),
            },
            key="boats_editor",
        )
        st.session_state["boats_rows"] = edited_df.to_dict("records")

        preview_chips = []
        for r in st.session_state["boats_rows"]:
            key = str(r.get("登番/選手名") or "").strip()
            info = engine.find_racer_in_db(db, key) if key else None
            if info:
                preview_chips.append(f"{boat_badge_html(r['枠'])} {info['name']} ({info['class_type']} / {info['win_nat']:.2f})")
            else:
                preview_chips.append(f"{boat_badge_html(r['枠'])} <span style='color:#DC2626;'>未照合({key or '空欄'})</span>")
        st.markdown("<div style='font-size:13px;line-height:2.0;'>" + " &nbsp; ".join(preview_chips) + "</div>", unsafe_allow_html=True)

        btn_c1, btn_c2 = st.columns([3, 1])
        with btn_c1:
            run_clicked = st.button("🚀 予測を実行する", type="primary", use_container_width=True)
        with btn_c2:
            if st.button("🔄 リセット", use_container_width=True):
                st.session_state["boats_rows"] = get_default_boats_rows()
                st.session_state["ai_notes"] = ""
                st.session_state["ai_meta_msg"] = ""
                st.session_state["last_prediction"] = None
                st.rerun()

    if run_clicked:
        try:
            race_df, warns = build_race_dataframe(st.session_state["boats_rows"], st.session_state["stadium"], db)
            for w_msg in warns:
                st.warning(w_msg)
            race_id = f"{st.session_state['race_date']}_{st.session_state['stadium']}_{int(st.session_state['race_no']):02d}R"
            with st.spinner("予測計算＆オッズ確認中..."):
                pred = run_web_prediction(
                    race_df=race_df,
                    stadium=st.session_state["stadium"],
                    race_id=race_id,
                    budget=int(st.session_state["budget"]),
                    is_womens_race=bool(st.session_state["is_womens_race"]),
                    wind_speed=float(st.session_state["wind_speed"]),
                    wind_dir=st.session_state["wind_dir"],
                    use_real_odds=bool(st.session_state["use_real_odds"]),
                    rel_filter_enabled=bool(st.session_state["rel_filter_enabled"]),
                    db=db,
                    save_log=True,
                )
            st.session_state["last_prediction"] = pred
        except Exception as e:
            st.error(f"予測実行エラー: {e}")

    if st.session_state.get("last_prediction"):
        st.divider()
        render_prediction_result(st.session_state["last_prediction"])


# ==========================================
# タブ2: 結果入力・収支ログ
# ==========================================
with tab_logs:
    logs = engine.load_prediction_logs()
    settled = [r for r in logs if r.get("result") is not None]
    main_settled = [r for r in settled if r.get("conf_rank") in engine.MAIN_TARGET_RANKS and r.get("main_bets")]
    ana_settled = [r for r in settled if r.get("conf_rank") in engine.ANA_TARGET_RANKS and r.get("bets")]

    st.subheader("📈 収支サマリー")
    m1, m2, m3 = st.columns(3)
    m1.metric("全予測ログ", f"{len(logs)} R", f"確定 {len(settled)} R")
    if main_settled:
        mm = engine._calc_subset_metrics(main_settled, bets_key="main_bets")
        m2.metric("本線 ROI (A/B)", f"{mm['roi']:.1f}%", f"{mm['profit']:+,.0f}円 ({mm['hits']}/{mm['n']})")
    else:
        m2.metric("本線 ROI (A/B)", "-")
    if ana_settled:
        am = engine._calc_subset_metrics(ana_settled, bets_key="bets")
        m3.metric("穴予想 ROI (C+〜)", f"{am['roi']:.1f}%", f"{am['profit']:+,.0f}円 ({am['hits']}/{am['n']})")
    else:
        m3.metric("穴予想 ROI (C+〜)", "-")

    st.divider()
    st.subheader("📝 レース結果の登録")
    if logs:
        recent_logs = list(reversed(logs[-20:]))
        options = {
            f"{r['race_id']} (ランク{r['conf_rank']} / {'確定:' + r['result']['comb'] if r.get('result') else '未確定'})": r["race_id"]
            for r in recent_logs
        }
        sel_label = st.selectbox("対象レースを選択", list(options.keys()))
        sel_id = options[sel_label]

        rc1, rc2 = st.columns(2)
        with rc1:
            res_comb = st.text_input("確定3連単 (例: 1-2-3)", value="")
        with rc2:
            res_odds = st.number_input("確定オッズ倍率 (例: 12.5)", min_value=1.0, max_value=2000.0, value=10.0, step=0.1)

        if st.button("💾 結果を保存する", type="primary", use_container_width=True):
            if not re.match(r"^[1-6]-[1-6]-[1-6]$", res_comb.strip()):
                st.error("3連単は `1-2-3` の形式で入力してください。")
            else:
                for i, rec in enumerate(logs):
                    if rec.get("race_id") == sel_id:
                        logs[i]["result"] = {
                            "comb": res_comb.strip(),
                            "odds": float(res_odds),
                            "payout_per_100": int(round(float(res_odds) * 100)),
                            "winner_boat": int(res_comb.strip()[0]),
                        }
                        break
                engine.write_prediction_logs(logs)
                st.success(f"{sel_id} の結果（{res_comb.strip()} / {res_odds:.1f}倍）を保存しました。")
                st.rerun()

        st.divider()
        jsonl_bytes = "\n".join(json.dumps(rec, ensure_ascii=False) for rec in logs).encode("utf-8")
        st.download_button(
            "⬇️ prediction_log.jsonl をダウンロード（バックアップ）",
            data=jsonl_bytes,
            file_name="prediction_log.jsonl",
            mime="application/x-ndjson",
            use_container_width=True,
        )
        st.caption("※ Streamlit Community Cloud はアプリ再起動時にサーバー上の一時ファイルが初期化されるため、必要に応じてログをダウンロードしてGitHubへ反映してください。")
    else:
        st.info("まだ予測ログがありません。レース予測を実行すると自動で記録されます。")


# ==========================================
# タブ3: 選手検索・DB管理
# ==========================================
with tab_db:
    st.subheader("🔍 選手コース別成績検索")
    q = st.text_input("選手名または4桁の登録番号を入力", placeholder="例: 4320 または 峰竜太")
    if q.strip():
        r_info = engine.find_racer_in_db(db, q.strip())
        if r_info:
            f_str = "F持ち" if r_info.get("is_flying") else "Fなし"
            st.success(
                f"**{r_info['reg_no']} {r_info['name']}**（{r_info.get('branch', '-')}/{r_info['class_type']}）｜ "
                f"全国勝率: {r_info['win_nat']:.2f} / 2連率: {r_info['2rate_nat']:.1f}% / 平均ST: {r_info.get('avg_st', 0.16):.2f} ({f_str})"
            )
            c_rows = []
            for c in range(1, 7):
                cs = r_info["courses"][str(c)]
                c_rows.append(
                    {
                        "コース": f"{c}C",
                        "進入数": cs.get("entries", "-"),
                        "1着率": f"{cs['1st']:.1f}%",
                        "2着率": f"{cs['2nd']:.1f}%",
                        "3着率": f"{cs['3rd']:.1f}%",
                        "平均ST": f"{cs.get('avg_st', 0.16):.2f}",
                        "まくり/差し": f"{cs['makuri']:.0f}% / {cs['sashi']:.0f}%",
                    }
                )
            st.dataframe(pd.DataFrame(c_rows), use_container_width=True, hide_index=True)
        else:
            st.warning("該当する選手が見つかりませんでした。")

    st.divider()
    st.subheader("📂 ファン手帳 (.lzh / .txt) から選手DBを更新")
    fan_files = st.file_uploader(
        "fan2410.lzh / fan2504.lzh / fan2510.lzh / fan2604.lzh 等をまとめてアップロード",
        type=["lzh", "txt"],
        accept_multiple_files=True,
    )
    if fan_files and st.button("🔨 選手DBを再構築する", use_container_width=True):
        import tempfile

        with st.spinner("ファン手帳データを解析中..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                paths = []
                for ff in fan_files:
                    p = os.path.join(tmpdir, ff.name)
                    with open(p, "wb") as out_f:
                        out_f.write(ff.getvalue())
                    paths.append(p)
                new_db = engine.build_db_from_fan_files(explicit_files=paths)
                if new_db:
                    reload_engine_resources()
                    st.success(f"DB更新完了！（全 {len(new_db):,} 名）")
                    st.rerun()
                else:
                    st.error("有効な選手レコードを読み込めませんでした。")