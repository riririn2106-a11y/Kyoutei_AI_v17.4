import os
import re
import sys
import csv
import glob
import json
import math
import struct
import datetime
import itertools
import urllib.request
import urllib.error
import pandas as pd

# スクリプト自身が置かれているフォルダ（OneDrive\Desktop\競艇ファイル など）を基準にする
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
DB_FILE = os.path.join(SCRIPT_DIR, "racer_db.json")
PARAMS_FILE = os.path.join(SCRIPT_DIR, "model_params.json")
LOG_FILE = os.path.join(SCRIPT_DIR, "prediction_log.jsonl")

# ランク別の買い目戦略区分
MAIN_TARGET_RANKS = {"A", "B"}               # 本線予想の対象ランク
ANA_TARGET_RANKS = {"C+", "C", "C-", "D", "E"}
ANA_STRICT_RACE_FILTER = True   # True: 穴予想は1枠B級かつ全国勝率5.0未満のみ（それ以外のC帯は見送り）
ANA_MAX_BETS = 4                # 穴予想の最大購入点数（条件未達時は無理に埋めず見送り）
ANA_VIRTUAL_HEADS = (2, 3, 4)   # 仮想オッズフォールバック時の対象1着艇番
ANA_VIRTUAL_MIN_PROB = 0.010    # 仮想オッズフォールバック時の最低予測確率 (1.0%)
ANA_VIRTUAL_MIN_ODDS = 15.0     # 仮想オッズフォールバック時の最低概算オッズ (上限なし)
ANA_VIRTUAL_MIN_MYOUMI = 1.10   # 仮想オッズフォールバック時の最低妙味度

# 予測信頼度フィルタ設定（案E: 明確さ30 / データ10 / 安定性60）
RELIABILITY_FILTER_ENABLED = True   # True: 予測信頼度が基準未満のレースは自動見送り（メニュー[9]でON/OFF切替可）
RELIABILITY_MIN_SCORE = 70.0        # 購入推奨の最低予測信頼度スコア（70.0 = 高(S)以上）

# --- v17: 追い風3区分（2m未満 / 2〜3m / 4m以上）× 女子限定戦（2コース重視）設定 ---
# 直近1年間（2025/12〜2026/11・戸田全37節・2,235R）および長期統計検証に基づく設定
TAIL_MID_WIND_THRESHOLD = 2.0   # 追い風 移行・警戒帯しきい値 (2〜3m)
STRONG_WIND_THRESHOLD = 4.0     # 強風しきい値 (4m以上：1号艇1着率が統計的有意に低下 p=0.015)
SEVERE_WIND_THRESHOLD = 6.0     # 爆風・安定板/2周戦警戒しきい値 (6m以上)

DEFAULT_V17_CONDITION_PARAMS = {
    "tail_mid_in_mult": 0.96,        # 追い風2〜3m: 1C小幅低下
    "tail_mid_c2_boost": 1.06,       # 追い風2〜3m: 2C差し微増
    "tail_mid_c4_boost": 1.05,       # 追い風2〜3m: 4C攻め微増
    "tail_strong_in_mult": 0.90,     # 追い風4m+: 1C有意低下
    "tail_strong_c2_boost": 1.18,    # 追い風4m+: 2C差し急増 (19〜20.5%)
    "tail_strong_c4_boost": 1.12,    # 追い風4m+: 4C差し・まくり増 (17.7〜18.4%)
    "head_strong_in_mult": 0.97,     # 向かい風4m+: 1Cは追い風ほど落ちない
    "head_strong_c2_boost": 0.88,    # 向かい風4m+: 2C差し激減
    "head_strong_c3_boost": 1.12,    # 向かい風4m+: 3Cまくり増
    "head_strong_c4_boost": 1.15,    # 向かい風4m+: 4Cまくり増
    "head_strong_makuri_mult": 1.18,
    "other_strong_in_mult": 0.92,    # 横風・不明4m+
    "other_strong_c2_boost": 1.05,
    "other_strong_c3_boost": 1.06,
    "other_strong_makuri_mult": 1.08,
    "severe_wind_in_mult": 0.94,     # 6m+・安定板/2周戦警戒
    "womens_in_mult": 0.96,          # 女子限定戦: 過度な1C減衰を回避（2年360R=42.2%）
    "womens_c2_boost": 1.15,         # 女子限定戦: 2C1着率突出（21.1%〜25.0%）を反映
    "womens_makuri_mult": 1.04,
}
V17_CONDITION_PARAMS = dict(DEFAULT_V17_CONDITION_PARAMS)


def get_condition_profile(is_womens_race=False, wind_speed=0.0, wind_dir="normal"):
    """
    v17: 追い風3区分（2m未満 / 2〜3m / 4m以上）・向かい風別系統・女子限定戦（2C重視）の
    組み合わせに応じた予測補正と穴狙いパラメータを返す。
    ※男女混合戦は1年間実測で1C 42.8%と男子戦より堅調なため is_womens_race=False とすること。
    wind_dir: "normal" (無風/横風/不明), "tail" (追い風), "head" (向かい風)
    """
    cp = V17_CONDITION_PARAMS
    ws = float(wind_speed or 0.0)
    wdir = str(wind_dir or "normal").lower()
    is_mid_tail = (wdir == "tail") and (TAIL_MID_WIND_THRESHOLD <= ws < STRONG_WIND_THRESHOLD)
    is_strong_wind = ws >= STRONG_WIND_THRESHOLD
    is_severe_wind = ws >= SEVERE_WIND_THRESHOLD

    in_mult = 1.00
    c2_boost = 1.00
    c3_boost = 1.00
    c4_boost = 1.00
    makuri_mult = 1.00
    labels = []

    if is_womens_race:
        in_mult *= cp["womens_in_mult"]
        c2_boost *= cp["womens_c2_boost"]
        makuri_mult *= cp["womens_makuri_mult"]
        labels.append("女子限定戦(2C重視)")

    if is_mid_tail:
        in_mult *= cp["tail_mid_in_mult"]
        c2_boost *= cp["tail_mid_c2_boost"]
        c4_boost *= cp["tail_mid_c4_boost"]
        labels.append(f"追い風{ws:.0f}m(2-3m警戒)")
    elif is_strong_wind:
        if wdir == "tail":
            in_mult *= cp["tail_strong_in_mult"]
            c2_boost *= cp["tail_strong_c2_boost"]
            c4_boost *= cp["tail_strong_c4_boost"]
            labels.append(f"追い風{ws:.0f}m(4m+差し急増)")
        elif wdir == "head":
            in_mult *= cp["head_strong_in_mult"]
            c2_boost *= cp["head_strong_c2_boost"]
            c3_boost *= cp["head_strong_c3_boost"]
            c4_boost *= cp["head_strong_c4_boost"]
            makuri_mult *= cp["head_strong_makuri_mult"]
            labels.append(f"向かい風{ws:.0f}m(まくり警戒)")
        else:
            in_mult *= cp["other_strong_in_mult"]
            c2_boost *= cp["other_strong_c2_boost"]
            c3_boost *= cp["other_strong_c3_boost"]
            makuri_mult *= cp["other_strong_makuri_mult"]
            labels.append(f"強風{ws:.0f}m")
    elif wdir == "tail" and ws > 0:
        labels.append(f"追い風{ws:.0f}m(2m未満)")

    if is_severe_wind:
        in_mult *= cp["severe_wind_in_mult"]
        labels.append("爆風/安定板警戒")

    in_mult = max(0.78, round(in_mult, 4))
    c2_boost = round(c2_boost, 4)
    c3_boost = round(c3_boost, 4)
    c4_boost = round(c4_boost, 4)
    makuri_mult = round(makuri_mult, 4)

    # 仮想オッズ時の対象頭・オッズ上限・最大点数
    if is_strong_wind and wdir == "head":
        virtual_heads = (3, 4, 5, 2)
    elif (is_strong_wind and wdir == "tail") or is_womens_race:
        virtual_heads = (2, 3, 4)
    else:
        virtual_heads = ANA_VIRTUAL_HEADS

    virtual_max_odds = 80.0 if (is_womens_race or is_strong_wind) else None
    # 爆風6m+・安定板/2周戦時は再現性低下に備え最大3点へ絞り込み（資金防衛）
    max_bets = 3 if is_severe_wind else ANA_MAX_BETS
    relax_race_filter = is_strong_wind  # 4m以上強風時は1枠勝率5.5未満・C-以下へ緩和

    return {
        "label": "・".join(labels) if labels else "通常一般戦",
        "is_womens_race": bool(is_womens_race),
        "wind_speed": ws,
        "wind_dir": wdir,
        "is_mid_tail": is_mid_tail,
        "is_strong_wind": is_strong_wind,
        "is_severe_wind": is_severe_wind,
        "in_mult": in_mult,
        "c2_boost": c2_boost,
        "c3_boost": c3_boost,
        "c4_boost": c4_boost,
        "makuri_mult": makuri_mult,
        "virtual_heads": virtual_heads,
        "virtual_max_odds": virtual_max_odds,
        "max_bets": max_bets,
        "relax_race_filter": relax_race_filter,
    }
MAIN_RANK_ORDER = ["A", "B"]
ANA_RANK_ORDER = ["C+", "C", "C-", "D", "E"]

# ==========================================
# 1. マスタデータ・場番号変換設定
# ==========================================
STADIUM_PROFILES = {
    "大村": {"in_base": 65.0, "local_weight": 1.0}, "徳山": {"in_base": 55.0, "local_weight": 1.0},
    "芦屋": {"in_base": 61.0, "local_weight": 1.0}, "下関": {"in_base": 60.0, "local_weight": 1.1},
    "若松": {"in_base": 58.0, "local_weight": 1.1}, "宮島": {"in_base": 57.0, "local_weight": 1.2},
    "丸亀": {"in_base": 56.0, "local_weight": 1.1}, "児島": {"in_base": 56.0, "local_weight": 1.1},
    "多摩川": {"in_base": 55.0, "local_weight": 1.0}, "尼崎": {"in_base": 55.0, "local_weight": 1.0},
    "津": {"in_base": 54.0, "local_weight": 1.1}, "住之江": {"in_base": 54.0, "local_weight": 1.1},
    "蒲郡": {"in_base": 53.0, "local_weight": 1.1}, "常滑": {"in_base": 53.0, "local_weight": 1.1},
    "三国": {"in_base": 53.0, "local_weight": 1.2}, "びわこ": {"in_base": 51.0, "local_weight": 1.5},
    "浜名湖": {"in_base": 51.0, "local_weight": 1.2}, "桐生": {"in_base": 50.0, "local_weight": 1.3},
    "鳴門": {"in_base": 48.0, "local_weight": 1.5}, "唐津": {"in_base": 51.0, "local_weight": 1.1},
    "福岡": {"in_base": 45.0, "local_weight": 2.5}, "平和島": {"in_base": 44.0, "local_weight": 2.0},
    "戸田": {"in_base": 43.0, "local_weight": 2.2}, "江戸川": {"in_base": 42.0, "local_weight": 3.0},
}

STADIUM_CODE_MAP = {
    1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖",
    7: "蒲郡", 8: "常滑", 9: "津", 10: "三国", 11: "びわこ", 12: "住之江",
    13: "尼崎", 14: "鳴門", 15: "丸亀", 16: "児島", 17: "宮島", 18: "徳山",
    19: "下関", 20: "若松", 21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村"
}

STADIUM_NAME_TO_CODE = {v: k for k, v in STADIUM_CODE_MAP.items()}


def normalize_stadium_name(raw_input):
    """場番号(2, 02, 02#など)や漢字名を正規の会場名に変換"""
    s = str(raw_input).strip().replace("#", "").replace("＃", "")
    s = s.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    if s.isdigit():
        return STADIUM_CODE_MAP.get(int(s), s)
    return s


def parse_st_value(st_str):
    """'.14' や 'F.11' などのST表記も自動で数値(0.14, -0.11)に変換"""
    s = str(st_str).strip().upper()
    if s.startswith("F"):
        val_str = s[1:]
        if val_str.startswith("."):
            val_str = "0" + val_str
        return -float(val_str)
    elif s.startswith("."):
        return float("0" + s)
    return float(s)


FRAME_BASE_PROB = {
    1: {"win_1st": 0.556, "win_2nd": 0.180, "win_3rd": 0.100},
    2: {"win_1st": 0.148, "win_2nd": 0.250, "win_3rd": 0.185},
    3: {"win_1st": 0.132, "win_2nd": 0.200, "win_3rd": 0.205},
    4: {"win_1st": 0.090, "win_2nd": 0.170, "win_3rd": 0.170},
    5: {"win_1st": 0.060, "win_2nd": 0.120, "win_3rd": 0.175},
    6: {"win_1st": 0.018, "win_2nd": 0.080, "win_3rd": 0.155},
}

DEVELOPMENT_BIAS = {
    1: {2: 1.2, 3: 1.1, 4: 0.9, 5: 0.8, 6: 0.7},
    2: {1: 1.3, 3: 1.0, 4: 1.0, 5: 0.8, 6: 0.7},
    3: {1: 1.1, 2: 1.1, 4: 1.1, 5: 0.9, 6: 0.7},
    4: {1: 0.7, 2: 0.6, 3: 0.7, 5: 1.4, 6: 1.3},
    5: {1: 0.8, 2: 0.7, 4: 1.1, 6: 1.2},
    6: {1: 0.9, 2: 0.8, 4: 1.0, 5: 1.1},
}

DEFAULT_WEIGHTS = {
    "win_rate_national": 0.4835,
    "rate_national": -0.9146,
    "win_rate_local": 0.2205,
    "rate_local": 1.8718,
    "motor": 0.5000,
    "st_relative": 1.1513,
    "raw_win_rate_add": 4.3199,
    "ex_score_weight": 1.0000,      # 展示評価への重み（自動修正対象）
    "makuri_attack_weight": 1.0000, # まくり・展開攻め評価への重み（自動修正対象）
    "course_1st_weight": 0.65,
    "course_2nd_weight": 0.50,
    "course_3rd_weight": 0.40,
    "f_1st_coef": 0.0110,           # 1着スコア係数（V15.8較正値）
    "in_prob_bias": 1.0000,         # イン1着確率の全体倍率補正
    "ana_min_myoumi": 1.20,         # 穴狙いモード（仮想オッズ時）の最低妙味度しきい値
    "ana_min_real_ev": 1.00,        # 穴狙いモード（実オッズ時）の最低実EVしきい値 (予測確率×実オッズ)
    "ana_max_1st_prob": 0.35,       # 穴狙いヘッド候補の最大1着率上限
}

WEIGHTS = DEFAULT_WEIGHTS.copy()
VENUE_WEIGHTS = {}  # {"戸田": {"trained_races": 200, "weights": {...}}, ...}
MIN_VENUE_ACTIVE_RACES = 20  # 場別専用重みを自動適用する最低学習レース数

CLASS_RELIABILITY = {"A1": 1.15, "A2": 1.00, "B1": 0.85, "B2": 0.50}
CLASS_POPULARITY = {"A1": 1.45, "A2": 1.10, "B1": 0.80, "B2": 0.50}

DEFAULT_COURSE_STATS = {
    "1": {"1st": 65.0, "2nd": 18.0, "3rd": 8.0, "makuri": 0.0, "sashi": 0.0, "avg_st": 0.15},
    "2": {"1st": 15.0, "2nd": 28.0, "3rd": 20.0, "makuri": 25.0, "sashi": 70.0, "avg_st": 0.16},
    "3": {"1st": 12.0, "2nd": 22.0, "3rd": 22.0, "makuri": 55.0, "sashi": 40.0, "avg_st": 0.16},
    "4": {"1st": 10.0, "2nd": 18.0, "3rd": 20.0, "makuri": 60.0, "sashi": 35.0, "avg_st": 0.17},
    "5": {"1st": 6.0,  "2nd": 12.0, "3rd": 18.0, "makuri": 40.0, "sashi": 55.0, "avg_st": 0.17},
    "6": {"1st": 2.0,  "2nd": 6.0,  "3rd": 12.0, "makuri": 30.0, "sashi": 60.0, "avg_st": 0.18},
}


# ==========================================
# 1.5 パラメータ永続化（共通＋場別専用モデルの自動切替）
# ==========================================
def _merge_with_default(raw_w):
    merged = DEFAULT_WEIGHTS.copy()
    if isinstance(raw_w, dict):
        for k, v in raw_w.items():
            if k in merged:
                merged[k] = float(v)
    return merged


def load_model_params():
    global WEIGHTS, VENUE_WEIGHTS
    VENUE_WEIGHTS = {}
    if os.path.exists(PARAMS_FILE):
        try:
            with open(PARAMS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            WEIGHTS = _merge_with_default(saved.get("weights", {}))
            raw_vw = saved.get("venue_weights", {})
            if isinstance(raw_vw, dict):
                for venue, v_entry in raw_vw.items():
                    if isinstance(v_entry, dict) and "weights" in v_entry:
                        VENUE_WEIGHTS[venue] = {
                            "trained_races": int(v_entry.get("trained_races", 0)),
                            "weights": _merge_with_default(v_entry["weights"]),
                        }
            return saved
        except Exception:
            pass
    WEIGHTS = DEFAULT_WEIGHTS.copy()
    return {
        "version": "V17-Wind3Tier-Women2C",
        "updated_at": None,
        "trained_races": 0,
        "weights": WEIGHTS.copy(),
        "venue_weights": {},
        "history": [],
    }


def save_model_params(meta_info):
    payload = {
        "version": "V17-Wind3Tier-Women2C",
        "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trained_races": meta_info.get("trained_races", 0),
        "weights": WEIGHTS.copy(),
        "venue_weights": VENUE_WEIGHTS,
        "history": meta_info.get("history", [])[-20:],
    }
    with open(PARAMS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def get_active_weights(stadium_name=None, min_venue_races=MIN_VENUE_ACTIVE_RACES):
    """
    指定会場の学習済み専用重みがあれば自動選択し、未達なら共通重みにフォールバックする。
    """
    if stadium_name:
        v_name = normalize_stadium_name(stadium_name)
        v_entry = VENUE_WEIGHTS.get(v_name)
        if v_entry and v_entry.get("trained_races", 0) >= min_venue_races:
            return v_entry["weights"], f"{v_name}専用 ({v_entry['trained_races']}R学習済)"
    return WEIGHTS, "全場共通モデル"


# ==========================================
# 2. 純Python製 LZH (-lh5-) 自動解凍エンジン（外部ツール不要）
# ==========================================
class PurePythonLH5Decoder:
    def __init__(self, data):
        self.data = data
        self.pos = 0
        self.bit_buf = 0
        self.bit_cnt = 0

    def get_bits(self, n):
        while self.bit_cnt < n:
            b = self.data[self.pos] if self.pos < len(self.data) else 0
            self.pos += 1
            self.bit_buf = (self.bit_buf << 8) | b
            self.bit_cnt += 8
        self.bit_cnt -= n
        val = (self.bit_buf >> self.bit_cnt) & ((1 << n) - 1)
        self.bit_buf &= (1 << self.bit_cnt) - 1
        return val

    def make_tree(self, nchar, bitlen, tablebits):
        table = [0] * (1 << tablebits)
        left = [0] * (2 * nchar)
        right = [0] * (2 * nchar)
        count = [0] * 17
        for length in bitlen:
            if 0 < length <= 16:
                count[length] += 1
        start = [0] * 18
        for i in range(1, 17):
            start[i + 1] = (start[i] + count[i]) << 1
        avail = nchar
        for ch in range(nchar):
            length = bitlen[ch]
            if length == 0:
                continue
            code = start[length]
            start[length] += 1
            if length <= tablebits:
                shift = tablebits - length
                base = code << shift
                for j in range(1 << shift):
                    if base + j < len(table):
                        table[base + j] = ch
            else:
                k = code >> (length - tablebits)
                p = table
                idx = k
                for i in range(length - tablebits, 0, -1):
                    if p[idx] == 0:
                        left[avail] = right[avail] = 0
                        p[idx] = avail
                        avail += 1
                    node = p[idx]
                    if (code >> (i - 1)) & 1:
                        p, idx = right, node
                    else:
                        p, idx = left, node
                p[idx] = ch
        return table, left, right, tablebits

    def decode_symbol(self, tree):
        table, left, right, tablebits, const_sym = tree
        if const_sym is not None:
            return const_sym
        while self.bit_cnt < 16:
            b = self.data[self.pos] if self.pos < len(self.data) else 0
            self.pos += 1
            self.bit_buf = (self.bit_buf << 8) | b
            self.bit_cnt += 8
        peek = (self.bit_buf >> (self.bit_cnt - tablebits)) & ((1 << tablebits) - 1)
        j = table[peek]
        if j < len(left) // 2:
            return j
        mask = 1 << (self.bit_cnt - tablebits - 1)
        while j >= len(left) // 2:
            j = right[j] if (self.bit_buf & mask) else left[j]
            mask >>= 1
        return j

    def read_pt_len(self, nn, nbit, i_special):
        n = self.get_bits(nbit)
        if n == 0:
            c = self.get_bits(nbit)
            return [0] * nn, (None, None, None, 0, c)
        pt_len = [0] * nn
        i = 0
        while i < n and i < nn:
            c = self.get_bits(3)
            if c == 7:
                while self.get_bits(1):
                    c += 1
            pt_len[i] = c
            i += 1
            if i == i_special:
                c = self.get_bits(2)
                for _ in range(c):
                    if i < nn:
                        pt_len[i] = 0
                        i += 1
        t, l, r, tb = self.make_tree(nn, pt_len, 8)
        return pt_len, (t, l, r, tb, None)

    def read_c_len(self, pt_len, pt_tree):
        NC = 510
        n = self.get_bits(9)
        if n == 0:
            c = self.get_bits(9)
            return [0] * NC, (None, None, None, 0, c)
        c_len = [0] * NC
        i = 0
        while i < n and i < NC:
            c = self.decode_symbol(pt_tree)
            if pt_tree[4] is None:
                l = pt_len[c]
                self.bit_cnt -= l
                self.bit_buf &= (1 << self.bit_cnt) - 1
            if c <= 2:
                if c == 0:
                    count = 1
                elif c == 1:
                    count = self.get_bits(4) + 3
                else:
                    count = self.get_bits(9) + 20
                for _ in range(count):
                    if i < NC:
                        c_len[i] = 0
                        i += 1
            else:
                c_len[i] = c - 2
                i += 1
        t, l, r, tb = self.make_tree(NC, c_len, 12)
        return c_len, (t, l, r, tb, None)

    def decompress(self, orig_size):
        out = bytearray()
        ring = bytearray(b" " * 16384)
        r_idx = 0
        block_rem = 0
        c_len = pt_len = c_tree = pt_tree = None

        while len(out) < orig_size:
            if block_rem == 0:
                block_rem = self.get_bits(16)
                pt_len, pt_tree = self.read_pt_len(19, 5, 3)
                c_len, c_tree = self.read_c_len(pt_len, pt_tree)
                pt_len, pt_tree = self.read_pt_len(14, 4, -1)
            block_rem -= 1
            c = self.decode_symbol(c_tree)
            if c_tree[4] is None:
                l = c_len[c]
                self.bit_cnt -= l
                self.bit_buf &= (1 << self.bit_cnt) - 1
            if c < 256:
                out.append(c)
                ring[r_idx] = c
                r_idx = (r_idx + 1) & 16383
            else:
                match_len = c - 253
                p = self.decode_symbol(pt_tree)
                if pt_tree[4] is None:
                    l = pt_len[p]
                    self.bit_cnt -= l
                    self.bit_buf &= (1 << self.bit_cnt) - 1
                if p == 0:
                    dist = 0
                elif p == 1:
                    dist = 1
                else:
                    dist = (1 << (p - 1)) + self.get_bits(p - 1)
                src = (r_idx - dist - 1) & 16383
                for _ in range(match_len):
                    b = ring[src]
                    out.append(b)
                    ring[r_idx] = b
                    r_idx = (r_idx + 1) & 16383
                    src = (src + 1) & 16383
                    if len(out) >= orig_size:
                        break
        return bytes(out)


def extract_lzh_pure_python(filepath):
    with open(filepath, "rb") as f:
        raw = f.read()
    idx = raw.find(b"-lh")
    if idx < 2:
        return raw.splitlines()
    start = idx - 2
    method = raw[start + 2:start + 7].decode("ascii", errors="ignore")
    comp_size, orig_size = struct.unpack("<II", raw[start + 7:start + 15])
    level = raw[start + 20]
    if level == 0:
        hdr_size = raw[start]
        data_offset = start + 2 + hdr_size
    elif level == 1:
        base_hdr_size = raw[start]
        name_len = raw[start + 21]
        pos = start + 22 + name_len + 3
        while pos + 2 <= len(raw):
            next_ext = struct.unpack("<H", raw[pos:pos + 2])[0]
            pos += 2
            if next_ext == 0:
                break
            pos += next_ext - 2
        data_offset = start + 2 + base_hdr_size
    else:
        return []

    comp_data = raw[data_offset:data_offset + comp_size]
    if method == "-lh0-":
        return comp_data.splitlines()
    elif method == "-lh5-":
        decoder = PurePythonLH5Decoder(comp_data)
        decompressed = decoder.decompress(orig_size)
        return decompressed.splitlines()
    return []


# ==========================================
# 3. ファン手帳データ検索＆DB構築モジュール
# ==========================================
def safe_int(b_slice):
    s = b_slice.decode("cp932", errors="ignore").strip()
    return int(s) if s.isdigit() else 0


def parse_fan_line(line_bytes):
    if len(line_bytes) < 402:
        return None
    try:
        reg_no = line_bytes[0:4].decode("cp932").strip()
        if not reg_no.isdigit():
            return None
        raw_name = line_bytes[4:20].decode("cp932", errors="ignore")
        clean_name = re.sub(r"[\s ]+", "", raw_name)
        branch = line_bytes[35:39].decode("cp932", errors="ignore").strip()
        class_type = line_bytes[39:41].decode("cp932", errors="ignore").strip()

        win_nat = safe_int(line_bytes[58:62]) / 100.0
        rate_nat = safe_int(line_bytes[62:66]) / 10.0
        avg_st = safe_int(line_bytes[79:82]) / 100.0

        year = safe_int(line_bytes[174:178])
        term = safe_int(line_bytes[178:179])
        period_key = year * 10 + term

        courses_raw = {}
        total_f = 0
        for c in range(1, 7):
            off1 = 82 + (c - 1) * 13
            entries = safe_int(line_bytes[off1:off1 + 3])
            c_st = safe_int(line_bytes[off1 + 7:off1 + 10]) / 100.0

            off2 = 198 + (c - 1) * 34
            cnt_1st = safe_int(line_bytes[off2:off2 + 3])
            cnt_2nd = safe_int(line_bytes[off2 + 3:off2 + 6])
            cnt_3rd = safe_int(line_bytes[off2 + 6:off2 + 9])
            cnt_f = safe_int(line_bytes[off2 + 18:off2 + 20])
            total_f += cnt_f

            courses_raw[str(c)] = {
                "entries": entries, "1st": cnt_1st, "2nd": cnt_2nd, "3rd": cnt_3rd,
                "f": cnt_f, "st": c_st
            }

        return {
            "reg_no": reg_no, "name": clean_name, "branch": branch,
            "class_type": class_type, "win_nat": win_nat, "2rate_nat": rate_nat,
            "avg_st": avg_st, "total_f": total_f, "period_key": period_key,
            "courses_raw": courses_raw
        }
    except Exception:
        return None


def find_fan_files_automatically():
    user_home = os.path.expanduser("~")
    search_dirs = [
        os.getcwd(),
        SCRIPT_DIR,
        os.path.join(user_home, "Downloads"),
        os.path.join(user_home, "Desktop"),
        os.path.join(user_home, "OneDrive", "Desktop"),
        os.path.join(user_home, "OneDrive", "デスクトップ"),
        os.path.join(user_home, "OneDrive", "Downloads"),
    ]
    found_files = set()
    for d in search_dirs:
        if not os.path.exists(d):
            continue
        for pat in ["*fan*.txt", "*fan*.lzh", "*FAN*.TXT", "*FAN*.LZH"]:
            for p in glob.glob(os.path.join(d, pat)):
                if os.path.isfile(p):
                    found_files.add(os.path.abspath(p))

    if found_files:
        return sorted(found_files)

    print("ℹ️ 自動検索でファイルが見つからなかったため、ファイル選択ウィンドウを開きます...")
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilenames(
            title="ファン手帳データ (fan2410, fan2504, fan2510, fan2604 等) を選択してください",
            filetypes=[("ファン手帳データ", "*.lzh *.txt"), ("すべてのファイル", "*.*")]
        )
        root.destroy()
        if selected:
            return sorted(selected)
    except Exception:
        pass

    print("\n👉 ファン手帳ファイルが入っているフォルダのパス（またはファイルのパス）を入力してください:")
    user_path = input("> ").strip().strip('"').strip("'")
    if os.path.isfile(user_path):
        return [os.path.abspath(user_path)]
    elif os.path.isdir(user_path):
        for pat in ["*fan*.txt", "*fan*.lzh", "*.txt", "*.lzh"]:
            for p in glob.glob(os.path.join(user_path, pat)):
                found_files.add(os.path.abspath(p))
        return sorted(found_files)
    return []


def estimate_kimarite(course_no, rate_1st, rate_2nd, c_st):
    c = int(course_no)
    base_m = int(DEFAULT_COURSE_STATS[str(c)]["makuri"])
    if c == 1:
        return 0.0, 0.0
    st_fast_bonus = max(-15.0, min(20.0, (0.16 - c_st) * 300.0)) if c_st > 0 else 0.0
    win_power_bonus = max(-10.0, min(20.0, (rate_1st - DEFAULT_COURSE_STATS[str(c)]["1st"]) * 0.8))
    makuri = round(max(5.0, min(90.0, base_m + st_fast_bonus + win_power_bonus)), 1)
    sashi = round(max(5.0, min(90.0, 95.0 - makuri)), 1)
    return makuri, sashi


def build_db_from_fan_files(explicit_files=None):
    candidates = explicit_files if explicit_files else find_fan_files_automatically()
    if not candidates:
        print("⚠️ ファン手帳ファイルが見つかりませんでした。")
        return None

    print(f"\n📂 検出されたファイル ({len(candidates)}件):")
    for p in candidates:
        print(f"  ・{p}")

    agg_racers = {}
    for fpath in candidates:
        print(f"  ⏳ {os.path.basename(fpath)} を解析中...")
        lines = extract_lzh_pure_python(fpath)
        count = 0
        for line_b in lines:
            parsed = parse_fan_line(line_b)
            if not parsed:
                continue
            count += 1
            reg = parsed["reg_no"]
            if reg not in agg_racers:
                agg_racers[reg] = {
                    "latest_period": -1,
                    "base_info": {},
                    "course_totals": {str(c): {"entries": 0, "1st": 0, "2nd": 0, "3rd": 0, "st_sum": 0.0, "st_cnt": 0} for c in range(1, 7)}
                }
            if parsed["period_key"] >= agg_racers[reg]["latest_period"]:
                agg_racers[reg]["latest_period"] = parsed["period_key"]
                agg_racers[reg]["base_info"] = parsed

            for c_str, c_data in parsed["courses_raw"].items():
                ct = agg_racers[reg]["course_totals"][c_str]
                ct["entries"] += c_data["entries"]
                ct["1st"] += c_data["1st"]
                ct["2nd"] += c_data["2nd"]
                ct["3rd"] += c_data["3rd"]
                if c_data["entries"] > 0 and c_data["st"] > 0:
                    ct["st_sum"] += c_data["st"] * c_data["entries"]
                    ct["st_cnt"] += c_data["entries"]
        print(f"  ✔ {os.path.basename(fpath)} : {count} 名の選手レコードを読込完了！")

    if not agg_racers:
        print("❌ 有効な選手レコードを読み込めませんでした。")
        return None

    db = load_racer_db() if os.path.exists(DB_FILE) else {}
    for reg, data in agg_racers.items():
        b_info = data["base_info"]
        name = b_info["name"]
        courses_final = {}

        for c in range(1, 7):
            c_str = str(c)
            ct = data["course_totals"][c_str]
            ent = ct["entries"]
            if ent >= 3:
                r1 = round((ct["1st"] / ent) * 100.0, 1)
                r2 = round((ct["2nd"] / ent) * 100.0, 1)
                r3 = round((ct["3rd"] / ent) * 100.0, 1)
                c_st = round(ct["st_sum"] / ct["st_cnt"], 2) if ct["st_cnt"] > 0 else b_info["avg_st"]
                m_rate, s_rate = estimate_kimarite(c, r1, r2, c_st)
                courses_final[c_str] = {
                    "1st": r1, "2nd": r2, "3rd": r3,
                    "makuri": m_rate, "sashi": s_rate, "avg_st": c_st, "entries": ent
                }
            else:
                courses_final[c_str] = DEFAULT_COURSE_STATS[c_str].copy()

        existing_local = db.get(name, {}).get("local", {})
        db[name] = {
            "reg_no": reg,
            "name": name,
            "branch": b_info["branch"],
            "class_type": b_info["class_type"],
            "win_nat": b_info["win_nat"],
            "2rate_nat": b_info["2rate_nat"],
            "avg_st": b_info["avg_st"],
            "is_flying": (b_info["total_f"] > 0),
            "local": existing_local,
            "courses": courses_final
        }

    save_racer_db(db)
    print(f"\n🎉 統合完了！ 全 {len(db)} 名の選手データを '{DB_FILE}' に保存しました。")
    return db


def load_racer_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_racer_db(db):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def find_racer_in_db(db, query_key):
    q = re.sub(r"[\s ]+", "", str(query_key))
    if q in db:
        return db[q]
    if q.isdigit():
        for k, v in db.items():
            if v.get("reg_no") == q:
                return v
    matches = [v for k, v in db.items() if k.startswith(q)]
    if len(matches) == 1:
        return matches[0]
    return None


# ==========================================
# 4. 当日レース入力パーサー（場番号＆ST表記ゆれ対応）
# ==========================================
def parse_daily_race_input(db):
    print("\n============================================================")
    print("🚤 当日レース直前情報入力（ファン手帳DB自動連携モード）")
    print("============================================================")
    raw_stadium = input("👉 開催会場名 または 場番号(01〜24) を入力してください (例: 戸田/02, 多摩川/05): ").strip()
    if not raw_stadium:
        return None, None, None

    stadium_name = normalize_stadium_name(raw_stadium)
    _, w_label = get_active_weights(stadium_name)
    if stadium_name not in STADIUM_PROFILES:
        print(f"⚠️ '{raw_stadium}' はマスタにないため標準水面として計算します。")
    else:
        print(f"🏟️  選択会場: 【 {stadium_name} 】 (適用モデル: {w_label})")

    race_no_raw = input("👉 レース番号(1〜12R / 過去日は '20261119 1R' 形式も可、空欄なら00R): ").strip()
    rno_tokens = re.split(r'\s+', race_no_raw)
    target_hd = datetime.datetime.now().strftime('%Y%m%d')
    rno_part = ""
    if len(rno_tokens) >= 2 and re.fullmatch(r'\d{8}', rno_tokens[0]):
        target_hd = rno_tokens[0]
        rno_part = rno_tokens[1]
    elif len(rno_tokens) == 1:
        rno_part = rno_tokens[0]
    rno_clean = rno_part.replace("R", "").replace("r", "")
    race_no = f"{int(rno_clean):02d}R" if rno_clean.isdigit() else "00R"
    race_id = f"{target_hd}_{stadium_name}_{race_no}"

    print("\n👉 1号艇〜6号艇の【当日直前データ】を入力してください。")
    print("【フォーマット（7項目）】:")
    print("  [登番/選手名] [モーター2連率] [展示ST(.14やF.11も可)] [チルト] [直線] [一周] [まわり足] (任意: [c進入] [F/F0] [L当地勝率,当地2連率])")
    print("------------------------------------------------------------")

    records = []
    boat_no = 1
    while boat_no <= 6:
        line = input(f"{boat_no}号艇 > ").strip()
        if not line:
            continue
        tokens = re.split(r'\s+', line)
        if len(tokens) < 7:
            print("⚠️ 項目数が足りません（最低7項目必要です）。再入力してください。")
            continue

        racer_key = tokens[0]
        racer_info = find_racer_in_db(db, racer_key)
        if racer_info is None:
            print(f"⚠️ 選手【{racer_key}】がDBに未登録です。新人(B2標準値: 勝率2.50)として仮処理しますか？ (Enterまたはyで続行 / nで再入力)")
            ans = input("> ").strip().lower()
            if ans in ["", "y", "yes"]:
                racer_info = {
                    "reg_no": racer_key if racer_key.isdigit() else "0000",
                    "name": f"未登録({racer_key})", "class_type": "B2",
                    "win_nat": 2.50, "2rate_nat": 8.0, "is_flying": False,
                    "local": {}, "courses": DEFAULT_COURSE_STATS
                }
            else:
                continue

        try:
            motor = float(tokens[1])
            st = parse_st_value(tokens[2])
            tilt = float(tokens[3])
            straight_t = float(tokens[4])
            turn_t = float(tokens[5])
            inspect_t = float(tokens[6])
        except ValueError as e:
            print(f"⚠️ 数値エラー: {e}。再入力してください。")
            continue

        target_course = boat_no
        is_flying = racer_info.get("is_flying", False)
        local_stats = racer_info.get("local", {}).get(stadium_name, {})
        win_loc = local_stats.get("win_loc", racer_info["win_nat"])
        rate_loc = local_stats.get("2rate_loc", racer_info["2rate_nat"])

        for opt in tokens[7:]:
            opt_low = opt.lower()
            if opt_low in ["f", "f1", "true"]:
                is_flying = True
            elif opt_low == "f0":
                is_flying = False
            elif opt_low.startswith("c") and opt_low[1:].isdigit() and 1 <= int(opt_low[1:]) <= 6:
                target_course = int(opt_low[1:])
            elif opt_low.startswith("l") and "," in opt_low[1:]:
                loc_parts = opt_low[1:].split(",")
                win_loc = float(loc_parts[0])
                rate_loc = float(loc_parts[1])

        c_stats = racer_info.get("courses", {}).get(str(target_course), DEFAULT_COURSE_STATS[str(target_course)])

        records.append({
            "boat_no": boat_no,
            "racer_name": racer_info.get("name", racer_key),
            "reg_no": racer_info.get("reg_no", ""),
            "class_type": racer_info["class_type"],
            "win_rate_national": racer_info["win_nat"],
            "2rate_national": racer_info["2rate_nat"],
            "win_rate_local": win_loc,
            "2rate_local": rate_loc,
            "motor_rate": motor,
            "exhibition_st": st,
            "tilt": tilt,
            "straight_time": straight_t,
            "turn_time": turn_t,
            "inspect_time": inspect_t,
            "target_course": target_course,
            "course_1st_rate": c_stats["1st"],
            "course_2nd_rate": c_stats["2nd"],
            "course_3rd_rate": c_stats["3rd"],
            "makuri_rate": c_stats["makuri"],
            "sashi_rate": c_stats["sashi"],
            "is_flying": is_flying
        })
        boat_no += 1

    df_res = pd.DataFrame(records)
    if df_res["target_course"].nunique() != 6:
        print("ℹ️ 進入コース指定に重複があるため、枠なり（1〜6コース）として処理します。")
        df_res["target_course"] = df_res["boat_no"]
        for idx, row in df_res.iterrows():
            c_str = str(row["target_course"])
            r_info = find_racer_in_db(db, row["racer_name"])
            if r_info:
                c_st = r_info.get("courses", {}).get(c_str, DEFAULT_COURSE_STATS[c_str])
                df_res.at[idx, "course_1st_rate"] = c_st["1st"]
                df_res.at[idx, "course_2nd_rate"] = c_st["2nd"]
                df_res.at[idx, "course_3rd_rate"] = c_st["3rd"]
                df_res.at[idx, "makuri_rate"] = c_st["makuri"]
                df_res.at[idx, "sashi_rate"] = c_st["sashi"]

    return df_res, stadium_name, race_id


def parse_split_inputs_legacy(base_text, course_text):
    """従来の2回分割入力モード（67項目＋コース別成績）パーサー"""
    base_tokens = re.split(r'\s+', base_text.strip())
    course_tokens = re.split(r'\s+', course_text.strip())
    if len(base_tokens) < 67:
        return None, None
    stadium_name = normalize_stadium_name(base_tokens[0])
    records = []
    b_idx, c_idx, boat_count = 1, 0, 1

    while b_idx < len(base_tokens) and boat_count <= 6:
        try:
            class_type = str(base_tokens[b_idx]).upper()
            win_nat = float(base_tokens[b_idx + 1])
            rate_nat = float(base_tokens[b_idx + 2])
            win_loc = float(base_tokens[b_idx + 3])
            rate_loc = float(base_tokens[b_idx + 4])
            motor = float(base_tokens[b_idx + 5])
            st = parse_st_value(base_tokens[b_idx + 6])
            tilt = float(base_tokens[b_idx + 7])
            straight_t = float(base_tokens[b_idx + 8])
            turn_t = float(base_tokens[b_idx + 9])
            inspect_t = float(base_tokens[b_idx + 10])
            b_idx += 11

            c_1st = float(course_tokens[c_idx])
            c_2nd = float(course_tokens[c_idx + 1])
            c_3rd = float(course_tokens[c_idx + 2])
            makuri = float(course_tokens[c_idx + 3])
            sashi = float(course_tokens[c_idx + 4])
            c_idx += 5

            is_flying = False
            target_course = boat_count
            for _ in range(2):
                if c_idx < len(course_tokens):
                    tok = course_tokens[c_idx].lower()
                    if tok in ['f', 'true']:
                        is_flying = True
                        c_idx += 1
                    elif tok.startswith('c') and tok[1:].isdigit() and 1 <= int(tok[1:]) <= 6:
                        target_course = int(tok[1:])
                        c_idx += 1
                    elif tok == '1' and not is_flying and (c_idx + 1 >= len(course_tokens) or not course_tokens[c_idx + 1].replace('.', '', 1).isdigit()):
                        is_flying = True
                        c_idx += 1

            records.append({
                "boat_no": boat_count, "racer_name": f"{boat_count}号艇", "reg_no": "----",
                "class_type": class_type, "win_rate_national": win_nat, "2rate_national": rate_nat,
                "win_rate_local": win_loc, "2rate_local": rate_loc, "motor_rate": motor, "exhibition_st": st, "tilt": tilt,
                "straight_time": straight_t, "turn_time": turn_t, "inspect_time": inspect_t,
                "target_course": target_course, "course_1st_rate": c_1st, "course_2nd_rate": c_2nd, "course_3rd_rate": c_3rd,
                "makuri_rate": makuri, "sashi_rate": sashi, "is_flying": is_flying
            })
            boat_count += 1
        except Exception:
            return None, None
    if len(records) != 6:
        return None, None
    df_res = pd.DataFrame(records)
    if df_res["target_course"].nunique() != 6:
        df_res["target_course"] = df_res["boat_no"]
    return df_res, stadium_name


# ==========================================
# 5. 予測エンジン＆本線(A/B)・穴予想(C+以下) 買い目生成
# ==========================================
def calculate_deviation_diff(series, reverse=False):
    values = pd.Series(series, copy=False).astype(float)
    if values.empty:
        return pd.Series(dtype=float)
    mean = values.mean()
    std = values.std(ddof=0) + 1e-5
    factor = -1.0 if reverse else 1.0
    return (10 * (values - mean) / std * factor).clip(lower=-25.0, upper=25.0)


def predict_race_v15(race_data, stadium_name, profiles, is_womens_race=False, wind_speed=0.0, wind_dir="normal"):
    df = race_data.copy()
    v_name = normalize_stadium_name(stadium_name)
    w, _ = get_active_weights(v_name)
    profile = profiles.get(v_name, {"in_base": 55.0, "local_weight": 1.0})

    cond_prof = get_condition_profile(is_womens_race=is_womens_race, wind_speed=wind_speed, wind_dir=wind_dir)
    in_scale = (profile["in_base"] / 55.0) * w.get("in_prob_bias", 1.0) * cond_prof["in_mult"]
    df["base_1st"] = df["target_course"].map(lambda x: FRAME_BASE_PROB[x]["win_1st"])
    df["base_2nd"] = df["target_course"].map(lambda x: FRAME_BASE_PROB[x]["win_2nd"])
    df["base_3rd"] = df["target_course"].map(lambda x: FRAME_BASE_PROB[x]["win_3rd"])

    orig_1st = df.loc[df["target_course"] == 1, "base_1st"].values
    if len(orig_1st) > 0:
        new_1st = min(0.92, max(0.10, orig_1st[0] * in_scale))
        df.loc[df["target_course"] == 1, "base_1st"] = new_1st
        remain_weight = (1.0 - new_1st) / (1.0 - orig_1st[0]) if orig_1st[0] != 1.0 else 1.0
        df.loc[df["target_course"] != 1, "base_1st"] *= remain_weight
        # v17: 追い風3区分・向かい風・女子限定戦に応じた2〜4コース個別補正（非イン合計を再正規化）
        c_boosts = {2: cond_prof["c2_boost"], 3: cond_prof["c3_boost"], 4: cond_prof["c4_boost"]}
        if any(abs(v - 1.0) > 1e-6 for v in c_boosts.values()):
            for c_no, b_mult in c_boosts.items():
                df.loc[df["target_course"] == c_no, "base_1st"] *= b_mult
            non_in_sum = df.loc[df["target_course"] != 1, "base_1st"].sum()
            if non_in_sum > 0:
                df.loc[df["target_course"] != 1, "base_1st"] *= (1.0 - new_1st) / non_in_sum

    df["diff_win_national"] = calculate_deviation_diff(df["win_rate_national"])
    df["diff_rate_national"] = calculate_deviation_diff(df["2rate_national"])
    w_national = w["win_rate_national"] * (0.5 if profile["local_weight"] >= 2.0 else 1.0)
    w_local = w["win_rate_local"] * profile["local_weight"]

    df["diff_win_local"] = calculate_deviation_diff(df["win_rate_local"])
    df["diff_rate_local"] = calculate_deviation_diff(df["2rate_local"])
    df["diff_motor"] = calculate_deviation_diff(df["motor_rate"])

    adjusted_st = df["exhibition_st"].copy().astype(float)
    ex_f_mask = adjusted_st < 0
    in_ex_f = bool((df.loc[df["target_course"] == 1, "exhibition_st"] < 0).any())
    adjusted_st.loc[ex_f_mask] = (0.17 + adjusted_st.loc[ex_f_mask].abs() * 0.4).clip(upper=0.28)
    adjusted_st[df["is_flying"] == True] += 0.03
    df["adjusted_st"] = adjusted_st
    df["st_relative_base"] = (adjusted_st.mean() - adjusted_st) / 0.01

    min_straight = df["straight_time"].astype(float).min()
    min_turn = df["turn_time"].astype(float).min()

    df["gap_straight"] = df["straight_time"].astype(float) - min_straight
    df["gap_turn"] = df["turn_time"].astype(float) - min_turn

    df["diff_straight"] = calculate_deviation_diff(df["straight_time"], reverse=True)
    df["diff_turn"] = calculate_deviation_diff(df["turn_time"], reverse=True)
    df["diff_inspect"] = calculate_deviation_diff(df["inspect_time"], reverse=True)

    df["spec_nobi"] = (df["diff_straight"] + df["diff_inspect"]) / 2.0
    df["spec_deashi"] = df["diff_turn"]

    df.loc[df["gap_straight"] <= 1e-5, "spec_nobi"] += 3.0
    df.loc[df["gap_turn"] <= 1e-5, "spec_deashi"] += 3.0

    tilt_vals = df["tilt"].astype(float)
    df.loc[tilt_vals >= 0.5, "spec_nobi"] += (tilt_vals[tilt_vals >= 0.5] * 4.0)
    df.loc[tilt_vals >= 0.5, "spec_deashi"] -= (tilt_vals[tilt_vals >= 0.5] * 2.0)

    df["ex_score"] = 0.0
    df.loc[df["target_course"].isin([1, 2, 3]), "ex_score"] = (df["spec_deashi"] * 1.4) + (df["spec_nobi"] * 1.1)
    df.loc[df["target_course"].isin([4, 5, 6]), "ex_score"] = (df["spec_nobi"] * 1.5) + (df["spec_deashi"] * 1.0)

    df["trust_factor"] = df["class_type"].map(CLASS_RELIABILITY).fillna(0.85)

    ex_w = w.get("ex_score_weight", 1.0)
    df["base_score"] = (
        df["diff_win_national"] * w_national + df["diff_rate_national"] * w["rate_national"]
        + df["diff_win_local"] * w_local + df["diff_rate_local"] * w["rate_local"]
        + df["diff_motor"] * w["motor"] + df["st_relative_base"] * w["st_relative"]
        + df["win_rate_national"] * w["raw_win_rate_add"] + df["ex_score"] * ex_w
    )
    df["pure_diff_score"] = df["base_score"] * df["trust_factor"]
    df.loc[df["is_flying"] == True, "pure_diff_score"] -= 10.0

    course_st_map = df.set_index("target_course")["adjusted_st"].to_dict()
    in_flying_vals = df.loc[df["target_course"] == 1, "is_flying"].values
    in_is_flying = bool(in_flying_vals[0]) if len(in_flying_vals) > 0 else False
    makuri_w = w.get("makuri_attack_weight", 1.0) * cond_prof["makuri_mult"]
    df["makuri_attack_feat"] = 0.0

    for idx, row in df.iterrows():
        c = int(row["target_course"])
        m_rate = float(row["makuri_rate"])
        s_rate = float(row["sashi_rate"])
        attack_bonus = 0.0

        if c >= 3 and (c - 1) in course_st_map:
            st_adv = course_st_map[c - 1] - row["adjusted_st"]
            if st_adv > 0.01 and m_rate >= 15.0:
                attack_bonus += (st_adv * 100.0) * (m_rate / 20.0) * 1.5

        if c >= 4:
            inner_sts = [course_st_map[ic] for ic in range(2, c) if ic in course_st_map]
            if inner_sts and min(inner_sts) - row["adjusted_st"] >= 0.03:
                attack_bonus += 5.0 * (m_rate / 20.0)

        if in_is_flying and c > 1:
            course_f_bonus = {2: 1.4, 3: 1.3, 4: 1.2, 5: 0.9, 6: 0.6}.get(c, 1.0)
            attack_power = (m_rate + s_rate) / 100.0
            attack_bonus += 6.0 * course_f_bonus * (1.0 + attack_power)

        df.at[idx, "makuri_attack_feat"] = attack_bonus
        df.at[idx, "pure_diff_score"] += attack_bonus * makuri_w

    f1_coef = w.get("f_1st_coef", 0.011)
    f_1st = 1.0 + df["pure_diff_score"] * f1_coef + (df["course_1st_rate"] * 0.01 * w["course_1st_weight"])
    f_2nd = 1.0 + df["pure_diff_score"] * 0.015 + (df["course_2nd_rate"] * 0.01 * w["course_2nd_weight"])
    f_3rd = 1.0 + df["pure_diff_score"] * 0.015 + (df["course_3rd_rate"] * 0.01 * w["course_3rd_weight"])

    df["pred_1st_prob"] = (df["base_1st"] * f_1st).clip(lower=0.001)
    df["pred_1st_prob"] /= df["pred_1st_prob"].sum()

    if in_ex_f:
        in_mask = df["target_course"] == 1
        if (df.loc[in_mask, "pred_1st_prob"] > 0.49).any():
            df.loc[in_mask, "pred_1st_prob"] = 0.49
            others_mask = ~in_mask
            others_sum = df.loc[others_mask, "pred_1st_prob"].sum()
            if others_sum > 0:
                df.loc[others_mask, "pred_1st_prob"] *= (1.0 - 0.49) / others_sum

    df["pred_2nd_prob"] = (df["base_2nd"] * f_2nd).clip(lower=0.001, upper=1.0)
    df["pred_3rd_prob"] = (df["base_3rd"] * f_3rd).clip(lower=0.001, upper=1.0)
    return df


def allocate_budget(bets_df, total_budget=1500, unit=100, equal_weight=False):
    if bets_df.empty:
        return bets_df
    df = bets_df.copy()
    if equal_weight:
        raw_alloc = pd.Series([unit] * len(df), index=df.index)
    else:
        weights = df["確率"] / df["確率"].sum()
        raw_alloc = (weights * total_budget / unit).round() * unit
        raw_alloc = raw_alloc.clip(lower=unit)

        diff = total_budget - raw_alloc.sum()
        if diff != 0:
            max_idx = df["確率"].idxmax()
            raw_alloc.loc[max_idx] = max(unit, raw_alloc.loc[max_idx] + diff)

    df["推奨金額"] = raw_alloc.astype(int)
    df["分岐オッズ"] = (df["推奨金額"].sum() / df["推奨金額"]).round(1)
    return df


def build_trifecta_table(res_df):
    """全120通りの3連単確率・推定市場人気・妙味度を生成する。"""
    prob_1st = res_df.set_index("boat_no")["pred_1st_prob"].to_dict()
    prob_2nd = res_df.set_index("boat_no")["pred_2nd_prob"].to_dict()
    prob_3rd = res_df.set_index("boat_no")["pred_3rd_prob"].to_dict()

    course_dict = res_df.set_index("boat_no")["target_course"].to_dict()
    makuri_dict = res_df.set_index("boat_no")["makuri_rate"].to_dict()
    sashi_dict = res_df.set_index("boat_no")["sashi_rate"].to_dict()
    class_dict = res_df.set_index("boat_no")["class_type"].to_dict()
    win_nat_dict = res_df.set_index("boat_no")["win_rate_national"].to_dict()

    trifecta_list = []
    boats = [1, 2, 3, 4, 5, 6]

    for p1, p2, p3 in itertools.permutations(boats, 3):
        c1 = course_dict[p1]
        base_bias = DEVELOPMENT_BIAS.get(c1, {}).copy()
        is_makuri_head = (c1 >= 3 and makuri_dict.get(p1, 0) > sashi_dict.get(p1, 0))

        def get_dynamic_bias(target_boat):
            tc = course_dict[target_boat]
            b_val = base_bias.get(tc, 1.0)
            if is_makuri_head:
                if tc == c1 - 1:
                    b_val *= 0.65
                elif tc == c1 + 1:
                    b_val *= 1.35
            return b_val

        rem_sum_2nd = sum(prob_2nd[b] * get_dynamic_bias(b) for b in boats if b != p1)
        p2_cond = (prob_2nd[p2] * get_dynamic_bias(p2)) / rem_sum_2nd if rem_sum_2nd > 0 else 0.2

        rem_sum_3rd = sum(prob_3rd[b] * get_dynamic_bias(b) for b in boats if b not in [p1, p2])
        p3_cond = (prob_3rd[p3] * get_dynamic_bias(p3)) / rem_sum_3rd if rem_sum_3rd > 0 else 0.2

        raw_prob = prob_1st[p1] * p2_cond * p3_cond
        trifecta_list.append({"買い目": f"{p1}-{p2}-{p3}", "prob": raw_prob})

    df_tri = pd.DataFrame(trifecta_list)
    df_tri["確率"] = df_tri["prob"] / df_tri["prob"].sum()

    def estimate_market_pop(comb):
        b1, b2, b3 = int(comb[0]), int(comb[2]), int(comb[4])

        def boat_pop_power(b):
            frame_w = {1: 2.2, 2: 1.2, 3: 1.1, 4: 1.0, 5: 0.7, 6: 0.5}.get(b, 1.0)
            cls_w = CLASS_POPULARITY.get(class_dict.get(b, "B1"), 0.8)
            rate_w = max(win_nat_dict.get(b, 5.0) / 5.5, 0.5)
            return frame_w * cls_w * rate_w

        return (boat_pop_power(b1) ** 1.3) * (boat_pop_power(b2) ** 0.8) * (boat_pop_power(b3) ** 0.5)

    df_tri["market_pop"] = df_tri["買い目"].apply(estimate_market_pop)
    df_tri["market_pop"] /= df_tri["market_pop"].sum()
    df_tri["妙味度"] = df_tri["確率"] / df_tri["market_pop"]
    df_tri["概算オッズ"] = (0.75 / df_tri["market_pop"]).clip(lower=1.5, upper=999.9).round(1)
    return df_tri


def _odds3t_table_combo_order():
    """BOAT RACE公式 3連単オッズ表 (td.oddsPoint 120セル) の出現順リストを返す。"""
    combos = []
    for block in range(5):
        for row in range(4):
            for c in range(1, 7):
                others = [b for b in range(1, 7) if b != c]
                b2 = others[block]
                b3 = [b for b in others if b != b2][row]
                combos.append(f"{c}-{b2}-{b3}")
    return combos


def parse_odds3t_html(html_text):
    """BOAT RACE公式 odds3t HTML から全120通りの3連単オッズ辞書を抽出する。"""
    raw_cells = re.findall(
        r'<td[^>]*class="[^"]*oddsPoint[^"]*"[^>]*>(.*?)</td>',
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if len(raw_cells) < 120:
        return None

    combos = _odds3t_table_combo_order()
    odds_map = {}
    for comb, raw_val in zip(combos, raw_cells[:120]):
        txt = re.sub(r"<[^>]+>", "", raw_val).strip().replace(",", "")
        try:
            val = float(txt)
            if val > 0:
                odds_map[comb] = val
        except ValueError:
            continue

    # 欠場艇がある場合でも最低60通り以上取れていれば有効とみなす
    return odds_map if len(odds_map) >= 60 else None


def parse_race_id_for_odds(race_id, stadium_name=None):
    """race_id (例: 20261125_戸田_05R / 20261125_02_05R) から (hd, jcd_str, rno) を抽出する。"""
    if not race_id:
        return None, None, None
    m = re.match(r"^(\d{8})_([^_]+)_(\d{1,2})R?$", str(race_id).strip())
    if not m:
        return None, None, None
    hd, st_token, rno_str = m.group(1), m.group(2), m.group(3)
    rno = int(rno_str)
    if not (1 <= rno <= 12):
        return None, None, None

    if st_token.isdigit():
        jcd_int = int(st_token)
    else:
        st_norm = normalize_stadium_name(stadium_name or st_token)
        jcd_int = STADIUM_NAME_TO_CODE.get(st_norm)
    if not jcd_int or not (1 <= jcd_int <= 24):
        return None, None, None

    return hd, f"{jcd_int:02d}", rno


def fetch_real_odds3t(hd, jcd, rno, timeout=10):
    """
    BOAT RACE公式サイトから締切前（または確定）3連単オッズ(全120通り)を取得する。
    取得失敗・発売前・オフライン時は None を返し、仮想オッズへ安全にフォールバックする。
    """
    url = f"https://www.boatrace.jp/owpc/pc/race/odds3t?rno={int(rno)}&jcd={int(jcd):02d}&hd={hd}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html_text = resp.read().decode("utf-8", errors="ignore")
        return parse_odds3t_html(html_text)
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def is_strict_ana_target_race(res_df, conf_rank=None, is_womens_race=False, wind_speed=0.0, wind_dir="normal"):
    """
    穴予想の対象レース判定。
    - 通常／女子戦（風速4m未満）: 1コース進入艇がB級かつ全国勝率5.0未満
    - 強風時（風速4m以上）: 上記に加え、1コース勝率5.5未満 または ランクC-以下も対象に緩和
    """
    in_rows = res_df[res_df["target_course"] == 1]
    if in_rows.empty:
        return False
    in_row = in_rows.iloc[0]
    cls_str = str(in_row.get("class_type", "")).upper()
    win_nat = float(in_row.get("win_rate_national", 0.0))
    base_ok = cls_str.startswith("B") and (win_nat < 5.0)
    cond_prof = get_condition_profile(is_womens_race=is_womens_race, wind_speed=wind_speed, wind_dir=wind_dir)
    if cond_prof["relax_race_filter"]:
        return base_ok or (win_nat < 5.5) or (conf_rank in {"C-", "D", "E"})
    return base_ok


def generate_ana_bets(
    res_df,
    conf_rank,
    budget=400,
    stadium_name=None,
    real_odds_map=None,
    min_real_ev=None,
    max_bets=None,
    strict_race_filter=ANA_STRICT_RACE_FILTER,
    is_womens_race=False,
    wind_speed=0.0,
    wind_dir="normal",
):
    """
    ランク C+ 以下 (C+, C, C-, D, E) 限定の厳選穴予想（最大4点・無理埋め廃止）。
    - レース選別 (strict_race_filter=True): 1枠B級かつ全国勝率5.0未満のみ対象（該当外は空DataFrame＝見送り）。
    - real_odds_map が渡された場合: 実EV (= 予測確率 × 実オッズ) >= 1.00 の非イン頭から確率上位 max_bets 点（オッズ上限なし・条件未達なら無理に埋めない）。
    - real_odds_map が None の場合: 2〜4号艇頭・確率1.0%以上・概算15倍以上（上限なし）・妙味度1.10以上から上位 max_bets 点（無理埋めなし）。
    ランク A・B は本線対象のため空 DataFrame を返す。
    """
    if conf_rank not in ANA_TARGET_RANKS:
        return pd.DataFrame()

    cond_prof = get_condition_profile(is_womens_race=is_womens_race, wind_speed=wind_speed, wind_dir=wind_dir)
    eff_max_bets = int(max_bets) if max_bets is not None else int(cond_prof["max_bets"])

    if strict_race_filter and not is_strict_ana_target_race(
        res_df, conf_rank=conf_rank, is_womens_race=is_womens_race, wind_speed=wind_speed, wind_dir=wind_dir
    ):
        return pd.DataFrame()

    df_tri = build_trifecta_table(res_df)
    w, _ = get_active_weights(stadium_name)
    boat_by_course = res_df.set_index("target_course")["boat_no"].to_dict()
    in_b = boat_by_course.get(1, 1)

    df_tri["is_non_in_head"] = ~df_tri["買い目"].str.startswith(f"{in_b}-")
    df_tri["head_boat"] = df_tri["買い目"].apply(lambda x: int(str(x).split("-")[0]))

    # --- 実オッズEVモード（締切前オッズ取得成功時：オッズ上限なし・EV>=1.0のみ厳選） ---
    if real_odds_map:
        df_real = df_tri.copy()
        df_real["実オッズ"] = df_real["買い目"].map(real_odds_map)
        df_real = df_real.dropna(subset=["実オッズ"]).copy()

        if not df_real.empty:
            ev_threshold = float(min_real_ev) if min_real_ev is not None else float(w.get("ana_min_real_ev", 1.00))
            df_real["実EV"] = df_real["確率"] * df_real["実オッズ"]

            # 非イン頭 かつ 実EV >= しきい値 のみ抽出（確率降順・最大 max_bets 点・無理埋め廃止）
            non_in_ev = df_real[
                (df_real["is_non_in_head"]) & (df_real["実EV"] >= ev_threshold)
            ].sort_values(by="確率", ascending=False)

            if non_in_ev.empty:
                return pd.DataFrame()

            selected = non_in_ev.head(eff_max_bets).copy()
            actual_budget = len(selected) * 100
            selected = allocate_budget(selected, total_budget=actual_budget, unit=100, equal_weight=True)
            selected["オッズ種別"] = "実オッズ"
            selected["穴区分"] = selected["実オッズ"].apply(
                lambda x: "万舟" if x >= 100.0 else ("大穴" if x >= 50.0 else "中穴")
            )
            return selected

    # --- 仮想オッズフォールバック（オフライン時・過去データ一括検証時） ---
    df_tri["ana_score"] = (df_tri["確率"] ** 0.65) * df_tri["妙味度"]

    virt_cond = (
        (df_tri["is_non_in_head"]) &
        (df_tri["head_boat"].isin(cond_prof["virtual_heads"])) &
        (df_tri["概算オッズ"] >= ANA_VIRTUAL_MIN_ODDS) &
        (df_tri["妙味度"] >= ANA_VIRTUAL_MIN_MYOUMI) &
        (df_tri["確率"] >= ANA_VIRTUAL_MIN_PROB)
    )
    if cond_prof["virtual_max_odds"] is not None:
        virt_cond = virt_cond & (df_tri["概算オッズ"] <= cond_prof["virtual_max_odds"])

    pool = df_tri[virt_cond].sort_values(by="ana_score", ascending=False)

    if pool.empty:
        return pd.DataFrame()

    selected = pool.head(eff_max_bets).copy()
    actual_budget = len(selected) * 100
    selected = allocate_budget(selected, total_budget=actual_budget, unit=100, equal_weight=True)
    selected["オッズ種別"] = "仮想オッズ"
    selected["穴区分"] = selected["概算オッズ"].apply(lambda x: "大穴" if x >= 80.0 else "中穴")
    return selected


def generate_custom_bets_v15(res_df, conf_rank, budget=1500):
    df_tri = build_trifecta_table(res_df)
    prob_1st = res_df.set_index("boat_no")["pred_1st_prob"].to_dict()
    course_dict = res_df.set_index("boat_no")["target_course"].to_dict()
    boat_by_course = res_df.set_index("target_course")["boat_no"].to_dict()
    makuri_dict = res_df.set_index("boat_no")["makuri_rate"].to_dict()
    sashi_dict = res_df.set_index("boat_no")["sashi_rate"].to_dict()
    boats = [1, 2, 3, 4, 5, 6]

    head_count_map = {"A": 4, "B": 6, "C+": 8, "C": 8, "C-": 8}

    if conf_rank in ["D", "E"]:
        in_b = boat_by_course.get(1, 1)
        enemy_probs = {b: prob_1st[b] for b in boats if b != in_b}
        sorted_enemies = sorted(enemy_probs.items(), key=lambda x: x[1], reverse=True)

        report_data = []
        for k_boat, k_prob in sorted_enemies[:2]:
            k_course = course_dict[k_boat]
            m_rate = makuri_dict.get(k_boat, 0.0)
            s_rate = sashi_dict.get(k_boat, 0.0)

            if m_rate >= s_rate and k_course > 2:
                t1_c = k_course + 1 if k_course + 1 <= 6 else 5
                t2_c = 5 if k_course != 5 else 6
                t1 = boat_by_course.get(t1_c, t1_c)
                t2 = boat_by_course.get(t2_c, t2_c)
                tactics = "まくり（外枠・ダッシュ追従軸）"
            else:
                t1 = in_b
                t2_c = k_course - 1 if k_course - 1 >= 2 else 2
                t2 = boat_by_course.get(t2_c, t2_c)
                tactics = "差し・まくり差し（イン残り・内寄り追従軸）"

            form_bets = set()
            for b3 in boats:
                if b3 not in (k_boat, t1): form_bets.add(f"{k_boat}-{t1}-{b3}")
                if b3 not in (k_boat, t2): form_bets.add(f"{k_boat}-{t2}-{b3}")
            for b2 in boats:
                if b2 not in (k_boat, t1): form_bets.add(f"{k_boat}-{b2}-{t1}")
                if b2 not in (k_boat, t2): form_bets.add(f"{k_boat}-{b2}-{t2}")

            sub_df = df_tri[df_tri["買い目"].isin(form_bets)].sort_values(by="確率", ascending=False).head(8)
            sub_df = allocate_budget(sub_df, total_budget=budget)

            report_data.append({
                "killer_boat": k_boat, "killer_course": k_course, "killer_prob": k_prob,
                "tactics": tactics, "target1": t1, "target2": t2, "bets_df": sub_df
            })
        return pd.DataFrame(), report_data
    else:
        df_sorted = df_tri.sort_values(by="確率", ascending=False)
        if conf_rank == "B":
            top6_cov = df_sorted.head(6)["確率"].sum()
            h_cnt = 8 if top6_cov < 0.38 else 6
        else:
            h_cnt = head_count_map.get(conf_rank, 6)

        main_bets_df = df_sorted.head(h_cnt).copy()
        main_bets_df = allocate_budget(main_bets_df, total_budget=budget)
        return main_bets_df, []


# ==========================================
# 6. 予測ログ記録・結果照合・本線(A/B)＆穴(C+以下) 分離バックテスト＆自動修正
# ==========================================
def _df_to_bets_payload(bets_df):
    payload = []
    if bets_df is not None and not bets_df.empty:
        for _, b in bets_df.iterrows():
            item = {
                "comb": b["買い目"],
                "pred_prob": round(float(b["確率"]), 6),
                "est_odds": float(b.get("概算オッズ", 0.0)),
                "myoumi": round(float(b.get("妙味度", 1.0)), 3),
                "bet_amount": int(b.get("推奨金額", 100)),
            }
            if "実オッズ" in b and pd.notna(b["実オッズ"]):
                item["real_odds"] = round(float(b["実オッズ"]), 1)
                item["real_ev"] = round(float(b.get("実EV", 0.0)), 3)
                item["odds_source"] = "real"
            else:
                item["odds_source"] = "virtual"
            payload.append(item)
    return payload


def save_prediction_log(race_id, stadium, res_df, conf_rank, ana_bets_df, main_bets_df=None, mode="auto"):
    boats_payload = []
    for _, r in res_df.sort_values("boat_no").iterrows():
        boats_payload.append({
            "boat_no": int(r["boat_no"]),
            "target_course": int(r["target_course"]),
            "pred_1st_prob": round(float(r["pred_1st_prob"]), 5),
            "ex_score": round(float(r["ex_score"]), 3),
            "makuri_attack_feat": round(float(r["makuri_attack_feat"]), 3),
            "pure_diff_score": round(float(r["pure_diff_score"]), 3),
            "course_1st_rate": round(float(r["course_1st_rate"]), 2),
        })

    # ランクA・Bは main_bets、ランクC+以下は bets (穴) に記録
    main_payload = _df_to_bets_payload(main_bets_df) if conf_rank in MAIN_TARGET_RANKS else []
    ana_payload = _df_to_bets_payload(ana_bets_df) if conf_rank in ANA_TARGET_RANKS else []

    entry = {
        "race_id": race_id,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stadium": normalize_stadium_name(stadium),
        "conf_rank": conf_rank,
        "mode": "main" if conf_rank in MAIN_TARGET_RANKS else "ana",
        "boats": boats_payload,
        "main_bets": main_payload,
        "bets": ana_payload,
        "result": None,
    }

    records = load_prediction_logs()
    updated = False
    for i, rec in enumerate(records):
        if rec.get("race_id") == race_id:
            if rec.get("result") is not None:
                entry["result"] = rec["result"]
            records[i] = entry
            updated = True
            break
    if not updated:
        records.append(entry)

    write_prediction_logs(records)
    return race_id


def load_prediction_logs():
    if not os.path.exists(LOG_FILE):
        return []
    records = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def write_prediction_logs(records):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def record_race_result_interactive():
    records = load_prediction_logs()
    if not records:
        print("⚠️ 予測ログがまだありません。先にレース予測を実行してください。")
        return

    pending = [r for r in records if r.get("result") is None]
    print(f"\n📝 未入力レース: {len(pending)}件 / 全ログ: {len(records)}件")
    target_list = pending[-10:] if pending else records[-10:]
    for idx, r in enumerate(target_list):
        status = "未確定" if r.get("result") is None else f"確定({r['result']['comb']} {r['result']['odds']}倍)"
        tag = "本線" if r.get("conf_rank") in MAIN_TARGET_RANKS else "穴予想"
        print(f"  [{idx}] {r['race_id']} ({r['stadium']} / ランク{r['conf_rank']}・{tag}) - {status}")

    sel = input("結果を入力する番号を選んでください (キャンセルはEnter) > ").strip()
    if not sel.isdigit() or int(sel) >= len(target_list):
        return

    target = target_list[int(sel)]
    comb = input("確定3連単 (例: 4-5-2) > ").strip()
    odds_raw = input("確定オッズ倍率 (例: 84.5 ※8450円なら84.5) > ").strip()
    if not re.match(r"^[1-6]-[1-6]-[1-6]$", comb):
        print("❌ 買い目形式が不正です (1-2-3 の形式で入力してください)。")
        return
    try:
        odds = float(odds_raw)
    except ValueError:
        print("❌ オッズの数値が不正です。")
        return

    target["result"] = {
        "comb": comb,
        "odds": odds,
        "payout_per_100": int(round(odds * 100)),
        "winner_boat": int(comb[0]),
    }

    for i, rec in enumerate(records):
        if rec["race_id"] == target["race_id"]:
            records[i] = target
            break
    write_prediction_logs(records)
    print(f"✅ {target['race_id']} の結果を保存しました！ ({comb} / {odds:.1f}倍)")


def _calc_subset_metrics(settled, bets_key="bets"):
    race_rows = []
    in_pred_sum = 0.0
    in_actual_wins = 0
    in_race_cnt = 0

    for rec in settled:
        res_comb = rec["result"]["comb"]
        res_odds = float(rec["result"]["odds"])
        win_boat = int(rec["result"]["winner_boat"])

        for b in rec.get("boats", []):
            if int(b["target_course"]) == 1:
                in_pred_sum += float(b["pred_1st_prob"])
                in_actual_wins += (1 if int(b["boat_no"]) == win_boat else 0)
                in_race_cnt += 1
                break

        race_bet = 0.0
        race_pay = 0.0
        for bet in rec.get(bets_key, []):
            amt = float(bet.get("bet_amount", 100))
            is_hit = (bet["comb"] == res_comb)
            race_bet += amt
            race_pay += (amt * res_odds) if is_hit else 0.0
        race_rows.append({"race_id": rec["race_id"], "bet": race_bet, "pay": race_pay})

    n = len(race_rows)
    total_bet = sum(x["bet"] for x in race_rows)
    total_pay = sum(x["pay"] for x in race_rows)
    hits = sum(1 for x in race_rows if x["pay"] > 0)
    roi = (total_pay / total_bet * 100.0) if total_bet > 0 else 0.0

    sorted_races = sorted(race_rows, key=lambda x: x["pay"], reverse=True)
    cut_1pct = max(1, int(math.floor(n * 0.01))) if hits > 0 else 0
    kept_1pct = sorted_races[min(cut_1pct, hits):]
    roi_trim_1pct = (sum(x["pay"] for x in kept_1pct) / sum(x["bet"] for x in kept_1pct) * 100.0) if kept_1pct else 0.0

    cut_hit10 = max(1, int(math.floor(hits * 0.10))) if hits > 0 else 0
    kept_hit10 = sorted_races[cut_hit10:]
    roi_trim_hit10 = (sum(x["pay"] for x in kept_hit10) / sum(x["bet"] for x in kept_hit10) * 100.0) if kept_hit10 else 0.0

    return {
        "n": n, "hits": hits, "hit_rate": (hits / n * 100.0) if n > 0 else 0.0,
        "total_bet": total_bet, "total_pay": total_pay, "profit": total_pay - total_bet,
        "roi": roi, "roi_trim_1pct": roi_trim_1pct, "roi_trim_hit10": roi_trim_hit10,
        "in_pred": (in_pred_sum / in_race_cnt * 100.0) if in_race_cnt > 0 else 0.0,
        "in_actual": (in_actual_wins / in_race_cnt * 100.0) if in_race_cnt > 0 else 0.0,
    }


def evaluate_logs_backtest(records=None):
    """
    1) 本線予想（ランク A・B）の的中率・収支・回収率
    2) 穴予想（ランク C+ 以下）の的中率・収支・回収率（ランク別・会場別・EV帯別）
    3) 合算トータル成績
    を分けて診断出力する。
    """
    if records is None:
        records = load_prediction_logs()
    all_settled = [r for r in records if r.get("result") is not None]
    if not all_settled:
        print("⚠️ 結果確定済みの予測ログがまだありません。")
        return None

    main_settled = [
        r for r in all_settled
        if r.get("conf_rank") in MAIN_TARGET_RANKS and r.get("main_bets")
    ]
    ana_settled = [
        r for r in all_settled
        if r.get("conf_rank") in ANA_TARGET_RANKS and r.get("bets")
    ]

    print("\n" + "=" * 78)
    print("📈 本線（ランクA・B）＆ 穴予想（ランクC+以下） バックテスト診断")
    print("=" * 78)
    print(f"全確定レース : {len(all_settled)} R (本線 A・B: {len(main_settled)} R / 穴予想 C+以下: {len(ana_settled)} R)")

    main_m = None
    if main_settled:
        main_m = _calc_subset_metrics(main_settled, bets_key="main_bets")
        print("-" * 78)
        print("🟩 【1. 本線予想 成績（ランク A・B / 予算1,500円傾斜配分）】")
        print(f"  的中       : {main_m['hits']} R / {main_m['n']} R (的中率: {main_m['hit_rate']:.2f}%)")
        print(f"  投資 / 払戻: {main_m['total_bet']:,.0f} 円 / {main_m['total_pay']:,.0f} 円 (純収支: {main_m['profit']:+,.0f} 円)")
        print(f"  回収率     : {main_m['roi']:.1f}% | 上位1%除外: {main_m['roi_trim_1pct']:.1f}% | イン予測→実測: {main_m['in_pred']:.1f}% → {main_m['in_actual']:.1f}%")
        by_m_rank = {}
        for rec in main_settled:
            by_m_rank.setdefault(rec["conf_rank"], []).append(rec)
        print(f"  {'ランク':<6} {'対象R':>6} {'的中':>5} {'的中率':>7} {'収支':>11} {'通常ROI':>8}")
        for rk in MAIN_RANK_ORDER:
            if rk in by_m_rank:
                rm = _calc_subset_metrics(by_m_rank[rk], bets_key="main_bets")
                print(f"  {rk:<6} {rm['n']:>5}R {rm['hits']:>5} {rm['hit_rate']:>6.1f}% {rm['profit']:>+10,.0f}円 {rm['roi']:>7.1f}%")

    ana_m = None
    if ana_settled:
        ana_m = _calc_subset_metrics(ana_settled, bets_key="bets")
        print("-" * 78)
        print("🏴‍☠️ 【2. 穴予想 成績（ランク C+ 以下 / 厳選6点・各100円）】")
        print(f"  的中       : {ana_m['hits']} R / {ana_m['n']} R (的中率: {ana_m['hit_rate']:.2f}%)")
        print(f"  投資 / 払戻: {ana_m['total_bet']:,.0f} 円 / {ana_m['total_pay']:,.0f} 円 (純収支: {ana_m['profit']:+,.0f} 円)")
        print(f"  回収率     : {ana_m['roi']:.1f}% | 上位1%除外: {ana_m['roi_trim_1pct']:.1f}% | 的中上位10%除外: {ana_m['roi_trim_hit10']:.1f}%")
        print(f"  イン1着率  : 予測平均 {ana_m['in_pred']:.1f}% vs 実測 {ana_m['in_actual']:.1f}%")

        by_a_rank = {}
        by_venue = {}
        ev_buckets = {
            "EV < 0.80": {"bet": 0.0, "pay": 0.0, "ev_sum": 0.0, "cnt": 0, "hit": 0},
            "0.80 - 1.00": {"bet": 0.0, "pay": 0.0, "ev_sum": 0.0, "cnt": 0, "hit": 0},
            "1.00 - 1.20": {"bet": 0.0, "pay": 0.0, "ev_sum": 0.0, "cnt": 0, "hit": 0},
            "1.20 - 1.50": {"bet": 0.0, "pay": 0.0, "ev_sum": 0.0, "cnt": 0, "hit": 0},
            "EV >= 1.50": {"bet": 0.0, "pay": 0.0, "ev_sum": 0.0, "cnt": 0, "hit": 0},
        }
        for rec in ana_settled:
            by_a_rank.setdefault(rec["conf_rank"], []).append(rec)
            v = normalize_stadium_name(rec.get("stadium", "不明"))
            by_venue.setdefault(v, []).append(rec)

            res_comb = rec["result"]["comb"]
            res_odds = float(rec["result"]["odds"])
            for bet in rec["bets"]:
                amt = float(bet.get("bet_amount", 100))
                is_hit = (bet["comb"] == res_comb)
                pay = amt * res_odds if is_hit else 0.0
                ref_odds = res_odds if is_hit else float(bet.get("est_odds", 0.0))
                ev = float(bet["pred_prob"]) * ref_odds
                if ev < 0.80: bk = "EV < 0.80"
                elif ev < 1.00: bk = "0.80 - 1.00"
                elif ev < 1.20: bk = "1.00 - 1.20"
                elif ev < 1.50: bk = "1.20 - 1.50"
                else: bk = "EV >= 1.50"
                ev_buckets[bk]["bet"] += amt
                ev_buckets[bk]["pay"] += pay
                ev_buckets[bk]["ev_sum"] += amt * ev
                ev_buckets[bk]["cnt"] += 1
                ev_buckets[bk]["hit"] += (1 if is_hit else 0)

        print("  [ランク別 穴予想内訳 (C+〜E)]")
        print(f"  {'ランク':<6} {'対象R':>6} {'的中':>5} {'的中率':>7} {'収支':>11} {'通常ROI':>8} {'上位1%除外':>10}")
        for rk in ANA_RANK_ORDER:
            if rk in by_a_rank:
                rm = _calc_subset_metrics(by_a_rank[rk], bets_key="bets")
                print(
                    f"  {rk:<6} {rm['n']:>5}R {rm['hits']:>5} {rm['hit_rate']:>6.1f}% "
                    f"{rm['profit']:>+10,.0f}円 {rm['roi']:>7.1f}% {rm['roi_trim_1pct']:>9.1f}%"
                )

        print("  [会場別 穴予想内訳 (ランク C+ 以下)]")
        print(f"  {'会場':<8} {'対象R':>6} {'的中率':>7} {'収支':>11} {'通常ROI':>8} {'上位1%除外':>10}")
        for v_name, v_recs in sorted(by_venue.items(), key=lambda kv: len(kv[1]), reverse=True):
            vm = _calc_subset_metrics(v_recs, bets_key="bets")
            print(
                f"  {v_name:<8} {vm['n']:>5}R {vm['hit_rate']:>6.1f}% "
                f"{vm['profit']:>+10,.0f}円 {vm['roi']:>7.1f}% {vm['roi_trim_1pct']:>9.1f}%"
            )

        print("  [期待値(EV)帯別 穴予想 理論EV vs 実回収率]")
        print(f"  {'EV帯':<14} {'点数':>6} {'的中':>4} {'理論EV':>8} {'実回収率':>9}")
        for label, bk in ev_buckets.items():
            if bk["cnt"] == 0:
                print(f"  {label:<14} {0:>6} {0:>4} {'-':>8} {'-':>9}")
            else:
                th_ev = bk["ev_sum"] / bk["bet"] * 100.0
                ac_roi = bk["pay"] / bk["bet"] * 100.0
                print(f"  {label:<14} {bk['cnt']:>6} {bk['hit']:>4} {th_ev:>7.1f}% {ac_roi:>8.1f}%")

    # 合算トータル
    tot_bet = (main_m["total_bet"] if main_m else 0.0) + (ana_m["total_bet"] if ana_m else 0.0)
    tot_pay = (main_m["total_pay"] if main_m else 0.0) + (ana_m["total_pay"] if ana_m else 0.0)
    tot_hits = (main_m["hits"] if main_m else 0) + (ana_m["hits"] if ana_m else 0)
    tot_n = (main_m["n"] if main_m else 0) + (ana_m["n"] if ana_m else 0)
    if tot_n > 0:
        print("-" * 78)
        print("💰 【3. 本線＋穴予想 合算トータル】")
        print(
            f"  対象 {tot_n} R | 的中 {tot_hits} R ({tot_hits / tot_n * 100:.1f}%) | "
            f"投資 {tot_bet:,.0f} 円 / 払戻 {tot_pay:,.0f} 円 | "
            f"純収支 {tot_pay - tot_bet:+,.0f} 円 (回収率 {tot_pay / tot_bet * 100:.1f}%)"
        )
    evaluate_probability_calibration(all_settled)
    print("=" * 78)
    return {"main": main_m, "ana": ana_m}


def _compute_calibrated_weights(settled, base_weights, lr=0.05):
    """確定ログ集合から4パラメータの較正済み重み辞書を計算して返す。"""
    new_w = base_weights.copy()
    in_pred_sum = 0.0
    in_actual_sum = 0.0
    in_cnt = 0

    grad_f1 = 0.0
    grad_ex = 0.0
    grad_makuri = 0.0
    total_boats = 0

    for rec in settled:
        win_boat = int(rec["result"]["winner_boat"])
        for b in rec["boats"]:
            y = 1.0 if int(b["boat_no"]) == win_boat else 0.0
            p = max(0.001, min(0.99, float(b["pred_1st_prob"])))
            err = y - p

            if int(b["target_course"]) == 1:
                in_pred_sum += p
                in_actual_sum += y
                in_cnt += 1

            grad_f1 += err * float(b.get("pure_diff_score", 0.0))
            grad_ex += err * float(b.get("ex_score", 0.0)) * 0.01
            grad_makuri += err * float(b.get("makuri_attack_feat", 0.0)) * 0.01
            total_boats += 1

    if in_cnt > 0:
        pred_in_rate = in_pred_sum / in_cnt
        actual_in_rate = in_actual_sum / in_cnt
        ratio = actual_in_rate / max(0.10, pred_in_rate)
        new_in_bias = new_w["in_prob_bias"] * (1.0 + 0.25 * (ratio - 1.0))
        new_w["in_prob_bias"] = round(max(0.75, min(1.25, new_in_bias)), 4)

    if total_boats > 0:
        l2 = 0.05
        step_f1 = (grad_f1 / total_boats) - l2 * (new_w["f_1st_coef"] - DEFAULT_WEIGHTS["f_1st_coef"])
        step_ex = (grad_ex / total_boats) - l2 * (new_w["ex_score_weight"] - DEFAULT_WEIGHTS["ex_score_weight"])
        step_mk = (grad_makuri / total_boats) - l2 * (new_w["makuri_attack_weight"] - DEFAULT_WEIGHTS["makuri_attack_weight"])

        new_w["f_1st_coef"] = round(max(0.005, min(0.022, new_w["f_1st_coef"] + lr * 0.02 * step_f1)), 5)
        new_w["ex_score_weight"] = round(max(0.50, min(1.80, new_w["ex_score_weight"] + lr * step_ex)), 4)
        new_w["makuri_attack_weight"] = round(max(0.50, min(2.00, new_w["makuri_attack_weight"] + lr * step_mk)), 4)

    return new_w


def calibrate_and_update_weights(records=None, min_races=10, min_venue_races=MIN_VENUE_ACTIVE_RACES, lr=0.05):
    """全確定ログの1着勝敗を使って、全場共通モデルおよび会場別専用モデルの確率較正を行う。"""
    global WEIGHTS, VENUE_WEIGHTS
    if records is None:
        records = load_prediction_logs()
    settled = [r for r in records if r.get("result") is not None and r.get("boats")]
    if len(settled) < min_races:
        print(f"⚠️ 自動修正には最低 {min_races} レースの確定ログが必要です（現在: {len(settled)} R）。")
        return False

    meta = load_model_params()
    old_global = WEIGHTS.copy()
    WEIGHTS = _compute_calibrated_weights(settled, WEIGHTS, lr=lr)

    by_venue = {}
    for rec in settled:
        v_name = normalize_stadium_name(rec.get("stadium", ""))
        if v_name:
            by_venue.setdefault(v_name, []).append(rec)

    updated_venues = []
    for v_name, v_recs in sorted(by_venue.items()):
        if len(v_recs) < min_venue_races:
            continue
        prev_vw = VENUE_WEIGHTS.get(v_name, {}).get("weights", DEFAULT_WEIGHTS.copy())
        new_vw = _compute_calibrated_weights(v_recs, prev_vw, lr=lr)
        VENUE_WEIGHTS[v_name] = {
            "trained_races": len(v_recs),
            "weights": new_vw,
        }
        updated_venues.append((v_name, len(v_recs), prev_vw, new_vw))

    history = meta.get("history", [])
    history.append({
        "at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "races": len(settled),
        "global": {k: WEIGHTS[k] for k in ["in_prob_bias", "f_1st_coef", "ex_score_weight", "makuri_attack_weight"]},
        "venues": {v: VENUE_WEIGHTS[v]["trained_races"] for v in VENUE_WEIGHTS},
    })
    meta["trained_races"] = len(settled)
    meta["history"] = history
    save_model_params(meta)

    print(f"\n🔧 【モデル自動修正完了】(全確定ログ {len(settled)} R の1着確率較正を反映)")
    print("  [全場共通モデル]")
    for k in ["in_prob_bias", "f_1st_coef", "ex_score_weight", "makuri_attack_weight"]:
        print(f"    ・{k:<22}: {old_global[k]} → {WEIGHTS[k]}")

    if updated_venues:
        print(f"  [会場別専用モデル (最低 {min_venue_races}R 以上で自動作成・予測時自動切替)]")
        for v_name, cnt, old_v, new_v in updated_venues:
            print(
                f"    🏟️ {v_name:<4} ({cnt:>4}R): "
                f"in補正 {old_v['in_prob_bias']}→{new_v['in_prob_bias']} | "
                f"1着係数 {old_v['f_1st_coef']}→{new_v['f_1st_coef']} | "
                f"展示 {old_v['ex_score_weight']}→{new_v['ex_score_weight']} | "
                f"まくり {old_v['makuri_attack_weight']}→{new_v['makuri_attack_weight']}"
            )
    else:
        print(f"  ℹ️ 会場別専用モデルの作成には各場 {min_venue_races}R 以上の確定ログが必要です。")

    print(f"💾 更新パラメータを '{PARAMS_FILE}' に保存しました。")
    return True



def calculate_prediction_reliability(res_df, stadium_name, db, cond_prof):
    """
    予測そのものの信頼度（0〜100点）を3要素から算出する（v17.4 案E配点: 明確さ30 / データ10 / 安定性60）。
    """
    # 1. 予測の明確さ (30点満点): 1着確率のエントロピー + 1位2位差
    probs = res_df["pred_1st_prob"].astype(float).clip(lower=1e-9).values
    entropy = -sum(p * math.log(p) for p in probs)
    norm_clarity = max(0.0, min(1.0, 1.0 - (entropy / math.log(6))))
    sorted_p = sorted(probs, reverse=True)
    top_gap = sorted_p[0] - sorted_p[1] if len(sorted_p) >= 2 else 0.0
    score_clarity = (norm_clarity * 21.0) + min(9.0, top_gap * 22.5)

    # 2. データ充実度 (10点満点): 選手コース実績数 + 会場別学習レース数
    entry_scores = []
    for _, row in res_df.iterrows():
        r_info = find_racer_in_db(db, row.get("reg_no") or row.get("racer_name")) if db else None
        c_str = str(int(row["target_course"]))
        entries = 0
        if r_info and c_str in r_info.get("courses", {}):
            entries = int(r_info["courses"][c_str].get("entries", 0))
        entry_scores.append(min(1.0, entries / 20.0) if entries >= 3 else 0.2)
    racer_data_ratio = sum(entry_scores) / max(1, len(entry_scores))

    v_name = normalize_stadium_name(stadium_name)
    v_races = VENUE_WEIGHTS.get(v_name, {}).get("trained_races", 0)
    venue_data_ratio = min(1.0, v_races / 300.0) if v_races >= MIN_VENUE_ACTIVE_RACES else 0.4
    score_data = ((racer_data_ratio * 23.0) + (venue_data_ratio * 12.0)) * (10.0 / 35.0)

    # 3. 条件安定性 (60点満点): F艇・強風・能力と展示の整合性
    penalty = 0.0
    ex_f_cnt = int((res_df["exhibition_st"].astype(float) < 0).sum())
    f_hold_cnt = int((res_df["is_flying"] == True).sum())
    penalty += ex_f_cnt * 3.5 + f_hold_cnt * 1.5

    if cond_prof.get("is_severe_wind"):
        penalty += 10.0
    elif cond_prof.get("is_strong_wind"):
        penalty += 5.0
    elif cond_prof.get("is_mid_tail"):
        penalty += 2.0

    top_row = res_df.sort_values("pred_1st_prob", ascending=False).iloc[0]
    if float(top_row.get("ex_score", 0.0)) < -10.0:
        penalty += 4.0

    score_stability = max(0.0, 25.0 - penalty) * (60.0 / 25.0)

    total_score = round(min(100.0, max(0.0, score_clarity + score_data + score_stability)), 1)
    if total_score >= 70:
        grade = "高(S)"
    elif total_score >= 55:
        grade = "やや高(A)"
    elif total_score >= 40:
        grade = "標準(B)"
    else:
        grade = "低(C・波乱/不確実)"

    return {
        "score": total_score,
        "grade": grade,
        "clarity": round(score_clarity, 1),
        "data": round(score_data, 1),
        "stability": round(score_stability, 1),
    }


def evaluate_probability_calibration(settled):
    """確定ログから1着予測確率の Brier Score と確率帯別キャリブレーションを算出"""
    bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 0.65), (0.65, 1.01)]
    bin_stats = {b: {"cnt": 0, "pred_sum": 0.0, "win_sum": 0} for b in bins}
    brier_sum = 0.0
    total_boats = 0

    for rec in settled:
        win_boat = int(rec["result"]["winner_boat"])
        for b in rec.get("boats", []):
            p = float(b["pred_1st_prob"])
            y = 1.0 if int(b["boat_no"]) == win_boat else 0.0
            brier_sum += (p - y) ** 2
            total_boats += 1
            for lo, hi in bins:
                if lo <= p < hi:
                    bin_stats[(lo, hi)]["cnt"] += 1
                    bin_stats[(lo, hi)]["pred_sum"] += p
                    bin_stats[(lo, hi)]["win_sum"] += int(y)
                    break

    brier = brier_sum / total_boats if total_boats > 0 else 0.0
    ece = sum(
        abs((st["pred_sum"] / st["cnt"]) - (st["win_sum"] / st["cnt"])) * (st["cnt"] / total_boats)
        for st in bin_stats.values() if st["cnt"] > 0
    ) if total_boats > 0 else 0.0

    print("-" * 78)
    print(f"🎯 【確率の統計的信頼度】 Brier Score: {brier:.4f} | 較正誤差(ECE): {ece*100:.2f}%")
    print(f"  {'予測確率帯':<12} {'艇数':>6} {'予測平均':>8} {'実測勝率':>8} {'ズレ':>8}")
    for (lo, hi), st in bin_stats.items():
        if st["cnt"] > 0:
            mp = st["pred_sum"] / st["cnt"] * 100
            ma = st["win_sum"] / st["cnt"] * 100
            print(f"  {int(lo*100):>2}%〜{int(min(hi,1.0)*100):<3}%   {st['cnt']:>6} {mp:>7.1f}% {ma:>7.1f}% {ma-mp:>+7.1f}%")


# ==========================================
# 7. レポート出力＆メインメニュー
# ==========================================
def print_race_result(res, stadium, conf_rank, in_boat, confidence_score, main_bets_df, report_data, budget, ana_bets_df=None, cond_label="通常一般戦", is_womens_race=False, wind_speed=0.0, wind_dir="normal", reliability=None, rel_blocked=False):
    _, w_label = get_active_weights(stadium)
    if conf_rank in MAIN_TARGET_RANKS:
        mode_label = f" [本線予想・{'8点拡張' if len(main_bets_df) == 8 else f'{len(main_bets_df)}点集中'}]"
    else:
        mode_label = " [穴予想対象]"
    print(f"\n🏟️  会場: {stadium} [{w_label} / 条件:{cond_label}] | 🎯 AI判定: 【 ランク {conf_rank}{mode_label} 】 (イン{in_boat}号艇 1着信頼度: {confidence_score*100:.1f}%)")
    if reliability:
        print(f"🔎 予測信頼度: {reliability['score']}点 [{reliability['grade']}] (明確さ:{reliability['clarity']}/30 | データ:{reliability['data']}/10 | 安定性:{reliability['stability']}/60)")
    print("----------------------------------------------------------------------------")
    print("📊 【各艇パワー＆直前展示評価一覧】")
    print("枠 | 進入 | 登番 | 選手名     | 級 | 展示ST(F) | コース1着率 | 展示指数 | 総合スコア | 1着率")
    for _, r in res.sort_values("target_course").iterrows():
        f_mark = "F" if r["is_flying"] else " "
        name_pad = str(r["racer_name"]).ljust(6, " ")[:6]
        reg_str = str(r.get("reg_no", "")).rjust(4)
        print(f" {int(r['boat_no'])} |  {int(r['target_course'])}C  | {reg_str} | {name_pad} | {r['class_type']} |  {r['exhibition_st']:+5.2f}({f_mark}) |    {r['course_1st_rate']:5.1f}%   |  {r['ex_score']:+6.1f}  |   {r['pure_diff_score']:+6.1f}   | {r['pred_1st_prob']*100:5.1f}%")
    print("============================================================================")

    if rel_blocked and reliability:
        print(f"⏭️  【見送り（信頼度フィルタ）】 予測信頼度 {reliability['score']}点 [{reliability['grade']}] が基準（{RELIABILITY_MIN_SCORE:.0f}点・高S以上）未満のため自動見送り")
        print("    ※全レースの買い目を表示したい場合はメインメニュー [9] で信頼度フィルタを OFF にしてください。")
        print("----------------------------------------------------------------------------")
    elif conf_rank in MAIN_TARGET_RANKS:
        print(f"🟩 ★ 本線三連単ターゲット（ランク {conf_rank} / {len(main_bets_df)}点 / 予算 {budget}円 自動傾斜配分） ★")
        for _, b_row in main_bets_df.iterrows():
            print(f"  【 {b_row['買い目']} 】 確率:{b_row['確率']*100:5.2f}% | 推奨:{b_row['推奨金額']:4d}円 (最低オッズ:{b_row['分岐オッズ']:4.1f}倍〜) [妙味:{b_row['妙味度']:.2f}]")
    else:
        if ana_bets_df is None or ana_bets_df.empty:
            if ANA_STRICT_RACE_FILTER and not is_strict_ana_target_race(res, conf_rank=conf_rank, is_womens_race=is_womens_race, wind_speed=wind_speed, wind_dir=wind_dir):
                print(f"⏭️  【見送り（ケン）】 ランク {conf_rank}：1枠がA級または全国勝率5.0以上のため穴狙い対象外（ムダ打ち回避）")
            else:
                print(f"⏭️  【見送り（ケン）】 ランク {conf_rank}：期待値・確率条件を満たす穴買い目がありません（無理買い回避）")
            print("----------------------------------------------------------------------------")
        else:
            is_real_mode = ("オッズ種別" in ana_bets_df.columns) and (ana_bets_df["オッズ種別"].iloc[0] == "実オッズ")
            mode_tag = "実オッズEVフィルタ・非イン頭" if is_real_mode else "仮想オッズ・厳選フィルタ"
            print(f"🏴‍☠️ ★ ランク {conf_rank} 中穴〜大穴 厳選{len(ana_bets_df)}点（{mode_tag} / 各100円 / 合計 {int(ana_bets_df['推奨金額'].sum())}円） ★")
            for _, b_row in ana_bets_df.iterrows():
                if is_real_mode:
                    print(
                        f"  [{b_row['穴区分']}] 【 {b_row['買い目']} 】 確率:{b_row['確率']*100:4.2f}% | "
                        f"実オッズ:{b_row['実オッズ']:6.1f}倍 | 実EV:{b_row['実EV']:.2f} | 推奨:{b_row['推奨金額']}円"
                    )
                else:
                    print(
                        f"  [{b_row['穴区分']}] 【 {b_row['買い目']} 】 確率:{b_row['確率']*100:4.2f}% | "
                        f"概算:{b_row['概算オッズ']:5.1f}倍 | 妙味:{b_row['妙味度']:.2f} | 推奨:{b_row['推奨金額']}円"
                    )
            print("----------------------------------------------------------------------------")

        if conf_rank in ["D", "E"] and report_data:
            print(f"🔥 【展開参考】イン飛び波乱レース！決まり手特性からの展開軸：")
            for idx, r_data in enumerate(report_data):
                print(f"  第{idx+1}候補：【 {r_data['killer_boat']}号艇 】 ({r_data['killer_course']}C / 1着率:{r_data['killer_prob']*100:.1f}%) ⚔️ {r_data['tactics']} → 軸:{r_data['target1']}・{r_data['target2']}号艇")


def run_prediction_pipeline(target_race_data, stadium, budget, race_id=None, save_log=True, use_real_odds=True, real_odds_map=None, is_womens_race=False, wind_speed=0.0, wind_dir="normal", db=None):
    stadium = normalize_stadium_name(stadium)
    if race_id is None:
        race_id = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{stadium}"
    if db is None:
        db = load_racer_db()
    cond_prof = get_condition_profile(is_womens_race=is_womens_race, wind_speed=wind_speed, wind_dir=wind_dir)
    res = predict_race_v15(target_race_data, stadium, STADIUM_PROFILES, is_womens_race=is_womens_race, wind_speed=wind_speed, wind_dir=wind_dir)
    in_boats = res.loc[res["target_course"] == 1, "boat_no"].values
    in_boat = int(in_boats[0]) if len(in_boats) > 0 else 1
    confidence_score = res.set_index("boat_no")["pred_1st_prob"].get(in_boat, 0.0)

    if confidence_score >= 0.75: conf_rank = "A"
    elif confidence_score >= 0.60: conf_rank = "B"
    elif confidence_score >= 0.50: conf_rank = "C+"
    elif confidence_score >= 0.40: conf_rank = "C"
    elif confidence_score >= 0.30: conf_rank = "C-"
    elif confidence_score >= 0.18: conf_rank = "D"
    else: conf_rank = "E"

    reliability = calculate_prediction_reliability(res, stadium, db, cond_prof)
    rel_blocked = bool(RELIABILITY_FILTER_ENABLED and reliability["score"] < RELIABILITY_MIN_SCORE)

    if rel_blocked:
        main_bets_df = pd.DataFrame()
        report_data = []
        ana_bets_df = pd.DataFrame()
    else:
        fetched_odds = real_odds_map
        ana_eligible = (conf_rank in ANA_TARGET_RANKS) and (not ANA_STRICT_RACE_FILTER or is_strict_ana_target_race(res, conf_rank=conf_rank, is_womens_race=is_womens_race, wind_speed=wind_speed, wind_dir=wind_dir))
        if ana_eligible and use_real_odds and fetched_odds is None:
            hd, jcd, rno = parse_race_id_for_odds(race_id, stadium)
            if hd and jcd and rno:
                print(f"🌐 公式3連単オッズを取得中... ({hd} {stadium} {rno}R)")
                fetched_odds = fetch_real_odds3t(hd, jcd, rno)
                if fetched_odds:
                    w_cur, _ = get_active_weights(stadium)
                    ev_th = w_cur.get("ana_min_real_ev", 1.00)
                    print(f"✅ 実オッズ {len(fetched_odds)}通りを取得 → 実EV >= {ev_th:.2f}（非イン頭・確率上位最大{ANA_MAX_BETS}点・オッズ上限なし）を適用します。")
                else:
                    print("⚠️ 公式オッズを取得できませんでした（発売前またはオフライン）。厳選仮想オッズモードで算出します。")

        main_bets_df, report_data = generate_custom_bets_v15(res, conf_rank, budget=budget)
        ana_bets_df = generate_ana_bets(res, conf_rank, budget=cond_prof["max_bets"] * 100, stadium_name=stadium, real_odds_map=fetched_odds, is_womens_race=is_womens_race, wind_speed=wind_speed, wind_dir=wind_dir)

    print_race_result(res, stadium, conf_rank, in_boat, confidence_score, main_bets_df, report_data, budget, ana_bets_df=ana_bets_df, cond_label=cond_prof["label"], is_womens_race=is_womens_race, wind_speed=wind_speed, wind_dir=wind_dir, reliability=reliability, rel_blocked=rel_blocked)

    if save_log:
        saved_id = save_prediction_log(race_id, stadium, res, conf_rank, ana_bets_df, main_bets_df=main_bets_df)
        print(f"\n💾 予測ログを保存しました (ID: {saved_id}) -> 後でメニュー[5]から結果を入力できます。")
    return res, conf_rank, main_bets_df, ana_bets_df


def prompt_race_condition():
    """v17 対話モード用：女子限定戦（男女混合はN）・風速・風向（2m以上で確認）の簡易入力"""
    w_in = input("👩 女子限定戦（オールレディース・ヴィーナス等 ※男女混合はN）ですか？ (y/N) [Enter=通常]: ").strip().lower()
    is_womens = w_in in ("y", "yes", "1", "はい")
    ws_in = input("💨 風速(m)を入力 [Enter=0 / 追い風は 2m未満・2〜3m・4m以上 で自動分岐]: ").strip()
    try:
        wind_speed = float(ws_in) if ws_in else 0.0
    except ValueError:
        wind_speed = 0.0
    wind_dir = "normal"
    if wind_speed >= TAIL_MID_WIND_THRESHOLD:
        wd_in = input("🧭 風向を選択 (1:追い風 / 2:向かい風 / Enter:横風・不明): ").strip()
        if wd_in == "1":
            wind_dir = "tail"
        elif wd_in == "2":
            wind_dir = "head"
    return is_womens, wind_speed, wind_dir


# v17 互換エイリアス（旧Boatrace_Collector等との互換維持）
predict_race_v17 = predict_race_v15
generate_custom_bets_v17 = generate_custom_bets_v15


if __name__ == "__main__":
    BUDGET = 1500
    meta = load_model_params()
    db = load_racer_db()

    if not db:
        imported = build_db_from_fan_files()
        if imported:
            db = imported

    while True:
        try:
            logs = load_prediction_logs()
            settled_cnt = sum(1 for x in logs if x.get("result") is not None)
            main_cnt = sum(1 for x in logs if x.get("result") is not None and x.get("conf_rank") in MAIN_TARGET_RANKS)
            ana_cnt = sum(1 for x in logs if x.get("result") is not None and x.get("conf_rank") in ANA_TARGET_RANKS)
            venue_list = ", ".join(f"{k}({v['trained_races']}R)" for k, v in VENUE_WEIGHTS.items()) or "未作成"
            print("\n============================================================")
            print("🔮 ボートレース改良　v17.4 | 追い風3区分×女子戦2C重視＆安定性重視信頼度(30/10/60)")
            print(f"   DB: {len(db)}名 | 確定ログ: {settled_cnt}R (本線 A・B: {main_cnt}R / 穴 C+以下: {ana_cnt}R)")
            rel_status = f"ON ({RELIABILITY_MIN_SCORE:.0f}点・高S以上のみ購入)" if RELIABILITY_FILTER_ENABLED else "OFF (全信頼度で購入)"
            print(f"   場別専用モデル: {venue_list}")
            print(f"   信頼度フィルタ: {rel_status}")
            print("============================================================")
            print("  [1] 🚤 当日レース予測（A・Bは本線 / C+以下は公式実オッズEV>=1.0の非イン頭6点）")
            print("  [2] 📂 ファン手帳データ (fan*.txt / .lzh) からDBを一括構築・更新")
            print("  [3] 🔍 選手のコース別成績・登録情報を検索")
            print("  [4] 📋 従来の2回分割入力モード（67項目＋コース別戦績）で予測")
            print("  [5] 📝 レース結果を入力してログと照合")
            print("  [6] 📊 バックテスト診断（本線 A・B と 穴 C+以下 を分離集計＋合算収支）")
            print("  [7] 🔧 蓄積ログから全場共通＆会場別モデルの重みを自動修正")
            print("  [8] ♻️ 重みパラメータ（共通＋場別）を初期値にリセット")
            print(f"  [9] 🛡️ 予測信頼度フィルタ ON/OFF 切替 (現在: {'ON' if RELIABILITY_FILTER_ENABLED else 'OFF'} / 基準{RELIABILITY_MIN_SCORE:.0f}点以上)")
            print("  [q] 終了")
            choice = input("\n番号を選択してください > ").strip().lower()

            if choice == "1":
                target_race_data, stadium, race_id = parse_daily_race_input(db)
                if target_race_data is not None:
                    is_w, ws, wd = prompt_race_condition()
                    run_prediction_pipeline(target_race_data, stadium, BUDGET, race_id=race_id, save_log=True, is_womens_race=is_w, wind_speed=ws, wind_dir=wd, db=db)
            elif choice == "2":
                imported = build_db_from_fan_files()
                if imported:
                    db = imported
            elif choice == "3":
                q = input("検索する選手名または登録番号を入力してください > ").strip()
                r_info = find_racer_in_db(db, q)
                if r_info:
                    f_str = "F持ち" if r_info.get("is_flying") else "Fなし"
                    print(f"\n👤 【{r_info['reg_no']} {r_info['name']}】 ({r_info['branch']} / {r_info['class_type']}) | 全国勝率: {r_info['win_nat']} / 2連対率: {r_info['2rate_nat']}% / 平均ST: {r_info.get('avg_st', 0.16)} ({f_str})")
                    print("  コース | 4期進入数 | 1着率 | 2着率 | 3着率 | 平均ST | 推定まくり/差し率")
                    for c in range(1, 7):
                        cs = r_info["courses"][str(c)]
                        ent = cs.get("entries", "-")
                        print(f"   {c}コース |   {str(ent).rjust(4)}走  | {cs['1st']:4.1f}% | {cs['2nd']:4.1f}% | {cs['3rd']:4.1f}% |  {cs.get('avg_st', 0.16):.2f}  | まくり{cs['makuri']:4.1f}% / 差し{cs['sashi']:4.1f}%")
                else:
                    print("❌ 該当する選手が見つかりませんでした。")
            elif choice == "4":
                print("\n👉 【1回目】基本情報を入力してください（場番号または会場名 + 6艇分データ）")
                base_input = ""
                while True:
                    line = sys.stdin.readline()
                    if not line or line.strip() == "": break
                    base_input += line + " "
                    if len(re.split(r'\s+', base_input.strip())) >= 67: break
                print("\n👉 【2回目】コース別戦績＆決まり手を入力してください。")
                course_input = ""
                while True:
                    line = sys.stdin.readline()
                    if not line or line.strip() == "": break
                    course_input += line + " "
                    if len(re.split(r'\s+', course_input.strip())) >= 30: break
                target_race_data, stadium = parse_split_inputs_legacy(base_input, course_input)
                if target_race_data is not None:
                    is_w, ws, wd = prompt_race_condition()
                    run_prediction_pipeline(target_race_data, stadium, BUDGET, save_log=True, is_womens_race=is_w, wind_speed=ws, wind_dir=wd, db=db)
                else:
                    print("❌ 解析失敗。入力データ数やフォーマットをご確認ください。")
            elif choice == "5":
                record_race_result_interactive()
            elif choice == "6":
                evaluate_logs_backtest()
            elif choice == "7":
                calibrate_and_update_weights(min_races=10, min_venue_races=MIN_VENUE_ACTIVE_RACES)
            elif choice == "8":
                WEIGHTS = DEFAULT_WEIGHTS.copy()
                VENUE_WEIGHTS = {}
                save_model_params({"version": "V17-Wind3Tier-Women2C", "trained_races": 0, "history": []})
                print("✅ 全場共通・会場別の重みパラメータをすべて初期値にリセットしました。")
            elif choice == "9":
                RELIABILITY_FILTER_ENABLED = not RELIABILITY_FILTER_ENABLED
                st_txt = f"ON ({RELIABILITY_MIN_SCORE:.0f}点・高S以上のみ購入推奨)" if RELIABILITY_FILTER_ENABLED else "OFF (全信頼度で買い目出力)"
                print(f"✅ 予測信頼度フィルタを 【 {st_txt} 】 に切り替えました。")
            elif choice == "q":
                print("システムを終了します。")
                break
        except KeyboardInterrupt:
            print("\n⚠️ Ctrl+C が押されたため入力をキャンセルし、メインメニューに戻ります。（終了する場合は q を入力してください）")
            continue