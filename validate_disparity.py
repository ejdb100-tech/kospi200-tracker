#!/usr/bin/env python3
"""
이격도 신호 검증 하네스 (validation harness, NOT a trading system)
================================================================
질문: 코스피200이 "과열(이격도 높음)"일 때 실제로 선행수익률이 낮아지는가?
      그리고 고정 임계값 대신 변동성 스케일링이 신호를 개선하는가?

세 가지 신호를 5/20/60일 선행수익률에 대해 비교:
  1) disparity   : 원본 이격도 = close/ma50*100  (고정 임계 130/120/105 평가 포함)
  2) z_vol       : 변동성 스케일 = (close/ma50 - 1) / (sigma_d * sqrt(50))
                   → "전고점 대비가 아니라 50일 변동성 몇 배만큼 위에 있나"
  3) z_roll      : 이격도의 롤링 z-score (적응형 임계값 대용)

누설 방지/검증 장치
  - Purged walk-forward: 버킷 경계는 train(과거)에서만, 평가는 test(미래)에서.
  - Embargo: fold 경계에서 선행수익률 창(h일)이 겹치지 않도록 h일 격리.
  - 중첩(overlap) 보정: h일 선행수익률은 h-1일 겹쳐 t값이 부풀려짐
      → 유의성은 (a) circular-shift 순열검정, (b) moving-block 부트스트랩으로.
  - 다중검정: 신호 3 × 호라이즌 3 = 9개 → BH(FDR) 보정 p값 함께 보고.

데이터는 --demo면 합성, 아니면 FinanceDataReader로 코스피200(KS200) 실데이터.
"""
import argparse
import sys
import numpy as np
import pandas as pd

SEED = 7


# ----------------------------------------------------------------------
# 데이터
# ----------------------------------------------------------------------
def load_real(start="2014-01-01"):
    # FinanceDataReader 사용(코스피200 = 'KS200'). pykrx는 KRX 보안정책 변경으로
    # 지수 조회가 깨져(KeyError: '지수명') 대체함.
    import FinanceDataReader as fdr
    df = fdr.DataReader("KS200", start)
    if df is None or df.empty or "Close" not in df.columns:
        raise RuntimeError("KS200 데이터를 받지 못했습니다. 네트워크를 확인하세요.")
    s = df["Close"].astype(float).dropna()
    s.index = pd.to_datetime(s.index)
    return s.rename("close")


def load_demo(n=2600, seed=SEED):
    """합성 가격: 추세 + 모멘텀 + 약한 평균회귀. 과열 구간까지 움직이게 튜닝. 도구 시연용."""
    rng = np.random.default_rng(seed)
    px = [330.0]
    mom = 0.0
    ma_window = 50
    for t in range(1, n):
        hist = px[-ma_window:]
        ma = np.mean(hist)
        ext = px[-1] / ma - 1.0
        mom = 0.92 * mom + 0.08 * rng.standard_normal()        # 추세(모멘텀) 레짐
        # 모멘텀이 이격을 키우고, 이격이 커지면 약하게 되돌림 → 과열↔해소 사이클
        mu = 0.06 / 252 + 0.0035 * mom - 0.10 * ext
        sig = 0.17 / np.sqrt(252) * (1 + 1.5 * abs(ext))        # 이격 크면 변동성↑
        px.append(px[-1] * np.exp(mu + sig * rng.standard_normal()))
    idx = pd.bdate_range("2015-01-02", periods=n)
    return pd.Series(px, index=idx, name="close")


# ----------------------------------------------------------------------
# 피처 / 라벨
# ----------------------------------------------------------------------
def build_features(close, W_roll=504):
    df = pd.DataFrame({"close": close})
    df["ma50"] = close.rolling(50).mean()
    df["disparity"] = close / df["ma50"] * 100
    ret = np.log(close).diff()
    sigma_d = ret.ewm(span=20, min_periods=20).std()
    ext = close / df["ma50"] - 1.0
    df["z_vol"] = ext / (sigma_d * np.sqrt(50))
    m = df["disparity"].rolling(W_roll, min_periods=W_roll // 2).mean()
    s = df["disparity"].rolling(W_roll, min_periods=W_roll // 2).std()
    df["z_roll"] = (df["disparity"] - m) / s
    return df


def forward_returns(close, horizons):
    out = {}
    for h in horizons:
        out[h] = close.shift(-h) / close - 1.0
    return out


# ----------------------------------------------------------------------
# 통계 유틸
# ----------------------------------------------------------------------
def spearman(a, b):
    return pd.Series(a).corr(pd.Series(b), method="spearman")


def circular_shift_pvalue(sig, fwd, n=2000, seed=SEED):
    """순환이동 순열검정: 자기상관 보존, 교차관계만 파괴 → Spearman의 null p값."""
    x = sig.values
    y = fwd.values
    obs = spearman(x, y)
    rng = np.random.default_rng(seed)
    N = len(y)
    cnt = 0
    for _ in range(n):
        k = int(rng.integers(1, N))
        s = spearman(x, np.roll(y, k))
        if abs(s) >= abs(obs):
            cnt += 1
    return obs, (cnt + 1) / (n + 1)


def block_bootstrap_spread(sig, fwd, q_lo, q_hi, block, n=1000, seed=SEED):
    """top-quantile vs bottom-quantile 선행수익률 차이의 moving-block 부트스트랩 CI."""
    lo, hi = np.quantile(sig, q_lo), np.quantile(sig, q_hi)
    top = fwd[sig >= hi]
    bot = fwd[sig <= lo]
    obs = top.mean() - bot.mean()
    vals = pd.DataFrame({"s": sig.values, "f": fwd.values})
    N = len(vals)
    rng = np.random.default_rng(seed)
    boot = []
    nblocks = int(np.ceil(N / block))
    for _ in range(n):
        starts = rng.integers(0, N - block + 1, size=nblocks)
        idx = np.concatenate([np.arange(s0, s0 + block) for s0 in starts])[:N]
        bs = vals.iloc[idx]
        l, hgh = np.quantile(bs["s"], q_lo), np.quantile(bs["s"], q_hi)
        t = bs.loc[bs["s"] >= hgh, "f"].mean()
        b = bs.loc[bs["s"] <= l, "f"].mean()
        boot.append(t - b)
    boot = np.array(boot)
    ci = np.nanpercentile(boot, [2.5, 97.5])
    p = 2 * min((boot >= 0).mean(), (boot <= 0).mean())
    return obs, ci, p


def bh_fdr(pvals):
    """Benjamini-Hochberg 보정 p값."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.clip(ranked, 0, 1)
    return out


# ----------------------------------------------------------------------
# Purged walk-forward
# ----------------------------------------------------------------------
def walk_forward(sig, fwd, k_folds, embargo, n_buckets):
    """train(과거)에서 분위경계 산출 → test에서 버킷별 선행수익률(OOS). 상-하 스프레드 평균."""
    n = len(sig)
    fold = n // k_folds
    spreads, lo_means, hi_means = [], [], []
    for i in range(1, k_folds):
        ts = i * fold
        te = n if i == k_folds - 1 else (i + 1) * fold
        tr_end = max(0, ts - embargo)
        if tr_end < n_buckets * 20:
            continue
        tr_s = sig.iloc[:tr_end]
        te_s = sig.iloc[ts:te]
        te_f = fwd.iloc[ts:te]
        edges = np.quantile(tr_s, np.linspace(0, 1, n_buckets + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        b = pd.cut(te_s, bins=edges, labels=False, include_lowest=True)
        g = te_f.groupby(b).mean()
        if 0 in g.index and (n_buckets - 1) in g.index:
            hi_means.append(g[n_buckets - 1])
            lo_means.append(g[0])
            spreads.append(g[n_buckets - 1] - g[0])
    if not spreads:
        return None
    return {
        "oos_top_minus_bottom": float(np.mean(spreads)),
        "oos_bottom_mean": float(np.mean(lo_means)),
        "oos_top_mean": float(np.mean(hi_means)),
        "n_folds": len(spreads),
    }


# ----------------------------------------------------------------------
# 메인 분석
# ----------------------------------------------------------------------
def fixed_bucket_table(disparity, fwd, h):
    edges = [-np.inf, 105, 120, 130, np.inf]
    labels = ["≤105 해소", "105-120 정상", "120-130 경계", "≥130 과열"]
    b = pd.cut(disparity, bins=edges, labels=labels)
    t = pd.DataFrame({"fwd": fwd, "bucket": b}).dropna()
    g = t.groupby("bucket", observed=True)["fwd"]
    tab = pd.DataFrame({"n": g.size(), f"mean_fwd{h}_%": g.mean() * 100,
                        "hit_up_%": g.apply(lambda x: (x > 0).mean() * 100)})
    return tab.round(2)


def run(close, horizons, k_folds, embargo, n_buckets, n_perm, do_plot):
    feat = build_features(close)
    fwd = forward_returns(close, horizons)
    signals = ["disparity", "z_vol", "z_roll"]

    print("=" * 68)
    print(f"표본: {close.index[0].date()} ~ {close.index[-1].date()}  ({len(close)}일)")
    print("=" * 68)

    # 고정 임계값(원본 주장) 직접 검증 — disparity만
    for h in horizons:
        aligned = pd.concat([feat["disparity"], fwd[h]], axis=1).dropna()
        print(f"\n[고정 임계값] disparity → {h}일 선행수익률")
        print(fixed_bucket_table(aligned.iloc[:, 0], aligned.iloc[:, 1], h).to_string())

    # 신호×호라이즌 매트릭스
    rows = []
    for s in signals:
        for h in horizons:
            d = pd.concat([feat[s], fwd[h]], axis=1).dropna()
            d.columns = ["sig", "fwd"]
            rho, p_perm = circular_shift_pvalue(d["sig"], d["fwd"], n=n_perm)
            sp, ci, p_boot = block_bootstrap_spread(
                d["sig"], d["fwd"], 0.2, 0.8, block=h, n=max(400, n_perm // 4))
            wf = walk_forward(d["sig"], d["fwd"], k_folds, embargo, n_buckets)
            rows.append({
                "signal": s, "h": h, "n": len(d),
                "spearman": round(rho, 3), "p_perm": round(p_perm, 4),
                "spread_IS_%": round(sp * 100, 3),
                "spread_CI_%": f"[{ci[0]*100:.2f},{ci[1]*100:.2f}]",
                "p_boot": round(p_boot, 4),
                "spread_OOS_%": None if wf is None else round(wf["oos_top_minus_bottom"] * 100, 3),
                "folds": None if wf is None else wf["n_folds"],
            })
    res = pd.DataFrame(rows)
    res["p_perm_BH"] = bh_fdr(res["p_perm"]).round(4)

    print("\n" + "=" * 68)
    print("신호 × 호라이즌 검증 매트릭스")
    print("  spearman<0 = 더 extended일수록 선행수익률 낮음 (이론과 일치 방향)")
    print("  spread = top20% − bottom20% 선행수익률. OOS는 purged walk-forward.")
    print("=" * 68)
    print(res.to_string(index=False))

    print("\n해석 가이드")
    print("  - p_perm_BH < 0.05 이고 spearman 부호가 IS/OOS spread와 일치해야 '진짜 신호'.")
    print("  - z_vol의 |spearman|/유의성이 disparity보다 크면 변동성 스케일링이 개선한 것.")
    print("  - spread_IS와 spread_OOS의 부호·크기가 크게 갈리면 과적합/누설 의심.")

    if do_plot:
        _plot(feat, fwd, horizons)
    res.to_csv("validation_results.csv", index=False)
    print("\n저장: validation_results.csv")
    return res


def _plot(feat, fwd, horizons):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("(matplotlib 없음 — 차트 생략)")
        return
    sigs = ["disparity", "z_vol", "z_roll"]
    fig, ax = plt.subplots(len(sigs), len(horizons), figsize=(4 * len(horizons), 3 * len(sigs)))
    for r, s in enumerate(sigs):
        for c, h in enumerate(horizons):
            d = pd.concat([feat[s], fwd[h]], axis=1).dropna()
            d.columns = ["sig", "fwd"]
            q = pd.qcut(d["sig"], 5, labels=False, duplicates="drop")
            m = (d["fwd"].groupby(q).mean() * 100)
            a = ax[r, c] if len(sigs) > 1 else ax[c]
            a.bar(m.index, m.values, color=["#22d3ee", "#3fb950", "#9aa", "#d9a528", "#f85149"][:len(m)])
            a.axhline(0, color="#888", lw=.6)
            a.set_title(f"{s} | {h}d", fontsize=9)
            a.set_xlabel("quintile (low->high)", fontsize=7)
            a.set_ylabel("mean fwd ret %", fontsize=7)
    fig.tight_layout()
    fig.savefig("validation_buckets.png", dpi=130)
    print("저장: validation_buckets.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="합성 데이터로 실행(네트워크 불필요)")
    ap.add_argument("--start", default="2014-01-01")
    ap.add_argument("--horizons", default="5,20,60")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--embargo", type=int, default=60)
    ap.add_argument("--buckets", type=int, default=5)
    ap.add_argument("--nperm", type=int, default=2000)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    horizons = [int(x) for x in a.horizons.split(",")]
    close = load_demo() if a.demo else load_real(a.start)
    run(close, horizons, a.folds, a.embargo, a.buckets, a.nperm, a.plot)


if __name__ == "__main__":
    main()
