"""값싼 후보 비교기 — 학습 없이 **저장된 OOF 확률만** 다시 채점한다.

아스트라(gpt-6-astra) 감사가 지적한 세 가지를 실측으로 닫기 위해 만들었다.

1. **후광 보정 끄기의 효과는 W1에서 잰 +0.19뿐이다.** 그 값을 최신 모델에 더해 79.84라고
   보고해 왔는데 이는 측정값이 아니다. 후광 보정은 Si 연결성과 두께까지 바꾸므로 모델이
   달라지면 효과도 달라진다. 최신 모델에서 직접 재야 한다.
2. **서로 다른 실험의 확률 앙상블은 아직 안 봤다.** fold 앙상블과 4방향 TTA는 이미 있으므로
   "앙상블 추가"는 새 수단이 아니지만, s15와 s20처럼 **다른 설정의 결합**은 미측정이다.
   두 실험의 fold 배정이 같아야 누수가 없다(아래 `_check_folds`가 강제한다).
3. **k 재추정을 너무 일찍 닫았다.** 확인된 것은 "W1·후광 on에서 60/80이 최적"뿐이고,
   s20·후광 off·앙상블에서는 예측 두께 분포가 달라져 최적 k도 달라질 수 있다.

왜 한 번의 순회로 다 재는가: 확률 파일 하나가 수십 MB라 후보마다 다시 읽으면 디스크만
갉는다. 이미지 하나를 열어 모든 후보를 계산하고 **지표만 누적**한다. 대회 산식이 이미지별
mIoU의 평균이라(`metrics.competition_score`) 예측 마스크를 모아 둘 필요가 없다.

k 격자가 공짜인 이유: 두께 계산(`component_thickness`의 거리 변환)은 k와 무관하다. 덩어리
두께를 한 번 구해 두면 k를 바꾸는 것은 임계값 비교 한 번뿐이다.

주의: 파일 이름을 `2026-09-w2-{격자}.md` 꼴로 지으면 안 된다. `scripts/w2_chain*.sh`가
그 이름의 존재로 격자 완료를 판정해 돌아야 할 격자를 건너뛴다.

실행: `python -m komap.experiments.cheap_wins`
"""
import argparse
import json
from pathlib import Path

import numpy as np

from komap import io
from komap.combine import paint_si, pred3_from_prob
from komap.config import CLS_SI3, ROOT
from komap.metrics import image_miou, iou_per_class
from komap.oof import list_oof, load_full_prob
from komap.postprocess import component_thickness
from komap.preprocess import anchor_normalize, group_of

DOC_DIR = ROOT / "docs" / "experiments"
RUNS_DIR = ROOT / "runs"
STEM = "2026-09-w2-cheap-wins"

# 비교할 실험. 이름 → runs 폴더. 앙상블은 이 둘의 확률 평균이다.
SOURCES = {"s15": "w2_res_s15", "s20": "w2_res_s20"}
ENSEMBLE = ("s15", "s20")

# k 격자. 기본값 60/80을 반드시 포함한다(현재 설정이 표에 있어야 증감을 읽는다).
K_A = (55, 60, 65)
K_B = (75, 80, 85)
K_DEFAULT = (60, 80)

N_CLASSES = 4
CLASS_NAMES = ["Al 기지", "Al₃Ni", "Primary Si", "Eutectic Si"]


def _check_folds(tables: dict[str, dict]):
    """모든 실험이 같은 fold 배정을 써야 앙상블에 누수가 없다.

    OOF 확률은 그 이미지를 학습하지 않은 fold의 모델이 낸 것이다. 두 실험의 분할이 다르면
    한쪽에서는 학습에 쓰인 이미지의 확률을 섞게 되어 점수가 부풀려진다."""
    names = [set(t) for t in tables.values()]
    if len({frozenset(n) for n in names}) != 1:
        raise ValueError("실험마다 OOF 이미지 집합이 다르다 — 앙상블하면 비교가 깨진다")
    base = next(iter(tables.values()))
    for tag, t in tables.items():
        bad = [n for n in t if t[n][0] != base[n][0]]
        if bad:
            raise ValueError(f"{tag}: fold 배정이 다른 이미지 {len(bad)}장 — 앙상블은 누수다")


def _variants(probs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """확률맵 후보. 앙상블은 원본 해상도로 되돌린 뒤 평균한다(배율이 달라도 합칠 수 있다)."""
    out = dict(probs)
    if all(t in probs for t in ENSEMBLE):
        out["+".join(ENSEMBLE)] = np.mean([probs[t] for t in ENSEMBLE], axis=0)
    return out


def _acc_key(source, halo, ka, kb):
    return f"{source}|halo={'on' if halo else 'off'}|k={ka}/{kb}"


def measure(runs_dir=RUNS_DIR, split="train", limit=0):
    """이미지 한 장을 열어 모든 후보를 채점하고 지표만 누적한다."""
    tables = {tag: list_oof(Path(runs_dir) / d) for tag, d in SOURCES.items()}
    _check_folds(tables)
    paths = [p for p in sorted(io.list_split(split)) if io.mask_path_for(p).exists()]
    if limit:
        paths = paths[:limit]

    acc: dict[str, dict] = {}
    for ip in paths:
        gray = io.load_gray(ip)
        h, w = gray.shape
        gn = anchor_normalize(gray)
        target = io.load_mask(io.mask_path_for(ip)).astype(np.int64)
        grp = group_of(w)
        probs = {tag: load_full_prob(tables[tag][ip.name][1], h, w) for tag in SOURCES}

        for source, prob in _variants(probs).items():
            for halo in (True, False):
                p3 = pred3_from_prob(prob, gn if halo else None)
                # 거리 변환은 k와 무관하다 — 한 번만 구하고 k 격자는 임계 비교만 한다.
                lab, thick = component_thickness(p3 == CLS_SI3)
                for ka in K_A:
                    for kb in K_B:
                        thr = w / (ka if grp == "A" else kb)
                        p4 = paint_si(p3, lab, thick > thr)
                        key = _acc_key(source, halo, ka, kb)
                        a = acc.setdefault(key, {"per_image": [], "per_class": {k: [] for k in range(N_CLASSES)}})
                        a["per_image"].append(image_miou(p4, target, N_CLASSES))
                        for k, v in iou_per_class(p4, target, N_CLASSES).items():
                            if v is not None:
                                a["per_class"][k].append(v)

    rows = []
    for key, a in acc.items():
        source, halo, kk = key.split("|")
        rows.append({
            "source": source, "halo": halo.split("=")[1], "k": kk.split("=")[1],
            "miou": float(np.mean(a["per_image"])),
            "per_class": {str(k): (float(np.mean(v)) * 100 if v else None) for k, v in a["per_class"].items()},
        })
    rows.sort(key=lambda r: -r["miou"])
    return {"n_images": len(paths), "split": split, "rows": rows}


def render(res):
    rows = res["rows"]
    base = next((r for r in rows if r["source"] == "s20" and r["halo"] == "on"
                 and r["k"] == f"{K_DEFAULT[0]}/{K_DEFAULT[1]}"), None)
    out = [f"# 값싼 후보 비교 — 학습 없이 저장된 OOF 확률만 재채점 ({res['split']} {res['n_images']}장)", "",
           "아스트라 감사의 지적 셋을 실측으로 닫는다: ① 후광 보정 끄기 효과를 **최신 모델에서**",
           "직접 잰다(기존 +0.19는 W1 값이다) ② 서로 다른 실험(s15·s20)의 확률 앙상블 ③ k 재추정.",
           "학습·추론은 없다. 두 실험의 fold 배정이 같음을 확인하고 돌렸으므로 앙상블에 누수가 없다.", ""]
    if base:
        out += [f"기준은 **s20 · 후광 on · k {K_DEFAULT[0]}/{K_DEFAULT[1]} = {base['miou']:.3f}**"
                " (격자 표에 보고된 그 값이다).", ""]

    out += ["## 상위 15개", "",
            "| 순위 | 확률 출처 | 후광 | k(A/B) | OOF4 | " + " | ".join(CLASS_NAMES) + " | 기준 대비 |",
            "|---:|---|---|---|---:|" + "---:|" * (len(CLASS_NAMES) + 1)]
    for i, r in enumerate(rows[:15], 1):
        pc = " | ".join(f"{r['per_class'].get(str(c)):.2f}" if r["per_class"].get(str(c)) is not None else "—"
                        for c in range(N_CLASSES))
        d = f"{r['miou'] - base['miou']:+.3f}" if base else "—"
        out.append(f"| {i} | {r['source']} | {r['halo']} | {r['k']} | **{r['miou']:.3f}** | {pc} | {d} |")

    out += ["", "## 축별로 갈라 보기", ""]
    for label, sel in (("후광 on/off (k 60/80 고정)", lambda r: r["k"] == "60/80"),
                       ("확률 출처 (후광·k는 각 출처의 최선)", None)):
        out += [f"### {label}", ""]
        if sel:
            sub = sorted([r for r in rows if sel(r)], key=lambda r: -r["miou"])
            out += ["| 출처 | 후광 | OOF4 |", "|---|---|---:|"]
            out += [f"| {r['source']} | {r['halo']} | {r['miou']:.3f} |" for r in sub]
        else:
            best = {}
            for r in rows:
                if r["source"] not in best or r["miou"] > best[r["source"]]["miou"]:
                    best[r["source"]] = r
            out += ["| 출처 | 최선 설정 | OOF4 |", "|---|---|---:|"]
            out += [f"| {s} | 후광 {r['halo']} · k {r['k']} | {r['miou']:.3f} |"
                    for s, r in sorted(best.items(), key=lambda x: -x[1]["miou"])]
        out.append("")

    out += ["## 읽는 법", "",
            "- **k를 고르는 것은 선택이다.** 같은 70장에서 여러 k를 보고 최고를 고르면 그만큼",
            "  낙관적으로 잡힌다. k를 바꿔 얻은 이득은 다른 축의 이득보다 할인해서 읽어야 한다.",
            "- 앙상블은 **원본 해상도로 되돌린 뒤** 평균한다. 배율이 다른 두 실험을 합칠 수 있는",
            "  이유이고, 두께 판정도 원본 좌표계에서 한 번만 한다.",
            "- 이 표는 train OOF다. valid는 별도로 확인해야 하며, Al₃Ni의 OOF↔valid 격차가",
            "  −7.98로 가장 크다는 점을 함께 본다(아스트라 감사).", ""]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N장만(빠른 확인용)")
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--docs-dir", default=None)
    a = ap.parse_args(argv)
    runs_dir = Path(a.runs_dir) if a.runs_dir else RUNS_DIR
    docs_dir = Path(a.docs_dir) if a.docs_dir else DOC_DIR

    res = measure(runs_dir, a.split, a.limit)
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / f"{STEM}.md").write_text(render(res), encoding="utf-8")
    (docs_dir / f"{STEM}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    top = res["rows"][0]
    print(json.dumps({"n_images": res["n_images"], "n_candidates": len(res["rows"]),
                      "best": {k: top[k] for k in ("source", "halo", "k", "miou")},
                      "doc": str(docs_dir / f"{STEM}.md")}, ensure_ascii=False))
    return res


if __name__ == "__main__":
    main()
