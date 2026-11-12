"""fold 하나를 학습한다. 기본 선택 기준은 검증 fold의 mIoU(전체 이미지 추론 기준).

타일 손실이 아니라 '전체 이미지 추론 mIoU'로 best를 고르는 이유: 대회 채점은 이미지 한 장
단위로 매겨진다. 타일 손실이 낮아도 타일 이음매·드문 클래스에서 무너지면 점수는 떨어지므로
실제 채점과 같은 방식으로 재 본다. Si 전용 손실에서는 si_pair를 선택 기준으로 쓴다.
"""
# PYTORCH_ENABLE_MPS_FALLBACK은 komap/__init__.py가 torch import 이전에 건다.
import argparse
import hashlib
import math
import json
import os
import random
import uuid
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from komap import hint as Hn
from komap import io
from komap.config import BOUNDARY_BAND_PX, N_FOLDS, ROOT, RUNS_DIR, SEED
from komap.edge_aux import (EDGE_AUX_DEFAULT_WEIGHT, attach_edge_head, boundary_target,
                            checkpoint_state_dict, edge_bce_loss)
from komap.cv import make_folds
from komap.dataset import SegDataset, load_full
from komap.infer import predict_probs
from komap.losses import LOSS_SCOPES, build_loss
from komap.metrics import competition_score, si_pair_score
from komap.model import build_model, pick_device
from komap.augment import PHOTOMETRIC_KINDS
from komap.oof import restore_full_prob
from komap.preprocess import NORM_MODES
from komap.si2 import OUTPUT_HEADS, expand_prob, output_channels


def _probability(text):
    """0~1의 유한한 확률만 받는다. 범위 밖·NaN·무한대를 조용히 받으면 체크포인트에 적힌 값과
    실제 동작이 어긋난다(음수·NaN은 항상 꺼짐, 1 초과는 항상 켜짐)."""
    value = float(text)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"확률은 0~1의 유한한 값이어야 한다: {text}")
    return value


def _nonnegative(text):
    """유한한 0 이상의 보조 손실 가중치만 받는다."""
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(f"가중치는 0 이상의 유한한 값이어야 한다: {text}")
    return value


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--weights", default="imagenet")
    ap.add_argument("--arch", default="unetplusplus")
    ap.add_argument("--loss", default="ce_dice")
    ap.add_argument("--class-weights", default=None, help="쉼표 구분, 예 1,1.5,1.25")
    ap.add_argument("--ignore-band", type=int, default=0)
    ap.add_argument("--n-classes", type=int, default=3)
    ap.add_argument("--n-folds", type=int, default=N_FOLDS, help="디버그용으로만 줄인다")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--resample-scale", type=float, default=1.0,
                    help="폭별 리샘플 배수에 공통으로 곱하는 전역 배율(입력 해상도 축)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--spi", type=int, default=8, help="이미지당 에폭 내 타일 수")
    ap.add_argument("--limit", type=int, default=0, help="디버그: 이미지 수 제한")
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--hint", action="store_true",
                    help="규칙 지도 4채널을 입력에 붙인다(캐스케이드)")
    ap.add_argument("--ridge", action="store_true",
                    help="얇은 판을 겨냥한 Hessian 능선 응답 1채널을 입력에 붙인다")
    ap.add_argument("--anchor-jitter", type=_probability, default=0.0,
                    help="정규화 뒤 앵커 밝기 재매핑을 적용할 확률(0~1)")
    ap.add_argument("--no-interp-geo", action="store_true",
                    help="Affine·ElasticTransform 기하 증강을 끈다")
    ap.add_argument("--norm", choices=NORM_MODES, default="anchor3",
                    help="회색조 정규화 방식")
    ap.add_argument("--no-size-correction", action="store_true",
                    help="폭별 픽셀 크기 보정을 끈다")
    ap.add_argument("--photometric", choices=PHOTOMETRIC_KINDS, default="measured",
                    help="학습 광도 증강 종류")
    ap.add_argument("--train-list", default=None)
    ap.add_argument("--val-list", default=None)
    ap.add_argument("--select-epoch", choices=("best", "last"), default="best")
    ap.add_argument("--oof-res", choices=("resampled", "original"), default="resampled")
    ap.add_argument("--loss-scope", choices=LOSS_SCOPES, default="all",
                    help="손실을 넣는 픽셀 범위. si = 정답이 Primary/Eutectic인 픽셀만(4클래스 전용)")
    ap.add_argument("--select-metric", choices=("miou", "si_pair"), default="miou",
                    help="best 에폭 선택 기준. si_pair = 정답 Si 픽셀 안 Primary/Eutectic IoU 평균(4클래스 전용)")
    ap.add_argument("--head", choices=OUTPUT_HEADS, default="full",
                    help="출력 헤드. si2 = Primary/Eutectic 2채널(4클래스 라벨·--loss-scope si·--select-metric si_pair 전용)")
    ap.add_argument("--edge-aux-weight", type=_nonnegative, default=EDGE_AUX_DEFAULT_WEIGHT,
                    help="경계 보조 헤드 BCE 가중 λ. 0이면 기존 동작. 3클래스·--head full 전용")
    args = ap.parse_args(argv)
    if (args.train_list is None) != (args.val_list is None):
        ap.error("--train-list와 --val-list는 함께 지정해야 한다")
    if args.loss_scope == "si" and args.n_classes != 4:
        ap.error("--loss-scope si는 4클래스 전용이다")
    if args.select_metric == "si_pair" and args.n_classes != 4:
        ap.error("--select-metric si_pair는 4클래스 전용이다")
    if args.head == "si2" and args.n_classes != 4:
        ap.error("--head si2는 --n-classes 4 전용이다")
    if args.head == "si2" and args.loss_scope != "si":
        ap.error("--head si2는 --loss-scope si가 필요하다")
    if args.head == "si2" and args.select_metric != "si_pair":
        ap.error("--head si2는 --select-metric si_pair가 필요하다")
    if args.edge_aux_weight > 0 and (args.n_classes != 3 or args.head != "full"):
        ap.error("--edge-aux-weight는 --n-classes 3·--head full 전용이다")
    return args


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, paths, args, device, oof_res="resampled"):
    """검증 이미지를 전체 추론해 (대회 산식 결과, 이미지별 확률맵)을 돌려준다.
    확률맵은 리샘플 해상도 그대로다 — OOF 앙상블은 같은 격자에서 평균내야 하기 때문이다."""
    preds, targets, probs, prs = [], [], {}, []
    for p in paths:
        xf, yf, meta = load_full(p, args.n_classes,
                              hint_thr=tuple(args.hint_thr) if getattr(args, "hint_thr", None) else None,
                              resample_scale=getattr(args, "resample_scale", 1.0),
                              ridge=getattr(args, "ridge", False),
                              norm=getattr(args, "norm", "anchor3"),
                              size_correction=not getattr(args, "no_size_correction", False))
        pr = predict_probs(model, xf, args.tile, 0.5, args.tta, device)
        pr = expand_prob(pr, getattr(args, "head", "full"))
        prs.append(pr)
        probs[p.name] = (restore_full_prob(pr, meta["height"], meta["width"])
                         if oof_res == "original" else pr)
        preds.append(pr.argmax(0))
        targets.append(yf)
    score = competition_score(preds, targets, args.n_classes)
    score["si_pair"] = si_pair_score(prs, targets) if args.n_classes == 4 else None
    return score, probs


def _list_paths(path):
    values = []
    for line in Path(path).read_text().splitlines():
        value = line.strip()
        if value:
            p = Path(value)
            values.append((ROOT / p).resolve() if not p.is_absolute() else p.resolve())
    return values


def _list_hash(path):
    normalized = "\n".join(line.strip() for line in Path(path).read_text().splitlines() if line.strip())
    return hashlib.sha256(normalized.encode()).hexdigest()


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save_probs(out, probs):
    for name, prob in probs.items():
        np.save(out / "oof_probs" / name.replace(".png", ".npy"), prob.astype(np.float16))


def step_loss(model, loss_fn, x, y, edge_aux_weight):
    """분할 손실과 선택적 경계 손실을 계산한다. 0이면 기존 경로 그대로다."""
    if edge_aux_weight <= 0:
        return loss_fn(model(x), y), {}
    seg_logits, edge_logits = model.forward_with_edge(x)
    seg = loss_fn(seg_logits, y)
    edge, valid = boundary_target(y, BOUNDARY_BAND_PX, n_classes=seg_logits.shape[1])
    edge_loss = edge_bce_loss(edge_logits, edge, valid)
    return seg + edge_aux_weight * edge_loss, {"seg": seg.item(), "edge": edge_loss.item()}


def run_fold(args):
    seed_all(args.seed)
    device = pick_device()
    list_mode = args.train_list is not None
    if list_mode:
        tr, va = _list_paths(args.train_list), _list_paths(args.val_list)
        overlap = sorted({p.name for p in tr} & {p.name for p in va})
        if overlap:
            raise ValueError(f"학습·검증 목록이 겹친다: {', '.join(overlap)}")
        missing_masks = [str(p) for p in tr + va if not io.mask_path_for(p).exists()]
        if missing_masks:
            raise FileNotFoundError(f"마스크가 없다: {missing_masks[0]}")
    else:
        paths = io.list_split("train")
        if args.limit:
            paths = paths[: args.limit]
        folds = make_folds([p.name for p in paths], args.n_folds, args.seed)
        tr = [p for p, f in zip(paths, folds, strict=True) if f != args.fold]
        va = [p for p, f in zip(paths, folds, strict=True) if f == args.fold]
    if not tr or not va:
        raise ValueError(f"fold {args.fold}: 학습 {len(tr)}장 · 검증 {len(va)}장 — 분할이 비었다")
    out = Path(args.runs_dir) / args.exp / f"fold{args.fold}"
    runs_root = Path(args.runs_dir).resolve()
    try:
        out.resolve().relative_to(runs_root)
    except ValueError as exc:
        raise ValueError(f"출력 경로가 runs-dir 밖이다: {out}") from exc
    (out / "oof_probs").mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex if list_mode else None
    if list_mode:
        for file in (out / "oof_probs").glob("*.npy"):
            file.unlink()
        for name in ("result.json", "args.json", "COMPLETE.json", "best.pt"):
            (out / name).unlink(missing_ok=True)
        extra = {"train_list_sha256": _list_hash(args.train_list),
                 "val_list_sha256": _list_hash(args.val_list), "n_train": len(tr),
                 "n_val": len(va), "run_id": run_id}
        for key, value in extra.items():
            setattr(args, key, value)

    # 임계값은 체크포인트 args에 남아 predict가 학습과 같은 힌트를 만들 수 있게 한다
    hint_thr = Hn.fit_thresholds_cached(args.runs_dir) if args.hint else None
    args.hint_thr = list(hint_thr) if hint_thr else None
    ds = SegDataset(tr, args.tile, train=True, n_classes=args.n_classes, samples_per_image=args.spi,
                    hint_thr=hint_thr, resample_scale=args.resample_scale, ridge=args.ridge,
                    anchor_jitter=args.anchor_jitter, interp_geo=not args.no_interp_geo,
                    norm=args.norm, size_correction=not args.no_size_correction,
                    photometric=args.photometric)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=0, drop_last=True)
    if len(dl) == 0:
        raise ValueError(f"배치가 만들어지지 않는다(표본 {len(ds)} < bs {args.bs})")
    model = build_model(args.encoder, args.weights, output_channels(args.n_classes, args.head), args.arch,
                        in_channels=Hn.input_channels(args.hint, args.ridge))
    if args.edge_aux_weight > 0:
        model = attach_edge_head(model, args.seed)
    model = model.to(device)
    cw = [float(v) for v in args.class_weights.split(",")] if args.class_weights else None
    loss_fn = build_loss(args.loss, args.n_classes, cw, args.ignore_band,
                         loss_scope=args.loss_scope, head=args.head)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(1, args.epochs * len(dl)))

    best = {"best_val_miou3": -1.0, "epoch": -1}
    with open(out / "log.jsonl", "w") as log:
        for ep in range(args.epochs):
            model.train()
            tl = []
            seg_l, edge_l = [], []
            for x, y in dl:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss, parts = step_loss(model, loss_fn, x, y, args.edge_aux_weight)
                loss.backward()
                opt.step()
                sched.step()
                tl.append(loss.item())
                if args.edge_aux_weight > 0:
                    seg_l.append(parts["seg"])
                    edge_l.append(parts["edge"])
            rec = {"epoch": ep, "train_loss": float(np.mean(tl))}
            if args.edge_aux_weight > 0:
                rec.update({"train_seg_loss": float(np.mean(seg_l)),
                            "train_edge_loss": float(np.mean(edge_l))})
            if args.select_epoch == "best":
                s, probs = evaluate(model, va, args, device, oof_res=args.oof_res)
                rec.update({"val_miou3": s["miou"], "per_class": s["per_class_mean"]})
                if args.select_metric == "si_pair":
                    if s["si_pair"] is None:
                        raise ValueError("선택 기준을 계산할 Si 픽셀이 검증 fold에 없다")
                    # si2의 4클래스 mIoU는 전개 확률 argmax가 항상 2·3이라 비Si가 오분류되고
                    # 클래스 0·1 IoU가 0인 참고값이다. 에폭 선택에는 쓰지 않는다.
                    rec["val_si_pair"] = s["si_pair"]
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log.flush()
            print(rec)
            if args.select_epoch == "best":
                selected = s["miou"] if args.select_metric == "miou" else s["si_pair"]
                previous = (best["best_val_miou3"] if args.select_metric == "miou"
                            else best.get("best_val_si_pair", -1.0))
            if args.select_epoch == "best" and selected > previous:
                best = {"best_val_miou3": s["miou"], "epoch": ep}
                if args.select_metric == "si_pair":
                    best.update({"best_val_si_pair": s["si_pair"], "select_metric": "si_pair"})
                if args.head == "si2":
                    best["head"] = "si2"
                # args는 argparse Namespace라 그대로는 직렬화되지 않는다. vars()로 평범한 dict로 바꾼다.
                torch.save({"model": checkpoint_state_dict(model), "args": vars(args)}, out / "best.pt")
                for n, pr in probs.items():
                    # float16으로 저장해 용량을 반으로 줄인다(확률값 정밀도는 앙상블에 충분하다).
                    np.save(out / "oof_probs" / n.replace(".png", ".npy"), pr.astype(np.float16))
        if args.select_epoch == "last":
            s, probs = evaluate(model, va, args, device, oof_res=args.oof_res)
            if args.select_metric == "si_pair" and s["si_pair"] is None:
                raise ValueError("선택 기준을 계산할 Si 픽셀이 검증 fold에 없다")
            best = {"best_val_miou3": s["miou"], "epoch": args.epochs - 1,
                    "select_epoch": "last", "n_train": len(tr), "n_val": len(va),
                    "n_oof_saved": len(probs), "run_id": run_id}
            if args.select_metric == "si_pair":
                best.update({"best_val_si_pair": s["si_pair"], "select_metric": "si_pair"})
            if args.head == "si2":
                best["head"] = "si2"
            torch.save({"model": checkpoint_state_dict(model), "args": vars(args)}, out / "best.pt")
            _save_probs(out, probs)
            rows = (out / "log.jsonl").read_text().splitlines()
            last = json.loads(rows[-1]); last.update({"val_miou3": s["miou"], "per_class": s["per_class_mean"]})
            if args.select_metric == "si_pair":
                last["val_si_pair"] = s["si_pair"]
            (out / "log.jsonl").write_text("\n".join(rows[:-1] + [json.dumps(last, ensure_ascii=False)]) + "\n")
    if list_mode and args.select_epoch == "best":
        best.update({"select_epoch": "best", "run_id": run_id})
    (out / "result.json").write_text(json.dumps(best, ensure_ascii=False))
    if list_mode:
        (out / "args.json").write_text(json.dumps(vars(args), ensure_ascii=False))
        files = {p.name: _sha256(p) for p in sorted((out / "oof_probs").glob("*.npy"))}
        complete = {"run_id": run_id, "args_sha256": _sha256(out / "args.json"),
                    "result_sha256": _sha256(out / "result.json"), "n_files": len(files), "files": files}
        tmp = out / "COMPLETE.json.tmp"
        tmp.write_text(json.dumps(complete, ensure_ascii=False))
        os.replace(tmp, out / "COMPLETE.json")
    return best


if __name__ == "__main__":
    print(run_fold(parse_args()))
