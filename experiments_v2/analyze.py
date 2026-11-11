import csv, json, os, statistics
from pathlib import Path

RES = Path(r"E:/tac_r2s2r_repro/experiments_v2/results")
FIG = Path(r"E:/tac_r2s2r_repro/experiments_v2/figures"); FIG.mkdir(parents=True, exist_ok=True)
OUT = Path(r"E:/tac_r2s2r_repro/experiments_v2/analysis"); OUT.mkdir(parents=True, exist_ok=True)

def load(name):
    with open(RES/name, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

main = load("v2_main.csv")
diag = load("v2_diagnosis.csv")
sweep = load("v2_bar_sweep.csv")

def f(x):
    try: return float(x)
    except: return None

BUDGETS = [256,512,1024,2048,4096]
ORDER = ["source_only","uniform_dr","task_weighted","tac_consistency","tw_bar","tac_bar","tw_ema","tw_vr","tacpp"]
SHORT = {"source_only":"S","uniform_dr":"DR","task_weighted":"TW","tac_consistency":"TAC",
         "tw_bar":"TW-BAR","tac_bar":"TAC-BAR","tw_ema":"TW-FTI","tw_vr":"TW-VR","tacpp":"TAC++"}

def ci95(vals):
    m = statistics.mean(vals)
    if len(vals)<2: return m,0.0
    h = 2.262*statistics.stdev(vals)/(len(vals)**0.5)  # t(9) ~2.262
    return m,h

# ---- main table ----
lines = ["# TAC-R2S2R v2 — main results (10 seeds, mean ± 95% CI)", "",
         "Accuracy on the 8192-sample target proxy. Uncalibrated probabilities.", ""]
lines.append("| Budget | " + " | ".join(SHORT[m] for m in ORDER) + " |")
lines.append("|---:|" + "|".join(["---:"]*len(ORDER)) + "|")
table = {}
for b in BUDGETS:
    row = [f"**{b}**"]
    for m in ORDER:
        g = [f(r["acc"]) for r in main if int(r["budget"])==b and r["method"]==m]
        if not g: row.append("—"); continue
        mu,h = ci95(g); table[(b,m)]=mu*100
        row.append(f"{mu*100:.2f} ± {h*100:.2f}")
    lines.append("| " + " | ".join(row) + " |")
(OUT/"table_main.md").write_text("\n".join(lines)+"\n", encoding="utf-8")

# ---- calibration ----
cal_rows=[]
for b in BUDGETS:
    g=[r for r in main if int(r["budget"])==b and r["method"]=="tacpp"]
    if not g: continue
    hce=[f(r["hce"]) for r in g]; hcec=[f(r["hce_cal"]) for r in g]
    ece=[f(r["ece"]) for r in g]; ecec=[f(r["ece_cal"]) for r in g]
    brier=[f(r["brier"]) for r in g]; brierc=[f(r["brier_cal"]) for r in g]
    T=[f(r["temperature"]) for r in g if r.get("temperature")]
    cal_rows.append(dict(budget=b, hce=statistics.mean(hce)*100, hce_cal=statistics.mean(hcec)*100,
                         ece=statistics.mean(ece)*100, ece_cal=statistics.mean(ecec)*100,
                         brier=statistics.mean(brier), brier_cal=statistics.mean(brierc),
                         T=statistics.mean(T) if T else 1.0))
call = ["# Calibration effect (TAC++ = TAC-BAR + post-hoc temperature scaling)", "",
        "| Budget | HCE% raw | HCE% cal | ECE% raw | ECE% cal | Brier raw | Brier cal | T |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|"]
for r in cal_rows:
    call.append(f"| {r['budget']} | {r['hce']:.2f} | {r['hce_cal']:.2f} | {r['ece']:.2f} | {r['ece_cal']:.2f} | "
                f"{r['brier']:.4f} | {r['brier_cal']:.4f} | {r['T']:.2f} |")
(OUT/"table_calibration.md").write_text("\n".join(call)+"\n", encoding="utf-8")

# ---- diagnosis: final-epoch stats per budget/base ----
def diag_final(base,b):
    rows=[r for r in diag if r["base"]==base and int(r["budget"])==b]
    if not rows: return None
    ep=max(int(r["epoch"]) for r in rows)
    fin=[r for r in rows if int(r["epoch"])==ep]
    g=lambda k: statistics.mean(f(r[k]) for r in fin)
    return dict(epoch=ep, target_acc=g("target_acc")*100, pred_entropy=g("pred_entropy"),
                rep_var=g("rep_var"), pen_over_bce=g("pen_over_bce"), pred_std=g("pred_std"))
dl=["# Collapse diagnosis — final-epoch instrumentation (target probe, 3 seeds)","",
    "| Budget | method | target acc% | pred entropy | rep var | pen/BCE |",
    "|---:|:---|---:|---:|---:|---:|"]
diag_tbl={}
for b in BUDGETS:
    for base in ["task_weighted","tac_consistency"]:
        s=diag_final(base,b)
        if not s: continue
        diag_tbl[(b,base)]=s
        dl.append(f"| {b} | {SHORT[base]} | {s['target_acc']:.2f} | {s['pred_entropy']:.4f} | {s['rep_var']:.4f} | {s['pen_over_bce']:.3f} |")
(OUT/"table_diagnosis.md").write_text("\n".join(dl)+"\n", encoding="utf-8")

# ---- bar sweep ----
sw=["# BAR alpha sweep (5 seeds)","",
    "| base | alpha | 1024 | 2048 | 4096 |","|---|---:|---:|---:|---:|"]
alphas=sorted({r["alpha"] for r in sweep}, key=lambda x: float(x))
for base in ["task_weighted","tac_consistency"]:
    for a in alphas:
        cells=[]
        for b in [1024,2048,4096]:
            g=[f(r["acc"]) for r in sweep if r["base"]==base and r["alpha"]==a and int(r["budget"])==b]
            cells.append(f"{statistics.mean(g)*100:.2f}" if g else "—")
        sw.append(f"| {SHORT[base]} | {a} | " + " | ".join(cells) + " |")
(OUT/"table_bar_sweep.md").write_text("\n".join(sw)+"\n", encoding="utf-8")

# ---- figures ----
figs=[]
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"figure.dpi":140})
    colors={"source_only":"#888","uniform_dr":"#b5651d","task_weighted":"#d62728",
            "tac_consistency":"#2ca02c","tw_bar":"#ff7f0e","tac_bar":"#17becf",
            "tw_ema":"#1f77b4","tw_vr":"#9467bd"}
    # Fig 1: accuracy vs budget (collapse curve)
    fig,ax=plt.subplots(figsize=(7.2,4.2))
    for m in ["source_only","uniform_dr","task_weighted","tw_bar","tw_vr","tac_consistency","tw_ema"]:
        ys=[table.get((b,m)) for b in BUDGETS]
        if all(y is None for y in ys): continue
        ax.plot(BUDGETS,[y if y is not None else float("nan") for y in ys],
                marker="o",ms=4,lw=1.8,color=colors.get(m,"#333"),label=SHORT[m])
    ax.axhline(50,ls=":",c="#bbb",lw=1)
    ax.set_xscale("log",base=2); ax.set_xticks(BUDGETS); ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.set_xlabel("synthetic budget (number of labelled source samples)")
    ax.set_ylabel("target accuracy (%)")
    ax.set_title("Budget-induced collapse of task-weighted consistency\nand recovery by frozen-teacher invariance (10 seeds)")
    ax.legend(ncol=4,fontsize=8,frameon=False,loc="lower left"); ax.grid(alpha=.25)
    fig.tight_layout(); p=FIG/"fig1_collapse_curve.png"; fig.savefig(p); plt.close(fig); figs.append(p)

    # Fig 2: diagnosis at budget 4096 (entropy + rep var vs epoch)
    fig,axes=plt.subplots(1,2,figsize=(9,3.6))
    for base,c in [("task_weighted","#d62728"),("tac_consistency","#2ca02c")]:
        rows=[r for r in diag if r["base"]==base and int(r["budget"])==4096]
        if not rows: continue
        eps=sorted({int(r["epoch"]) for r in rows})
        ent=[statistics.mean(f(r["pred_entropy"]) for r in rows if int(r["epoch"])==e) for e in eps]
        rv=[statistics.mean(f(r["rep_var"]) for r in rows if int(r["epoch"])==e) for e in eps]
        ta=[statistics.mean(f(r["target_acc"]) for r in rows if int(r["epoch"])==e)*100 for e in eps]
        axes[0].plot(eps,ent,color=c,lw=1.8,label=SHORT[base]+" entropy")
        axes[0].plot(eps,rv,color=c,lw=1.4,ls="--",label=SHORT[base]+" rep-var")
        axes[1].plot(eps,ta,color=c,lw=1.9,label=SHORT[base])
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("target probe"); axes[0].set_title("Entropy / representation variance")
    axes[0].legend(fontsize=7,frameon=False); axes[0].grid(alpha=.25)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("target acc (%)"); axes[1].set_title("Target accuracy during training (budget 4096)")
    axes[1].legend(fontsize=8,frameon=False); axes[1].grid(alpha=.25)
    fig.tight_layout(); p=FIG/"fig2_diagnosis.png"; fig.savefig(p); plt.close(fig); figs.append(p)

    # Fig 3: calibration HCE before/after
    fig,ax=plt.subplots(figsize=(6.4,3.6))
    x=range(len(cal_rows)); w=0.38
    ax.bar([i-w/2 for i in x],[r["hce"] for r in cal_rows],w,label="HCE raw",color="#d62728")
    ax.bar([i+w/2 for i in x],[r["hce_cal"] for r in cal_rows],w,label="HCE calibrated",color="#2ca02c")
    ax.set_xticks(list(x)); ax.set_xticklabels([str(r["budget"]) for r in cal_rows])
    ax.set_xlabel("budget"); ax.set_ylabel("HCE (%)"); ax.set_title("Temperature scaling reduces confident-error rate")
    ax.legend(frameon=False); ax.grid(alpha=.25,axis="y")
    fig.tight_layout(); p=FIG/"fig3_calibration.png"; fig.savefig(p); plt.close(fig); figs.append(p)
    print("FIGURES_OK", [str(p) for p in figs])
except Exception as e:
    print("FIG_SKIP", repr(e))

# ---- headline json ----
head = dict(table={f"{b}|{m}":round(v,3) for (b,m),v in table.items()},
            diagnosis={f"{b}|{k}":v for (b,k),v in diag_tbl.items()},
            calibration=cal_rows, figures=[str(p) for p in figs])
(OUT/"v2_headline.json").write_text(json.dumps(head,indent=2), encoding="utf-8")
print("ANALYSIS_DONE")
for m in ["task_weighted","tw_bar","tw_ema","tac_consistency"]:
    print(m, [f"{table.get((b,m),float('nan')):.1f}" for b in BUDGETS])
