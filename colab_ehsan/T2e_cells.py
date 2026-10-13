# T2e follow-up prompt checks for MedCLIPSeg_Colab.ipynb (v0.1)
# Paste each "# %%" block into its own cell, right after the T2d cells.
# Run order (GPU runtime): Section 0 -> helpers cell -> T2d build cell (optional) -> T2e-1 -> T2e-2 -> T2e-3

# %%
# ==== T2e-1. Follow-up prompt edits: build the prompt files (BUSI test set) ====
# Needs: Section 0 + the helpers cell (evaluate, load_per_image, show_results, save_tables).
import re
BUS = globals().get("BUS", ["BUSI", "BUSBRA", "BUSUC", "BUID", "UDIAT"])   # load_per_image() needs it
P2E = "data/BUSI/Prompts_Folder/"
PAT = re.compile(r"^(One|Two|Three|Four|Five) (small|medium|large) ([\w-]+?)-shaped "
                 r"(benign|malignant|regular|irregular) (tumors?) at the (.+?) in the breast ultrasound image\.$")
NORMAL_TXT = "No tumor in the breast ultrasound image."
FULL_LESION_TXT = "One medium oval-shaped benign tumor at the center in the breast ultrasound image."
END = " in the breast ultrasound image."

def edit2(s, kind):
    s = str(s).strip()
    if s == NORMAL_TXT:                                   # normal images: only one edit touches them
        return FULL_LESION_TXT if kind == "normal_as_full_lesion" else s
    m = PAT.match(s)
    if m is None:                                         # fail loudly instead of silently keeping the prompt
        raise ValueError(f"unparsed prompt: {s!r}")
    cnt, size, shape, cls, noun, loc = m.groups()
    if kind.startswith("size_") and kind != "size_only":
        size = kind[len("size_"):]                        # size_small / size_medium / size_large
    elif kind == "negated_full":  cnt = "No"              # "No small oval-shaped ... tumor at ..."
    elif kind == "size_only":     return f"{cnt} {size} {noun}{END}"
    elif kind == "location_only": return f"{cnt} {noun} at the {loc}{END}"
    elif kind not in ("rebuild", "normal_as_full_lesion"):
        raise ValueError(kind)
    return f"{cnt} {size} {shape}-shaped {cls} {noun} at the {loc}{END}"

EDITS2 = ["size_small", "size_medium", "size_large",          # causal test: does the size word set mask extent?
          "negated_full",                                     # negation inside the full template
          "size_only", "location_only",                       # which attribute restores detection vs. generic
          "normal_as_full_lesion"]                            # normals get a detailed lesion prompt (controlled B1)

base2 = pd.read_excel(P2E + "Test_text_original.xlsx")
assert all(edit2(s, "rebuild") == str(s).strip() for s in base2.Description), "parser does not reproduce the originals"
for k in EDITS2:
    e = base2.copy(); e["Description"] = e.Description.map(lambda s: edit2(s, k))
    n_changed = (e.Description != base2.Description.str.strip()).sum()
    e.to_excel(P2E + f"Test_text_{k}.xlsx", index=False)
    print(f"{k:22s} {n_changed:2d}/78 prompts changed | e.g. {e.Description.iloc[0] if k != 'normal_as_full_lesion' else e.Description.iloc[1]}")

# %%
# ==== T2e-2. Run them (GPU, ~2.5 min each, ~18 min total) ====
for k in EDITS2:
    evaluate("BUSI", prompt=k, tag="T2e-edit")
show_results("T2e")
save_tables()

# %%
# ==== T2e-3. Analysis (CPU is fine; reads the per-image CSVs on Drive) ====
def _pi(prompt):
    try: return load_per_image("BUSI", prompt).set_index("Name")
    except FileNotFoundError: print(f"  (missing per-image results for '{prompt}' - run evaluate() for it)"); return None

orig = _pi("original")
lesions = orig.index[~orig.gt_empty.astype(bool)]
normals = orig.index[orig.gt_empty.astype(bool)]
stated = orig.loc[lesions, "prompt_text"].astype(str).str.extract(r"^\w+ (small|medium|large) ")[0]
SIZES = ["small", "medium", "large"]

# (a) Size sweep: same image, only the size word changes -------------------------------------------
size_conds = {"original": "original", "none": "no_size", "small": "size_small", "medium": "size_medium", "large": "size_large"}
st = {lab: t for lab, p in size_conds.items() if (t := _pi(p)) is not None}
ratio = pd.DataFrame({lab: t.loc[lesions, "pred_area"] / t.loc[lesions, "gt_area"] for lab, t in st.items()})
dsc = pd.DataFrame({lab: 100 * t.loc[lesions, "DSC"] for lab, t in st.items()})
ratio["true size"] = dsc["true size"] = stated
T2e_ratio = ratio.groupby("true size").median().reindex(SIZES).round(2)
T2e_ratio.insert(0, "n", stated.value_counts().reindex(SIZES))
T2e_dsc = dsc.groupby("true size").mean().reindex(SIZES).round(1)
print("Median predicted/true lesion area - rows: size word in the ORIGINAL prompt, cols: size word GIVEN"); display(T2e_ratio)
print("Mean lesion DSC (%)"); display(T2e_dsc)
if all(s in st for s in SIZES):
    a = pd.DataFrame({s: st[s].loc[lesions, "pred_area"] for s in SIZES})
    mono = ((a.small < a.medium) & (a.medium < a.large)).mean()
    print(f"Lesions whose predicted area grows small < medium < large: {100 * mono:.0f}%  "
          f"(median area ×{(a.large / a.small.clip(lower=1e-6)).median():.2f} from 'small' to 'large')")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for s in SIZES:
        ax.plot(SIZES, T2e_ratio.loc[s, SIZES], marker="o", label=f"true size: {s}")
    ax.axhline(1, color="gray", lw=0.8, ls="--"); ax.set_yscale("log")
    ax.set_xlabel("size word given in the prompt"); ax.set_ylabel("median predicted / true area")
    ax.set_title("Does the size word set the mask extent?", fontsize=10); ax.legend(fontsize=8); plt.tight_layout(); plt.show()

# (b) Presence: which wording makes lesions disappear? --------------------------------------------
pres_conds = ["original", "negated_full", "location_only", "size_only", "generic", "lesion_as_normal"]
pt = {p: t for p in pres_conds if (t := _pi(p)) is not None}
missed = {p: set(t.loc[lesions].query("pred_empty == True").index) for p, t in pt.items()}
T2e_presence = pd.DataFrame({p: {"example prompt": t.loc[lesions[0], "prompt_text"],
                                 "lesion DSC": round(100 * t.loc[lesions, "DSC"].mean(), 1),
                                 "lesions missed": len(missed[p]),
                                 "median pred/true area": round((t.loc[lesions, "pred_area"] / t.loc[lesions, "gt_area"]).median(), 2)}
                             for p, t in pt.items()}).T
print("Lesion images (65) under presence-related edits"); display(T2e_presence)
T2e_overlap = pd.DataFrame({a: {b: len(missed[a] & missed[b]) for b in missed} for a in missed})
print("Missed-lesion overlap (cell = number of lesions missed under BOTH; diagonal = missed under that edit)"); display(T2e_overlap)

# (c) Normals: does a detailed lesion prompt create a lesion? ---------------------------------------
norm_conds = ["original", "normal_as_lesion", "normal_as_full_lesion"]
nt = {p: t for p in norm_conds if (t := _pi(p)) is not None}
T2e_normals = pd.DataFrame({p: {"prompt given to normals": t.loc[normals[0], "prompt_text"],
                                "normals with a false positive": int((t.loc[normals, "pred_empty"] != True).sum()),
                                "mean predicted area (% image)": round(100 * t.loc[normals, "pred_area"].mean(), 2)}
                            for p, t in nt.items()}).T
print("Normal images (13)"); display(T2e_normals)

for name, t in {"T2e_size_area_ratio": T2e_ratio, "T2e_size_dsc": T2e_dsc, "T2e_presence": T2e_presence,
                "T2e_missed_overlap": T2e_overlap, "T2e_normals": T2e_normals}.items():
    TABLES[name] = t
save_tables()
