import io
import json
import numpy as np
import pandas as pd
import streamlit as st
from ortools.sat.python import cp_model

# ---------------------------
# App & global settings
# ---------------------------
st.set_page_config(page_title="Dilan's WoWtimizer", layout="wide")

SCORE_SCALE = 10  # convert weighted score to integers for CP-SAT
DEFAULT_MAX_LOSS_PCT = 0.03
ONYXIA_URL = "https://static.wikia.nocookie.net/wowpedia/images/7/71/Lady_Onyxia.jpg"

ALL_STATS = [
    "Attack Power","Strength","Agility","Intellect","Spell Power",
    "Critical Strike Rating","Haste Rating","Armor Penetration",
    "Stamina","Hit Rating","Expertise Rating"
]

# ---------------------------
# WoW theme helpers
# ---------------------------
WOW_CLASS_COLORS = {
    "Warrior": "#C79C6E",
    "Paladin": "#F58CBA",
    "Hunter":  "#ABD473",
    "Rogue":   "#FFF569",
    "Priest":  "#FFFFFF",
    "Death Knight": "#C41F3B",
    "Shaman":  "#0070DE",
    "Mage":    "#40C7EB",
    "Warlock": "#8787ED",
    "Monk":    "#00FF96",
    "Druid":   "#FF7D0A",
    "Demon Hunter": "#A330C9",
}

def inject_wow_theme(primary="#C79C6E"):
    st.markdown(f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;800&family=Inter:wght@400;600&display=swap');

      :root {{
        --wow-primary: {primary};
        --wow-gold: #b08d57;
        --wow-ink: #f4e6c1;
        --wow-panel: rgba(20,16,12,0.65);
        --wow-panel-strong: rgba(20,16,12,0.85);
      }}

      .stApp {{
        background: radial-gradient(1200px 600px at 20% -10%, #242018 0%, #0e0b08 55%, #080705 100%);
        color: var(--wow-ink);
        font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      }}

      h1, h2, h3 {{
        font-family: 'Cinzel', serif !important;
        color: var(--wow-primary) !important;
        text-shadow: 0 0 6px rgba(0,0,0,.6);
        letter-spacing: .3px;
      }}

      /* Hero banner */
      .wow-hero {{
        border:1px solid var(--wow-gold);
        border-radius:14px;
        height:300px;
        padding:16px 18px;
        margin:8px 0 18px 0;
        display:flex;
        align-items:flex-end;
        background-image: linear-gradient(180deg, rgba(0,0,0,.55), rgba(0,0,0,.55)), url('{ONYXIA_URL}');
        background-size: cover;
        background-position: center;
      }}

      /* Panels & tables */
      div[data-testid="stExpander"] > details {{
        border: 1px solid var(--wow-gold);
        border-radius: 12px;
        background: var(--wow-panel);
      }}
      div[data-testid="stExpander"] summary {{
        color: var(--wow-ink) !important;
      }}
      div[data-testid="stDataFrame"] > div {{
        background: var(--wow-panel-strong);
        border: 1px solid var(--wow-gold);
        border-radius: 10px;
        padding: 4px;
      }}

      /* Buttons & downloads */
      .stButton>button, .stDownloadButton>button {{
        background: linear-gradient(180deg, var(--wow-primary), #3b2f1f);
        color: #fff;
        border: 1px solid var(--wow-gold);
        border-radius: 10px;
        box-shadow: 0 0 0 1px rgba(0,0,0,.5) inset, 0 1px 4px rgba(0,0,0,.4);
      }}

      /* KPI metrics */
      div[data-testid="stMetric"] {{
        background: var(--wow-panel);
        border: 1px solid var(--wow-gold);
        border-radius: 10px;
        padding: 10px;
      }}
    </style>
    """, unsafe_allow_html=True)

def wow_hero(title="Get yourself dragon-ready.", subtitle="You deserve that epic head."):
    st.markdown(
        f"""
        <div class="wow-hero">
          <div>
            <div style="font-family:'Cinzel',serif;font-size:34px;color:var(--wow-primary);line-height:1.1;">
              {title}
            </div>
            <div style="color:var(--wow-ink);opacity:.95;margin-top:4px">{subtitle}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------
# Helpers (data IO & scoring)
# ---------------------------
def read_items(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".json"):
        data = json.loads(file.getvalue().decode("utf-8"))
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(file)
    return df

def clean_items(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["slot", "Name"]:
        if col not in df.columns:
            raise ValueError(f"Items file is missing required column: {col}")
    for c in ALL_STATS:
        if c not in df.columns: df[c] = 0
    for c in ALL_STATS + ["Cost_num"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["slot"] = df["slot"].astype(str)
    df["Name"] = df["Name"].astype(str)
    return df

def read_weights(file) -> dict:
    wdf = pd.read_csv(file)
    if not {"Attribute","Value"}.issubset(wdf.columns):
        raise ValueError("Stat weights CSV must have columns: Attribute, Value")
    wdf["Value"] = pd.to_numeric(wdf["Value"], errors="coerce").fillna(0)
    return {r["Attribute"]: float(r["Value"]) for _, r in wdf.iterrows()}

def read_constraints(file) -> dict:
    cdf = pd.read_csv(file)
    if not {"Constraint","Minimum"}.issubset(cdf.columns):
        raise ValueError("Constraints CSV must have columns: Constraint, Minimum")
    cdf["Minimum"] = pd.to_numeric(cdf["Minimum"], errors="coerce").fillna(0)
    return {r["Constraint"]: float(r["Minimum"]) for _, r in cdf.iterrows()}

def compute_score_row(row, weights_keys, w):
    return float(sum(float(row[s]) * float(w[s]) for s in weights_keys))

def make_scored_tables(items_df, weights):
    obj_stats = [k for k in weights.keys() if k in items_df.columns]
    is_ring = items_df["slot"].str.lower().str.contains("ring")
    ring_df = items_df[is_ring].copy()
    nonring_df = items_df[~is_ring].copy()

    nonring_df["Score"] = nonring_df.apply(lambda r: compute_score_row(r, obj_stats, weights), axis=1)
    ring_groups = ring_df.groupby("Name", as_index=False).first()
    ring_groups["slot"] = "Ring"
    ring_groups["Score"] = ring_groups.apply(lambda r: compute_score_row(r, obj_stats, weights), axis=1)
    return nonring_df, ring_groups, obj_stats

def _slot_order(slots):
    return [*sorted([s for s in slots if "ring" not in s.lower()]), "Ring"]

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

# ---------------------------
# OR-Tools models (optimal & budget)
# ---------------------------
def solve_optimal(nonring_df, ring_groups, obj_stats, constraints_map):
    slots = sorted(nonring_df["slot"].unique().tolist())
    model = cp_model.CpModel()

    x_vars = {}
    for slot in slots:
        for idx, row in nonring_df[nonring_df["slot"] == slot].iterrows():
            x_vars[(slot, idx)] = model.NewBoolVar(f"x_{slot}_{idx}")
    y_vars = {}
    for j, row in ring_groups.iterrows():
        y_vars[j] = model.NewBoolVar(f"y_ring_{j}")

    def score_scaled_row(row):
        return int(round(compute_score_row(row, obj_stats, weights) * SCORE_SCALE))

    score_terms = []
    for (slot, idx), var in x_vars.items():
        score_terms.append(score_scaled_row(nonring_df.loc[idx]) * var)
    for j, var in y_vars.items():
        score_terms.append(score_scaled_row(ring_groups.loc[j]) * var)
    model.Maximize(sum(score_terms))

    for slot in slots:
        model.Add(sum(x_vars[(s,i)] for (s,i) in x_vars if s == slot) == 1)
    model.Add(sum(y_vars.values()) == 2)

    for stat, minimum in constraints_map.items():
        stat_terms = []
        for (slot, idx), var in x_vars.items():
            stat_terms.append(int(nonring_df.loc[idx, stat]) * var)
        for j, var in y_vars.items():
            stat_terms.append(int(ring_groups.loc[j, stat]) * var)
        model.Add(sum(stat_terms) >= int(minimum))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, None, status

    picked = []
    for (slot, idx), var in x_vars.items():
        if solver.Value(var):
            picked.append(nonring_df.loc[idx].copy())
    for j, var in y_vars.items():
        if solver.Value(var):
            r = ring_groups.loc[j].copy()
            picked.append(r)
    chosen_df = pd.DataFrame(picked)

    order = _slot_order(slots)
    chosen_df["__k"] = chosen_df["slot"].apply(lambda s: (order.index(s) if s in order else 999, s))
    chosen_df = chosen_df.sort_values("__k").drop(columns="__k")

    totals = {c: (chosen_df[c].sum() if c in chosen_df.columns else 0) for c in ALL_STATS}
    total_score = sum(totals.get(s,0) * weights.get(s,0) for s in obj_stats)
    total_price = float(chosen_df["Cost_num"].sum()) if "Cost_num" in chosen_df.columns else 0.0

    return chosen_df, {"totals": totals, "score": total_score, "price": total_price}, status

def solve_budget(nonring_df, ring_groups, obj_stats, constraints_map, score_target, cost_epsilon=0):
    slots = sorted(nonring_df["slot"].unique().tolist())

    def score_scaled_row(row):
        return int(round(compute_score_row(row, obj_stats, weights) * SCORE_SCALE))

    m1 = cp_model.CpModel()
    bx = {}
    for slot in slots:
        for idx, row in nonring_df[nonring_df["slot"] == slot].iterrows():
            bx[(slot, idx)] = m1.NewBoolVar(f"bx_{slot}_{idx}")
    by = {j: m1.NewBoolVar(f"by_{j}") for j, _ in ring_groups.iterrows()}

    cost_terms = []
    for (slot, idx), var in bx.items():
        cost = int(nonring_df.loc[idx, "Cost_num"]) if "Cost_num" in nonring_df.columns else 0
        cost_terms.append(cost * var)
    for j, var in by.items():
        cost = int(ring_groups.loc[j, "Cost_num"]) if "Cost_num" in ring_groups.columns else 0
        cost_terms.append(cost * var)

    score_terms = []
    for (slot, idx), var in bx.items():
        score_terms.append(score_scaled_row(nonring_df.loc[idx]) * var)
    for j, var in by.items():
        score_terms.append(score_scaled_row(ring_groups.loc[j]) * var)

    for slot in slots:
        m1.Add(sum(bx[(s,i)] for (s,i) in bx if s == slot) == 1)
    m1.Add(sum(by.values()) == 2)

    for stat, minimum in constraints_map.items():
        stat_terms = []
        for (slot, idx), var in bx.items():
            stat_terms.append(int(nonring_df.loc[idx, stat]) * var)
        for j, var in by.items():
            stat_terms.append(int(ring_groups.loc[j, stat]) * var)
        m1.Add(sum(stat_terms) >= int(minimum))

    m1.Add(sum(score_terms) >= int(round(score_target * SCORE_SCALE)))
    m1.Minimize(sum(cost_terms))
    s1 = cp_model.CpSolver()
    s1.parameters.max_time_in_seconds = 20
    st1 = s1.Solve(m1)
    if st1 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, None, st1

    min_cost = int(s1.ObjectiveValue())

    m2 = cp_model.CpModel()
    cx = {}
    for slot in slots:
        for idx, row in nonring_df[nonring_df["slot"] == slot].iterrows():
            cx[(slot, idx)] = m2.NewBoolVar(f"cx_{slot}_{idx}")
    cy = {j: m2.NewBoolVar(f"cy_{j}") for j, _ in ring_groups.iterrows()}

    cost_terms2 = []
    score_terms2 = []
    for (slot, idx), var in cx.items():
        cost = int(nonring_df.loc[idx, "Cost_num"]) if "Cost_num" in nonring_df.columns else 0
        cost_terms2.append(cost * var)
        score_terms2.append(score_scaled_row(nonring_df.loc[idx]) * var)
    for j, var in cy.items():
        cost = int(ring_groups.loc[j, "Cost_num"]) if "Cost_num" in ring_groups.columns else 0
        cost_terms2.append(cost * var)
        score_terms2.append(score_scaled_row(ring_groups.loc[j]) * var)

    for slot in slots:
        m2.Add(sum(cx[(s,i)] for (s,i) in cx if s == slot) == 1)
    m2.Add(sum(cy.values()) == 2)

    for stat, minimum in constraints_map.items():
        stat_terms = []
        for (slot, idx), var in cx.items():
            stat_terms.append(int(nonring_df.loc[idx, stat]) * var)
        for j, var in cy.items():
            stat_terms.append(int(ring_groups.loc[j, stat]) * var)
        m2.Add(sum(stat_terms) >= int(minimum))

    m2.Add(sum(score_terms2) >= int(round(score_target * SCORE_SCALE)))
    m2.Add(sum(cost_terms2) <= min_cost + cost_epsilon)
    m2.Maximize(sum(score_terms2))
    s2 = cp_model.CpSolver()
    s2.parameters.max_time_in_seconds = 20
    st2 = s2.Solve(m2)
    if st2 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, None, st2

    picked = []
    for (slot, idx), var in cx.items():
        if s2.Value(var):
            picked.append(nonring_df.loc[idx].copy())
    for j, var in cy.items():
        if s2.Value(var):
            r = ring_groups.loc[j].copy()
            picked.append(r)
    chosen_df = pd.DataFrame(picked)

    order = _slot_order(slots)
    chosen_df["__k"] = chosen_df["slot"].apply(lambda s: (order.index(s) if s in order else 999, s))
    chosen_df = chosen_df.sort_values("__k").drop(columns="__k")

    totals = {c: (chosen_df[c].sum() if c in chosen_df.columns else 0) for c in ALL_STATS}
    score = sum(totals.get(s,0) * weights.get(s,0) for s in obj_stats)
    price = float(chosen_df["Cost_num"].sum()) if "Cost_num" in chosen_df.columns else 0.0

    return chosen_df, {"totals": totals, "score": score, "price": price}, st2

# ---------------------------
# UI
# ---------------------------
st.title("🗡️ Dilan's WoWtimizer")

with st.sidebar:
    st.markdown("### ⚙️ Upload & Settings")
    f_items = st.file_uploader("Items (CSV or JSON)", type=["csv","json"])
    f_weights = st.file_uploader("Stat weights (CSV)", type=["csv"])
    f_constraints = st.file_uploader("Constraints (CSV)", type=["csv"])

    st.markdown("#### Theme")
    class_choice = st.selectbox("Theme color (WoW class)", list(WOW_CLASS_COLORS.keys()), index=3)

    max_loss = st.slider("Budget: allowed score loss (%)", 0.0, 20.0, DEFAULT_MAX_LOSS_PCT*100, 0.5) / 100.0
    run = st.button("Run optimization")

# Apply theme and fixed hero
inject_wow_theme(WOW_CLASS_COLORS[class_choice])
wow_hero()

if not run:
    st.info("Upload the three CSV files and click **Run optimization**.")
    st.stop()

# ---------------------------
# Run pipeline
# ---------------------------
try:
    items_df = clean_items(read_items(f_items))
    weights = read_weights(f_weights)
    constraints_map = read_constraints(f_constraints)
except Exception as e:
    st.error(str(e))
    st.stop()

nonring_df, ring_groups, obj_stats = make_scored_tables(items_df, weights)

st.subheader("All items by slot")
st.caption("Sorted by item score calculated based on stat weights you provided")
for slot in sorted(nonring_df["slot"].unique().tolist()):
    with st.expander(f"{slot}", expanded=False):
        df = nonring_df[nonring_df["slot"] == slot].copy()
        show_cols = ["slot","Name"] + [c for c in ALL_STATS if c in df.columns]
        if "Cost_num" in df.columns: show_cols += ["Cost_num"]
        show_cols += ["Score"]
        st.dataframe(
            df.sort_values("Score", ascending=False)[show_cols],
            use_container_width=True,
            column_config={
                "Cost_num": st.column_config.NumberColumn("Cost (g)", format="%d g"),
                "Score": st.column_config.NumberColumn("Score", format="%.1f"),
            }
        )

with st.expander("Rings (unique by Name)", expanded=False):
    rg = ring_groups.copy().sort_values("Score", ascending=False)
    ring_cols = ["slot","Name"] + [c for c in ALL_STATS if c in rg.columns]
    if "Cost_num" in rg.columns: ring_cols += ["Cost_num"]
    ring_cols += ["Score"]
    st.dataframe(
        rg[ring_cols],
        use_container_width=True,
        column_config={
            "Cost_num": st.column_config.NumberColumn("Cost (g)", format="%d g"),
            "Score": st.column_config.NumberColumn("Score", format="%.1f"),
        }
    )

# Optimal solve
chosen_df, meta, status = solve_optimal(nonring_df, ring_groups, obj_stats, constraints_map)
if chosen_df is None:
    st.error(f"Optimal solver status: {status}. No feasible solution.")
    st.stop()

best_cols = ["slot","Name"] + [c for c in ALL_STATS if c in chosen_df.columns]
if "Cost_num" in chosen_df.columns: best_cols += ["Cost_num"]
if "Score" in chosen_df.columns:   best_cols += ["Score"]

st.subheader("🏆🇧🇮🇸 List")
st.markdown("###### For true min-maxers with deep pockets.")
st.dataframe(
    chosen_df[best_cols],
    use_container_width=True,
    column_config={
        "Cost_num": st.column_config.NumberColumn("Cost (g)", format="%d g"),
        "Score": st.column_config.NumberColumn("Score", format="%.1f"),
    }
)
st.write(f"**Total price:** {meta['price']:.0f}g")
st.write(f"**Total weighted score:** {meta['score']:.2f}")

st.download_button("Download best set CSV", data=df_to_csv_bytes(chosen_df[best_cols]), file_name="best_item_set.csv")

# Budget solve
target_score = (1.0 - float(max_loss)) * float(meta["score"])
budget_df, budget_meta, bstatus = solve_budget(nonring_df, ring_groups, obj_stats, constraints_map, target_score)

if budget_df is None:
    st.error(f"Budget solver status: {bstatus}. Budget alternative not found.")
    st.stop()

bcols = ["slot","Name"] + [c for c in ALL_STATS if c in budget_df.columns]
if "Cost_num" in budget_df.columns: bcols += ["Cost_num"]
if "Score" in budget_df.columns:    bcols += ["Score"]

st.subheader(f"💸 Budget set (≤ {max_loss*100:.1f}% score loss)")
st.markdown("###### For petty penny-pinchers")
st.dataframe(
    budget_df[bcols],
    use_container_width=True,
    column_config={
        "Cost_num": st.column_config.NumberColumn("Cost (g)", format="%d g"),
        "Score": st.column_config.NumberColumn("Score", format="%.1f"),
    }
)
st.write(f"**Budget total price:** {budget_meta['price']:.0f}g")
st.write(f"**Budget total weighted score:** {budget_meta['score']:.2f}")

# Comparison
gold_saved = meta["price"] - budget_meta["price"]
score_drop_pct = 100.0 * max(meta["score"] - budget_meta["score"], 0) / meta["score"] if meta["score"] > 0 else 0.0
cost_saved_pct = 100.0 * max(gold_saved, 0) / meta["price"] if meta["price"] > 0 else 0.0

st.subheader("↔️ Comparison")
st.markdown("###### Know what you're trading off")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Optimal price", f"{meta['price']:.0f}g")
c2.metric("Budget price", f"{budget_meta['price']:.0f}g", f"-{cost_saved_pct:.2f}%")
c3.metric("Score drop", f"{score_drop_pct:.2f}%")
c4.metric("Gold saved", f"{max(gold_saved,0):.0f}g")

st.write(
    f"For **{max(gold_saved,0):.0f}g less (-{cost_saved_pct:.2f}%)**, "
    f"you can get this set that is only **{score_drop_pct:.2f}%** worse than the best option."
)

st.download_button("Download budget set CSV", data=df_to_csv_bytes(budget_df[bcols]), file_name="budget_item_set.csv")



