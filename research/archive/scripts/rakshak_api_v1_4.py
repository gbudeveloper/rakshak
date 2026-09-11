from __future__ import annotations
import sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import joblib, numpy as np, pandas as pd, torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

SCRIPT_DIR=Path(__file__).resolve().parent
ROOT=SCRIPT_DIR.parent
MODEL_FILE=ROOT/"experiments"/"rakshak_final_v1.2"/"rakshak_final_model.joblib"
PRED_FILE=ROOT/"experiments"/"industrial_safety_predictions_v1.3.csv"
POOL_FILE=ROOT/"data"/"annotations"/"industrial_safety_annotation_pool.csv"
EXTRACT_FILE=ROOT/"experiments"/"safety_information_extraction_v0.4"/"safety_extraction_flat.csv"
REVIEW_DIR=ROOT/"data"/"reviews"; DB_FILE=REVIEW_DIR/"rakshak_reviews.db"
MODEL_NAME="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
sys.path.insert(0,str(SCRIPT_DIR))
from train_rakshak_final_pipeline_v1_2 import build_pathway_features
from evidence_engine_v1_5 import analyze
from rakshak_reviewer_policy_v1_4 import policy

app=FastAPI(title="RAKSHAK / SIF-Insight API",version="1.4.0",
 description="SIF precursor triage, persistent review, and descriptive HSSE analytics.")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=False,
 allow_methods=["GET","POST","OPTIONS"],allow_headers=["*"])

class PredictRequest(BaseModel):
    report_id:str|None=None; description:str=Field(min_length=1)
class BatchItem(BaseModel):
    report_id:str|None=None; description:str=Field(min_length=1)
class BatchRequest(BaseModel):
    records:list[BatchItem]=Field(min_length=1,max_length=500)
class ReviewRequest(BaseModel):
    report_id:str=Field(min_length=1)
    decision:str=Field(pattern="^(CONFIRMED|REJECTED|NEEDS_MORE_INFO)$")
    reviewer:str=Field(default="demo-reviewer",min_length=1,max_length=120)
    note:str=Field(default="",max_length=4000)

_MODEL_PACKAGE=None; _MODEL=None; _ENCODER=None; _DEVICE=None
_EXTRACT_CACHE=None; _POOL_CACHE=None

def db():
    REVIEW_DIR.mkdir(parents=True,exist_ok=True)
    c=sqlite3.connect(DB_FILE); c.row_factory=sqlite3.Row; return c

def init_db():
    with db() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS reviews(
        id INTEGER PRIMARY KEY AUTOINCREMENT,report_id TEXT NOT NULL,decision TEXT NOT NULL,
        reviewer TEXT NOT NULL,note TEXT NOT NULL DEFAULT '',created_at TEXT NOT NULL)""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_reviews_report ON reviews(report_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_reviews_created ON reviews(created_at)")
        c.commit()

@app.on_event("startup")
def startup(): init_db()

def load_resources():
    global _MODEL_PACKAGE,_MODEL,_ENCODER,_DEVICE
    if _MODEL_PACKAGE is None:
        if not MODEL_FILE.exists(): raise RuntimeError(f"Frozen model not found: {MODEL_FILE}")
        _MODEL_PACKAGE=joblib.load(MODEL_FILE); _MODEL=_MODEL_PACKAGE["model"]
        if getattr(_MODEL,"n_features_in_",None)!=422: raise RuntimeError("Frozen model schema mismatch.")
    if _ENCODER is None:
        _DEVICE="cuda" if torch.cuda.is_available() else "cpu"
        _ENCODER=SentenceTransformer(MODEL_NAME,device=_DEVICE); _ENCODER.max_seq_length=256
    return _MODEL_PACKAGE,_MODEL,_ENCODER,_DEVICE

def build_features(texts):
    pkg,model,enc,device=load_resources()
    emb=enc.encode(texts,batch_size=8,show_progress_bar=False,convert_to_numpy=True,normalize_embeddings=False,device=device).astype(np.float32)
    if emb.shape!=(len(texts),384): raise RuntimeError(f"Embedding schema mismatch: {emb.shape}")
    names=None; rows=[]
    for t in texts:
        d=build_pathway_features(t); n=list(d.keys())
        if names is None:names=n
        elif n!=names:raise RuntimeError("Pathway feature ordering changed.")
        rows.append([d[k] for k in names])
    path=np.asarray(rows,dtype=np.float32)
    full=np.hstack([emb,path]); keep=np.asarray(pkg["feature_keep_mask"],dtype=bool)
    if full.shape[1]!=422 or keep.size!=422:raise RuntimeError("422-feature schema mismatch.")
    X=full[:,keep]
    if getattr(model,"n_features_in_",None)!=X.shape[1]:raise RuntimeError("Model input mismatch.")
    return X

def lsr_candidates(text,e):
    t=text.lower(); hz=str(e.get("hazards","")); ex=str(e.get("exposures","")); pw=str(e.get("pathways",""))
    rules={"Bypassing Safety Controls":["bypass","override","disable","safety control","barrier"],
    "Confined Space":["confined space","tank","vessel"],"Driving":["vehicle","truck","driver","driving","collision","forklift","excavator","loader"],
    "Energy Isolation":["energized","electric","voltage","isolation","lockout","tagout","de-energized","stored energy"],
    "Hot Work":["welding","cutting","grinding","hot work","ignition","oxyfuel"],"Line of Fire":["line of fire","struck","projectile","dropped object","moving object","pressure release"],
    "Safe Mechanical Lifting":["lifting","lifted","suspended load","crane","rigging"],"Work Authorisation":["permit","authorization","authorisation"],
    "Working at Height":["height","fall","scaffold","ladder","tower","platform"]}
    out=[]
    for rule,terms in rules.items():
        matched=sorted({x for x in terms if x in t}); score=min(.60,.20*len(matched))
        if rule=="Driving" and ("vehicle" in hz or "vehicle_exposure" in ex or "vehicle_person_collision" in pw):score+=.4
        elif rule=="Energy Isolation" and ("electrical" in hz or "electrical_exposure" in ex or "electrical_contact" in pw):score+=.4
        elif rule=="Working at Height" and ("fall_height" in hz or "fall_exposure" in ex or "fall_from_height" in pw):score+=.4
        elif rule=="Line of Fire" and ("line_of_fire" in ex or "struck_by_projectile" in pw or "vehicle_person_collision" in pw):score+=.4
        elif rule=="Hot Work" and ("fire_explosion" in hz or "hot work" in t or "oxyfuel" in t):score+=.4
        elif rule=="Confined Space" and any(x in t for x in ["confined space","tank","vessel"]):score+=.4
        elif rule=="Safe Mechanical Lifting" and any(x in t for x in ["lifting","crane","suspended load"]):score+=.4
        elif rule=="Bypassing Safety Controls" and any(x in t for x in ["bypass","override","disable","barrier"]):score+=.4
        elif rule=="Work Authorisation" and any(x in t for x in ["permit","authorization","authorisation"]):score+=.4
        score=min(1.0,score)
        if score>0:out.append({"rule":rule,"candidate_score":round(score,4),"matched_terms":matched[:8]})
    return sorted(out,key=lambda x:(-x["candidate_score"],x["rule"]))[:3]

def load_meta():
    global _POOL_CACHE,_EXTRACT_CACHE
    if _POOL_CACHE is None:
        if not POOL_FILE.exists():raise RuntimeError(f"Annotation pool not found: {POOL_FILE}")
        pool=pd.read_csv(POOL_FILE); pool["report_id"]=pool["report_id"].astype(str)
        _POOL_CACHE=pool.drop_duplicates("report_id")
    if _EXTRACT_CACHE is None and EXTRACT_FILE.exists():
        ext=pd.read_csv(EXTRACT_FILE); ext["report_id"]=ext["report_id"].astype(str)
        _EXTRACT_CACHE=ext.drop_duplicates("report_id")
    return _POOL_CACHE,_EXTRACT_CACHE

def meta_df():
    pred=pd.read_csv(PRED_FILE); pred["report_id"]=pred["report_id"].astype(str)
    pool,ext=load_meta()
    meta_cols=[c for c in ["report_id","site","location","department","activity","asset","report_type","timestamp"] if c in pool.columns]
    m=pool[meta_cols].copy()
    if ext is not None:
        ec=[c for c in ["report_id","activities","activity_evidence","barriers","barrier_evidence","activity_count","barrier_count"] if c in ext.columns]
        m=m.merge(ext[ec].drop_duplicates("report_id"),on="report_id",how="left")
        if "activity" in m.columns:
            blank=m["activity"].isna()|m["activity"].astype(str).str.strip().eq("")
        else:
            blank=pd.Series(True,index=m.index)
        if "activities" in m.columns:
            m.loc[blank,"activity"]=m.loc[blank,"activities"]
    return pred.merge(m.drop_duplicates("report_id"),on="report_id",how="left",validate="one_to_one")

def tokens(s):
    vals=[]
    for x in s.fillna("").astype(str):
        vals += [v.strip() for v in x.replace("|",",").split(",") if v.strip() and v.strip().lower()!="nan"]
    return pd.Series(vals,dtype="string")

def top_counts(s,limit):
    c=tokens(s).value_counts().head(limit)
    return [{"category":str(k),"count":int(v)} for k,v in c.items()]

def density(df,col,limit):
    if col not in df.columns:return []
    x=df.copy();x[col]=x[col].fillna("").astype(str).str.strip();x=x[x[col]!=""]
    if not len(x):return []
    x["_yes"]=pd.to_numeric(x["sif_precursor_probability"],errors="coerce")>=.5
    g=x.groupby(col).agg(records=("report_id","count"),model_yes=("_yes","sum"),avg_probability=("sif_precursor_probability","mean")).reset_index()
    g["precursor_density"]=g.model_yes/g.records;g=g.sort_values(["precursor_density","records"],ascending=False).head(limit)
    return [{"category":str(r[col]),"records":int(r.records),"model_yes":int(r.model_yes),
             "precursor_density":round(float(r.precursor_density),4),"avg_probability":round(float(r.avg_probability),4)} for _,r in g.iterrows()]

def predict_records(records):
    texts=[r.description.strip() for r in records]; X=build_features(texts); probs=_MODEL.predict_proba(X)[:,1]
    out=[]
    for item,text,p in zip(records,texts,probs):
        e=analyze(text);p=float(p)
        pr,reason=policy({"sif_precursor_probability":p,"hazard_signal":e.get("hazard_signal",0),"exposure_signal":e.get("exposure_signal",0),"pathway_signal":e.get("pathway_signal",0),"complete_pathway":e.get("complete_pathway",0)})
        out.append({"report_id":item.report_id,"description":text,"sif_precursor_probability":round(p,6),"prediction_at_0_50":bool(p>=.5),
        "review":{"priority":pr,"reason":reason,"human_review_required":True},
        "evidence":{"hazards":e.get("hazards",""),"exposures":e.get("exposures",""),"pathways":e.get("pathways",""),"hazard_evidence":e.get("hazard_evidence",""),"exposure_evidence":e.get("exposure_evidence",""),"pathway_evidence":e.get("pathway_evidence",""),"hazard_signal":e.get("hazard_signal",0),"exposure_signal":e.get("exposure_signal",0),"pathway_signal":e.get("pathway_signal",0),"pathway_signal_strength":e.get("pathway_signal_strength",0),"complete_pathway":e.get("complete_pathway",0),"sif_context_status":e.get("sif_context_status","UNKNOWN"),"pathway_assessment":e.get("pathway_assessment","")},
        "lsr_candidates":lsr_candidates(text,e),"governance":{"model_input":"Narrative description only","autonomous_sif_decision":False,"requires_hsse_review":True}})
    return out

@app.get("/health")
def health():
    try:
        _,m,_,d=load_resources();return {"status":"ok","service":"RAKSHAK / SIF-Insight","api_version":app.version,"model_version":"rakshak_final_v1.2","review_policy":"v1.4","evidence_engine":"v1.5","device":d,"model_input_dimension":int(m.n_features_in_),"persistent_review_store":str(DB_FILE),"human_in_loop":True}
    except Exception as e:raise HTTPException(503,str(e))

@app.get("/model-info")
def model_info():
    try:
        pkg,m,_,d=load_resources();return {"system":"RAKSHAK / SIF-Insight","api_version":app.version,"model_version":"rakshak_final_v1.2","encoder":MODEL_NAME,"embedding_dimension":384,"pathway_features":38,"raw_features":422,"model_input_dimension":int(m.n_features_in_),"reference_threshold":float(pkg.get("reference_threshold",.5)),"review_policy":"v1.4","evidence_engine":"v1.5","device":d,"training_records":70,"locked_test_records":20,"autonomous_decision":False}
    except Exception as e:raise HTTPException(503,str(e))

@app.post("/predict")
def predict(req:PredictRequest):
    try:return predict_records([BatchItem(report_id=req.report_id,description=req.description)])[0]
    except Exception as e:raise HTTPException(500,str(e))
@app.post("/predict/batch")
def predict_batch(req:BatchRequest):
    try:return {"records":predict_records(req.records)}
    except Exception as e:raise HTTPException(500,str(e))

@app.post("/reviews")
def create_review(req:ReviewRequest):
    try:
        now=datetime.now(timezone.utc).isoformat()
        with db() as c:
            cur=c.execute("INSERT INTO reviews(report_id,decision,reviewer,note,created_at) VALUES(?,?,?,?,?)",(req.report_id,req.decision,req.reviewer,req.note,now));c.commit()
        return {"status":"saved","review_id":cur.lastrowid,"report_id":req.report_id,"decision":req.decision,"reviewer":req.reviewer,"created_at":now}
    except Exception as e:raise HTTPException(500,str(e))
@app.get("/reviews")
def reviews(report_id:str|None=None,limit:int=Query(100,ge=1,le=1000)):
    try:
        with db() as c:
            rows=c.execute("SELECT id,report_id,decision,reviewer,note,created_at FROM reviews WHERE report_id=? ORDER BY id DESC LIMIT ?",(report_id,limit)).fetchall() if report_id else c.execute("SELECT id,report_id,decision,reviewer,note,created_at FROM reviews ORDER BY id DESC LIMIT ?",(limit,)).fetchall()
        return {"reviews":[dict(x) for x in rows]}
    except Exception as e:raise HTTPException(500,str(e))
@app.get("/reviews/{report_id}")
def report_reviews(report_id:str):
    try:
        with db() as c:rows=c.execute("SELECT id,report_id,decision,reviewer,note,created_at FROM reviews WHERE report_id=? ORDER BY id DESC",(report_id,)).fetchall()
        return {"report_id":report_id,"reviews":[dict(x) for x in rows]}
    except Exception as e:raise HTTPException(500,str(e))
@app.get("/review-stats")
def review_stats():
    try:
        with db() as c:
            total=c.execute("SELECT COUNT(*) n FROM reviews").fetchone()["n"];dec=c.execute("SELECT decision,COUNT(*) n FROM reviews GROUP BY decision").fetchall()
        return {"total_reviews":int(total),"by_decision":[dict(x) for x in dec]}
    except Exception as e:raise HTTPException(500,str(e))

@app.get("/analytics/overview")
def analytics_overview():
    try:
        d=meta_df();p=pd.to_numeric(d.sif_precursor_probability,errors="coerce")
        return {"records":len(d),"model_yes_ge_0_50":int((p>=.5).sum()),"high_ge_0_75":int((p>=.75).sum()),"very_high_ge_0_90":int((p>=.9).sum()),"mean_probability":round(float(p.mean()),4),"median_probability":round(float(p.median()),4),"complete_pathways":int(pd.to_numeric(d.complete_pathway,errors="coerce").fillna(0).eq(1).sum()),"priority_distribution":d.review_priority.value_counts().reindex(["P1","P2","P3","P4","P5"],fill_value=0).to_dict(),"activity_metadata_available":bool("activity" in d.columns and d["activity"].fillna("").astype(str).str.strip().ne("").any()),"warning":"Model-predicted YES is a triage signal, not ground-truth SIF prevalence."}
    except Exception as e:raise HTTPException(500,str(e))
@app.get("/analytics/hazards")
def analytics_hazards(limit:int=Query(15,ge=1,le=50)):
    try:return {"items":top_counts(meta_df().hazards,limit)}
    except Exception as e:raise HTTPException(500,str(e))
@app.get("/analytics/exposures")
def analytics_exposures(limit:int=Query(15,ge=1,le=50)):
    try:return {"items":top_counts(meta_df().exposures,limit)}
    except Exception as e:raise HTTPException(500,str(e))
@app.get("/analytics/pathways")
def analytics_pathways(limit:int=Query(15,ge=1,le=50)):
    try:return {"items":top_counts(meta_df().pathways,limit)}
    except Exception as e:raise HTTPException(500,str(e))
@app.get("/analytics/lsr")
def analytics_lsr(limit:int=Query(15,ge=1,le=50)):
    try:
        d=meta_df();counts={}
        for t in d.description.fillna("").astype(str):
            for x in lsr_candidates(t,analyze(t)):counts[x["rule"]]=counts.get(x["rule"],0)+1
        return {"items":[{"category":k,"count":v} for k,v in sorted(counts.items(),key=lambda z:(-z[1],z[0]))[:limit]],"authoritative":False}
    except Exception as e:raise HTTPException(500,str(e))
@app.get("/analytics/activities")
def analytics_activities(limit:int=Query(15,ge=1,le=50)):
    try:
        d=meta_df()
        if "activity" not in d.columns or not d["activity"].fillna("").astype(str).str.strip().ne("").any():
            return {"items":[],"status":"NO_ACTIVITY_METADATA","message":"No activity metadata is available in the merged source for this dataset."}
        return {"items":density(d,"activity",limit),"status":"OK","source":"canonical metadata with extractor fallback"}
    except Exception as e:raise HTTPException(500,str(e))
@app.get("/analytics/locations")
def analytics_locations(limit:int=Query(15,ge=1,le=50)):
    try:return {"items":density(meta_df(),"location",limit)}
    except Exception as e:raise HTTPException(500,str(e))
@app.get("/analytics/sites")
def analytics_sites(limit:int=Query(15,ge=1,le=50)):
    try:return {"items":density(meta_df(),"site",limit)}
    except Exception as e:raise HTTPException(500,str(e))
@app.get("/analytics/departments")
def analytics_departments(limit:int=Query(15,ge=1,le=50)):
    try:return {"items":density(meta_df(),"department",limit)}
    except Exception as e:raise HTTPException(500,str(e))
