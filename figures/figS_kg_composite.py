#!/usr/bin/env python3
"""
Supplementary Figure — the biophysical knowledge graph (composite).

(A) the ACTUAL 32-organism graph, force-directed, organisms tinted by kingdom so
    the four clades cluster; (B) how each training window's context features are
    read WITHOUT the true element node (anti-leak); (C) how the 76-d vector fuses
    into the profile trunk.

Counts from knowledge_graph_reprofiled_32/kg_stats.json; branch shapes from
train_taxonomy_multitask.py (KGFeatureProvider; kg_branch) and the 'both' run config.

  python figures/figS_kg_composite.py  ->  figures/figS_kg_composite.{pdf,png}
"""
import json, numpy as np, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import networkx as nx, pandas as pd

KGDIR="knowledge_graph_reprofiled_32"
INK,MUTED,RULE,PANEL = "#1b1a24","#6b6577","#d6dae0","#f7f8fa"
RED,BIOc = "#c22", "#2f7fd0"
# 4 kingdom hues (organisms + their kingdom node share the hue)
KC={"animalia":"#d1495b","fungi":"#e08a1e","plantae":"#2e8b57","protista":"#6a4ca8"}
ELEMc,PARAMc,HOTc = "#2b7fd0","#12897b","#f2b8d4"
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["DejaVu Sans"],
                     "pdf.fonttype":42,"ps.fonttype":42})

nodes=pd.read_csv(f"{KGDIR}/kg_nodes.csv"); edges=pd.read_csv(f"{KGDIR}/kg_edges.csv")
org2king={str(r["source"]).split(":")[-1]: str(r["target"]).split(":")[-1]
          for _,r in edges[edges.edge_type=="BELONGS_TO"].iterrows()}  # organism -> kingdom
G=nx.Graph(); nt={}; nm={}
for _,r in nodes.iterrows():
    G.add_node(r["node_id"]); nt[r["node_id"]]=r["node_type"]; nm[r["node_id"]]=str(r["node_id"]).split(":")[-1]
for _,r in edges.iterrows():
    if r["source"] in nt and r["target"] in nt: G.add_edge(r["source"],r["target"])
pos=nx.spring_layout(G,k=0.42,iterations=170,seed=7)

def ncolor(nid):
    t=nt[nid]
    if t=="kingdom":  return KC.get(nm[nid],"#888")
    if t=="organism": return KC.get(org2king.get(nm[nid],""),"#888")
    return {"element":ELEMc,"parameter":PARAMc,"hotspot":HOTc}[t]

fig=plt.figure(figsize=(8.2,8.8)); fig.patch.set_facecolor("white")

# ========== PANEL A — force-directed render ==========
axA=fig.add_axes([0.02,0.40,0.96,0.55]); axA.axis("off")
hot_e=[]; back_e=[]
for u,v in G.edges():
    (hot_e if (nt[u]=="hotspot" or nt[v]=="hotspot") else back_e).append((u,v))
nx.draw_networkx_edges(G,pos,edgelist=hot_e,ax=axA,edge_color="#f0c3d9",width=0.22,alpha=0.32)
nx.draw_networkx_edges(G,pos,edgelist=back_e,ax=axA,edge_color="#b06",width=0.8,alpha=0.55)
SIZE={"kingdom":560,"organism":170,"element":300,"parameter":420,"hotspot":6}
for t in ["hotspot","organism","element","parameter","kingdom"]:
    nl=[n for n in G.nodes if nt[n]==t]
    nx.draw_networkx_nodes(G,pos,nodelist=nl,node_color=[ncolor(n) for n in nl],
        node_size=SIZE[t],edgecolors="white" if t!="hotspot" else "none",
        linewidths=0.6 if t!="hotspot" else 0,ax=axA,alpha=0.96 if t!="hotspot" else 0.65)
for n in G.nodes:
    if nt[n] in ("kingdom","parameter"):
        axA.text(pos[n][0],pos[n][1],nm[n],fontsize=5.6,ha="center",va="center",
                 color="white",fontweight="bold",zorder=6)
axA.set_xlim(min(p[0] for p in pos.values())-.08,max(p[0] for p in pos.values())+.08)
axA.set_ylim(min(p[1] for p in pos.values())-.08,max(p[1] for p in pos.values())+.08)
fig.text(0.02,0.975,"A",fontsize=13,fontweight="bold",color=INK)
fig.text(0.055,0.975,"The biophysical knowledge graph — 1,757 nodes, 7,265 edges",
         fontsize=11.5,fontweight="bold",color=INK)
fig.text(0.055,0.958,"the actual 32-organism graph, force-directed; organisms tinted by kingdom "
         "(clades cluster)",fontsize=8.4,color=MUTED)
# legend
h1=[Line2D([0],[0],marker="o",color="none",markerfacecolor=KC[k],markeredgecolor="white",
    markersize=8,label=k) for k in ["animalia","fungi","plantae","protista"]]
h2=[Line2D([0],[0],marker="o",color="none",markerfacecolor=c,markeredgecolor="white",markersize=ms,label=l)
    for c,ms,l in [(ELEMc,8,"element (14)"),(PARAMc,9,"parameter (7)"),(HOTc,5,"hotspot (1,700)")]]
leg=axA.legend(handles=h1+h2,loc="lower left",frameon=True,fontsize=7.2,labelspacing=.5,
    borderpad=.7,title="kingdoms (node + its organisms)  ·  other types",title_fontsize=7.4,ncol=1)
leg.get_frame().set_edgecolor(RULE)

# ========== PANELS B & C — schematic ==========
axS=fig.add_axes([0,0.015,1,0.365]); axS.set_xlim(0,100); axS.set_ylim(0,100); axS.axis("off")
def box(x,y,w,h,fc="white",ec=RULE,lw=1.0,z=2,r=1.4):
    axS.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0,rounding_size={r}",fc=fc,ec=ec,lw=lw,zorder=z))
def nodebox(cx,cy,w,h,label,sub,fc,ec,tc=INK,fs=8):
    box(cx-w/2,cy-h/2,w,h,fc=fc,ec=ec,lw=1.3,z=4,r=1.2)
    axS.text(cx,cy+(1.4 if sub else 0),label,ha="center",va="center",fontsize=fs,fontweight="bold",color=tc,zorder=5)
    if sub: axS.text(cx,cy-2.3,sub,ha="center",va="center",fontsize=6.4,color=MUTED,zorder=5)
def flow(p1,p2,c=MUTED,lw=1.2,ls="-"):
    axS.annotate("",xy=p2,xytext=p1,arrowprops=dict(arrowstyle="-|>",color=c,lw=lw,ls=ls,mutation_scale=8,shrinkA=5,shrinkB=5))
def panel(x,w,y,h,letter,title,ec=RULE):
    box(x,y,w,h,fc=PANEL,ec=ec,lw=1.0,r=1.4)
    axS.text(x+1.8,y+h-4.0,letter,fontsize=12.5,fontweight="bold",color=INK,va="center")
    axS.text(x+6.0,y+h-4.0,title,fontsize=9.2,fontweight="bold",color=INK,va="center")

# ---- B (left) : anti-leak context extraction ----
panel(1.5,55,5,90,"B","Per-window features — context nodes only (anti-leak)",ec=RED)
axS.text(4.5,84,"one training window:",fontsize=7.8,fontweight="bold",color=INK)
nodebox(17,74,22,8,"kingdom node","used","#efe9f6",KC["protista"],fs=7.4)
nodebox(17,60,22,8,"organism node","used","#efe9f6",KC["animalia"],fs=7.4)
nodebox(17,45,24,8,"true element node","EXCLUDED","#fdeceb",RED,tc=RED,fs=7.4)
axS.plot([5,29],[48,42],color=RED,lw=1.7,zorder=6)
axS.text(17,35,"including it would leak the label",fontsize=6.6,color=RED,ha="center")
box(32,44,23,36,fc="white",ec=RULE,lw=1,r=1)
axS.text(33.5,75,"per node, keep:",fontsize=7.6,fontweight="bold",color=INK)
axS.text(33.5,69,"• 10 structural",fontsize=7.0,color=INK)
axS.text(33.5,62.5,"• 28 profile summary",fontsize=7.0,color=INK)
axS.text(33.5,58,"  (mean/std/peak × 7 params)",fontsize=6.3,color=MUTED)
axS.text(33.5,51,"drop identity indices",fontsize=6.6,color=RED)
box(32,22,23,7,fc="#fbe9f3",ec=PARAMc,lw=1.3,r=1)
axS.text(43.5,25.5,"kg_features = 76-d",ha="center",va="center",fontsize=7.8,fontweight="bold",color="#a01b3a")
flow((28,60),(32,62),PARAMc,1.2)
flow((43.5,44),(43.5,29.5),PARAMc,1.2)

# ---- C (right) : fusion ----
panel(58.5,40,5,90,"C","Fusion into the trunk",ec=PARAMc)
chain=[(65,76,"kg_features","76","#fbe9f3"),(78,76,"LayerNorm","",  "white"),(91,76,"Dense","96","white")]
for i,(cx,cy,lab,sub,fc) in enumerate(chain):
    nodebox(cx,cy,12,8,lab,sub,fc,PARAMc,tc=("#a01b3a" if fc!="white" else INK),fs=7.0)
    if i: flow((chain[i-1][0]+6,76),(cx-6,76),PARAMc,1.1)
nodebox(72,60,12,8,"Dropout","","white",PARAMc,fs=7.0)
nodebox(85,60,12,8,"Dense","48","white",PARAMc,fs=7.0)
flow((91,72),(85,64),PARAMc,1.1); flow((78,60),(79,60),PARAMc,1.1)
nodebox(72,42,22,8,"profile trunk","256-d","#eaf2fc",BIOc,tc=BIOc,fs=7.4)
nodebox(92,50,8,8,"⊕","","white",INK,fs=10)
flow((85,56),(90,52),PARAMc,1.1); flow((83,44),(89,48),BIOc,1.1)
axS.text(78,30,"→ Dense 192 → shared representation → heads",ha="center",va="center",
         fontsize=7.0,color=MUTED)

fig.text(0.5,0.006,"KG branch active only in-distribution — withheld from every cross-kingdom "
         "transfer result (cohort-derived features leak across a held-out kingdom).",
         ha="center",fontsize=7.6,color=MUTED,style="italic")

fig.savefig("figures/figS_kg_composite.pdf")
fig.savefig("figures/figS_kg_composite.png",dpi=200)
print("wrote figures/figS_kg_composite.{pdf,png}")
