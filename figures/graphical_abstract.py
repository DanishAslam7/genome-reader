#!/usr/bin/env python3
"""
Graphical abstract — static, publication-ready.

Three honest beats, left to right:
  (1) input   — DNA read as physics: 7 biophysical parameters, no sequence
  (2) model   — the multitask CNN in 3D; gradient reversal -> species-invariant
  (3) result  — the real transfer-distance gradient + the mechanism finding

All numbers from RESULTS_LEDGER.md. 3D blocks use the interactive figure's projection.

  python figures/graphical_abstract.py  ->  figures/graphical_abstract.{png,pdf}
"""
import numpy as np, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, Polygon, Ellipse

BIO,ELEM,GRL,SEQ = "#2f6fd0","#e8930c","#d63a57","#12a05a"
INK,MUTED,RULE,PAPER,PANEL = "#1b1a24","#6b6577","#dfdce8","#ffffff","#f6f5fa"
HB="#c2418c"
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["DejaVu Sans"],
                     "pdf.fonttype":42,"ps.fonttype":42})

fig=plt.figure(figsize=(13.8,6.4),dpi=100); fig.patch.set_facecolor(PAPER)
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis("off")
ax.set_autoscale_on(False)

ax.text(3,94.5,"DNA biophysical structure is a universal, interpretable genomic code",
        fontsize=18.5,fontweight="bold",color=INK)
ax.text(3,89.6,"A multitask network predicts genomic elements from DNA mechanics alone — "
        "across 32 organisms and four eukaryotic kingdoms",fontsize=11.2,color=MUTED)

def zone(x,letter,title):
    ax.text(x,84,letter,fontsize=12.5,fontweight="bold",color=INK)
    ax.text(x+2.8,84,title,fontsize=11.5,fontweight="bold",color=INK,va="center")
def arrow(x1,x2,y,c=MUTED):
    ax.annotate("",xy=(x2,y),xytext=(x1,y),arrowprops=dict(arrowstyle="-|>",color=c,lw=2.2,
        mutation_scale=16,shrinkA=0,shrinkB=0))

# ================= ZONE A — input =================
zone(2,"1","Read DNA as physics")
xs=np.linspace(0,1,120); hx=3.5+xs*9.5
ax.plot(hx,72+2.4*np.sin(xs*9),color=BIO,lw=2.2)
ax.plot(hx,72+2.4*np.sin(xs*9+math.pi),color="#9bb8e6",lw=2.2)
for i in range(0,120,10):
    ax.plot([hx[i]]*2,[72+2.4*np.sin(xs[i]*9),72+2.4*np.sin(xs[i]*9+math.pi)],color=RULE,lw=1)
ax.text(16,72,"→",fontsize=16,color=MUTED,va="center",ha="center")
labs=["H-bond","stacking","solvation","base-pair","intra-bp","backbone","inter-bp"]
cols=[HB,HB,"#b8c0cc","#b8c0cc","#b8c0cc","#b8c0cc",BIO]
t=np.linspace(0,1,160)
for i,(lb,c) in enumerate(zip(labs,cols)):
    yy=66-i*3.0
    sig=(np.sin(t*(5+i)+i)*0.6+np.sin(t*(11+i))*0.35); sig=sig/(np.abs(sig).max()+1e-9)
    ax.plot(19+t*11,yy+sig*1.05,color=c,lw=1.5)
    ax.text(18.2,yy,lb,fontsize=7.0,color=MUTED,ha="right",va="center")
ax.text(2,40,"Seven mechanical parameters per position —\nno nucleotide sequence.",
        fontsize=10.4,color=INK,va="top",linespacing=1.5)
ax.text(2,33,"H-bonding and stacking are up-weighted by a\nfixed, disclosed prior (Sharma et al. 2025).",
        fontsize=8.6,color=MUTED,va="top",linespacing=1.5,style="italic")
arrow(31,34,55)

# ================= ZONE B — 3D model =================
zone(35,"2","The model, in three dimensions")
ay,axr=-0.6,-0.26
cA,sA,cB,sB=math.cos(ay),math.sin(ay),math.cos(axr),math.sin(axr)
def rot(x,y,z):
    x1=x*cA+z*sA; z1=-x*sA+z*cA; return (x1, y*cB - z1*sB, y*sB + z1*cB)
BX,BY,BS=48.5,54,0.72
def pj(v):
    p=1/(1-v[2]*0.02); return (BX+v[0]*BS*p, BY+v[1]*BS*p, v[2])
CSc=[(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]
FC=[([4,5,6,7],(0,0,1)),([0,3,2,1],(0,0,-1)),([1,2,6,5],(1,0,0)),
    ([0,4,7,3],(-1,0,0)),([3,7,6,2],(0,1,0)),([0,1,5,4],(0,-1,0))]
lm=math.sqrt(0.35**2+0.6**2+0.72**2); LI=(0.35/lm,0.6/lm,0.72/lm)
def h2r(h):h=h.lstrip('#');return tuple(int(h[i:i+2],16)/255 for i in(0,2,4))
def box(cx,cy,cz,hx,hy,hz,color,z=3):
    pc=[pj(rot(cx+c[0]*hx,cy+c[1]*hy,cz+c[2]*hz)) for c in CSc]; base=h2r(color); fl=[]
    for idx,n in FC:
        nr=rot(*n)
        if nr[2]<=0.02: continue
        dep=sum(pc[i][2] for i in idx)/4
        br=0.55+0.45*max(0,nr[0]*LI[0]+nr[1]*LI[1]+nr[2]*LI[2]); fl.append((dep,idx,br))
    for dep,idx,br in sorted(fl,key=lambda q:q[0]):
        ax.add_patch(Polygon([(pc[i][0],pc[i][1]) for i in idx],closed=True,
            facecolor=tuple(min(1,x*br) for x in base),edgecolor="white",lw=0.7,zorder=z))
    return pj(rot(cx,cy,cz))
blocks=[(-11,0,0,1.0,5.0,0.6,BIO,"profile"),(-7,0,0,1.1,5.0,2.9,BIO,""),
        (-3.5,0,0,1.1,3.7,2.9,BIO,""),(-0.3,0,0,1.1,2.5,2.9,BIO,""),(2.9,0,0,1.1,2.5,2.9,BIO,""),
        (6.8,0,0,1.7,1.7,1.7,"#3a3450","shared"),(10.8,0,0,2.1,2.1,2.1,ELEM,"element")]
cen={}
for i,(x,y,z,hx,hy,hz,c,nm) in enumerate(blocks):
    p=box(x,y,z,hx,hy,hz,c,z=3+i)
    if nm: cen[nm]=p
oc=box(9.5,5.2,3,1.2,1.2,1.2,GRL,z=12); kc=box(9.5,8.2,-2,1.0,1.0,1.0,GRL,z=12)
for hc in (oc,kc):
    ax.annotate("",xy=(cen["shared"][0],cen["shared"][1]),xytext=(hc[0],hc[1]),
        arrowprops=dict(arrowstyle="-|>",color=GRL,lw=1.5,mutation_scale=9,shrinkA=5,shrinkB=7))
ax.text(cen["profile"][0],cen["profile"][1]+6.2,"475 × 7\nprofile",fontsize=7.4,color=BIO,
        ha="center",va="bottom",fontweight="bold",linespacing=1.2)
ax.text(cen["element"][0]+1.5,cen["element"][1]-4.4,"element\nhead",fontsize=8,color=ELEM,
        ha="center",va="top",fontweight="bold",linespacing=1.2)
ax.text(kc[0]+2.6,kc[1]+0.6,"gradient-reversal\nadversaries",fontsize=7.2,color=GRL,
        ha="left",va="center",linespacing=1.2)
ax.text(35,39,"A ConvNeXt trunk (475→119) reads the profile\n"
        "into a shared representation. Gradient-reversal\n"
        "adversaries erase organism and kingdom identity,\n"
        "so the features transfer across species.",
        fontsize=9.5,color=INK,va="top",linespacing=1.55)
arrow(64,67,55)

# ================= ZONE C — results =================
zone(68,"3","Structure travels — and is interpretable")
cax=fig.add_axes([0.705,0.545,0.235,0.225]); cax.set_facecolor("white")
bio=[0.685,0.542,0.569,0.386]; seq=[0.696,0.547,0.569,0.321]; xs=np.arange(4)
cax.axhspan(0.18,0.25,color="#eeeef2",zorder=0)
cax.text(3.28,0.215,"chance",fontsize=6.4,color=MUTED,ha="right",va="center")
cax.plot(xs,bio,"-o",color=BIO,lw=2.4,ms=6,label="biophysics",zorder=3)
cax.plot(xs,seq,"--s",color=SEQ,lw=2.0,ms=5,label="+ sequence",zorder=3)
for i,b in enumerate(bio):
    cax.text(i,b+0.024,f"{b:.3f}",fontsize=6.4,color=BIO,ha="center",fontweight="bold")
cax.annotate("sequence\nhurts",xy=(3,0.321),xytext=(2.35,0.28),fontsize=7.2,color=SEQ,
    ha="center",arrowprops=dict(arrowstyle="-|>",color=SEQ,lw=1.3))
cax.set_xticks(xs); cax.set_xticklabels([]); cax.set_xlim(-0.3,3.35); cax.set_ylim(0.15,0.80)
cax.set_ylabel("element accuracy",fontsize=7.6); cax.tick_params(labelsize=6.6)
cax.legend(fontsize=7.2,loc="upper right",frameon=False,handlelength=1.6)
for s in ("top","right"): cax.spines[s].set_visible(False)
cax.set_title("increasing distance from training  →",fontsize=7.6,color=MUTED,loc="left",pad=3)

# ---- organism silhouettes along the transfer axis (test organism gets more distant) ----
XP=[72.43,78.87,85.31,91.75]; YICO=48.5; YLAB=44.0
def png(path,cx,cy,h,z=6):
    im=mpimg.imread(path); ar=im.shape[1]/im.shape[0]; w=h*ar*(6.4/13.8)
    ax.imshow(im,extent=[cx-w/2,cx+w/2,cy-h/2,cy+h/2],origin="upper",zorder=z,aspect="auto")
def diatom(cx,cy,z=6):   # pennate frustule — pointed lens with striae
    ax.add_patch(Ellipse((cx,cy),3.4,1.5,facecolor=INK,edgecolor="none",zorder=z))
    for dx in (-0.9,-0.3,0.3,0.9):
        ax.plot([cx+dx,cx+dx],[cy-0.5,cy+0.5],color="white",lw=0.8,zorder=z+1)
def bacterium(cx,cy,z=6): # bacillus capsule + flagella
    ax.add_patch(FancyBboxPatch((cx-1.5,cy-0.6),3.0,1.2,boxstyle="round,pad=0,rounding_size=0.6",
        facecolor=INK,edgecolor="none",zorder=z))
    tt=np.linspace(0,1,30)
    for off in (-0.35,0.35):
        ax.plot(cx+1.5+tt*1.7, cy+off+0.18*np.sin(tt*13),color=INK,lw=0.9,zorder=z)
# dotted connectors from each data point down to its icon
for i,xp in enumerate(XP):
    ax.plot([xp,xp],[YICO+2.2, 52.4],color=RULE,lw=0.8,ls=(0,(1,1.5)),zorder=2)
png("figures/phylopic/human.png",XP[0],YICO,7.0)
png("figures/phylopic/arabidopsis.png",XP[1],YICO,6.2)
diatom(XP[2],YICO); bacterium(XP[3],YICO)
for xp,lb in zip(XP,["in-distribution","held-out\nkingdom","unseen\ndiatom","prokaryote"]):
    ax.text(xp,YLAB,lb,fontsize=7.2,color=INK,ha="center",va="top",linespacing=1.2)

ax.text(68,37.5,"An entire kingdom held out, biophysics transfers as well as sequence "
        "(0.542 vs 0.547);\non organisms as distant as diatoms and prokaryotes it does better — "
        "adding\nsequence even hurts.",fontsize=9.2,color=INK,va="top",linespacing=1.5)
ax.add_patch(FancyBboxPatch((67.5,7.5),30,8.4,boxstyle="round,pad=0.3,rounding_size=1.2",
        fc=PANEL,ec=HB,lw=1.4))
ax.add_patch(plt.Circle((70.6,11.7),1.5,color=HB,zorder=4))
ax.text(70.6,11.7,"H",color="white",fontsize=10,ha="center",va="center",fontweight="bold",zorder=5)
ax.text(73.4,13.6,"Interpretable mechanism",fontsize=9.4,color=INK,fontweight="bold",va="center")
ax.text(73.4,10.2,"Promoter recognition depends on hydrogen bonding\n(−0.447 accuracy when that channel is removed).",
        fontsize=8.4,color=MUTED,va="center",linespacing=1.4)

ax.add_line(plt.Line2D([3,97],[4.8,4.8],color=RULE,lw=1))
ax.text(3,2.3,"32 organisms · 4 eukaryotic kingdoms · zero-shot to an unseen diatom lineage and to prokaryotes · "
        "biophysics-only model, no sequence",fontsize=8.6,color=MUTED)

fig.savefig("figures/graphical_abstract.png",dpi=300,facecolor=PAPER,bbox_inches="tight")
fig.savefig("figures/graphical_abstract.pdf",facecolor=PAPER,bbox_inches="tight")
print("wrote figures/graphical_abstract.{png,pdf}")
