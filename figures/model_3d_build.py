import json

def block(sp, ch, hx=1.5): return dict(hx=hx, hy=round(1.0+sp/475*8,2), hz=round(0.6+ch/256*4.2,2))
def vec(dim, hx=1.6):
    s=round(0.8+dim/200*2.2,2); return dict(hx=hx, hy=s, hz=s)
def head(nc, big=False):
    s=round((1.5 if big else 1.0)+nc/32*1.6,2); return dict(hx=(1.7 if big else 1.1), hy=s, hz=s)

L=[]
def add(id,name,x,y,z,group,geom,shape="",params="",role="",why="",label=""):
    d=dict(id=id,name=name,x=x,y=y,z=z,group=group,shape=shape,params=params,role=role,why=why,label=label)
    d.update(geom); L.append(d)

add("profile","Biophysical profile",-42,0,0,"bio",block(475,7),"475 × 7","","input · 7 mechanical parameters",
    "The only input to the reported model: seven DNA biophysical/mechanical parameters per position "
    "(hydrogen-bonding, stacking, solvation, backbone and base-pair geometry) — with no nucleotide identity. "
    "The network reads DNA as a physical object, not a string of letters.","profile")
add("scale","ParameterScale",-35,0,0,"prior",block(475,7,hx=0.6),"× fixed weights","0 (non-trainable)",
    "fixed prior channel weighting",
    "A non-trainable per-channel multiplier (hbond, stack ×2.0; four channels ×1.4; inter ×1.0) taken from a "
    "prior molecular-dynamics study and fixed before any training. Disclosed so the model is never credited "
    "with 'discovering' a weighting it was handed.","× scale")
add("stem","Stem convolution",-28,0,0,"bio",block(475,256),"k=7 → 256","12,800","lift 7 → 256 channels","","stem")
add("stage1","ConvNeXt stage 1",-20,0,0,"bio",block(238,256),"475 → 238","","downsample ×2",
    "Three ConvNeXt-style stages, one block each: depthwise k=7 conv → LayerNorm → Dense 1024 (4× expand) "
    "→ Dense 256, with a residual connection. Stride-2 downsampling halves the position axis at each stage.","stage 1")
add("stage2","ConvNeXt stage 2",-12,0,0,"bio",block(119,256),"238 → 119","","downsample ×2","","stage 2")
add("stage3","ConvNeXt stage 3",-4,0,0,"bio",block(119,256),"119 × 256","trunk ≈ 1.79 M","final trunk stage","","stage 3")
add("pool","Pool → dense",6,0,0,"bio",vec(128),"→ 128","98,688","collapse positions","","pool")
add("fuse","Fuse",14,0,0,"bio",vec(176),"⊕ → 176","62,288","concat trunk + derived branches",
    "The pooled trunk is concatenated with the two always-on derived branches to form the 176-dimensional "
    "shared vector.","fuse")
add("shared","Shared representation",22,0,0,"shared",vec(176),"176-dim","","read by all nine heads",
    "One vector feeds every head. Because two heads are adversarial (right), the trunk is pushed to make this "
    "vector predict elements while NOT revealing which organism or kingdom a window came from — the origin of "
    "cross-species transfer.","SHARED")

add("complexity","Profile-complexity branch",5,-12,6,"bio",vec(48),"14 stats/ch → 48","","derived · always on","")
add("startstop","Start/stop-contrast branch",5,-17,-5,"bio",vec(32),"L/C/R windows → 32","","derived · always on","")
add("sequence","Sequence branch",0,12,7,"seq",block(501,4),"one-hot 501×4 → 96","+424,256","optional ablation arm",
    "Not in the reported model. Added, it costs +19.3% parameters and buys +1.08 pp element accuracy "
    "in-distribution — but 0.00 on an unseen diatom lineage and −6.5 pp on prokaryotes. Sequence's value "
    "decays with evolutionary distance and can turn negative; biophysics does not.","sequence")
add("kg","Knowledge-graph branch",5,17,-6,"kg",vec(48),"context nodes → 48","","off for all transfer",
    "Injects organism/kingdom context-node features. Cohort-derived, so it leaks across a held-out kingdom — "
    "switched OFF for every cross-kingdom and external result. Contributes +1.38 pp only in-distribution.","KG")

add("element","Element head",34,0,0,"elem",head(5,big=True),"5 classes","loss ×2.5","REPORTED — the target",
    "The only head reported in the paper: promoter, gene and exon boundaries, start and stop codons. It reads "
    "element identity from the species-invariant shared representation.","element")
add("organism","Organism head",31,8,6,"grl",head(32),"32 classes","loss ×1.0 · GRL λ=0.005","adversary (gradient reversal)",
    "Gradient reversal: on backprop its gradient is negated, so the trunk is trained to FAIL at telling "
    "organisms apart. This erases organism identity from the features — the mechanism behind cross-organism "
    "generalization. Red particles show the reversed signal flowing back.","organism")
add("kingdom","Kingdom head",31,12,-4,"grl",head(4),"4 classes","loss ×0.5 · GRL λ=0.08","adversary (gradient reversal)",
    "The same adversary at kingdom level, with a stronger λ=0.08. It is the direct engine of cross-KINGDOM "
    "universality — the paper's headline transfer result.","kingdom")
add("human","Human UTR/enhancer head",31,-7,6,"aux",head(3),"3 classes","loss ×1.0","auxiliary · masked to human","")
add("biogroup","Biological-group head",31,-11,-4,"aux",head(4),"4 classes","loss ×0.8","auxiliary","")
add("coding","Coding detector",31,-15,6,"aux",head(2),"2 classes","loss ×0.5","auxiliary","")
add("pair","Start/stop-pair head",31,16,6,"aux",head(2),"2 classes","loss ×0.4","auxiliary","")
add("coarse","Coarse-element head",31,-19,-5,"aux",head(7),"7 classes","loss ×0.3","auxiliary","")
add("utr","UTR-subtype head",31,20,-4,"aux",head(2),"2 classes","loss ×0.2","auxiliary","")

E=[]
def e(s,t,k="fwd"): E.append(dict(s=s,t=t,k=k))
spine=["profile","scale","stem","stage1","stage2","stage3","pool","fuse","shared"]
for a,b in zip(spine,spine[1:]): e(a,b)
for b in ["complexity","startstop","sequence","kg"]: e(b,"fuse")
for hh in ["element","human","biogroup","coding","pair","coarse","utr"]: e("shared",hh)
e("shared","organism","grl"); e("shared","kingdom","grl")

arch=dict(layers=L, edges=E)
json.dump(arch, open("figures/arch_3d.json","w"))
tpl=open("figures/model_3d_template.html").read()
data=json.dumps(arch, separators=(",",":"))
pilot=open("figures/pilot_samples.json").read().strip()
body=tpl.replace("__ARCHDATA__",data).replace("__PILOT__",pilot)
open("figures/model_3d_artifact.html","w").write(body)
doc=('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
'<title>Model Architecture — 3D</title>\n'
'<meta name="description" content="Interactive 3D model of the multitask convolutional architecture.">\n'
'</head>\n<body>\n'+body+'\n</body>\n</html>\n')
open("figures/model_3d.html","w").write(doc)
print("layers",len(L),"edges",len(E),"— rebuilt")
