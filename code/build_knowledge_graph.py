#!/usr/bin/env python3
"""
Build knowledge graph from genomic biophysical profile data.

Node types:
  - Kingdom   (4)   : animalia, fungi, plantae, protista
  - Organism  (11)  : fly, human, arabidopsis, ...
  - Element   (14)  : gs, prom, 5utrs, es, cds, ... (genomic boundary regions)
  - Parameter (7)   : bbone, bp, hbond, inter, intra, sol, stack
  - Hotspot   (<=600): high-signal regions per (organism, element, parameter)

Edge types:
  - BELONGS_TO          : Organism -> Kingdom
  - PRECEDES            : Element  -> Element   (canonical genomic linear order)
  - HAS_ELEMENT         : Organism -> Element   (element exists in this genome)
  - MEASURED_BY         : Element  -> Parameter (with per-organism profile stats as attrs)
  - HOTSPOT_IN_ORGANISM : Hotspot  -> Organism
  - HOTSPOT_IN_ELEMENT  : Hotspot  -> Element
  - HOTSPOT_IN_PARAMETER: Hotspot  -> Parameter
  - CORRELATES_WITH     : Parameter <-> Parameter (Pearson r >= threshold, from mean profiles)

Outputs (knowledge_graph/):
  kg.graphml      -- NetworkX GraphML (human-readable XML)
  kg.pkl          -- pickled NetworkX MultiDiGraph
  kg_nodes.csv    -- flat node table  (Neo4j-compatible)
  kg_edges.csv    -- flat edge table  (Neo4j-compatible)
  kg_stats.json   -- summary statistics
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

warnings.filterwarnings("ignore")

# --- Biological metadata -----------------------------------------------------

ORGANISM_KINGDOM = {
    "arabidopsis": "plantae",
    "fly":         "animalia",
    "human":       "animalia",
    "klactis":     "fungi",
    "mouse":       "animalia",
    "pfalciparum": "protista",
    "rat":         "animalia",
    "scerevisiae": "fungi",
    "worm":        "animalia",
    "ylipolytica": "fungi",
    "zeamays":     "plantae",
    # New NCBI RefSeq organisms (extracted_new).
    "toxoplasma_gondii":         "protista",
    "tetrahymena_thermophila":   "protista",
    "chlamydomonas_reinhardtii": "protista",
    "trypanosoma_brucei":        "protista",
    "aspergillus_fumigatus":     "fungi",
    "aspergillus_niger":         "fungi",
    "schizosaccharomyces_pombe": "fungi",
    "cryptococcus_neoformans":   "fungi",
    "neurospora_crassa":         "fungi",
    "populus_trichocarpa":       "plantae",
    "triticum_aestivum":         "plantae",
    "physcomitrium_patens":      "plantae",
    "glycine_max":               "plantae",
    "solanum_lycopersicum":      "plantae",
    "oryza_sativa":              "plantae",
    # Balancing additions (animalia + protista -> 8 each).
    "danio_rerio":               "animalia",
    "xenopus_tropicalis":        "animalia",
    "gallus_gallus":             "animalia",
    "dictyostelium_discoideum":  "protista",
    "giardia_intestinalis":      "protista",
    "leishmania_major":          "protista",
}

ORGANISM_SCIENTIFIC = {
    "arabidopsis": "Arabidopsis thaliana",
    "fly":         "Drosophila melanogaster",
    "human":       "Homo sapiens",
    "klactis":     "Kluyveromyces lactis",
    "mouse":       "Mus musculus",
    "pfalciparum": "Plasmodium falciparum",
    "rat":         "Rattus norvegicus",
    "scerevisiae": "Saccharomyces cerevisiae",
    "worm":        "Caenorhabditis elegans",
    "ylipolytica": "Yarrowia lipolytica",
    "zeamays":     "Zea mays",
    # New NCBI RefSeq organisms (extracted_new).
    "toxoplasma_gondii":         "Toxoplasma gondii",
    "tetrahymena_thermophila":   "Tetrahymena thermophila",
    "chlamydomonas_reinhardtii": "Chlamydomonas reinhardtii",
    "trypanosoma_brucei":        "Trypanosoma brucei",
    "aspergillus_fumigatus":     "Aspergillus fumigatus",
    "aspergillus_niger":         "Aspergillus niger",
    "schizosaccharomyces_pombe": "Schizosaccharomyces pombe",
    "cryptococcus_neoformans":   "Cryptococcus neoformans",
    "neurospora_crassa":         "Neurospora crassa",
    "populus_trichocarpa":       "Populus trichocarpa",
    "triticum_aestivum":         "Triticum aestivum",
    "physcomitrium_patens":      "Physcomitrium patens",
    "glycine_max":               "Glycine max",
    "solanum_lycopersicum":      "Solanum lycopersicum",
    "oryza_sativa":              "Oryza sativa",
    # Balancing additions (animalia + protista -> 8 each).
    "danio_rerio":               "Danio rerio",
    "xenopus_tropicalis":        "Xenopus tropicalis",
    "gallus_gallus":             "Gallus gallus",
    "dictyostelium_discoideum":  "Dictyostelium discoideum",
    "giardia_intestinalis":      "Giardia intestinalis",
    "leishmania_major":          "Leishmania major",
}

ELEMENT_LABEL = {
    "gs":    "Gene Start",
    "ge":    "Gene End",
    "prom":  "Promoter",
    "5utrs": "5-prime UTR Start",
    "5utre": "5-prime UTR End",
    "ens":   "Intron Start (5-prime Splice Site)",
    "ene":   "Intron End (3-prime Splice Site)",
    "es":    "Exon Start (3-prime Splice Site)",
    "ee":    "Exon End (5-prime Splice Site)",
    "cds":   "Coding Sequence",
    "stac":  "Start Codon",
    "stoc":  "Stop Codon",
    "3utrs": "3-prime UTR Start",
    "3utre": "3-prime UTR End",
}

ELEMENT_TYPE = {
    "gs":    "gene_boundary",
    "ge":    "gene_boundary",
    "prom":  "regulatory",
    "5utrs": "utr",
    "5utre": "utr",
    "ens":   "splice_site",
    "ene":   "splice_site",
    "es":    "splice_site",
    "ee":    "splice_site",
    "cds":   "coding",
    "stac":  "codon",
    "stoc":  "codon",
    "3utrs": "utr",
    "3utre": "utr",
}

# Canonical upstream -> downstream genomic order
GENOMIC_ORDER = [
    "gs", "prom", "5utrs", "5utre",
    "ens", "es", "stac", "cds", "stoc", "ee", "ene",
    "3utrs", "3utre", "ge",
]

PARAM_LABEL = {
    "bbone": "Backbone Geometry",
    "bp":    "Base-Pair Geometry",
    "hbond": "Hydrogen Bond Energy",
    "inter": "Inter-base Interaction Energy",
    "intra": "Intra-base Stacking Energy",
    "sol":   "Solvation Energy",
    "stack": "Base Stacking Energy",
}

PARAM_TYPE = {
    "bbone": "backbone_geometry",
    "bp":    "base_pair_geometry",
    "hbond": "hydrogen_bonding",
    "inter": "inter_base_interaction",
    "intra": "intra_base_stacking",
    "sol":   "solvation",
    "stack": "base_stacking",
}


# --- Helpers -----------------------------------------------------------------

def safe_float(x):
    """Return float or None if NaN."""
    f = float(x)
    return None if (f != f) else round(f, 6)  # NaN check without math


def profile_summary(vec):
    """Compact summary of a 1-D profile vector (ignores NaN)."""
    valid = vec[~np.isnan(vec)]
    if len(valid) == 0:
        return {"mean": None, "std": None, "peak_pos": None, "peak_val": None}
    return {
        "mean":     round(float(valid.mean()), 6),
        "std":      round(float(valid.std()),  6),
        "peak_pos": int(np.nanargmax(vec)),
        "peak_val": round(float(np.nanmax(vec)), 6),
    }


# --- Main --------------------------------------------------------------------

def build_kg(
    meta_path,
    hotspot_path,
    profiles_path,
    encoder_path,
    out_dir,
    corr_threshold=0.5,
    organism_name=None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Load data ------------------------------------------------------------
    print("Loading data ...")
    meta     = pd.read_parquet(meta_path)
    hotspots = pd.read_csv(hotspot_path)
    profiles = np.load(profiles_path)        # (11, 14, 7, 475)
    enc      = json.load(open(encoder_path))

    kingdoms_map   = enc["kingdom"]    # name -> idx
    organisms_map  = enc["organism"]
    elements_map   = enc["element"]
    parameters_map = enc["parameter"]

    param_names = [k for k, _ in sorted(parameters_map.items(), key=lambda x: x[1])]
    elem_names  = [k for k, _ in sorted(elements_map.items(),   key=lambda x: x[1])]

    selected_organisms = sorted(organisms_map)
    if organism_name is not None:
        if organism_name not in organisms_map:
            raise ValueError(f"Unknown organism: {organism_name}")
        meta = meta[meta.organism == organism_name].copy()
        hotspots = hotspots[hotspots.organism == organism_name].copy()
        selected_organisms = [organism_name]

    print(f"  metadata  : {meta.shape}")
    print(f"  hotspots  : {hotspots.shape}")
    print(f"  profiles  : {profiles.shape}")
    print(f"  kingdoms  : {sorted(kingdoms_map)}")
    print(f"  organisms : {selected_organisms}")
    print(f"  elements  : {sorted(elements_map)}")
    print(f"  parameters: {param_names}")

    # -- Create graph ---------------------------------------------------------
    G = nx.MultiDiGraph(
        name="Genomic Biophysical Profile KG",
        description=(
            "Knowledge graph encoding biophysical signatures of genomic elements "
            "across genomic organisms and elements, parameterised by 7 biophysical parameters."
        ),
    )

    # -- 1. Kingdom nodes -----------------------------------------------------
    print("Adding kingdom nodes ...")
    selected_kingdoms = sorted({ORGANISM_KINGDOM[name] for name in selected_organisms})
    for name, idx in kingdoms_map.items():
        if name not in selected_kingdoms:
            continue
        n_orgs = sum(1 for organism in selected_organisms if ORGANISM_KINGDOM[organism] == name)
        G.add_node(f"kingdom:{name}",
                   node_type="kingdom",
                   name=name,
                   kingdom_idx=idx,
                   n_organisms=n_orgs)

    # -- 2. Organism nodes + BELONGS_TO ---------------------------------------
    print("Adding organism nodes ...")
    for name in selected_organisms:
        idx = organisms_map[name]
        kname  = ORGANISM_KINGDOM[name]
        n_elem = int(meta[meta.organism == name]["element"].nunique())
        n_seq  = int(meta[meta.organism == name]["n_rows"].max())
        G.add_node(f"organism:{name}",
                   node_type="organism",
                   name=name,
                   scientific_name=ORGANISM_SCIENTIFIC[name],
                   kingdom=kname,
                   organism_idx=idx,
                   n_elements=n_elem,
                   n_sequences=n_seq)
        G.add_edge(f"organism:{name}", f"kingdom:{kname}",
                   edge_type="BELONGS_TO", relation="belongs_to")

    # -- 3. Element nodes + PRECEDES ------------------------------------------
    print("Adding element nodes ...")
    organism_idx_for_summary = None if organism_name is None else organisms_map[organism_name]
    for name, idx in elements_map.items():
        order_pos = GENOMIC_ORDER.index(name) if name in GENOMIC_ORDER else -1
        if organism_idx_for_summary is None:
            mp = profiles[:, idx, :, :]
            mean_profile = np.nanmean(mp, axis=0)
        else:
            mean_profile = profiles[organism_idx_for_summary, idx, :, :]
        per_param_summary = {
            param_names[p]: profile_summary(mean_profile[p])
            for p in range(len(param_names))
        }
        G.add_node(f"element:{name}",
                   node_type="element",
                   name=name,
                   label=ELEMENT_LABEL.get(name, name),
                   element_type=ELEMENT_TYPE.get(name, "other"),
                   element_idx=idx,
                   genomic_order=order_pos,
                   profile_summary=json.dumps(per_param_summary))

    # PRECEDES edges
    for i in range(len(GENOMIC_ORDER) - 1):
        a, b = GENOMIC_ORDER[i], GENOMIC_ORDER[i + 1]
        if a in elements_map and b in elements_map:
            G.add_edge(f"element:{a}", f"element:{b}",
                       edge_type="PRECEDES",
                       relation="precedes",
                       order_rank=i)

    # -- 4. Parameter nodes ---------------------------------------------------
    print("Adding parameter nodes ...")
    for name, idx in parameters_map.items():
        G.add_node(f"parameter:{name}",
                   node_type="parameter",
                   name=name,
                   label=PARAM_LABEL[name],
                   physical_type=PARAM_TYPE[name],
                   parameter_idx=idx)

    # -- 5. HAS_ELEMENT edges: Organism -> Element ----------------------------
    print("Adding HAS_ELEMENT edges ...")
    oe_pairs = meta.drop_duplicates(["organism", "element"])
    for _, row in oe_pairs.iterrows():
        org, elem = row["organism"], row["element"]
        n_seq = int(meta[(meta.organism == org) & (meta.element == elem)]["n_rows"].iloc[0])
        G.add_edge(f"organism:{org}", f"element:{elem}",
                   edge_type="HAS_ELEMENT",
                   relation="has_element",
                   n_sequences=n_seq)

    # -- 6. MEASURED_BY edges: Element -> Parameter (with stats) --------------
    print("Adding MEASURED_BY edges ...")
    for _, row in meta.iterrows():
        org, elem, param = row["organism"], row["element"], row["parameter"]
        oidx = organisms_map[org]
        eidx = elements_map[elem]
        pidx = parameters_map[param]
        vec  = profiles[oidx, eidx, pidx, :]   # (475,)
        ps   = profile_summary(vec)
        G.add_edge(f"element:{elem}", f"parameter:{param}",
                   edge_type="MEASURED_BY",
                   relation="measured_by",
                   organism=org,
                   n_sequences=int(row["n_rows"]),
                   profile_mean=ps["mean"],
                   profile_std=ps["std"],
                   profile_peak_pos=ps["peak_pos"],
                   profile_peak_val=ps["peak_val"])

    # -- 7. Hotspot nodes + edges ---------------------------------------------
    print(f"Adding {len(hotspots)} hotspot nodes ...")
    for i, row in hotspots.iterrows():
        org, elem, param, method = (
            row["organism"], row["element"], row["parameter"], row["method"])
        hid = f"hotspot:{org}:{elem}:{param}:{row['start']}:{row['end']}:{method}"
        G.add_node(hid,
                   node_type="hotspot",
                   organism=org,
                   element=elem,
                   parameter=param,
                   method=method,
                   start=int(row["start"]),
                   end=int(row["end"]),
                   width=int(row["width"]),
                   peak_pos=int(row["peak_pos"]),
                   peak_val=float(row["peak_val"]),
                   mean_val=float(row["mean_val"]),
                   n_sequences=int(row["n_sequences"]))
        G.add_edge(hid, f"organism:{org}",
                   edge_type="HOTSPOT_IN_ORGANISM", relation="hotspot_in_organism")
        G.add_edge(hid, f"element:{elem}",
                   edge_type="HOTSPOT_IN_ELEMENT", relation="hotspot_in_element")
        G.add_edge(hid, f"parameter:{param}",
                   edge_type="HOTSPOT_IN_PARAMETER", relation="hotspot_in_parameter")

    # -- 8. CORRELATES_WITH: Parameter <-> Parameter --------------------------
    print("Computing parameter-parameter correlation edges ...")
    if organism_idx_for_summary is None:
        corr_source = profiles
    else:
        corr_source = profiles[[organism_idx_for_summary], :, :, :]
    flat = corr_source.reshape(-1, len(param_names), profiles.shape[-1])
    param_means = np.nanmean(flat, axis=-1)                            # (154, 7)
    valid_rows  = ~np.any(np.isnan(param_means), axis=1)
    param_means_clean = param_means[valid_rows]
    if param_means_clean.shape[0] > 1:
        corr_mat = np.corrcoef(param_means_clean.T)                    # (7, 7)
        for i in range(len(param_names)):
            for j in range(i + 1, len(param_names)):
                r = float(corr_mat[i, j])
                if abs(r) >= corr_threshold:
                    pi, pj = param_names[i], param_names[j]
                    G.add_edge(f"parameter:{pi}", f"parameter:{pj}",
                               edge_type="CORRELATES_WITH", relation="correlates_with",
                               pearson_r=round(r, 4))
                    G.add_edge(f"parameter:{pj}", f"parameter:{pi}",
                               edge_type="CORRELATES_WITH", relation="correlates_with",
                               pearson_r=round(r, 4))
        print(f"  Correlation matrix (threshold={corr_threshold}):")
        header = "         " + "  ".join(p.ljust(5) for p in param_names)
        print(f"  {header}")
        for i, pi in enumerate(param_names):
            row_str = "  " + pi.ljust(6) + " : " + "  ".join(
                f"{corr_mat[i,j]:+.2f}" for j in range(len(param_names)))
            print(row_str)
    else:
        print("  Warning: not enough valid samples to compute correlations.")

    # -- Summary --------------------------------------------------------------
    stats = {
        "n_nodes":      G.number_of_nodes(),
        "n_edges":      G.number_of_edges(),
        "n_organisms":  len(selected_organisms),
        "n_kingdoms":   len(selected_kingdoms),
        "n_elements":   len(elements_map),
        "n_parameters": len(parameters_map),
        "n_hotspots":   len(hotspots),
        "organism_filter": organism_name,
        "node_types":   {},
        "edge_types":   {},
    }
    for _, d in G.nodes(data=True):
        t = d.get("node_type", "unknown")
        stats["node_types"][t] = stats["node_types"].get(t, 0) + 1
    for _, _, d in G.edges(data=True):
        t = d.get("edge_type", "unknown")
        stats["edge_types"][t] = stats["edge_types"].get(t, 0) + 1

    print("\n=== Knowledge Graph ===")
    print(f"  Nodes : {stats['n_nodes']}")
    print(f"  Edges : {stats['n_edges']}")
    for t, c in stats["node_types"].items():
        print(f"    node/{t:<28} {c:>5}")
    for t, c in stats["edge_types"].items():
        print(f"    edge/{t:<28} {c:>5}")

    # -- Save outputs ---------------------------------------------------------
    print(f"\nSaving outputs to {out_dir} ...")

    json.dump(stats, open(out_dir / "kg_stats.json", "w"), indent=2)
    print("  kg_stats.json")

    nx.write_graphml(G, str(out_dir / "kg.graphml"))
    print("  kg.graphml")

    with open(out_dir / "kg.pkl", "wb") as fh:
        pickle.dump(G, fh, protocol=4)
    print("  kg.pkl")

    node_rows = []
    for nid, data in G.nodes(data=True):
        row = {"node_id": nid}
        row.update(data)
        node_rows.append(row)
    pd.DataFrame(node_rows).to_csv(out_dir / "kg_nodes.csv", index=False)
    print("  kg_nodes.csv")

    edge_rows = []
    for u, v, data in G.edges(data=True):
        row = {"source": u, "target": v}
        row.update(data)
        edge_rows.append(row)
    pd.DataFrame(edge_rows).to_csv(out_dir / "kg_edges.csv", index=False)
    print("  kg_edges.csv")

    print("\nDone.")
    return G


# --- CLI ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Build genomic biophysical KG")
    p.add_argument("--root", default=".",
                   help="Root directory of the genome_reader project (holds metadata.parquet etc.)")
    p.add_argument("--meta",     default=None,
                   help="Path to metadata.parquet (default: <root>/metadata.parquet)")
    p.add_argument("--hotspots", default=None,
                   help="Path to hotspots.csv (default: <root>/explore_out/hotspots.csv)")
    p.add_argument("--profiles", default=None,
                   help="Path to mean_profiles.npy (default: <root>/explore_out/mean_profiles.npy)")
    p.add_argument("--encoders", default=None,
                   help="Path to encoders.json (default: <root>/dataset/encoders.json)")
    p.add_argument("--out",      default=None,
                   help="Output directory (default: <root>/knowledge_graph or <root>/knowledge_graph/<organism> when --organism is set)")
    p.add_argument("--organism", default=None,
                   help="Optional organism filter for building one organism-specific KG")
    p.add_argument("--corr-thresh", type=float, default=0.5,
                   help="Pearson r threshold for CORRELATES_WITH edges (default: 0.5)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = Path(args.root)

    meta_path     = Path(args.meta)     if args.meta     else root / "metadata.parquet"
    hotspot_path  = Path(args.hotspots) if args.hotspots else root / "explore_out/hotspots.csv"
    profiles_path = Path(args.profiles) if args.profiles else root / "explore_out/mean_profiles.npy"
    encoder_path  = Path(args.encoders) if args.encoders else root / "dataset/encoders.json"
    if args.out:
        out_dir = Path(args.out)
    elif args.organism:
        out_dir = root / "knowledge_graph" / args.organism
    else:
        out_dir = root / "knowledge_graph"

    for f in [meta_path, hotspot_path, profiles_path, encoder_path]:
        if not f.exists():
            raise FileNotFoundError(f"Required file not found: {f}")

    build_kg(
        meta_path=meta_path,
        hotspot_path=hotspot_path,
        profiles_path=profiles_path,
        encoder_path=encoder_path,
        out_dir=out_dir,
        corr_threshold=args.corr_thresh,
        organism_name=args.organism,
    )
