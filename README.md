# ATLAS

<p align="center">
  <img src="ATLAS.png" alt="ATLAS Methodology" width="100%">
</p>

ATLAS is a topology-aware multimodal graph learning pipeline for cancer tissue-of-origin prediction that integrates pathology foundation-model embeddings, spatial tissue organization, and somatic genomic features.

The repository includes scripts for genomic feature extraction and filtering, multimodal cohort construction, spatial graph generation, cross-validation splitting, spatial-aware and spatial-null model training, spatial attention visualization, and external validation on CPTAC using five-fold probability averaging.

## Methodology

ATLAS integrates spatial pathology and somatic genomics to predict cancer tissue of origin.

The workflow consists of:

1. **Precomputed UNI2-h tissue tile embeddings**  
   Whole-slide images are represented using `N × 1536` tile-level UNI2-h feature embeddings.

2. **Somatic genomic feature extraction**  
   Mutation, copy-number alteration, and SBS96 mutational-signature features are extracted for each patient.

3. **IntOGen genomic filtering and fusion**  
   Mutation and CNA features are restricted to IntOGen cancer-driver genes, while SBS96 features are retained. The resulting genomic modalities are fused into a patient-level genomic representation.

4. **Matched multimodal cohort construction**  
   Patients with both pathology and genomic data are matched to construct the multimodal TCGA cohort.

5. **Spatial graph construction**  
   A fixed-radius proximity graph is built from tissue tile coordinates. Each tissue tile is represented as a node, and edges connect spatially neighboring tissue regions.

6. **Topology-aware multimodal graph learning**  
   Spatial pathology features are processed using a Kolmogorov-Arnold Graph Attention Network, while genomic features are encoded separately and integrated with the pathology representation.

7. **Attention and global mean pooling**  
   Spatial attention pooling and global mean pooling aggregate node-level tissue information into a slide-level pathology representation.

8. **Multimodal representation**  
   The spatial pathology representation is combined with the genomic representation to form an integrated patient-level representation.

9. **Tissue-of-origin prediction**  
   The multimodal representation is used to classify the predicted cancer tissue of origin.

The learned spatial attention weights can also be projected back onto the whole-slide image to provide spatial interpretability.

## Repository Structure

```text
ATLAS/
├── ATLAS.png
│
├── data/
│   ├── tcga/
│   │   ├── genomics/
│   │   │   └── scripts/
│   │   │       ├── extract_tcga_mutations.py
│   │   │       ├── extract_tcga_cna.py
│   │   │       ├── extract_tcga_sbs96.py
│   │   │       └── fuse_tcga_genomics.py
│   │   │
│   │   ├── multimodal/
│   │   │   └── scripts/
│   │   │       ├── match_tcga_cohort.py
│   │   │       └── create_cv_splits.py
│   │   │
│   │   ├── spatial-aware/
│   │   │   └── build_spatial_aware_graphs.py
│   │   │
│   │   └── spatial-null/
│   │       └── build_spatial_null_graphs.py
│   │
│   └── cptac/
│       └── genomics/
│           └── scripts/
│               ├── extract_cptac_mutations.py
│               ├── extract_cptac_cna.py
│               ├── extract_cptac_sbs96.py
│               └── fuse_cptac_genomics.py
│
├── models/
│   ├── spatial-aware/
│   │   ├── figures/
│   │   └── scripts/
│   │       └── train_spatial_aware.py
│   │
│   └── spatial-null/
│       ├── figures/
│       └── scripts/
│           └── train_spatial_null.py
│
├── interpretability/
│   ├── figures/
│   └── scripts/
│       └── plot_spatial_attention.py
│
├── validation/
│   ├── build_cptac_graphs.py
│   └── validate_cptac.py
│
├── LICENSE
└── README.md
