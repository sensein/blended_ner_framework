Primary [encoding probes](MOLECULAR_MARKER) generation was performed from [oligonucleotide pools](CELLULAR_COMMUNITY) as described previously [42][43][44]. [Oligonucleotide pools](CELLULAR_COMMUNITY) and primers were purchased from Twist Biosciences and Integrated DNA Technologies (IDT), respectively. Oligo pools were amplified through PCR with approximately 18 cycles using Phusion Plus PCR Master Mix (Thermo Fisher Scientific, F631L). The resulting PCR product was purified and concentrated with Zymo DNA clean & concentrator-25 (Zymo Research, D4006). RNA was generated from the purified PCR-amplified DNA through in-vitro transcription (IVT) using HiScribe T7 ARCA mRNA kit (NEB, E2060S).

Next, single strand DNA (ssDNA) was generated from the IVT-derived RNA through reverse transcription (RT) using Thermo Scientific Maxima Reverse Transcriptase (Thermo Fisher Scientific, EP0743). The RT product went through alkaline hydrolysis (1:1 of 1M NaOH and 0.5M EDTA) to remove RNA followed by ssDNA ([encoding probes](MOLECULAR_MARKER)) purification using Zymo Oligo Clean & Concentrator (Zymo Research, D4061). The concentration and purity of the [encoding probes](MOLECULAR_MARKER) were determined through NanoDrop while the size of the [encoding probes](MOLECULAR_MARKER) was verified through Agilent TapeStation (RRID:SCR_019547). During the RT step, the forward primer introduced an acrydite anchor into the [encoding probes](MOLECULAR_MARKER) which allowed the [encoding probes](MOLECULAR_MARKER) to covalently bind with the polyacrylamide gel post encoding probes hybridization.

Adaptor and readout probe preparation:

[Adaptor probes](MOLECULAR_MARKER) (total 28 [adaptors](SPATIAL_MODULE)) and [readout probes](MOLECULAR_MARKER) (total 3 [readout probes](MOLECULAR_MARKER) each tagged with one of the three fluorophores:

5Alex750N/3AlexF750N, 5Cy5/3Cy5Sp, 5Cy3/3Cy3Sp) were purchased from IDT.

[Adaptor probes](MOLECULAR_MARKER) and [readout probes](MOLECULAR_MARKER) were diluted with a pre-hybridization buffer (PreHB, 0.1% Tween-20, 35% formamide, 2xSSC) with dilution factors of 1:1000 and 1:2000, respectively.

5. Silanization and PDL-coating of coverslips: 50mm x 64mm glass coverslips ([CellPath](NEURAL_PATHWAY), SAF-5064-02A) were incubated in 18.5% HCl in methanol for 30 min at room temperature (RT) followed by 4 washes with MiliQ water. Then coverslips were incubated in 70% ethanol for 5 min at RT followed by drying at 60°C for 30 min. Next, coverslips were incubated in a solution containing 0.1% triethylamine and 0.2% allyl trichlorosilane in chloroform for 30 min at RT. Coverslips were then washed with chloroform followed by washing with 100% ethanol. Then coverslips were dried at 60°C for 30 min. Coverslips were stacked on top of each other with a working solution of Poly-D-Lysine (PDL, SIgmaAldrich, A-003-M) (1 mg PDL/ml + RNase inhibitor) in between coverslips and incubated for ~3 hours. Next, coverslips were dried at 60°C for 30 min. Finally, silanized and PDL-coated coverslips were stored at -20°C until use (good for 6 months).

MERFISH+ sample preparation:

Human postmortem basal ganglia samples in OCT block were sectioned (16 µm thick) using cryostat (Leica CM1850) at -20°C and mounted onto PDL-coated pre-silanized coverslips as aforedescribed. The mounted tissue sections were air dried at room temperature for 5 minutes followed by fixation with 4% paraformaldehyde (PFA) in PBS for 15 min. The PFA-fixed samples were stored in a 10% glucose solution in PBS (with RNase inhibitor, 1:1000) at -80°C for future use. The PFA-fixed frozen samples were thawed and incubated with 70% (v/v) ethanol for 1 hour at room temperature. The samples were then incubated with 5% SDS (in 2xSSC) for 10 minutes at room temperature followed by three washes with 2xSSC.

Samples were incubated with PreHB for 15 minutes at room temperature. The coverslip containing the sample was transferred to a new Petri dish and 200 µl of encoding probe hybridization buffer containing 200 µg of encoding probes were added onto the sample.

The encoding probe hybridization buffer consisted of 50% (v/v) formamide in 2xSSC and 10% (wt/v) dextran sulfate, with the RNase inhibitor added. Parafilm was placed gently on the top of the sample to prevent evaporation of the encoding probe hybridization buffer (10% dextran sulfate, 50% formamide, 2xSSC) and incubated in humidified incubator for 18-24 hours at 47°C. After encoding probe hybridization, samples were washed with PreHB twice 15 minutes each at room temperature. Next, 4% polyacrylamide gel embedding of samples was performed to anchor RNA molecules at place. Samples were then post-fixed with 4% PFA for 10 minutes at room temperature.

Samples were incubated with digestion buffer containing 2% SDS in 2x SSC and 2%

proteinase K overnight at 37°C. To quench sample autofluorescence, samples (still in digestion buffer) were incubated in Vizgen photobleacher for at least 3 hours. Finally, samples were washed with 2x SSC three times and proceeded to image acquisition.

Drift correction:

To estimate the drift between hybridization rounds, we detected image features corresponding to local minima and local maxima intensity points within the DAPI channel using the same procedure as in the spot detection step. These features were then used to register the images from different hybridization rounds using phase cross-correlation.

Decoding:

We decoded the detected fluorescent spots as described previously 42.

Briefly, we matched each spot in an image to spots that are within a set distance threshold in other images across the different hybridization rounds. We then compared the intensity pattern formed by the colocalizing spots to the binary words in the codebook in order to decode their gene identity. Finally, we filtered the molecules based on their correlation to the point spread function and the distance of their normalized brightness from the matched binary word.

Cell segmentation:

First, we deconvolved the flat-field corrected DAPI images using the Wiener algorithm, with the parameter beta set to 0.01, in order to improve the contrast of the nuclei boundaries. Then for each z-plane, we performed cell segmentation using the Cellpose 'nuclei' model 79 with the parameters: diameter = 23, flow_threshold = -10 and cellprob_threshold = -10. Next, we stitched the segmented images to get a 3D reconstruction of the cell masks. We then applied the masks to the decoded molecule in order to assign them to cells based on their spatial position. Finally, we generated a cell-by-gene count matrix by aggregating the transcript count of each gene in every cell. In addition, we generated cell metadata that included spatial information such as the XYZ location and volume of the imaged cells.

Cell type clustering and cell type annotation

Pre-processing:

We preprocessed the cell by gene count matrix as follows. First, cells with total transcript counts greater than 99 percentile or less 1 percentile were filtered out. Second, cells with very large (above 98 percentile) and very small (below 2 percentile) volume were removed. Such cells arise from segmentation artifacts and usually represent incomplete or doublet cells (two different cells segmented as one).

The count matrix was then normalized by volume of cells, in order to account for the effect of cell size variation on the expression level of genes. The matrix was again normalized by the total count of transcripts in the cells in order to reduce noise arising from the difference in the relative expression of genes between cells. Finally, the data was log-normalized to ensure all genes had the same expression baseline and hence minimized the risk of masking low abundant genes.

Clustering and annotation:

We performed clustering analysis using classical singlecell clustering pipeline from the Scanpy package 80. First, the graph representation of the count matrix was generated using the neighbors module, scanpy.pp.neighbors(use_rep = 'X', n_neighbors = 15). Then, the communities formed in the graph were clustered using the Leiden clustering algorithm, scanpy.tl.leiden(resolution = 1). Finally, the UMAP embedding of the clustered data was obtained by projecting the high dimensional clustering into 2D space, scanpy.tl.umap(resolution = 1, min_dist = 0.1). The annotation of cell types was done by mapping the data to the Cross-species Basal ganglion cell type taxonomy from the Allen institute through cell_type_mapper (the python backend for MapMyCells, RRID:SCR_024672) using two precomputed supporting files shared through the BICAN consortium: query_markers (a lookup table of marker genes for the taxonomy) and precomputed_stats (HDF5 file that defines the taxonomy).

We performed the subclustering analysis for astrocytes using only astrocytic genes in order to avoid artifacts from cross-talk or contamination from other cell types. To identify the genes, we applied the Wilcoxon rank sum method on the pre-processed count matrix as implemented in the rank_genes_groups function from the Scanpy package 80.

Next, we reduced the data to cells that were labeled as astrocytes at the Subclass level and to genes that were differentially expressed in them. Following this, we performed further quantity control to remove outlier cells with volume and transcript count below 1 percentile or greater than 99th percentile. Next, we did Principal Component Analysis (PCA) and harmony integration on the normalized count matrix using the functions sc.pl.pca and sc.external.pp.harmony_intergrate from Scanpy respectively. We then constructed the KNN graph based on the principal components from harmony integration using the parameters n_neighbors = 10 and n_pcs = 20. Lastly, we ran

Leiden clustering with a resolution of 0.6.

Spatial module analysis

We performed spatial module analysis with two different approaches, ( 1 however, the internal globus pallidus module partially overlapped with adjacent white matter tracts; its boundary was refined using the spatial location of GPi Shell neurons, which delineated the GPi border with high precision.

To identify spatial modules corresponding to the striosome and matrix compartment, we followed similar but slightly different procedure to that used in global spatial modules analysis. The main differences were that we only used MSN cell types when building the cell type composition matrix and we only included the cells in the striatum for the analysis as the Striosome and Matrix compartments exist only in this part of the Basal ganglia. In detail, we first built a K-nearest neighbor graph based on the spatial location of MSN subtypes using the scikit-learn package 82, then we queried the graph to identify 50 nearest MSNs for each cell in the striatum. Next, as explained in 14, we defined the composition matrix as distance weighted frequency count of neighboring MSN subtypes.

This matrix was then L2 normalized and clustered using the Leiden algorithm from the Scanpy package. The same analysis approach was used in human and mouse datasets.

Whole transcriptome based spatial module analysis:

To do spatial module analysis using gene expression, we integrated the expression values with the spatial information of cells. First, we calculated the spatial distance of each cell to neighboring cells located within a radius r using the function radius_neighbors_graph from scikitlearn 82. Next, we took the negative exponential of the distance values and normalized them by the total distance to get a gaussian distribution. The resulting values were then multiplied with the gene expression matrix to get spatially weighted average gene expression for each cell. Next, using the Scanpy package 80, we identified the top 5000 highly variable genes, which we then used for PCA and KNN graph computation. Finally, we did Leiden clustering to identify the spatial modules. Similar to the spatial modules analysis based on cell type composition, we did the clustering hierarchically -starting from high level to finer spatial modules. And the parameters were tuned at each hierarchy until the modules matched the anatomical regions.

Cell type enrichment analysis

Cell type enrichment analyses across spatial anatomical modules: The enrichment analysis was done following the method described in Zhang et al. 14. We first computed the confusion matrix between spatial modules and cell types at the group level using the pivot_table function from the pandas package. Next, for each cell type, we calculated its expected average density in every spatial module as the product of the fraction of cells that belonged to the cell type and the fraction of cells found in the spatial modules. Then the enrichment score of a cell type in a spatial module was computed as the ratio of its cell density and expected average cell density in the spatial module. To quantify the significance of the result, we used a permutation test where we shuffled cell type labels and recomputed the enrichment score. We repeated this step 1000 times to generate a null distribution of the enrichment score. Then the p-value was computed as the fraction of values in the distribution that are at least as large as the observed enrichment score.

Finally, we corrected the p-values for multiple testing using the Bonferroni method.

Cell type enrichment analyses across the matrix and striosome compartments:

We quantified the enrichment of cell types within the Matrix and Striosome compartments using the Cochran-Mantel-Haenszel statistics 83. For each cell type, we built contingency tables that summarize the number of cells in the compartments from each individual sample. Then, we computed the common odds-ratio as:

Where, K: is the total number of samples, ‫ܣ‬: is the number of cells in sample i that belongs to a cell type of interest and are within the brain region of interest.

‫ܤ‬: is the number of cells in sample i that belongs to a cell type of interest but are not within the brain region of interest.

We computed the significance of the estimated values using a permutation test. More specifically, we randomized the distance values and recomputed the regression and, then we obtained the explained variance for the estimated values. We repeated the permutation 1000 times to generate a null distribution, the p-value was then calculated as the fraction of the explained variance that were at least as greater as the observed explained variance.

Analysis of gene expression gradients in the striatum using stereo-seq data

To quantify the gene expression gradient along the ventromedial to dorsolateral (VM-DL) axis, we used the internal capsule as our reference since it is located approximately along this axis. We first rotated the brain sections so that the internal capsule aligns with the horizontal axis. Then we digitized the rotated coordinate values using the combination of the functions digitize and histogram from the NumPy package (RRID:SCR_008633). Next, we grouped the cells located within the same bin (digitized coordinates) and took the average of their spatially weighted gene expression values.

Following this, we quantified the expression gradient of each gene by calculating the correlation between the bin average gene expression and the axis coordinates using the pearsonr function from SciPy. Finally, we integrated the results from all stereo-seq samples by computing the mean and std of the correlation coefficients. We only kept genes whose standard deviation is below 0.2 and reported their mean correlation coefficient as final value.

Gene Ontology enrichment analysis for spatial modules

We performed gene ontology enrichment analysis using the enricher function from the gget package using the "ontology" database. For each spatial module, we first identified the marker genes that were differentially expressed in each spatial module. Then we obtained the top 10 ontology terms in relation to these genes by running the enricher function. Next, we mapped each term to their parent (broad) categories by parsing the directed acyclic graph (DAG) built from the basic GO ontology file -'go-basic.obo' using the GODag module from goatools package. Finally, we computed the frequency of the parent terms in each module and calculated the proportion of the frequencies across the regions to quantify the relative abundance across the regions.

Density-based clustering of subcellular transcript distributions

For each cell type, we first selected 100 cells whose differential gene expression correlated most with the average (pseudobulk) differential gene expression of the cell type. Then, we quantified the enrichment of every gene within the intracellular space of the cells as the fold change between transcript counts inside and outside the cell's segmentation mask. We excluded genes from further analysis if their enrichment was less than 3-fold or if they were not differentially expressed. Next, we computed the median distance between neighboring transcripts in both the somatic and putative distal neuritic spaces using the KDTree function from the SciPy package 81. Following this, we performed density-based clustering to identify the somatic transcripts corresponding to each cell. To achieve this, we used the DBSCAN function from the scikit-learn package (RRID:SCR_002577) 82 with the maximum distance parameter set to the average of the median transcript distances in somatic and putative distal neuritic spaces. Next, we classified the nonsomatic transcripts as neuritic if they are within 20 microns from the nearest somatic transcript.

Identifying mouse homologues of human basal ganglia cell types

To identify the mouse homologues of the cell types detected in our human Basal ganglia MERFISH dataset, we used the imputed whole mouse brain MERFISH spatial transcriptomic dataset from an earlier publication 10. First, we manually insepcted the mouse MERFISH data to identify tissue sections that contain brain regions corresponding to those covered in the human samples. We selected the sections of C57BL6J-638850.39 -50, which show the caudoputamen and globus pallidus regions.