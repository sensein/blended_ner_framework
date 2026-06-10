Then we used the parcellation_division label "STR" and "PAL" in the dataset to select all cells located in the mouse [striatum](Brain_Region) and [pallidum](Brain_Region). Next, we filtered the data by removing low abundance cell types; cell types with less than 150 cell counts. We then mapped the [mouse genes](GENE) to their human orthologs using the mouse human homologous gene table from the Mouse Genome Informatics (MGI) database. Next, we identified the [genes](GENE) shared between the mouse and human [MERFISH](MERFISH_PLATFORM) dataset and reduced the gene expression matrix in both mouse and human to the common [genes](GENE). Following this, we computed the pseudobulk gene expression of each cell type at the supertype level for the mouse and at the group level for the human. Then we performed the Pearson correlation between the mouse and human expression values. Through this approach, we identified the mouse homologs of the human D1 Matrix and Striosome [MSNs](NEURON_TYPE).

ACKNOWLEDGEMENTS

We thank members of the Xu laboratory for their valuable feedback and discussion throughout this project. We are grateful to Dr. Ruth Walker for her insightful comments.

We also acknowledge Drs. Elizabeth and Edwin Monuki, along with our UCI colleagues, for their contributions to establishing the UCI BICAN Brain Bank and supporting the collection of neurotypical human brain samples. This work was in part funded by the NIH grant [UM1MH130994 Center for Multiomic Human Brain Cell Atlas](GRANT) and UC Irvine Center for Neural Circuit technology development funds. This publication was supported and coordinated through the Brain Initiative Cell Atlas Network (BICAN, RRID:SCR_022794).

Supplementary Materials

Supplementary Movie 1. [MERFISH](MERFISH_PLATFORM)+ spatial transcriptomics of a large human [basal ganglia](Brain_Region) section visualized across scales. The movie depicts successively greater magnified views starting from multi-centimeter tissue architecture to subcellular resolution scale, revealing transcriptionally distinct cell types and local cellular neighborhoods, and individual RNA transcripts visualized at submicrometer precision (color-coded by gene identity).

Table S1. Human brain sample information. Detailed information on human donor identification number, age, sex, race/ethnicity, postmortem interval (PMI) from time of death to tissue freezing, RNA integrity number (RIN) indicating sample quality, cause of death, section AP, subregions profiled. Supplemental Data for the information related to the [MERFISH+](MERFISH_PLATFORM) encoding probes, adaptor probes, and readout probes and the gene panel codebook. Stereo-seq Analysis Workflow STOmics RRID:SCR_025001 STAR http://code.google.com/p/rna-star/ RRID:SCR_004463 Ensembl Ensembl RRID:SCR_002344 OpenCV http://opencv.org RRID:SCR_015526 NumPy http://www.numpy.org RRID:SCR_008633 SciPy http://www.scipy.org RRID:SCR_008058 statsmodel http://www.statsmodels.org/ RRID:SCR_016074 scikit-learn http://scikit-learn.org RRID:SCR_002577 Agilent TapeStation Laptop Agilent RRID:SCR_019547 Other 50mm x 64mm glass coverslips CellPath SAF-5064-02A Allen Institute for Brain Science http://www.brain-map.org RRID:SCR_006491 BRAIN Initiative Cell Atlas Network BRAIN Initiative Cell Atlas Network RRID:SCR_022794 Brain Image Library (BIL) Brain Image Library (BIL) RRID:SCR_017272 NeMOarchive Neuroscience Multi-omic Archive (NeMO Archive) RRID:SCR_016152 Echo Revolution microscope Discover Echo RRID: SCR_027699 MGI DNBSEQ-T7 Genetic Sequencer MGI RRID:SCR_024847 Complete Genomics Complete Genomics RRID:SCR_027007 PacGenomics PacGenomics RRID: SCR_027700

RESOURCE AVAILABILITY

Supplemental

Figure Legends

Material availability

Oligonucleotide probe sequences used for imaging can be found in Supplemental Data.

These probes or materials for making these probes can be purchased from commercial sources, as detailed in the Key Resources Table.

Data and code availability

Image and sequencing data are in the process of being deposited at the Brain Image Board (IRB). These adult human postmortem brain samples were from white males of 36 to 53 years old and showed no evidence of neurodegenerative pathology (Table S1).

Preparation of multi-centimeter tissue sections for Stereo-seq and [MERFISH+](MERFISH_PLATFORM)

molecules; 405 nm for DAPI) at 0.4µm step size (25 frames per field of view/channel, total frames=100/field of view). After image acquisition of a round, fluorescence signals were removed through flowing stripping buffer (80% formamide in 0.8xSSC+0.1% Tween-20) and 2xSSC before proceeding to next round of hybridization and imaging.

Bioinformatics

Stereo-seq data processing

We processed the Stereo-seq data using the standard analysis workflow software. The pipeline takes three files as an input: (1) paired-end FASTQ files, (2) mask file and (3) mosaic image of nuclei-staining. The first read in the fastq files contains coordinate ID (CID) -spatial location identifier barcode sequence that is unique to each DNA nano ball on the stereo-seq chip, and the Molecular ID (MID) -artificially synthesized mRNA-specific sequence that helps differentiate the number of reads contributed by mRNA expression level due to amplification. The CIDs were first aligned and matched with the values in the mask file which contains the mapping of the CID barcodes to the actual spatial coordinates in the tissue. This allows the identification of the spatial location of the mRNAs within the tissue section. Reads that have MID sequence with more than one N bases or more than one bases with quality less than Q10 were excluded. The second read in the fastq file contains the actual sequence of the captured mRNA. These were aligned to a reference genome using the STAR aligner. The reference genome ‫ܥ‬: is the number of cells in sample i that does not belong to a cell type of interest but are within the brain region of interest.

‫ܦ‬: is the number of cells in sample i that does not belong to a cell type of interest and are not within the brain region of interest.

ܶ: is the total number of cells in the data.

Next, we quantified the significance of the result using the permutation test, as described above.

Cell type organization and gene expression gradients with respect to [striosome](Brain_Region) borders

We first identified the contours for striosome borders from the spatial map of the brain section using the function findcontours from the Opencv package. Then we used the KDTree module from SciPy to find the distance of each cell from the contour of the nearest striosome. We set the distance of cells within the striosome compartment as positive and negative for those that are outside. Then we binned the distance into bins of size 50 microns and computed the mean abundance of cell type within each bin.

When doing the calculation, we only considered cells that are 200 microns away from the striosome border and 500 microns into the center of the striosome. We also removed cell types that were very sparse -with less than 1% abundance near the striosome.

To quantify the gene expression gradient with respect to the border striosomes, we computed the mean log1p gene expression of cells that are within the binned distances. Then, we performed lowess regression using the statsmodel package to quantify the dependence of the mean gene expression on the distance of cells from the striosomes.

Computing the volume of cells using somatic transcripts:

To quantify the volume of a cell, we generated a tetrahedral mesh from the point cloud formed by the cell's somatic transcripts. For this, we performed 3D Delaunay triangulation using the function delaunay_3d from the pyvista package with the parameter alpha set to 3 times the median distance between neighboring transcripts. We then triangulated the mesh and computed its volume using the triangulate function. Following this procedure, we quantified the volume of 100 cells sampled from each cell type. Finally, we took the median of the cell's volume to get a representative value for each cell type. We filtered out cells that had artifacts in the labeling of their somatic transcripts. Such artifacts arise when cells are surrounded by density transcript from nearby cells or their processes.

Quantifying the projection of neurons using distribution of subcellular transcripts

As the [MERFISH](MERFISH_PLATFORM) imaging data is acquired on a field of view by field of view basis, the decoded transcript information is organized separately for each FOV. For this reason, we performed the quantification of a neuron projection to a target region by aggregating the information within each FOV. For every gene in each FOV, we first computed the count of transcripts that are outside a cell's segmentation mask.

Then we took the log of the counts and normalized them across genes and all FOVs. Next, we computed the differential gene expression between [STRd D1 Matrix](NEURON_TYPE) and [STRd D2 Matrix](NEURON_TYPE) MSNs. The differential expression was quantified as a difference-in-difference: the deviation of STRd D1 Matrix MSN's pseudobulk expression from the across-cell-type mean, minus the deviation of STRd D2 matrix MSN's pseudobulk expression from that mean. Finally, we quantified the projection strength as the dot product between log count of extracellular transcripts and the differential gene expression scores.