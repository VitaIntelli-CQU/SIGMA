# SIGMA

model_image

This repository provides the code and tutorials for SIGMA. The tutorials reproduce the main experiments on simulated and real spatial multi-omics datasets.

## Environment

The experiments were run with:

- Python 3.9.23
- R 4.3.0 (2023-04-21), used for `mclust` clustering
- PyTorch 2.4.1 with CUDA 12.1

Create the environment:

```bash
conda create -n sigma python=3.9.23
conda activate sigma
pip install -r requirements.txt
```

Install the PyTorch Geometric dependency wheels:

```bash
pip install torch-scatter-2.1.2+pt24cu121-cp39-cp39-linux_x86_64.whl
pip install torch-sparse-0.6.18+pt24cu121-cp39-cp39-linux_x86_64.whl
```

Install the R package required by `mclust` clustering:

```r
install.packages("mclust")
```

## Data Availability

The data paths used in the tutorials are consistent with the SMART project layout. By default, the notebooks expect datasets under `../SIGMA_data/`. If your data are stored elsewhere, update the `file_fold` variable in the corresponding notebook.

| Notebook          | Dataset                                         | Modalities     | Data source                                                  |
| ----------------- | ----------------------------------------------- | -------------- | ------------------------------------------------------------ |
| `tutorial1.ipynb` | Simulated spatial multi-omics dataset           | RNA, ADT, ATAC | SMART data repository: <https://doi.org/10.5281/zenodo.17093158> |
| `tutorial2.ipynb` | 10X Visium Human Lymph Node                     | RNA, ADT       | GEO accession GSE263617: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE263617> |
| `tutorial3.ipynb` | MISAR-seq mouse brain dataset                   | RNA, ATAC      | National Genomics Data Center accession OEP003285: <https://www.biosino.org/node/project/detail/OEP003285> |
| `tutorial4.ipynb` | P22 mouse brain spatial CUT&Tag-RNA-seq dataset | RNA, H3K27me3  | GEO accession GSE205055: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205055>; UCSC browser: <https://brain-spatial-omics.cells.ucsc.edu> |
| `tutorial5.ipynb` | 10X Visium Human Tonsil dataset                 | RNA, ADT       | Zenodo: <https://zenodo.org/records/12654113/preview/data_imputation.zip?include_deleted=0#tree_item0> |

## Running Tutorials

Start Jupyter and run the selected notebook:

```bash
jupyter lab
```

The five tutorials correspond to the five datasets listed above. Each notebook includes data loading, preprocessing, model training, clustering, and visualization for the corresponding experiment.

## Citation

If you use this code or datasets in your work, please cite the associated paper and the original data sources.
