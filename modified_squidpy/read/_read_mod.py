from __future__ import annotations

from typing import Union  # noqa: F401
from typing import Any
from pathlib import Path
import os
import re
import json

from scanpy import logging as logg
from anndata import AnnData

from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd

from squidpy.read._utils import _load_image, _read_counts
from squidpy.datasets._utils import PathLike
from squidpy._constants._pkg_constants import Key


#****************
from scipy.sparse import csr_matrix, vstack, coo_matrix
import dask.dataframe as ddf
#***************

__all__ = ["visium", "vizgen", "nanostring"]


def nanostring_mod(
    path: str | Path,
    *,
    slide : str,
    counts_file: str,
    meta_file: str,
    fov_file: str | None = None,
) -> AnnData:
    """
    Read *Nanostring* formatted dataset.

    In addition to reading the regular *Nanostring* output, it loads the metadata file, BUT NOT *CellComposite* and *CellLabels*
    directories containing the images and optionally the field of view file.

    .. seealso::

        - `Nanostring Spatial Molecular Imager <https://nanostring.com/products/cosmx-spatial-molecular-imager/>`_.
        - :func:`squidpy.pl.spatial_scatter` on how to plot spatial data.

    Parameters
    ----------
    path
        Path to the root directory containing *Nanostring* files.
    sample
        Slide name
    counts_file
        File containing the counts. Typically ends with *_exprMat_file.csv*.
    meta_file
        File containing the spatial coordinates and additional cell-level metadata.
        Typically ends with *_metadata_file.csv*.
    fov_file
        File containing the coordinates of all the fields of view.

    Returns
    -------
    Annotated data object with the following keys:

        - :attr:`anndata.AnnData.obsm` ``['spatial']`` -  local coordinates of the centers of cells.
          :attr:`anndata.AnnData.obsm` ``['spatial_fov']`` - global coordinates of the centers of cells in the
          field of view.
        - :attr:`anndata.AnnData.uns` ``['spatial']['{fov}']['images']`` - *hires* and *segmentation* images.
        - :attr:`anndata.AnnData.uns` ``['spatial']['{fov}']['metadata']]['{x,y}_global_px']`` - coordinates of the field of view.
          Only present if ``fov_file != None``.
    """  # noqa: E501

    path, fov_key = Path(path), "fov"
    cell_id_key = "cell_ID"
    # cell_id is c_2_1_1, c_2_1_2.. in metadata file
    # cell_ID is 1, 2, 3.. in metadata file 
    # cell_ID is 1, 2, 3.. in exprMAT file

    print("Step 1. Reading metadata file")
    obs = pd.read_csv(path / meta_file, header=0)
    # Fix : replace "concatenated fov_cellID" by MultiIndex then collapse because Method in Anndata does not support MultiIndex
    obs.set_index([fov_key, cell_id_key], drop=False, inplace=True)
    obs.index = ["_".join(map(str, (slide,) + idx)) for idx in obs.index]
    obs[fov_key] = pd.Categorical(obs[fov_key].astype(str))
    
    print("Step 2. Reading counts file")
    cols = pd.read_csv(path / counts_file, header =0, nrows =0) 
    # Fix : chunk and sparsify matrix upon reading exprMAT csv to save RAM
    chunksize = 10 **4
    
    with pd.read_csv(path / counts_file, header=0, chunksize = chunksize) as reader:
        i = 0
        for chunk in reader:
            i +=1
            if i == 1:
                A = coo_matrix(chunk)
            else :
                A = vstack([A, coo_matrix(chunk)])              
    B = A.tocsr()

    print("Step 3. Generating counts sparse matrix with correct columns")
    counts = pd.DataFrame.sparse.from_spmatrix(B, columns = list(cols))

    # Fix : replace "concatenated fov_cellID" by MultiIndex then collapsing index because Method in Anndata does not support MultiIndex
    counts.set_index([fov_key, cell_id_key], drop=False, inplace=True)
    counts.index = ["_".join(map(str, (slide,) + idx)) for idx in counts.index]
    
    # Fix drop cell_ID and fov; they get recorded as genes otherwise
    counts.drop(columns=["fov","cell_ID"], inplace=True)

    common_index = obs.index.intersection(counts.index)
    print("\tCommon index length : ", len(common_index))

    print("Step 4. Generating AnnData")
    adata = AnnData(
        csr_matrix(counts.loc[common_index, :].values),
        dtype=counts.values.dtype,
        obs=obs.loc[common_index, :],
        uns={Key.uns.spatial: {}},
    )
    adata.var_names = counts.columns
    
    adata.obsm[Key.obsm.spatial] = adata.obs[["CenterX_local_px", "CenterY_local_px"]].values
    adata.obsm["spatial_fov"] = adata.obs[["CenterX_global_px", "CenterY_global_px"]].values
    adata.obs.drop(columns=["CenterX_local_px", "CenterY_local_px"], inplace=True)

    for fov in adata.obs[fov_key].cat.categories:
        adata.uns[Key.uns.spatial][f"{slide}_{fov}"] = {
            "images": {},
            "scalefactors": {"tissue_hires_scalef": 1, "spot_diameter_fullres": 1},
        }

    print("Step 5. Reading image directories if existing")       
    file_extensions = (".jpg", ".png", ".jpeg", ".tif", ".tiff")

    pat = re.compile(r".*_F(\d+)")
    for subdir in ["CellComposite", "CellLabels"]:
        # Fix : do not read images folders in new output version
        if os.path.isdir(path / subdir):
            kind = "hires" if subdir == "CellComposite" else "segmentation"
            for fname in os.listdir(path / subdir):
                if fname.endswith(file_extensions):
                    # Fix : add slide name to fov as dict key
                    fov = str(int(pat.findall(fname)[0]))
                    adata.uns[Key.uns.spatial][f"{slide}_{fov}"]["images"][kind] = _load_image(path / subdir / fname)
    
        else:
            print(f"\tNo  {subdir} image directory in output. Skipping.")

    print("Step 6. Reading fov file")
    if fov_file is not None:
        
        fov_positions = pd.read_csv(path / fov_file, header=0)
        
        if fov_key in fov_positions.columns :
            fov_positions.set_index(fov_key, inplace = True, drop = False)
        # Fix : File has no 'fov' column, it shows as 'FOV'. Export format from Nanostring has changed.
        elif fov_key.upper() in fov_positions.columns :
            fov_positions.set_index(fov_key.upper(), inplace = True, drop = False)
        else :
            print("\tNo 'fov' or 'FOV' column in positions file")
            return
        
        for fov, row in fov_positions.iterrows():
            # Fix : record only positions for fov's present in exprMAT and metadata     
            try :
                adata.uns[Key.uns.spatial][f"{slide}_{fov}"]["metadata"] = row.to_dict()
            except:
                print(f"\tFOV {fov} in positions file but not metadata file, skipped")
    return adata
