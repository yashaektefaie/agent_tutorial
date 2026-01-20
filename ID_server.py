# depmap_server.py

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional, Dict

import pandas as pd
import requests
from mcp.server.fastmcp import FastMCP

# -----------------------------------------------------------------------------
# STEP 0: Configure MCP server
# -----------------------------------------------------------------------------

PORT = int(os.environ.get("PORT", "8000"))

# Name shown to the MCP client (e.g., in tool list)
mcp = FastMCP(
    name="depmap-crispr",
    json_response=True,      # return JSON-structured results
    stateless_http=True,     # good default for HTTP/remote deployment
    host="0.0.0.0",   
    port=PORT,       
)

# URL to the DepMap CRISPR Gene Effect dataset (use your real URL here)
CRISPR_GENE_EFFECT_URL = "https://www.dropbox.com/scl/fi/p4h4gmdbus4nndrxk081r/CRISPRi_Pseudomonas_aeruginosa_UCBPP-PA14_guidesCRISPRi1_genes_counts.tsv?rlkey=mt9pdamr0r9zr6x760et3dtiu&st=fxc7mdse&dl=1" 
METADATA_URL = "https://www.dropbox.com/scl/fi/plas9uwyz66driom6f7nu/MOCP-0162_key.txt?rlkey=ptwrrmd9o7xfrhrbgp981x6vx&st=jaz3fgr4&dl=1" 
GFF_URL = "https://www.dropbox.com/scl/fi/l2cp1bw1qrtnmteveae4s/Pseudomonas_aeruginosa_UCBPP-PA14_guidesCRISPRi1_ALL.gff?rlkey=7vyn0krh9rzt37ur3gs3stkis&st=lvwmafvh&dl=1"

# -----------------------------------------------------------------------------
# STEP 1: Original helper functions (your "backbone")
# -----------------------------------------------------------------------------

def download_if_missing(url: str = CRISPR_GENE_EFFECT_URL, local_path: str = "data/CRISPRGeneEffectID.csv") -> str:
    """
    Download the CSV once if it doesn't exist locally.
    Returns the local path.
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if not os.path.exists(local_path):
        # NOTE: For stdio MCP servers, avoid print() and use logging to stderr.
        # For HTTP transport, print() is okay.
        print(f"[depmap-crispr] Downloading DepMap file from {url} ...")
        resp = requests.get(url)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(resp.content)
        print(f"[depmap-crispr] Saved to {local_path}")

    return local_path


@lru_cache(maxsize=1)
def load_url(url: str = CRISPR_GENE_EFFECT_URL,
            local_path: str = "data/CRISPRGeneEffectID.csv",
            first_col: bool = False               ) -> pd.DataFrame:
    """
    Cached loader for the CRISPR gene effect matrix.

    Downloads once (if needed) and then reuses the cached CSV.
    Returns a DataFrame with cell lines as rows and genes as columns.
    """
    path = download_if_missing(url, local_path)
    if first_col:
        if 'CRISPRGeneEffectID' in path or 'metadata' in path:
            df = pd.read_csv(path, index_col=0, sep="\t", engine="python", on_bad_lines="warn")
        else:
            df = pd.read_csv(path, index_col=0, engine="python", on_bad_lines="warn")
    else:
        if 'CRISPRGeneEffectID' in path or 'metadata' in path:
            df = pd.read_csv(path, sep="\t", engine="python", on_bad_lines="warn")
        else:
            df = pd.read_csv(path, engine="python", on_bad_lines="warn")
    return df


def return_gene_expression_vector(strain_ID:str, 
                      url:str = CRISPR_GENE_EFFECT_URL,
                      local_path:str = "data/CRISPRGeneEffectID.csv") -> Dict[str, float]:
    """
    Return the gene expression vector for a given strain ID.

    Parameters
    ----------
    strain_ID : str
        Strain ID.
    url: str
        URL to download the CSV file from.
    local_path: str
        Local path to store the CSV file.

    Returns
    -------
    dict or None
        Gene expression vector as a dict, or None if not found.
    """
    df = load_url(url, local_path)

    if strain_ID not in df.columns:
        return None
    
    return  (
        df[['Geneid', strain_ID]]
        .set_index('Geneid')[strain_ID]
        .to_dict()
    )

def return_ID_top_count(gene:str, top_k:int = 5,
                      url:str = CRISPR_GENE_EFFECT_URL,
                      local_path:str = "data/CRISPRGeneEffectID.csv") -> Dict[str, float]:
    """
    Return the IDs with the top gene counts for a given gene symbol. 

    Parameters
    ----------
    gene : str
        Gene symbol, e.g. "KRAS".
    top_k : int
        Number of top matches to return.
    url: str
        URL to download the CSV file from.
    local_path: str
        Local path to store the CSV file.

    Returns
    -------
    list or None
        List of top K matching gene IDs, or None if not found.
    """
    df = load_url(url, local_path)

    if gene not in df['Geneid'].values:
        return None
    
    row = df.loc[
        df['Geneid'] == gene
    ].iloc[0]

    topk = (
        row
        .drop('Geneid')          # remove identifier
        .sort_values(ascending=False)
        .head(top_k)
        .to_dict()
    )

    return topk

def gather_strain_metadata(
    strain_id: str,
    keep_cols: Optional[list] = None,
    url: str = METADATA_URL,
    local_path: str = "data/metadata.csv"
) -> Optional[dict]:
    """
    Return metadata for a strain.

    Typical columns in Model.csv include:
      - Dose
      - Compound
      - Media
      - DNA_concentration
      - etc.

    Parameters
    ----------
    strain_id : str
        Strain ID.
    keep_cols : list or None
        If provided, restrict output to these columns .

    Returns
    -------
    dict or None
        Metadata as a dict, or None if cell line not found.
    """
    df = load_url(url, local_path)

    if strain_id not in df['Sample_ID'].values:
        return None

    row = df[df['Sample_ID'] == strain_id]
    if row.empty:
        return None

    series = row.iloc[0]
    if keep_cols is not None:
        cols = ["Sample_ID"] + [c for c in keep_cols if c in series.index]
        series = series[cols]

    return series.to_dict()


# -----------------------------------------------------------------------------
# STEP 2: Expose the functions as MCP tools
# -----------------------------------------------------------------------------

@mcp.tool()
def get_gene_expression_vector_tool(strain_id: str) -> dict:
    """
    Return the gene expression vector for a given strain ID.

    Args:
        strain_id: Strain ID (e.g., a sample column name from the counts matrix).

    Returns:
        A JSON-serializable dict with:
            - strain_id
            - expression (dict mapping gene IDs to counts, or null)
            - found (bool)
            - message (string, optional error message)
    """
    expression = return_gene_expression_vector(strain_ID=strain_id)

    if expression is None:
        return {
            "strain_id": strain_id,
            "expression": None,
            "found": False,
            "message": "Strain ID not found in the counts matrix.",
        }

    return {
        "strain_id": strain_id,
        "expression": expression,
        "found": True,
    }


@mcp.tool()
def get_top_count_ids_tool(gene: str, top_k: int = 5) -> dict:
    """
    Return the IDs with the top gene counts for a given gene symbol.

    Args:
        gene: Gene identifier (e.g., from the Geneid column).
        top_k: Number of top matches to return (default: 5).

    Returns:
        A JSON-serializable dict with:
            - gene
            - top_k
            - top_ids (dict mapping strain IDs to counts, or null)
            - found (bool)
            - message (string, optional error message)
    """
    top_ids = return_ID_top_count(gene=gene, top_k=top_k)

    if top_ids is None:
        return {
            "gene": gene,
            "top_k": top_k,
            "top_ids": None,
            "found": False,
            "message": "Gene not found in the counts matrix.",
        }

    return {
        "gene": gene,
        "top_k": top_k,
        "top_ids": top_ids,
        "found": True,
    }


@mcp.tool()
def get_strain_metadata_tool(strain_id: str, keep_cols: Optional[list] = None) -> dict:
    """
    Retrieve metadata for a specific strain.

    Args:
        strain_id: Strain/Sample ID.
        keep_cols: Optional list of column names to include in the output.

    Returns:
        A JSON-serializable dict with:
            - strain_id
            - metadata (dict or null)
            - found (bool)
            - message (string, optional error message)
    """
    metadata = gather_strain_metadata(strain_id=strain_id, keep_cols=keep_cols)

    if metadata is None:
        return {
            "strain_id": strain_id,
            "metadata": None,
            "found": False,
            "message": "Strain ID not found in metadata.",
        }

    return {
        "strain_id": strain_id,
        "metadata": metadata,
        "found": True,
    }

# -----------------------------------------------------------------------------
# STEP 3: Run the MCP server
# -----------------------------------------------------------------------------

# def main():
#     """
#     Entry point. Starts the MCP server.

#     For local dev / HTTP transport, this will listen on http://localhost:8000/mcp
#     by default when using `transport="streamable-http"`.
#     """
#     #mcp.run(transport="streamable-http")
#     mcp.run()


# if __name__ == "__main__":
#     # main()
#     mcp.run(transport="http", port=8000)

# -----------------------------------------------------------------------------
# STEP 3: Run the MCP server
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Render sets the PORT environment variable.
    # FastMCP automatically binds to 0.0.0.0 and uses this port.
    os.environ.setdefault("PORT", os.environ.get("PORT", "8000"))

    # Start a Streamable HTTP MCP server.
    # This will expose your server at:  http://<host>:<PORT>/mcp
    mcp.run(transport="streamable-http")

# if __name__ == "__main__":
#     import contextlib
#     from fastapi import FastAPI
#     from fastapi.middleware.cors import CORSMiddleware
#     import uvicorn

#     @contextlib.asynccontextmanager
#     async def lifespan(app: FastAPI):
#         # REQUIRED when serving via uvicorn instead of mcp.run()
#         async with mcp.session_manager.run():
#             yield

#     app = FastAPI(lifespan=lifespan, redirect_slashes=False)

#     # CORS so the browser-based Inspector can do OPTIONS preflight
#     app.add_middleware(
#         CORSMiddleware,
#         allow_origins=["*"],
#         allow_credentials=True,
#         allow_methods=["*"],
#         allow_headers=["*"],
#     )

#     # IMPORTANT: mount at "/" because the streamable_http_app already serves at /mcp
#     app.mount("/", mcp.streamable_http_app())

#     uvicorn.run(app, host="127.0.0.1", port=8000)