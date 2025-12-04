# depmap_server.py

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import pandas as pd
import requests
from mcp.server.fastmcp import FastMCP

# -----------------------------------------------------------------------------
# STEP 0: Configure MCP server
# -----------------------------------------------------------------------------

# Name shown to the MCP client (e.g., in tool list)
mcp = FastMCP(
    name="depmap-crispr",
    json_response=True,      # return JSON-structured results
    stateless_http=True,     # good default for HTTP/remote deployment
)

# URL to the DepMap CRISPR Gene Effect dataset (use your real URL here)
CRISPR_GENE_EFFECT_URL = "https://www.dropbox.com/scl/fi/oxqalmas5igfhkcxnrrer/CRISPRGeneEffectTrunc.csv?rlkey=rt0pnygna3s11hisfwwalcd7p&st=c9tkgsj8&dl=0"  # TODO: replace
LOCAL_PATH = "data/CRISPRGeneEffect.csv"


# -----------------------------------------------------------------------------
# STEP 1: Original helper functions (your "backbone")
# -----------------------------------------------------------------------------

def download_if_missing(url: str = CRISPR_GENE_EFFECT_URL, local_path: str = LOCAL_PATH) -> str:
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
def load_crispr_gene_effect(url: str = CRISPR_GENE_EFFECT_URL,
                            local_path: str = LOCAL_PATH) -> pd.DataFrame:
    """
    Cached loader for the CRISPR gene effect matrix.

    Downloads once (if needed) and then reuses the cached CSV.
    Returns a DataFrame with cell lines as rows and genes as columns.
    """
    path = download_if_missing(url, local_path)
    df = pd.read_csv(path, index_col=0)  # first column = DepMap_ID or similar

    # Normalize gene column names to upper case
    df.columns = [c.upper() for c in df.columns]

    return df


def lookup_crispr_gene_effect(gene: str,
                              depmap_id: str,
                              url: str = CRISPR_GENE_EFFECT_URL,
                              local_path: str = LOCAL_PATH) -> Optional[float]:
    """
    Pure Python helper that does the actual DepMap lookup.
    """
    df = load_crispr_gene_effect(url, local_path)

    gene = gene.upper()

    if gene not in df.columns:
        # Could also raise, but returning None and an error message is gentler.
        return None

    if depmap_id not in df.index:
        return None

    val = df.loc[depmap_id, gene]

    try:
        return float(val)
    except Exception:
        return None

def get_expression_tpm_log1p(
    gene: str,
    depmap_id: str,
    gene_col_candidates=("HugoSymbol", "gene", "GeneSymbol"),
) -> Optional[float]:
    """
    Look up log1p(TPM) expression for a gene in a cell line.

    Assumes a matrix where:
      - Rows = genes, with a gene-name column (e.g. 'HugoSymbol')
      - Columns = DepMap_IDs (e.g. 'ACH-000001', 'ACH-000002', ...)

    Parameters
    ----------
    gene : str
        Gene symbol, e.g. "KRAS".
    depmap_id : str
        DepMap cell line ID.
    gene_col_candidates : tuple
        Column names to try for the gene symbol column.

    Returns
    -------
    float or None
        log1p(TPM) value, or None if not found.
    """
    df = _load_expression_csv()

    gene_col = None
    for col in gene_col_candidates:
        if col in df.columns:
            gene_col = col
            break

    if gene_col is None:
        raise ValueError(
            f"Could not find a gene symbol column among {gene_col_candidates}. "
            "Check your truncated expression file."
        )

    if depmap_id not in df.columns:
        # truncated matrix may not include this cell line
        return None

    row = df.loc[df[gene_col] == gene]
    if row.empty:
        return None

    value = row.iloc[0][depmap_id]
    if pd.isna(value):
        return None
    return float(value)

# -----------------------------------------------------------------------------
# STEP 2: Expose the function as an MCP tool
# -----------------------------------------------------------------------------

@mcp.tool()
def get_crispr_gene_effect(gene: str, depmap_id: str) -> dict:
    """
    Look up the CRISPR gene effect score for a specific gene in a specific cell line.

    Args:
        gene: HGNC gene symbol (e.g., "TP53", "KRAS", "BRCA1").
        depmap_id: DepMap cell line ID (e.g., "ACH-000001").

    Returns:
        A JSON-serializable dict with:
            - gene
            - depmap_id
            - effect (float or null)
            - found (bool)
            - message (string, optional error message)
    """
    score = lookup_crispr_gene_effect(gene=gene, depmap_id=depmap_id)

    if score is None:
        return {
            "gene": gene.upper(),
            "depmap_id": depmap_id,
            "effect": None,
            "found": False,
            "message": (
                "Gene or cell line not found in CRISPRGeneEffect matrix, "
                "or value could not be parsed as a float."
            ),
        }

    return {
        "gene": gene.upper(),
        "depmap_id": depmap_id,
        "effect": score,
        "found": True,
    }


# -----------------------------------------------------------------------------
# STEP 3: Run the MCP server
# -----------------------------------------------------------------------------

def main():
    """
    Entry point. Starts the MCP server.

    For local dev / HTTP transport, this will listen on http://localhost:8000/mcp
    by default when using `transport="streamable-http"`.
    """
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()