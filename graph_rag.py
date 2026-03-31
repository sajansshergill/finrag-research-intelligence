"""
graph_rag.py
------------
GraphRAG: analyst relationship mapping via NetworkX.

Graph structure:
  Nodes: Analyst, Sector, Region, Company
  Edges:
    Analyst --covers--> Sector     (weight = report count)
    Analyst --covers--> Region     (weight = report count)
    Analyst --covers--> Company    (weight = report count)
    Sector  --contains-> Company   (membership)

Use cases:
  1. Find analysts who cross-cover related sectors (e.g., Energy + Materials)
  2. Surface reports from analysts with deep regional expertise
  3. Expand a company query to analysts who cover peer companies
  4. Multi-hop: "analysts who cover EM AND have changed a Buy recommendation"

Graph traversal enriches retrieval results — a query about one sector
also surfaces reports from analysts known to cover closely related sectors,
even if the exact keyword match is weak.
"""

from __future__ import annotations
import json
from collections import defaultdict

import networkx as nx


# ── Graph builder ─────────────────────────────────────────────────────────────

class AnalystGraph:
    """
    Builds and queries a NetworkX bipartite graph from the research corpus.
    Node types: 'analyst', 'sector', 'region', 'company'
    """

    def __init__(self):
        self.G = nx.Graph()

    def build_from_corpus(self, corpus_path: str) -> None:
        """Build the graph from the raw JSONL corpus."""
        with open(corpus_path) as f:
            reports = [json.loads(line) for line in f if line.strip()]

        edge_weights: dict[tuple, int] = defaultdict(int)

        for r in reports:
            analyst = f"analyst::{r['analyst_name']}"
            sector  = f"sector::{r['sector']}"
            region  = f"region::{r['region']}"
            company = f"company::{r['company']}"

            # Add nodes with type metadata
            for node, ntype in [
                (analyst, "analyst"), (sector, "sector"),
                (region, "region"),  (company, "company"),
            ]:
                if not self.G.has_node(node):
                    self.G.add_node(node, node_type=ntype,
                                    label=node.split("::")[1])

            # Add edges with weight accumulation
            for edge in [(analyst, sector), (analyst, region),
                         (analyst, company), (sector, company)]:
                edge_weights[edge] += 1

        for (u, v), weight in edge_weights.items():
            self.G.add_edge(u, v, weight=weight)

        print(f"✓ Graph built: {self.G.number_of_nodes()} nodes, "
              f"{self.G.number_of_edges()} edges")

    # ── Query methods ─────────────────────────────────────────────────────────

    def get_analysts_for_sector(self, sector: str,
                                min_reports: int = 2) -> list[str]:
        """Return analysts who cover a given sector with min_reports threshold."""
        node = f"sector::{sector}"
        if not self.G.has_node(node):
            return []
        neighbors = [
            self.G.nodes[n]["label"]
            for n in self.G.neighbors(node)
            if self.G.nodes[n]["node_type"] == "analyst"
            and self.G[node][n]["weight"] >= min_reports
        ]
        return sorted(neighbors,
                      key=lambda a: -self.G[node][f"analyst::{a}"]["weight"])

    def get_analysts_for_region(self, region: str,
                                min_reports: int = 2) -> list[str]:
        """Return analysts with strong regional coverage."""
        node = f"region::{region}"
        if not self.G.has_node(node):
            return []
        return [
            self.G.nodes[n]["label"]
            for n in self.G.neighbors(node)
            if self.G.nodes[n]["node_type"] == "analyst"
            and self.G[node][n]["weight"] >= min_reports
        ]

    def get_related_sectors(self, sector: str, hops: int = 2) -> list[str]:
        """
        Find sectors connected to the given sector within N hops.
        Two sectors are related if analysts who cover one also tend to cover
        the other (shared analyst nodes).
        """
        sector_node = f"sector::{sector}"
        if not self.G.has_node(sector_node):
            return []

        # Find analysts who cover this sector
        analysts = [n for n in self.G.neighbors(sector_node)
                    if self.G.nodes[n]["node_type"] == "analyst"]

        # Find all sectors those analysts also cover
        related = set()
        for analyst in analysts:
            for neighbor in self.G.neighbors(analyst):
                if (self.G.nodes[neighbor]["node_type"] == "sector"
                        and neighbor != sector_node):
                    related.add(self.G.nodes[neighbor]["label"])

        return sorted(related)

    def get_analyst_profile(self, analyst_name: str) -> dict:
        """Return a structured profile for a named analyst."""
        node = f"analyst::{analyst_name}"
        if not self.G.has_node(node):
            return {}

        neighbors = list(self.G.neighbors(node))
        sectors  = [self.G.nodes[n]["label"] for n in neighbors
                    if self.G.nodes[n]["node_type"] == "sector"]
        regions  = [self.G.nodes[n]["label"] for n in neighbors
                    if self.G.nodes[n]["node_type"] == "region"]
        companies= [self.G.nodes[n]["label"] for n in neighbors
                    if self.G.nodes[n]["node_type"] == "company"]

        return {
            "analyst": analyst_name,
            "sectors":   sectors,
            "regions":   regions,
            "companies": companies,
            "degree":    self.G.degree(node),
        }

    def expand_query_context(self, sectors: list[str] | None = None,
                             regions: list[str] | None = None,
                             min_reports: int = 1) -> list[str]:
        """
        Given sectors/regions from a user query, return analyst names
        with coverage overlap — used to expand metadata filter scope.
        """
        analyst_names = set()

        for sector in (sectors or []):
            analyst_names.update(
                self.get_analysts_for_sector(sector, min_reports)
            )
        for region in (regions or []):
            analyst_names.update(
                self.get_analysts_for_region(region, min_reports)
            )

        return list(analyst_names)

    def compute_analyst_centrality(self) -> dict[str, float]:
        """
        Betweenness centrality for analyst nodes.
        High-centrality analysts bridge multiple sectors/regions —
        useful for surfacing cross-coverage insights.
        """
        centrality = nx.betweenness_centrality(self.G, weight="weight",
                                               normalized=True)
        return {
            k.split("::")[1]: v
            for k, v in centrality.items()
            if self.G.nodes[k]["node_type"] == "analyst"
        }


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    graph = AnalystGraph()
    graph.build_from_corpus("data/raw/corpus.jsonl")

    print("\n── Sample graph queries ─────────────────────────────────")

    em_analysts = graph.get_analysts_for_sector("Technology", min_reports=1)
    print(f"\nAnalysts covering Technology (top 5): {em_analysts[:5]}")

    related = graph.get_related_sectors("Energy")
    print(f"\nSectors related to Energy: {related[:5]}")

    if em_analysts:
        profile = graph.get_analyst_profile(em_analysts[0])
        print(f"\nProfile for {profile['analyst']}:")
        print(f"  Sectors  : {profile['sectors']}")
        print(f"  Regions  : {profile['regions']}")
        print(f"  Degree   : {profile['degree']}")