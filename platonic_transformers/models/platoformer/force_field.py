import math

import torch
import torch.nn as nn
from .platoformer import PlatonicTransformer
from .chg_spin_emb import ChgSpinEmbedding


class PlatonicForceField(nn.Module):
    """Wrapper around PlatonicTransformer for molecular force field prediction.

    Matches the interface expected by the scaling-laws training pipeline:
    forward(data) -> {"energy": Tensor, "forces": Tensor}

    Charge/spin conditioning follows the eSEN/UMA recipe: separate Random Fourier
    Features (`pos_emb`) for charge and spin, concat, mix through Linear+SiLU,
    then either add to or concat onto the per-atom embedding before the first
    PlatonicLinear. Injection is at input only (eSEN injects at every layer; we
    don't, for simplicity).
    """

    def __init__(
        self,
        max_num_elements: int = 100,
        hidden_dim: int = 128,
        nhead: int = 8,
        num_layers: int = 9,
        solid_name: str = "tetrahedron",
        ffn_dim_factor: int = 4,
        dropout: float = 0.0,
        rope_sigma: float = 1.0,
        learned_freqs: bool = True,
        dense_mode: bool = True,
        embed_init_std: float = None,
        layer_scale_init_value: float = None,
        attention: bool = False,
        avg_num_nodes: float = 1.0,
        rope_on_values: bool = False,
        activation: str = "gelu",
        readout_activation: str = None,  # None → keep legacy nn.GELU readout
        atom_embed: str = "learned",
        drop_path_rate: float = 0.0,
        freq_init: str = "random",
        attention_backend: str = "scatter",
        qk_norm: bool = False,
        swiglu: bool = False,
        qk_dim_factor: int = 1,
        v_dim_factor: int = 1,
        rope_v_independent: bool = False,
        use_key: bool = False,
        norm_type: str = "layernorm",   # "layernorm" or "rmsnorm" (LLaMA-style)
        # Charge/spin conditioning (eSEN/UMA recipe)
        chgspin_mode: str = "off",          # "off" | "add" | "concat"
        chgspin_emb_dim: int = None,        # pos_emb size; None → embed_dim
        chgspin_mix_init_std: float = 0.02, # init std of the chgspin_mix Linear weights
        chgspin_layerwise: bool = False,    # additive per-block injection (hidden_dim space)
        chgspin_layerwise_gate: bool = False,  # learnable per-block γ_chgspin gate on the additive injection
        chgspin_film: bool = False,         # per-block FiLM modulation; mutually exclusive with chgspin_layerwise
        # Radius graph (sparse attention): None → dense within-graph attention.
        # When set (e.g. 2.0 Å), each forward pass builds a radius_graph(pos, r)
        # once and reuses it across all blocks. Requires dense_mode=False and
        # attention_backend='scatter'. Flash attention is not compatible.
        interaction_radius: float = None,
        cutoff_p: int = 6,
        max_num_neighbors: int = 1000,
        # Dual-stage local→global blocks (AllScAIP-style). When True, each
        # "logical layer" becomes a pair of PlatonicBlocks: local (uses
        # radius edge_index) followed by global (full within-graph). Doubles
        # per-layer params and FLOPs at the same num_layers.
        local_global: bool = False,
    ):
        super().__init__()

        from .groups import PLATONIC_GROUPS
        num_G = PLATONIC_GROUPS[solid_name.lower()].G
        embed_dim = hidden_dim // num_G  # per-group-element atom embedding
        self.num_G = num_G
        self.hidden_dim = hidden_dim

        self.atom_embed_type = atom_embed
        if atom_embed == "khot":
            from .khot_embeddings import KHOT_EMBEDDINGS
            khot_dim = len(next(iter(KHOT_EMBEDDINGS.values())))
            khot_table = torch.zeros(max_num_elements, khot_dim)
            for z, vec in KHOT_EMBEDDINGS.items():
                if z < max_num_elements:
                    khot_table[z] = torch.tensor(vec, dtype=torch.float32)
            self.register_buffer("khot_table", khot_table)
            self.atom_embedding = nn.Linear(khot_dim, embed_dim, bias=False)
        else:
            self.atom_embedding = nn.Embedding(max_num_elements, embed_dim)
            init_std = embed_init_std if embed_init_std is not None else 1.0 / math.sqrt(embed_dim)
            nn.init.normal_(self.atom_embedding.weight, std=init_std)

        # --- Charge/Spin: eSEN recipe (pos_emb x2 + Linear+SiLU mix) ---
        # chgspin_mode controls the *input* injection only:
        #   "off":    no input injection;                                input_dim = embed_dim
        #   "add":    mixed (size embed_dim) is added to atom_emb;        input_dim = embed_dim
        #   "concat": mixed (size embed_dim) is concatted onto atom_emb;  input_dim = 2 * embed_dim
        # Layerwise/FiLM are independent: they need the embedders/mix even if
        # chgspin_mode == "off" (so the conditioning signal exists per block).
        assert chgspin_mode in ("off", "add", "concat"), f"chgspin_mode={chgspin_mode!r}"
        self.chgspin_mode = chgspin_mode

        # Validation: layerwise and FiLM are mutually exclusive; layerwise_gate
        # only makes sense with layerwise=True.
        if chgspin_layerwise and chgspin_film:
            raise ValueError("chgspin_layerwise and chgspin_film are mutually exclusive.")
        if chgspin_layerwise_gate and not chgspin_layerwise:
            raise ValueError("chgspin_layerwise_gate requires chgspin_layerwise=True.")

        self.chgspin_layerwise = chgspin_layerwise
        self.chgspin_layerwise_gate = chgspin_layerwise_gate
        self.chgspin_film = chgspin_film

        needs_chgspin_embedders = (chgspin_mode != "off") or chgspin_layerwise or chgspin_film
        if not needs_chgspin_embedders:
            self.charge_embedder = None
            self.spin_embedder = None
            self.chgspin_mix = None
        else:
            D = chgspin_emb_dim if chgspin_emb_dim is not None else embed_dim
            self.charge_embedder = ChgSpinEmbedding(
                embedding_type="pos_emb", embedding_target="charge", embedding_size=D,
            )
            self.spin_embedder = ChgSpinEmbedding(
                embedding_type="pos_emb", embedding_target="spin", embedding_size=D,
            )
            # eSEN: Linear(2*D, embed_dim) followed by SiLU. The init std is tunable
            # because pos_emb has per-row norm ~sqrt(D/2); with std=0.02 the post-SiLU
            # signal is ~1.4× the atom embedding init (std=1/sqrt(embed_dim)).
            self.chgspin_mix = nn.Linear(2 * D, embed_dim)
            nn.init.normal_(self.chgspin_mix.weight, std=chgspin_mix_init_std)
            nn.init.constant_(self.chgspin_mix.bias, 0)

        # Input shape determined by chgspin_mode only (concat doubles the input).
        if chgspin_mode == "concat":
            input_dim = 2 * embed_dim
        else:
            input_dim = embed_dim

        self.transformer = PlatonicTransformer(
            input_dim=input_dim,
            input_dim_vec=0,
            hidden_dim=hidden_dim,
            output_dim=1,
            output_dim_vec=1,
            nhead=nhead,
            num_layers=num_layers,
            solid_name=solid_name,
            spatial_dim=3,
            dense_mode=dense_mode,
            scalar_task_level="node",
            vector_task_level="node",
            mean_aggregation=False,
            dropout=dropout,
            ffn_dim_factor=ffn_dim_factor,
            rope_sigma=rope_sigma,
            learned_freqs=learned_freqs,
            layer_scale_init_value=layer_scale_init_value,
            attention=attention,
            rope_on_values=rope_on_values,
            activation=activation,
            readout_activation=readout_activation,
            drop_path_rate=drop_path_rate,
            freq_init=freq_init,
            attention_backend=attention_backend,
            qk_norm=qk_norm,
            swiglu=swiglu,
            qk_dim_factor=qk_dim_factor,
            v_dim_factor=v_dim_factor,
            rope_v_independent=rope_v_independent,
            use_key=use_key,
            norm_type=norm_type,
            chgspin_layerwise=chgspin_layerwise,
            chgspin_layerwise_gate=chgspin_layerwise_gate,
            chgspin_film=chgspin_film,
            chgspin_embed_dim=embed_dim,
            chgspin_init_gamma=layer_scale_init_value if layer_scale_init_value is not None else 1.0,
            interaction_radius=interaction_radius,
            cutoff_p=cutoff_p,
            max_num_neighbors=max_num_neighbors,
            local_global=local_global,
        )

        self.avg_num_nodes = avg_num_nodes

    def forward(self, data):
        atomic_numbers = data["atomic_numbers"].long()
        pos = data["pos"]
        batch = data["batch"]

        # Embed atomic numbers
        if self.atom_embed_type == "khot":
            x = self.atom_embedding(self.khot_table[atomic_numbers])
        else:
            x = self.atom_embedding(atomic_numbers)

        # Charge/spin conditioning (eSEN recipe). The mixed signal feeds three
        # independent paths: input add/concat (chgspin_mode), per-block additive
        # injection (chgspin_layerwise, optionally gated), and per-block FiLM
        # modulation (chgspin_film). Layerwise & FiLM consume mixed_per_node
        # directly inside the transformer.
        chgspin_mixed_per_node = None
        if self.charge_embedder is not None:
            chg = self.charge_embedder(data["charge"].view(-1).float())
            spn = self.spin_embedder(data["spin"].view(-1).float())
            mixed = torch.nn.functional.silu(self.chgspin_mix(torch.cat([chg, spn], dim=-1)))
            chgspin_mixed_per_node = mixed[batch]  # (N, embed_dim)
            if self.chgspin_mode == "add":
                x = x + chgspin_mixed_per_node
            elif self.chgspin_mode == "concat":
                x = torch.cat([x, chgspin_mixed_per_node], dim=-1)
            # chgspin_mode == "off": signal still computed for layerwise/film, no input add

        # Only pass the per-node conditioning if a per-block path actually needs it.
        if not (self.chgspin_layerwise or self.chgspin_film):
            chgspin_mixed_per_node = None

        scalars, vectors = self.transformer(
            x=x, pos=pos, batch=batch,
            avg_num_nodes=self.avg_num_nodes,
            chgspin_mixed_per_node=chgspin_mixed_per_node,
        )

        # Precision experiment: sum per-atom energies in fp64 to avoid ~N·ε·|x|
        # accumulator error when summing thousands of atoms (≈1e-3 relative error
        # at N=8000 in fp32). Then cast both energy and forces back to fp32 to match
        # downstream loss/metric expectations — the fp64 detour is contained to the
        # readout subnetwork + this reduction.
        num_graphs = batch.max() + 1
        energy_fp64 = torch.zeros(num_graphs, device=scalars.device, dtype=torch.float64)
        energy_fp64.index_add_(0, batch, scalars.squeeze(-1).double())
        energy = energy_fp64.to(torch.float32)

        forces = vectors.squeeze(-2).to(torch.float32)
        return {"energy": energy, "forces": forces}


def build_force_field(model_cfg) -> "PlatonicForceField":
    """Instantiate a :class:`PlatonicForceField` from a ``model:`` config block.

    ``model_cfg`` is the ``model`` section of an OMol config (``configs/omol.yaml``)
    as a plain ``dict`` or an ``ml_collections.ConfigDict``; it is also what the
    Lightning checkpoints written by ``mains/main_omol.py`` store under
    ``hyper_parameters["config"]["model"]``. The key -> constructor-argument mapping
    mirrors ``OMolModel.__init__`` so that a checkpoint's ``net.*`` weights load
    into the returned module unchanged.
    """
    def get(key, default=None):
        if isinstance(model_cfg, dict):
            return model_cfg.get(key, default)
        return getattr(model_cfg, key, default)

    return PlatonicForceField(
        max_num_elements=int(get("max_num_elements", 100)),
        hidden_dim=int(get("hidden_dim")),
        nhead=int(get("num_heads")),
        num_layers=int(get("num_layers")),
        solid_name=str(get("solid_name", "tetrahedron")),
        ffn_dim_factor=int(get("ffn_dim_factor", 4)),
        dropout=float(get("dropout", 0.0)),
        rope_sigma=float(get("rope_sigma", 1.0)),
        learned_freqs=bool(get("learned_freqs", True)),
        dense_mode=bool(get("dense_mode", False)),
        embed_init_std=get("embed_init_std", None),
        layer_scale_init_value=get("layer_scale_init_value", None),
        attention=bool(get("attention", True)),
        avg_num_nodes=float(get("avg_num_nodes", 1.0)),
        rope_on_values=bool(get("rope_on_values", False)),
        activation=str(get("activation", "gelu")),
        readout_activation=get("readout_activation", None),
        atom_embed=str(get("atom_embed", "learned")),
        drop_path_rate=float(get("drop_path_rate", 0.0)),
        freq_init=str(get("freq_init", "random")),
        attention_backend=str(get("attention_backend", "scatter")),
        qk_norm=bool(get("qk_norm", False)),
        swiglu=bool(get("swiglu", False)),
        qk_dim_factor=int(get("qk_dim_factor", 1)),
        v_dim_factor=int(get("v_dim_factor", 1)),
        rope_v_independent=bool(get("rope_v_independent", False)),
        use_key=bool(get("use_key", False)),
        norm_type=str(get("norm_type", "layernorm")),
        chgspin_mode=str(get("chgspin_mode", "off")),
        chgspin_emb_dim=get("chgspin_embed_dim", None),
        chgspin_mix_init_std=float(get("chgspin_mix_init_std", 0.02)),
        chgspin_layerwise=bool(get("chgspin_layerwise", False)),
        chgspin_layerwise_gate=bool(get("chgspin_layerwise_gate", False)),
        chgspin_film=bool(get("chgspin_film", False)),
        interaction_radius=get("interaction_radius", None),
        cutoff_p=int(get("cutoff_p", 6)),
        max_num_neighbors=int(get("max_num_neighbors", 1000)),
        local_global=bool(get("local_global", False)),
    )
