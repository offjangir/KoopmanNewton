"""Check the generated Kepler data against its metadata (shape, ranges, parameter ranges)."""
import sys, importlib.util, numpy as np, torch
d = sys.argv[1]
m = torch.load(f"{d}/metadata.pt", weights_only=False)
print("metadata", m)
X = np.concatenate([torch.load(f"{d}/trajectories_chunk_{i:06d}.pt").numpy() for i in range(m["num_chunks"])])
print("shape", X.shape, "finite", np.isfinite(X).all())
print("x range", X[..., 0].min(), X[..., 0].max(), "y range", X[..., 1].min(), X[..., 1].max())
P = []
for i in range(m["num_chunks"]):
    s = importlib.util.spec_from_file_location("p", f"{d}/orbital_params_chunk_{i:06d}.py"); mod = importlib.util.module_from_spec(s); s.loader.exec_module(mod); P += mod.orbital_params
e = np.array([p["e"] for p in P]); a = np.array([p["a"] for p in P])
print("n params", len(P), "e range", e.min(), e.max(), "a range", a.min(), a.max())
r = np.linalg.norm(X, axis=-1)
print("r range", r.min(), r.max())
