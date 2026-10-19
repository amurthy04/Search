"""Convert a BERT-style embedding model to CoreML (fixed batch x seq) and benchmark on ANE/GPU/CPU."""
import sys, time, numpy as np, torch
import coremltools as ct
from transformers import AutoModel, AutoTokenizer
sys.path.insert(0, ".")
from search.minibert import MiniBert

name = sys.argv[1] if len(sys.argv) > 1 else "BAAI/bge-small-en-v1.5"
B = int(sys.argv[2]) if len(sys.argv) > 2 else 32
S = int(sys.argv[3]) if len(sys.argv) > 3 else 256
out = f"data/index/coreml_{name.split('/')[-1]}_b{B}_s{S}.mlpackage"

tok = AutoTokenizer.from_pretrained(name)
w = MiniBert.from_hf(name, "cls", fixed=(B, S))
# sanity: compare to HF reference on real text
_hf = AutoModel.from_pretrained(name, torch_dtype=torch.float32).eval()
_enc = tok(["the cat sat on the mat", "web crawler latency"] * (B // 2), return_tensors="pt", padding="max_length", max_length=S)
with torch.no_grad():
    _ref = _hf(**_enc).last_hidden_state[:, 0]; _ref = _ref / _ref.norm(dim=1, keepdim=True)
    _mine = w(_enc["input_ids"], _enc["attention_mask"])
print("minibert vs hf cos:", (_ref * _mine).sum(1).tolist(), flush=True)
ids = torch.randint(100, 1000, (B, S)); mask = torch.ones(B, S, dtype=torch.long)
t0 = time.time()
traced = torch.jit.trace(w, (ids, mask))
ml = ct.convert(traced, inputs=[ct.TensorType(name="input_ids", shape=(B, S), dtype=np.int32),
                                ct.TensorType(name="attention_mask", shape=(B, S), dtype=np.int32)],
                outputs=[ct.TensorType(name="emb")], compute_precision=ct.precision.FLOAT16,
                minimum_deployment_target=ct.target.macOS15, convert_to="mlprogram")
ml.save(out)
print(f"converted in {time.time()-t0:.0f}s -> {out}", flush=True)

ref = w(ids, mask).detach().numpy()
for cu, label in [(ct.ComputeUnit.CPU_AND_NE, "ANE"), (ct.ComputeUnit.CPU_AND_GPU, "GPU"), (ct.ComputeUnit.ALL, "ALL")]:
    t0 = time.time()
    m = ct.models.MLModel(out, compute_units=cu)
    load = time.time() - t0
    inp = {"input_ids": ids.numpy().astype(np.int32), "attention_mask": mask.numpy().astype(np.int32)}
    r = m.predict(inp)["emb"]
    cos = float(np.mean(np.sum(r * ref, axis=1)))
    n = 10
    t0 = time.perf_counter()
    for _ in range(n): m.predict(inp)
    dt = time.perf_counter() - t0
    print(f"{label}: load {load:.1f}s  {B*n/dt:.0f} seq/s  (batch {B} x seq {S}, cos vs torch {cos:.4f})", flush=True)
