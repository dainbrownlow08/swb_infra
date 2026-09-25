"""openSMILE 3.0 prosodyShs over Switchboard side recordings and MISC task recordings -> per-file npz tracks.

One detector for both corpora (MISC's shipped tracks came from an older openSMILE build: loudness
matches ours at r .9998, F0/voicing do not). Config: opensmile_conf/prosody/prosodyShs.conf (upstream
with two include lines changed); the shared/ includes are linked from the installed `opensmile` package.

Outputs (10 ms frames; f0 Hz with 0 = unvoiced, voicing probability, pcm_loudness):
  swb  -> utterances_v2/derived/opensmile/sw0XXXX.{A,B}.npz   (3,988 sides; sources from .swb_extract_index.json)
  misc -> corpus/misc/derived_opensmile_v3/Pxx_Ty.npz          (220 recordings)
Existing outputs are skipped, so a run resumes. Run from the repo root:
  PYTHONPATH=src python3 analysis/c_alpha/run_opensmile.py swb|misc [workers]
"""
import glob, json, os, shutil, sys, tempfile, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONF_SRC = HERE / "opensmile_conf" / "prosody" / "prosodyShs.conf"
_smile = None


def build_conf_tree() -> Path:
    """Temp dir with our prosody/prosodyShs.conf and a link to the package's shared/ includes."""
    import opensmile
    tmp = Path(tempfile.mkdtemp(prefix="osconf_"))
    (tmp / "prosody").mkdir()
    shutil.copy(CONF_SRC, tmp / "prosody" / "prosodyShs.conf")
    os.symlink(Path(opensmile.__file__).parent / "core" / "config" / "shared", tmp / "shared")
    return tmp / "prosody" / "prosodyShs.conf"


def work(args):
    src, dst = args
    global _smile
    import soundfile as sf, opensmile
    if _smile is None:
        _smile = opensmile.Smile(feature_set=str(build_conf_tree()), feature_level="lld", verbose=False)
    try:
        y, sr = sf.read(src, dtype="float32")
        if y.ndim > 1:
            y = y.mean(1)
        d = _smile.process_signal(y, sr)
        np.savez_compressed(dst, f0=d["F0final_sma"].to_numpy(np.float32),
                            voicing=d["voicingFinalUnclipped_sma"].to_numpy(np.float32),
                            loudness=d["pcm_loudness_sma"].to_numpy(np.float32))
        return dst, len(d)
    except Exception as e:  # one bad file must not kill a 4,000-file pass
        return dst, f"ERR {e}"


def main(which: str, workers: int = 12) -> int:
    if which == "swb":
        idx = json.load(open(ROOT / ".swb_extract_index.json"))
        out = ROOT / "utterances_v2" / "derived" / "opensmile"
        jobs = [(e["path"], str(out / (Path(e["path"]).stem + ".npz"))) for e in idx]
    elif which == "misc":
        out = ROOT / "corpus" / "misc" / "derived_opensmile_v3"
        srcs = sorted(glob.glob(str(ROOT / "corpus/misc/MISC_Public_Release/Audio/*/P*_T*.wav")))
        jobs = [(p, str(out / (Path(p).stem + ".npz"))) for p in srcs]
    else:
        raise SystemExit("usage: run_opensmile.py swb|misc [workers]")
    out.mkdir(parents=True, exist_ok=True)
    jobs = [j for j in jobs if not Path(j[1]).exists()]
    t0 = time.time(); n_err = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (dst, res) in enumerate(ex.map(work, jobs, chunksize=4), 1):
            if isinstance(res, str):
                n_err += 1; print("  ", dst, res, flush=True)
            if i % 500 == 0:
                print(f"  {i:,}/{len(jobs):,} [{time.time() - t0:.0f}s]", flush=True)
    print(f"{which}: {len(jobs):,} files, {n_err} errors, {time.time() - t0:.0f}s")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 12))
