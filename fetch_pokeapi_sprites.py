"""Download reproducible public sprite banks for external-data matching experiments.

Downloaded images stay under data/external/ and are not included in Git.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


BASE = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon"
PATHS = {
    "default": "{id}.png",
    "official-artwork": "other/official-artwork/{id}.png",
    "home": "other/home/{id}.png",
}


def download(pokemon_id: int, variant: str, directory: Path) -> tuple[int, str]:
    target = directory / f"{pokemon_id}.png"
    if target.is_file() and target.stat().st_size > 100:
        return pokemon_id, "cached"
    url = f"{BASE}/{PATHS[variant].format(id=pokemon_id)}"
    request = urllib.request.Request(url, headers={"User-Agent": "PokeWars research/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                content = response.read()
            if not content.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError(f"Unexpected non-PNG response for {url}")
            temporary = target.with_suffix(".tmp")
            temporary.write_bytes(content)
            temporary.replace(target)
            return pokemon_id, "downloaded"
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return pokemon_id, "missing"
            error = str(exc)
        except (OSError, ValueError) as exc:
            error = str(exc)
        time.sleep(0.5 * (attempt + 1))
    return pokemon_id, f"error: {error}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(PATHS), required=True)
    parser.add_argument("--first-id", type=int, default=1)
    parser.add_argument("--last-id", type=int, default=1025)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/external/pokeapi"))
    args = parser.parse_args()
    if not (1 <= args.first_id <= args.last_id <= 1025) or not 1 <= args.workers <= 32:
        parser.error("Invalid ID range or worker count")
    directory = args.output_dir / args.variant
    directory.mkdir(parents=True, exist_ok=True)
    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, i, args.variant, directory): i
                   for i in range(args.first_id, args.last_id + 1)}
        for completed, future in enumerate(as_completed(futures), 1):
            pokemon_id, status = future.result()
            results[pokemon_id] = status
            if completed % 100 == 0 or completed == len(futures):
                print(f"{args.variant}: {completed}/{len(futures)} processed", flush=True)
    summary = {
        "variant": args.variant,
        "source": f"{BASE}/{PATHS[args.variant]}",
        "repository": "https://github.com/PokeAPI/sprites",
        "first_id": args.first_id,
        "last_id": args.last_id,
        "counts": {status: sum(value == status for value in results.values())
                   for status in sorted(set(results.values()))},
        "missing_ids": [i for i, status in sorted(results.items()) if status == "missing"],
        "error_ids": {str(i): status for i, status in sorted(results.items())
                      if status.startswith("error:")},
    }
    (directory / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(summary["counts"], flush=True)
    if summary["error_ids"]:
        raise RuntimeError(f"Failed to fetch {len(summary['error_ids'])} images")


if __name__ == "__main__":
    main()
