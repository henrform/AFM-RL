import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from env.afm_env import AfmEnvironment

CHUNK_SIZE = 16
ANGLES = [i*10 for i in range(36)] #0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
TXS = [i*3 for i in range(7)]
TYS = [i*3 for i in range(7)]


def chunk_list(lst: list, chunk_size: int) -> list[list]:
    """Split *lst* into chunks of at most *chunk_size* elements."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def process_chunk(chunk_idx: int, total: int, chunk: list[dict],
                  surface_path: str, params_path: str, save_path: str,
                  i_platform: int) -> str:
    """Process a single chunk of scan_params and save the resulting views."""
    print(f"[Chunk {chunk_idx + 1}/{total}] Starting ({len(chunk)} views)...")
    env = AfmEnvironment(
        surface_configs=[{
            'surface_path': surface_path,
            'params_path': params_path,
            'scan_params': chunk,
        }],
        i_platform=i_platform,
    )
    env.save_to_file(save_path, surface_idx=0)
    del env
    return f"[Chunk {chunk_idx + 1}/{total}] Done."


def main():
    parser = argparse.ArgumentParser(description="Pre-generate AFM environment views in chunks.")
    parser.add_argument("surface_path", type=str, help="Path to the .xyz surface file.")
    parser.add_argument("params_path", type=str, help="Path to the .ini params file.")
    parser.add_argument("save_path", type=str, help="Directory where views are saved.")
    parser.add_argument(
        "--chunk_size", type=int, default=CHUNK_SIZE,
        help=f"Number of scan_params per chunk (default: {CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of chunks to process concurrently (default: 1).",
    )
    parser.add_argument(
        "--i_platform", type=int, default=0,
        help="OpenCL platform index passed to every worker (default: 0).",
    )
    args = parser.parse_args()

    scan_params = [
        {'angle_deg': angle_deg, 'tx': tx, 'ty': ty}
        for angle_deg in ANGLES
        for tx in TXS
        for ty in TYS
    ]

    # Skip params whose view folder already exists
    def view_folder_exists(sp: dict) -> bool:
        ang, tx, ty = sp['angle_deg'], sp['tx'], sp['ty']
        folder = os.path.join(args.save_path, f"view_ang{ang:g}_tx{tx:g}_ty{ty:g}")
        return os.path.isdir(folder)

    remaining = [sp for sp in scan_params if not view_folder_exists(sp)]
    skipped = len(scan_params) - len(remaining)
    if skipped:
        print(f"Skipping {skipped} already-existing view(s).")

    if not remaining:
        print("All views already exist. Nothing to do.")
        return

    chunks = chunk_list(remaining, args.chunk_size)
    total = len(chunks)
    print(f"Total views: {len(remaining)}, chunks: {total} (size ≤ {args.chunk_size}), workers: {args.workers}")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_chunk, i, total, chunk,
                args.surface_path, args.params_path, args.save_path,
                args.i_platform,
            ): i
            for i, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            print(future.result())

    print("All chunks processed.")


if __name__ == "__main__":
    main()