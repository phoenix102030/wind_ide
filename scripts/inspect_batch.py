import argparse
from src.data.offline_dataset import load_offline_nwp_one_height, OfflineIDEPatchDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nwp-file", type=str, required=True)
    p.add_argument("--height", type=int, default=100)
    p.add_argument("--seq-len", type=int, default=4)
    args = p.parse_args()

    bundle = load_offline_nwp_one_height(args.nwp_file, height=args.height)
    ds = OfflineIDEPatchDataset(bundle, seq_len=args.seq_len)
    batch = ds[0]
    for k, v in batch.items():
        print(k, tuple(v.shape), v.dtype)


if __name__ == "__main__":
    main()