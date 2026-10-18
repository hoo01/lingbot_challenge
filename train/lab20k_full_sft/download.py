"""Download only the fixed full-SFT +6000 inference experiment, not the old archive."""
import argparse
import json
from pathlib import Path
from contract import validate_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    release = json.loads(Path(__file__).with_name('release.json').read_text())
    revision = release['revision']
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('Expected the immutable published HF commit, not main')
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=release['repo_id'], revision=revision,
                      allow_patterns=[release['prefix']+'**'], local_dir=args.output)
    bundle, _ = validate_bundle(args.output/release['prefix'])
    print(bundle)


if __name__ == '__main__':
    main()
