"""Project a retained complete IK candidate under an explicit nominal contract."""

import argparse
import json
from pathlib import Path
from xmimic.artifact_cache import ArtifactCache
from xmimic.retarget_workflow import qualify_cached_candidate, recipe_from_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("source-cache", "recipe", "robot", "cache"):
        parser.add_argument("--" + field, required=True, type=Path)
    parser.add_argument("--execution-ref", required=True)
    args = parser.parse_args()
    result = qualify_cached_candidate(
        args.source_cache,
        recipe_from_json(args.recipe.read_bytes()),
        args.robot,
        cache=ArtifactCache(args.cache),
        execution_ref=args.execution_ref,
    )
    print(
        json.dumps(
            {
                "cache_key": result.cache_key,
                "manifest_sha256": result.manifest_sha256,
                "disposition": result.disposition.value,
                "cache_hit": result.cache_hit,
                "directory": str(result.directory),
            }
        )
    )
