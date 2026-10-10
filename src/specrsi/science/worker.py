"""One JSON request per fresh isolated simulator process."""
from __future__ import annotations
import json
import sys
from .registry import evaluate


def main():
    request=json.loads(sys.stdin.read())
    result=evaluate(request["domain"],request["candidate"],scenario_set=request["scenario_set"],case_id=request.get("case_id",0))
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
