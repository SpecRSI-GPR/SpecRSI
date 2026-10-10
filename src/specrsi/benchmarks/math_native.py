"""Execute pinned AFlow MATH grading methods inside a bounded offline sandbox."""
from __future__ import annotations
import ast,json,tempfile
from pathlib import Path
from .base import ScoreResult, MissingArtifactError
from .sandbox import ROOT,run_isolated

def math_score(prediction: str,reference: dict) -> ScoreResult:
    source=ROOT/'third_party/aflow/benchmarks/math.py'
    # Import only original scoring methods; optimizer dependencies and logging
    # are unnecessary. Source methods are preserved verbatim by AST extraction.
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MATHBenchmark')
    names={'extract_model_answer','calculate_score','math_equal','is_digit','parse_digits','symbolic_equal'}
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
    scorer=ast.ClassDef(name='MATHBenchmark',bases=[],keywords=[],body=methods,decorator_list=[])
    script='from typing import Any, Tuple\nimport re, regex\nfrom math import isclose\nfrom sympy import N,simplify\nfrom sympy.parsing.latex import parse_latex\nfrom sympy.parsing.sympy_parser import parse_expr\nimport json\n'
    script+=ast.unparse(ast.fix_missing_locations(scorer))
    script+='\nr=json.load(open("request.json"))\ns,a=MATHBenchmark().calculate_score(r["gold"],r["prediction"])\nprint(json.dumps({"score":s,"extracted_answer":a}))\n'
    with tempfile.TemporaryDirectory(prefix='specrsi-math-') as directory:
        work=Path(directory); (work/'score.py').write_text(script)
        (work/'request.json').write_text(json.dumps({'gold':reference['solution'],'prediction':prediction}))
        result=run_isolated(['/math/bin/python','/work/score.py'],work,timeout=20,memory_mb=2048,
                            mounts={str(ROOT/'.envs/math'):'/math'})
    if result['timeout']: return ScoreResult(0.,{'parser_timeout':True,'metric':'aflow_math_equal'})
    if result['exit_code']!=0: raise MissingArtifactError('MATH grader runtime failed: '+result['stderr'][-2000:])
    response=json.loads(result['stdout'].strip().splitlines()[-1])
    return ScoreResult(float(response['score']),{'metric':'aflow_math_equal','extracted_answer':response['extracted_answer']})
