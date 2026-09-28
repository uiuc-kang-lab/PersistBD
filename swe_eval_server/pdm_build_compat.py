"""Keep PDM's helper dependencies compatible with Python 3.8 testbeds."""
from dataclasses import replace


PY38_PDM_COMPAT = """if python -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 8))'; then
  python -m pip install --disable-pip-version-check --upgrade --target "${PDM_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/pdm}/.compat_libs" 'dep-logic==0.5.2' 'packaging==24.2'
fi"""


def prepare_pdm_build_spec(spec):
    """Copy only affected setup scripts; preserve instance and test definitions."""
    if spec.repo != "pydantic/pydantic" or PY38_PDM_COMPAT in spec.repo_script_list:
        return spec
    for index, command in enumerate(spec.repo_script_list):
        if "pdm add pre-commit" in command:
            commands = list(spec.repo_script_list)
            commands.insert(index, PY38_PDM_COMPAT)
            return replace(spec, repo_script_list=commands)
    return spec


def prepare_pdm_build_specs(dataset):
    # SWE-bench accepts TestSpec objects as well as dataset dictionaries.
    try:
        from swebench.harness.test_spec import make_test_spec
    except ImportError:
        from swebench.harness.test_spec.test_spec import make_test_spec
    return [prepare_pdm_build_spec(make_test_spec(row)) for row in dataset]
