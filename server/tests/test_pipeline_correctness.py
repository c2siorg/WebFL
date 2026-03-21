import os
import re
import yaml
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DOCKERFILE_PATH = os.path.join(REPO_ROOT, "server", "Dockerfile")
APP_PY_PATH = os.path.join(REPO_ROOT, "server", "app.py")
WORKFLOW_PATH = os.path.join(REPO_ROOT, ".github", "workflows", "server-docker.yml")


@pytest.fixture(scope="module")
def dockerfile_lines():
    with open(DOCKERFILE_PATH) as f:
        return f.read().splitlines()


@pytest.fixture(scope="module")
def app_py_source():
    with open(APP_PY_PATH) as f:
        return f.read()


@pytest.fixture(scope="module")
def workflow():
    with open(WORKFLOW_PATH) as f:
        data = yaml.safe_load(f)
    # pyyaml treats bare `on:` as a boolean True key, fix that
    if True in data and "on" not in data:
        data["on"] = data.pop(True)
    return data


@pytest.fixture(scope="module")
def workflow_raw():
    with open(WORKFLOW_PATH) as f:
        return f.read()


# --- Dockerfile ---

def test_dockerfile_base_image(dockerfile_lines):
    # want slim to keep the image size down
    assert any(line.strip() == "FROM python:3.10-slim" for line in dockerfile_lines)


def test_dockerfile_installs_dependencies(dockerfile_lines):
    has_copy_req = any(
        re.match(r"^COPY\s+requirements\.txt", line.strip())
        for line in dockerfile_lines
    )
    has_pip_install = any("pip install -r requirements.txt" in line for line in dockerfile_lines)
    assert has_copy_req
    assert has_pip_install


def test_dockerfile_workdir_and_copy(dockerfile_lines):
    assert any(line.strip() == "WORKDIR /app" for line in dockerfile_lines)
    assert any(re.match(r"^COPY\s+\.\s+\.", line.strip()) for line in dockerfile_lines)


def test_dockerfile_expose_and_cmd(dockerfile_lines):
    assert any(line.strip() == "EXPOSE 5000" for line in dockerfile_lines)
    assert any(
        re.match(r'^CMD\s+\["python",\s*"app\.py"\]', line.strip())
        for line in dockerfile_lines
    )


# --- app.py env config ---

def test_app_reads_env_vars(app_py_source):
    # server should pick these up at runtime, not hardcode them
    for var in ("WEBFL_SERVER_HOST", "WEBFL_SERVER_PORT", "WEBFL_DEBUG"):
        assert f'"{var}"' in app_py_source or f"'{var}'" in app_py_source


# --- workflow triggers ---

def test_workflow_path_filters(workflow):
    # only rebuild when server code or the workflow itself changes
    expected_paths = {"server/**", ".github/workflows/server-docker.yml"}
    for trigger in ("push", "pull_request"):
        paths = set(workflow["on"][trigger].get("paths", []))
        assert expected_paths.issubset(paths)


def test_workflow_manual_dispatch(workflow):
    # handy for forcing a rebuild without a code change
    assert "workflow_dispatch" in workflow["on"]


# --- build steps ---

def test_workflow_buildx_step(workflow):
    steps = workflow["jobs"]["docker"]["steps"]
    assert any("docker/setup-buildx-action" in str(step.get("uses", "")) for step in steps)


def test_workflow_build_cache(workflow):
    # GHA cache cuts rebuild time significantly on unchanged layers
    steps = workflow["jobs"]["docker"]["steps"]
    build_step = next(
        (s for s in steps if "docker/build-push-action" in str(s.get("uses", ""))), None
    )
    assert build_step is not None
    with_block = build_step.get("with", {})
    assert with_block.get("cache-from") == "type=gha"
    assert with_block.get("cache-to") == "type=gha,mode=max"


def test_workflow_build_context_and_file(workflow):
    steps = workflow["jobs"]["docker"]["steps"]
    build_step = next(
        (s for s in steps if "docker/build-push-action" in str(s.get("uses", ""))), None
    )
    assert build_step is not None
    with_block = build_step.get("with", {})
    assert with_block.get("context") == "./server"
    assert with_block.get("file") == "./server/Dockerfile"


def test_workflow_push_condition(workflow):
    # build on PRs for validation, but only push to GHCR on actual merges
    steps = workflow["jobs"]["docker"]["steps"]
    build_step = next(
        (s for s in steps if "docker/build-push-action" in str(s.get("uses", ""))), None
    )
    assert build_step is not None
    assert build_step.get("with", {}).get("push") == "${{ github.event_name != 'pull_request' }}"


# --- image tagging ---

def test_workflow_image_tagging(workflow_raw):
    # latest on main, branch name otherwise, pr ref for PRs, sha for traceability
    assert "type=raw,value=latest,enable={{is_default_branch}}" in workflow_raw
    assert "type=ref,event=branch" in workflow_raw
    assert "type=ref,event=pr" in workflow_raw
    assert "type=sha,prefix=sha-" in workflow_raw


# --- concurrency ---

def test_workflow_concurrency(workflow):
    # cancel stale runs so a fast follow-up push doesn't get blocked
    concurrency = workflow.get("concurrency", {})
    assert concurrency.get("cancel-in-progress") is True
    assert "github.ref" in concurrency.get("group", "")
