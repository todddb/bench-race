"""
Lightweight tests for agent_reachable field in central API endpoints.

Tests verify that:
- Code structure includes agent_reachable field in API responses
- Both top-level and capabilities.agent_reachable are present

Run with: pytest tests/test_api_agent_reachable.py -v
"""

from __future__ import annotations

from pathlib import Path


def test_api_status_sets_agent_reachable_top_level():
    """Verify /api/status sets agent_reachable at top level of machine objects."""
    app_py = Path("central/app.py").read_text(encoding="utf-8")

    # Find the api_status function
    assert "def api_status():" in app_py, "/api/status endpoint not found"

    # Check that agent_reachable is set in success case
    assert '"agent_reachable": True,' in app_py, \
        "agent_reachable not set to True in success case"

    # Check that agent_reachable is set in error case
    assert '"agent_reachable": False,' in app_py, \
        "agent_reachable not set to False in error case"


def test_api_status_sets_agent_reachable_in_capabilities():
    """Verify /api/status also sets agent_reachable in capabilities object."""
    app_py = Path("central/app.py").read_text(encoding="utf-8")

    # Check that capabilities.agent_reachable is set
    assert '"capabilities": {\n                        "agent_reachable": True,' in app_py or \
           '"capabilities": {"agent_reachable": True,' in app_py, \
        "capabilities.agent_reachable not set to True in success case"

    assert '"capabilities": {"agent_reachable": False,' in app_py, \
        "capabilities.agent_reachable not set to False in error case"


def test_frontend_js_uses_agent_reachable_fallback():
    """Verify frontend JS implements agent_reachable fallback chain."""
    app_js = Path("central/static/js/app.js").read_text(encoding="utf-8")

    # Check for fallback logic
    assert "machine.agent_reachable" in app_js, \
        "Frontend JS doesn't check machine.agent_reachable"

    assert "machine.capabilities?.agent_reachable" in app_js, \
        "Frontend JS doesn't check machine.capabilities.agent_reachable as fallback"

    assert "machine.reachable" in app_js, \
        "Frontend JS doesn't check machine.reachable as legacy fallback"


def test_templates_conditionally_render_initial_status():
    """Verify templates conditionally render status based on agent_reachable."""
    index_html = Path("central/templates/index.html").read_text(encoding="utf-8")
    image_html = Path("central/templates/image.html").read_text(encoding="utf-8")

    # Check index.html has conditional rendering
    assert "{% if m.agent_reachable is defined and m.agent_reachable == true %}" in index_html, \
        "index.html doesn't conditionally render Online status"

    assert "{% elif m.agent_reachable is defined and m.agent_reachable == false %}" in index_html, \
        "index.html doesn't conditionally render Offline status"

    assert '{% else %}' in index_html and 'Checking...' in index_html, \
        "index.html doesn't have fallback to Checking... status"

    # Check image.html has conditional rendering
    assert "{% if m.agent_reachable is defined and m.agent_reachable == true %}" in image_html, \
        "image.html doesn't conditionally render Online status"

    assert "{% elif m.agent_reachable is defined and m.agent_reachable == false %}" in image_html, \
        "image.html doesn't conditionally render Offline status"

    assert '{% else %}' in image_html and 'Checking...' in image_html, \
        "image.html doesn't have fallback to Checking... status"


def test_minimal_agent_yaml_example_exists():
    """Verify minimal agent.yaml.example was created with required fields."""
    yaml_path = Path("agent/config/agent.yaml.example")
    assert yaml_path.exists(), "agent.yaml.example not found"

    content = yaml_path.read_text(encoding="utf-8")

    # Check for required fields
    assert "machine_id:" in content, "machine_id field missing"
    assert "label:" in content, "label field missing"
    assert "bind_host:" in content, "bind_host field missing"
    assert "bind_port:" in content, "bind_port field missing"
    assert "ollama_base_url:" in content, "ollama_base_url field missing"
    assert "central_base_url:" in content, "central_base_url field missing"

    # Check that it emphasizes minimal config
    assert "Minimal" in content or "minimal" in content, \
        "agent.yaml.example should emphasize minimal configuration"


def test_linux_installer_exists_and_is_executable():
    """Verify Linux installer script exists and is executable."""
    installer = Path("scripts/install_agent_linux.sh")
    assert installer.exists(), "install_agent_linux.sh not found"
    assert installer.stat().st_mode & 0o111, "install_agent_linux.sh not executable"

    content = installer.read_text(encoding="utf-8")

    # Check for key functionality
    assert "detect_cpu_cores" in content, "CPU detection missing"
    assert "detect_ram_gb" in content, "RAM detection missing"
    assert "detect_gpu_info" in content, "GPU detection missing"
    assert "generate_agent_yaml" in content, "YAML generation missing"
    assert "test_health_endpoint" in content, "Health endpoint test missing"
    assert "test_capabilities_endpoint" in content, "Capabilities endpoint test missing"


def test_macos_installer_exists_and_is_executable():
    """Verify macOS installer script exists and is executable."""
    installer = Path("scripts/install_agent_macos.sh")
    assert installer.exists(), "install_agent_macos.sh not found"
    assert installer.stat().st_mode & 0o111, "install_agent_macos.sh not executable"

    content = installer.read_text(encoding="utf-8")

    # Check for key functionality
    assert "sysctl" in content, "macOS system detection missing"
    assert "detect_cpu_cores" in content, "CPU detection missing"
    assert "detect_ram_gb" in content, "RAM detection missing"
    assert "detect_gpu_info" in content, "GPU detection missing"
    assert "generate_agent_yaml" in content, "YAML generation missing"
    assert "test_health_endpoint" in content, "Health endpoint test missing"
    assert "test_capabilities_endpoint" in content, "Capabilities endpoint test missing"
