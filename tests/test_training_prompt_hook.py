from hooks.route_training_prompt import build_hook_output


def _context(prompt):
    return build_hook_output({"prompt": prompt})["hookSpecificOutput"][
        "additionalContext"
    ]


def test_qwen3_sft_prompt_adds_mandatory_orchestration_route():
    context = _context("执行 Qwen3 SFT 训练")
    assert "Mandatory compute gate" in context
    assert "compute-orchestrator" in context
    assert "Mandatory safety gates" in context


def test_generic_gpu_task_is_detected():
    assert build_hook_output({"prompt": "请用GPU执行这个计算任务"}) is not None


def test_cuda_inference_is_detected():
    assert build_hook_output({"prompt": "运行 CUDA 推理并记录显存峰值"}) is not None


def test_explicit_cluster_name_is_detected():
    assert build_hook_output({"prompt": "切换到GPU-32G跑程序"}) is not None


def test_gpu_96g_context_routes_to_runtime_and_llm_references():
    context = _context("用GPU-96G执行LLM训练")
    assert "references/GPU-96G-runtime.md" in context
    assert "references/GPU-96G-training.md" in context
    assert "official templates" in context


def test_gpu_96g_context_routes_non_llm_checks_to_reference():
    context = _context("用GPU-96G训练YOLO")
    assert "classify LLM versus non-LLM" in context
    assert "named Conda environment" in context
    assert "dependency boundary" in context
    assert "built-in operator checklist" in context


def test_deepseek_chinese_training_prompt_is_detected():
    assert build_hook_output({"prompt": "开始 DeepSeek-V3 微调"}) is not None


def test_implicit_gpu_task_gets_resource_assessment_context():
    context = _context("运行一个大型张量计算任务")
    assert "Compute classification gate" in context
    assert "meaningfully possible" in context


def test_chinese_text_immediately_after_model_name_is_detected():
    assert build_hook_output({"prompt": "执行qwen3模型sft训练"}) is not None


def test_cpu_candidate_gets_classification_not_explicit_gpu_claim():
    context = _context("训练一个随机森林")
    assert "Compute classification gate" in context
    assert "Mandatory compute gate" not in context


def test_non_string_prompt_is_ignored():
    assert build_hook_output({"prompt": None}) is None


def test_long_prompt_keeps_compute_route_short_and_mandatory():
    context = _context(("背景信息。" * 50_000) + "最终任务需要GPU训练")
    assert "Mandatory compute gate" in context
    assert "compute-orchestrator" in context
    assert len(context) < 1_500
