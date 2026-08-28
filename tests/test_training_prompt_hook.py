from hooks.route_training_prompt import build_hook_output


def test_qwen3_sft_prompt_adds_orchestration_context():
    result = build_hook_output({"prompt": "执行 Qwen3 SFT 训练"})
    assert result is not None
    context = result["hookSpecificOutput"]["additionalContext"]
    assert "compute-orchestrator" in context
    assert "query available clusters" in context


def test_generic_gpu_task_is_detected():
    assert build_hook_output({"prompt": "请用GPU执行这个计算任务"}) is not None


def test_cuda_inference_is_detected():
    assert build_hook_output({"prompt": "运行 CUDA 推理并记录显存峰值"}) is not None


def test_explicit_cluster_name_is_detected():
    assert build_hook_output({"prompt": "切换到gpu-32G跑程序"}) is not None


def test_gpu_96g_context_requires_official_template_and_vram_budget():
    result = build_hook_output({"prompt": "用gpu-96G执行训练"})
    context = result["hookSpecificOutput"]["additionalContext"]
    assert "gitlab.zhejianglab.com/nh-megatron/nhmegatron" in context
    assert "plugin-local snapshot" in context
    assert "before making any web request" in context
    assert "cloning is optional" in context
    assert "relative to the nhmegatron repository root" in context
    assert "Never implement gpu-96G training logic or launchers from scratch" in context
    assert "same-family, same-architecture" in context
    assert "96 GiB" in context


def test_deepseek_chinese_training_prompt_is_detected():
    assert build_hook_output({"prompt": "开始 DeepSeek-V3 微调"}) is not None


def test_implicit_gpu_task_gets_resource_assessment_context():
    result = build_hook_output({"prompt": "运行一个大型张量计算任务"})
    context = result["hookSpecificOutput"]["additionalContext"]
    assert "determine" in context
    assert "meaningful possibility of GPU use" in context


def test_chinese_text_immediately_after_model_name_is_detected():
    assert build_hook_output({"prompt": "执行qwen3模型sft训练"}) is not None


def test_cpu_candidate_gets_assessment_not_explicit_gpu_claim():
    result = build_hook_output({"prompt": "训练一个随机森林"})
    context = result["hookSpecificOutput"]["additionalContext"]
    assert "first determine" in context
    assert "explicitly indicates a GPU workload" not in context


def test_non_string_prompt_is_ignored():
    assert build_hook_output({"prompt": None}) is None
