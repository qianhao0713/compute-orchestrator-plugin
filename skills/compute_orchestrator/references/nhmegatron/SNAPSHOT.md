# nhmegatron V5000 example snapshot

- Upstream: `https://gitlab.zhejianglab.com/nh-megatron/nhmegatron.git`
- Branch: `main`
- Commit: `49ebf44cee9fe35eff7cbf7937066dc0ee0e22d6`
- Imported path: `zj_examples/V5000`
- Imported on: `2026-08-13`
- Contents: 76 text scripts and configuration files, preserving upstream paths

Security sanitization: hard-coded DingTalk webhook tokens in `monitor_job.py`,
`qwen2_5/monitor_job_qwen25.sh`, and `qwen3/monitor_job_qwen3.sh` were replaced
with `REPLACE_WITH_YOUR_DINGTALK_TOKEN`. Other imported text files match the
recorded upstream commit byte-for-byte.

The two upstream tokenizer binaries ending in `.model` are not included because
they are runtime data rather than scripts. This snapshot is for offline template
discovery, inspection, resource estimation, and minimal script derivation. It is
not a complete `nhmegatron` checkout and must not replace the V5000 platform's
installed code at execution time.
