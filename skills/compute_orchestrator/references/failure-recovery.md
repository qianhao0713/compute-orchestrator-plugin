# Failure recovery

Do not classify every failure as insufficient hardware. Inspect logs and
distinguish memory exhaustion, leaks, invalid shapes, software defects,
unsupported operators/environments, distributed-launch errors, data/storage
problems, and transient infrastructure failures.

For a confirmed resource failure:

1. capture the exact error, `traceId` when applicable, and peak usage;
2. update the resource estimate from measurements;
3. prefer a semantics-preserving code/configuration change and smoke-test it;
4. otherwise select the next supported, currently available tier;
5. restart the inspection and provisioning workflow;
6. do not blindly repeat or continually increase resources.

After repeated failures with the same root cause, stop and report the evidence.
