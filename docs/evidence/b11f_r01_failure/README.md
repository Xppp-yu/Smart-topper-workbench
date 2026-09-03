# B11F AutoDL R01 failure evidence

This directory preserves the immutable terminal evidence for
`EXP-SLP-B11F-PM-FINAL-FIT-20260903-AUTODL-R01`.

## Provenance

- AutoDL export archive: `B11F_R01_FAILURE_EVIDENCE.tar.gz` (not committed because repository
  policy prohibits committing archives).
- Remote and local archive SHA-256:
  `ab5d97a2d9a6cf703e36efc1dc9815f2fdb4ff80fc819654c6fe318277731c98`.
- The seven text/JSON files were copied byte-for-byte from that archive. Their checksums are in
  `SHA256SUMS`; an independent local recomputation matched every entry.
- `operator_traceback.png` and `operator_terminal_audit.png` are the original operator-captured
  terminal screenshots. They provide the traceback and the terminal-carrier inspection that
  accompanied the exported machine-readable files.

## Independently checkable conclusions

- Root markers: `FAILED.json=EXISTS`; `RUNNING/STOPPED/DONE=ABSENT`.
- Failure: `AttributeError: 'numpy.ndarray' object has no attribute 'to'` at class-weight device
  conversion, before the DataLoader training loop begins.
- Identity: runner `9af268fa168207a269abbef22e522ac04fd6b6c5`, `git_dirty=false`.
- TEST carriers: `test_access=false` and `test_rows/test_labels/test_onehot=0`.
- Environment file SHA-256: `feb853112e6acffd736f351b2ac8d13daeb8ec99698765b6dccf0ab1c2635021`;
  persisted and observed hashes match.
- Budget file SHA-256: `c88df3ba0591354793ea7bc273229efe756f5d9de37befc883b5385db4f0d7df`;
  state is `FAILED`, the fixed budget is 2,700 seconds, and elapsed wall time is about 2.036 seconds.

The screenshots are operator evidence; the exported JSON, inventories, and checksums are the
primary machine-readable evidence. This evidence does not authorize reuse, resume, or overwrite
of R01, any new GPU run, or any TEST access.
