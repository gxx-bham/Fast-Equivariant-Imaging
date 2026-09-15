Tiny path check that `--unit nbuf4 --n-buf 4` honors N_buf=4 and stores 4 distinct (y, A) from T1.

- `N_buf=4`, `n_distinct_ya=4`, `n_distinct_A=4`, `holds_t1_A=true`, all `accels=4`
- Distinct `y_id` / `mask_id` per slot
- `deepinv_version=0.3.5`
- `go_kill.verdict=INCONCLUSIVE` because `--tiny`

Not a go/kill run.
