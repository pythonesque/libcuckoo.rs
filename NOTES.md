* Using ti.ti in `snapshot_table_nolock` results in significiant code size decrease (at least when I tried it).
* Same goes for passing a `(key, value)` tuple into `cuckoo_insert_loop` instead of individual values.
