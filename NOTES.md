* Seem to save some instructions converting `insert_into_table` into a closure (in place).
* Same goes for passing a `(key, value)` tuple into `cuckoo_insert_loop` instead of individual values.

Correct lock striping:

Map each thread to a core.  Support at most N concurrent writers where N can be atomically fetched.
Version counters are not stored in hashed locations.  Instead, they are stored at direct array offsets
and can be determined entirely by the bucket physical address.
The lowest bit must be reserved for all readers.  That is, *any* readers of that table entry set that bit when they start reading and
clear it when they finish.
The next N bits of each version counter are set iff the thread is currently inserting something into the "bucket."
E.g. with 4 threads, bits 1-4 represent current thread accesses by writers.
The next lg(N) bits must be reserved for the version--they represent the maximum number of increments that may occur on
N different threads without any of the N threads inserting twice.  E.g. with 4 threads, bits 5-6 must be used for version.
The next bit represents overflow; if this bit is set, we enter the slow path.  This involves acquiring a lock,
which in turn requires a manually maintained reference count.
Assuming that each thread loads the version before storing it, the following bit being set means it's time to take a lock.
The following bit represents a lock: if this bit is set, all reads must take a lock on the row from then un
