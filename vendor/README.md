# Local registry accessor

`dypdl-heuristic-search/` is the crates.io source of **0.8.0**, already selected
by this project's Cargo.lock. Cargo's patch section makes didp-mpi and
didp-yaml use the same copy. No search behavior or dependency versions change.

The only source change adds `StateRegistry::signature_count()`, returning
the existing hash map's length in O(1). This counts shared signature keys,
including an empty group left by `insert_with` when a replacement is pruned.
The accessor avoids extra hashing or a separate per-signature tracking table.

The base crates.io checksum is
`7d00babdebd60bab7aefd6c0585e777c250fb770a8eb26d571d52428d5c7c4f8`.
The original MIT and Apache-2.0 license texts are included with the crate.
Once the upstream dependency exposes this accessor, the local patch can be
removed after updating to that release.
