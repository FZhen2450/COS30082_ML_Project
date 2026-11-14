# Operational note, 14 Nov 2026 (about 01:50 MYT)
Owner instruction (chat): Katana is dedicated to AI work; use it fully ("push it to the border").
Effect from now on:
- the protocol_v4 "resource_limits" GPU cap (<= 6.0 GiB, one GPU job at a time) is lifted;
- independent jobs may run in parallel; DataLoader workers may use all 12 CPUs.
Unchanged:
- scientific settings (declared batch size, epochs, seeds, splits, gates);
- authorised scope (Stage H, Stage I, J0 and the single-fold J1-J3 pilot; no full Stage J sweep).
Runtime and throughput figures taken under parallel load are labelled as such.
