# Prime Number Algorithms for Compute Benchmarks

bench-race includes a CPU-bound compute benchmark that counts all prime numbers
up to N. This is a pure-CPU workload (no GPU involvement) designed to compare
raw processing throughput across machines.

Three algorithms are available, each with different performance characteristics.
This document explains how they work, when to use each one, and what the
tradeoffs are.

---

## Table of Contents

- [Overview](#overview)
- [Algorithm 1: Segmented Sieve (Default)](#algorithm-1-segmented-sieve-default)
- [Algorithm 2: Simple Sieve](#algorithm-2-simple-sieve)
- [Algorithm 3: Trial Division](#algorithm-3-trial-division)
- [Comparison Summary](#comparison-summary)
- [Choosing an Algorithm](#choosing-an-algorithm)
- [How the Compute Endpoint Works](#how-the-compute-endpoint-works)
- [Interpreting Results](#interpreting-results)
- [Suggested Screenshots](#suggested-screenshots)

---

## Overview

All three algorithms answer the same question: *How many primes are there
less than or equal to N?* The answer is well-known in number theory (the prime
counting function, pi(N)), so correctness can be verified independently.

| Algorithm | Time Complexity | Space Complexity | Recommended N | Typical Runtime |
|-----------|----------------|-----------------|---------------|-----------------|
| **Segmented Sieve** | O(n log log n) | O(sqrt(n)) | 50,000,000 | 2-15s |
| **Simple Sieve** | O(n log log n) | O(n) | 20,000,000 | 2-10s |
| **Trial Division** | O(n * sqrt(n) / ln(n)) | O(n / ln(n)) | 2,000,000 | 10-30s |

The demo presets in the UI are calibrated to produce 2-30 second runtimes on
typical modern hardware -- long enough to generate meaningful performance data
and interesting progress updates, short enough to keep the demo interactive.

---

## Algorithm 1: Segmented Sieve (Default)

The segmented sieve is the default and recommended algorithm. It is an
optimization of the Sieve of Eratosthenes that processes numbers in
fixed-size segments rather than allocating one giant array.

### How It Works

**Phase 1: Find base primes**

First, find all primes up to sqrt(N) using the simple sieve. These are the
"base primes" -- every composite number up to N must have at least one prime
factor <= sqrt(N).

```
For N = 50,000,000:
  sqrt(50,000,000) = 7,071
  Sieve [0..7071] to find base primes: 2, 3, 5, 7, 11, ..., 7069
  There are 900 base primes below 7,071
```

**Phase 2: Process segments**

Divide the range [2, N] into segments of size max(sqrt(N), 1,000,000).
For each segment:

1. Create a boolean array for the segment, initialized to all `true` (prime).
2. For each base prime p, mark all multiples of p within this segment as
   composite. The starting point for marking is max(p^2, ceil(low/p)*p).
3. Count the remaining `true` entries -- these are primes.
4. Emit a progress update if the interval has elapsed.
5. Yield to the event loop (`await asyncio.sleep(0)`) so the agent stays
   responsive.

```
Segment [2 .. 1,000,001]:
  For p=2: mark 2,4,6,8,...
  For p=3: mark 3,9,15,21,...
  For p=5: mark 25,35,55,...
  ...
  Count survivors = 78,498 primes in this segment

Segment [1,000,002 .. 2,000,001]:
  Repeat with same base primes, different starting offsets
  ...
```

**Phase 3: Sum up**

The total prime count is the sum of primes found in all segments.

### Implementation

From `agent/agent_app.py` (lines 246-284):

```python
async def _run_segmented_sieve(n, stream_first_k, progress_interval_s, emit_line):
    sqrt_n = int(math.isqrt(n))
    base_flags = await _simple_sieve(sqrt_n)
    base_primes = [i for i in range(2, sqrt_n + 1) if base_flags[i]]
    segment_size = max(sqrt_n, 1_000_000)
    count = 0
    start_time = time.perf_counter()
    last_progress = start_time

    low = 2
    while low <= n:
        high = min(low + segment_size - 1, n)
        size = high - low + 1
        segment = bytearray(b"\x01") * size
        for p in base_primes:
            start = max(p * p, ((low + p - 1) // p) * p)
            if start > high:
                continue
            for multiple in range(start, high + 1, p):
                segment[multiple - low] = 0
        for idx, is_prime in enumerate(segment):
            if is_prime:
                count += 1
        now = time.perf_counter()
        if progress_interval_s > 0 and now - last_progress >= progress_interval_s:
            pct = min(100.0, (high / n) * 100.0)
            elapsed = now - start_time
            await emit_line(
                f"Progress: {pct:.0f}% | primes so far: {count:,} | elapsed: {elapsed:.1f}s"
            )
            last_progress = now
        low = high + 1
        await asyncio.sleep(0)
    return count
```

### Why It's the Default

- **Cache-friendly.** Each segment fits in L2/L3 cache, so marking composites
  operates on hot memory. The simple sieve's single giant array may thrash the
  cache for large N.
- **Low memory.** Uses O(sqrt(N) + segment_size) bytes, not O(N). For
  N = 50M, the segmented sieve uses ~1 MB per segment vs ~50 MB for the
  simple sieve.
- **Scalable.** Would be the natural starting point for multi-threaded
  parallelism (each thread processes a different segment).

---

## Algorithm 2: Simple Sieve

The simple sieve is a straightforward implementation of the Sieve of
Eratosthenes. It allocates a single array for the entire range [0, N]
and marks composites in one pass.

### How It Works

1. Allocate a `bytearray` of size N+1, initialized to `0x01` (prime).
2. Mark 0 and 1 as not prime.
3. For each number p from 2 to sqrt(N):
   - If p is still marked prime, mark all multiples of p starting from p^2
     as composite.
4. Count the `0x01` entries remaining.

```
For N = 20,000,000:
  Allocate 20 MB bytearray
  p=2: mark 4,6,8,10,... as composite
  p=3: mark 9,15,21,... as composite
  p=5: mark 25,35,55,...
  (skip p=4 -- already marked composite)
  ...
  Continue to p = 4,472 (sqrt of 20M)
  Count survivors
```

### Implementation

The core sieve function (`agent/agent_app.py`, lines 234-243):

```python
async def _simple_sieve(n: int) -> bytearray:
    is_prime = bytearray(b"\x01") * (n + 1)
    if n >= 0:
        is_prime[0:2] = b"\x00\x00"
    limit = int(math.isqrt(n))
    for p in range(2, limit + 1):
        if is_prime[p]:
            start = p * p
            is_prime[start:n + 1:p] = b"\x00" * ((n - start) // p + 1)
    return is_prime
```

The benchmark wrapper adds progress reporting and event-loop yields
(lines 287-311):

```python
async def _run_simple_sieve(n, stream_first_k, progress_interval_s, emit_line):
    flags = await _simple_sieve(n)  # Sieve the entire range at once
    count = 0
    start_time = time.perf_counter()
    last_progress = start_time
    for i in range(2, n + 1):
        if flags[i]:
            count += 1
        if progress_interval_s > 0:
            now = time.perf_counter()
            if now - last_progress >= progress_interval_s:
                pct = min(100.0, (i / n) * 100.0)
                elapsed = now - start_time
                await emit_line(
                    f"Progress: {pct:.0f}% | primes so far: {count:,} | elapsed: {elapsed:.1f}s"
                )
                last_progress = now
        if i % 10000 == 0:
            await asyncio.sleep(0)
    return count
```

### Key Observations

- **Uses Python's slice assignment** for marking composites:
  `is_prime[start:n+1:p] = b"\x00" * count`. This is a CPython-optimized
  operation that runs in C, making the sieve step fast despite being Python.
- **Two-phase cost.** The sieve itself (marking composites) is very fast.
  The counting loop (iterating all N entries) is the slow part, because
  it's a Python-level `for` loop over millions of entries.
- **Memory.** Allocates N+1 bytes. For N = 20M, that's ~20 MB. For N = 1B,
  it would be ~1 GB -- which is why the segmented sieve exists.

### When to Use It

The simple sieve is useful when:
- You want to understand the basic algorithm before looking at the segmented
  variant.
- N is small enough (< 50M) that the O(N) memory is acceptable.
- You want a clean baseline to compare against the segmented sieve's
  cache behavior.

---

## Algorithm 3: Trial Division

Trial division is the simplest primality test: check each candidate number
against all previously found primes up to its square root.

### How It Works

1. Start with 2 as the first prime.
2. For each odd candidate from 3 to N:
   a. Compute sqrt(candidate).
   b. Check if any known prime p <= sqrt(candidate) divides the candidate.
   c. If no prime divides it, it's prime -- add it to the list.
3. Return the count.

```
Checking candidate = 29:
  sqrt(29) = 5.38, so check primes up to 5
  Is 29 divisible by 2? No
  Is 29 divisible by 3? No
  Is 29 divisible by 5? No
  --> 29 is prime

Checking candidate = 33:
  sqrt(33) = 5.74, so check primes up to 5
  Is 33 divisible by 2? No
  Is 33 divisible by 3? Yes (33 = 3 * 11)
  --> 33 is composite
```

### Implementation

From `agent/agent_app.py` (lines 314-354):

```python
async def _run_trial_division(n, stream_first_k, progress_interval_s, emit_line):
    primes: List[int] = []
    count = 0
    start_time = time.perf_counter()
    last_progress = start_time

    if n >= 2:
        primes.append(2)
        count = 1

    candidate = 3
    while candidate <= n:
        is_prime = True
        limit = int(math.isqrt(candidate))
        for p in primes:
            if p > limit:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
            count += 1
        if progress_interval_s > 0:
            now = time.perf_counter()
            if now - last_progress >= progress_interval_s:
                pct = min(100.0, (candidate / n) * 100.0)
                elapsed = now - start_time
                await emit_line(
                    f"Progress: {pct:.0f}% | primes so far: {count:,} | elapsed: {elapsed:.1f}s"
                )
                last_progress = now
        if candidate % 10001 == 1:
            await asyncio.sleep(0)
        candidate += 2
    return count
```

### Key Observations

- **Intentionally slow.** The UI emits a note: *"Trial division is
  intentionally slow and for demo/education."* It exists as a baseline to
  show how much faster sieve algorithms are.
- **Skips even numbers.** Only checks odd candidates (candidate += 2), since
  all even numbers > 2 are composite.
- **Divides only by primes.** Rather than checking all numbers up to sqrt(n),
  it only checks known primes. This is optimal for trial division but still
  far slower than sieving.
- **Stores all primes.** The primes list grows as primes are found, using
  O(pi(N)) memory (about N / ln(N) entries).
- **Event loop yields.** Every 10,001 candidates, yields to the async event
  loop so the agent remains responsive.

### Time Complexity Breakdown

For each candidate c, we check primes up to sqrt(c). The number of primes
up to sqrt(c) is approximately sqrt(c) / ln(sqrt(c)). Summing over all
candidates from 2 to N:

```
Total divisions ~= sum_{c=2}^{N} sqrt(c) / ln(sqrt(c))
                ~= O(N * sqrt(N) / ln(N))
                ~= O(N^1.5 / ln(N))
```

For N = 2,000,000, this is roughly 10^8 division operations -- manageable but
slow. For N = 50,000,000, it would be roughly 10^10 -- painfully slow. This is
why the recommended N for trial division is only 2M.

---

## Comparison Summary

### Speed for Reference Values

Using pi(N) = known prime count:

| N | pi(N) | Segmented Sieve | Simple Sieve | Trial Division |
|---|-------|-----------------|-------------|----------------|
| 2,000,000 | 148,933 | ~0.1s | ~0.5s | ~10-30s |
| 20,000,000 | 1,270,607 | ~0.5s | ~2-8s | Impractical |
| 50,000,000 | 3,001,134 | ~2-15s | ~5-20s | Impractical |

*(Times are approximate and vary by hardware. These are Python implementations,
not C.)*

### Memory Usage

| Algorithm | Memory for N=50M | Memory for N=1B |
|-----------|-----------------|-----------------|
| Segmented Sieve | ~1 MB per segment | ~1 MB per segment |
| Simple Sieve | ~50 MB | ~1 GB |
| Trial Division | ~24 MB (primes list) | ~430 MB (primes list) |

### Algorithmic Characteristics

| Property | Segmented Sieve | Simple Sieve | Trial Division |
|----------|----------------|-------------|----------------|
| Cache-friendly | Yes (segments fit in cache) | No (single large array) | Yes (primes list is sequential) |
| Parallelizable | Yes (independent segments) | No (single array) | Limited (dependencies between candidates) |
| Streaming progress | Yes (after each segment) | Yes (during counting loop) | Yes (after each candidate) |
| Educational value | Medium | High (classic algorithm) | Highest (most intuitive) |

---

## Choosing an Algorithm

### Use Segmented Sieve (Default) When:

- You want the **fastest results** for large N.
- You're comparing hardware throughput and want the algorithm itself to not
  be the bottleneck.
- You want **low memory** usage, even for very large N.
- You're benchmarking with N > 20M.

### Use Simple Sieve When:

- You want a **simpler algorithm** that's easier to understand and verify.
- N is moderate (< 50M) and memory isn't a concern.
- You want to **compare cache effects** -- the simple sieve's single large
  array may show different performance characteristics on machines with
  different cache sizes.

### Use Trial Division When:

- You want an **educational demonstration** of the simplest primality test.
- You want to show the **dramatic speedup** of sieve algorithms vs naive
  approaches (100x+ difference).
- You need a **long-running workload** that produces continuous progress
  updates for a demo, even with small N.
- You're testing the agent's **event-loop responsiveness** under CPU load
  (trial division yields every 10K candidates).

---

## How the Compute Endpoint Works

### API

```
POST /api/compute
Content-Type: application/json

{
    "algorithm": "segmented_sieve",    // or "simple_sieve" or "trial_division"
    "n": 50000000,                     // Count primes <= N
    "threads": 1,                      // (not yet implemented)
    "repeat_index": 1,                 // Which repetition this is
    "progress_interval_s": 1.0,        // Progress update frequency
    "job_id": "optional-uuid"          // Optional; auto-generated if omitted
}
```

### Execution Flow

1. Agent receives the request and creates a `job_id`.
2. Agent emits `compute_line` event: `"Compute: Count primes <= 50,000,000"`.
3. The selected algorithm runs, emitting progress updates via `compute_line`
   events at the configured interval.
4. On completion, the agent emits a final `compute_done` event with metrics.
5. The response is also returned synchronously to the HTTP caller.

### Event Streaming

During execution, the agent broadcasts `compute_line` events via WebSocket:

```json
{"job_id": "abc123", "type": "compute_line", "payload": {
    "line": "Progress: 45% | primes so far: 1,425,172 | elapsed: 6.3s"
}}
```

On completion, a `compute_done` event:

```json
{"job_id": "abc123", "type": "compute_done", "payload": {
    "ok": true,
    "algorithm": "segmented_sieve",
    "n": 50000000,
    "primes_found": 3001134,
    "elapsed_ms": 14200,
    "primes_per_sec": 211347.5,
    "threads_requested": 1,
    "threads_used": 1,
    "repeat_index": 1
}}
```

---

## Interpreting Results

### Key Metrics

- **primes_found** -- Total primes discovered. Verify against known values
  (e.g., pi(50M) = 3,001,134).
- **elapsed_ms** -- Wall-clock time for the computation. This is the primary
  metric for comparing hardware.
- **primes_per_sec** -- Throughput. Higher is better. Useful for comparing
  the same algorithm across different machines.

### What Affects Results

1. **CPU single-thread performance.** All three algorithms currently run
   single-threaded. Machines with higher single-core IPC and clock speed
   will perform better.
2. **Cache size.** The segmented sieve benefits from larger L2/L3 caches.
3. **Python implementation.** These are pure Python implementations. The
   sieve's slice assignment is the only C-level optimization. Performance
   will differ from compiled implementations by 10-100x.
4. **Event-loop overhead.** The `await asyncio.sleep(0)` yields add small
   overhead. This is intentional to keep the agent responsive during
   long-running computations.

### Comparing Machines

For meaningful hardware comparison:

1. Use the **same algorithm and N** across all machines.
2. Run **multiple repeats** (e.g., repeat=3) and look at the median.
3. Use the **segmented sieve** for the most consistent results.
4. Compare **primes_per_sec** rather than elapsed time, as it normalizes
   for the workload size.

---

## Suggested Screenshots

1. **Compute benchmark running** -- showing the compute UI with progress
   updates streaming for multiple machines side-by-side.

2. **Algorithm selector dropdown** -- showing the three algorithm options
   with their recommended N values.

3. **Compute results comparison** -- showing final metrics (primes/sec,
   elapsed time) for the same algorithm run on different machines.

4. **Trial division vs segmented sieve** -- running both on the same
   machine with N=2M to show the dramatic speed difference.
