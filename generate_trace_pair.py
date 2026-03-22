"""
Cache trace pair generator for a two-core system.

Hardware:
  - Private L1 per core : 2 KB, 2-way, 32-byte lines, PLRU
  - Shared  L2           : 16 KB, 2-way, 32-byte lines, PLRU (inclusive)

Public API
----------
    trace1, trace2, ifl = generate_trace_pair(trace_len=16, nop_prob=0.25)

    ifl = |cyc_T1_concurrent - cyc_T1_solo|
        + |cyc_T2_concurrent - cyc_T2_solo|

Trace generation
----------------
Each trace is a sequence of exactly *trace_len* instructions drawn
independently and uniformly at random:
  - NOP              with probability nop_prob
  - LDR Rx, [addr]   otherwise, where addr is sampled uniformly from all
                     line-aligned addresses in a 4× L2-capacity address space

IFL is a pure emergent property of whichever addresses happen to map to
the same L2 sets, computed by running the cycle-accurate simulator.

Concurrency model
-----------------
Cycle-accurate interleaving: each core issues its next instruction as soon
as its own clock allows (after the previous instruction's latency elapses).
A min-heap ordered by (next_free_cycle, core_id) determines which core
issues next.  Both cores share a single L2 Cache object, so a miss by one
core fills L2 and can evict a line the other core has placed there, turning
a cheap L2 hit into an expensive memory fetch.

Core 0 issues first on cycle ties (lower core_id wins in the heap), keeping
the simulation deterministic.

PLRU convention (2-way)
-----------------------
    state = index of the LRU way (the next victim).
    On access/install of way W:  state ← 1 − W   (the other way is now LRU).
    On miss: victim = state.
"""

import random
from typing import Optional

# ── hardware constants ────────────────────────────────────────────────────────
L1_SIZE   = 2  * 1024      # 2 KB
L2_SIZE   = 16 * 1024      # 16 KB
LINE_SIZE = 32
WAYS      = 2

L1_SETS = L1_SIZE  // (WAYS * LINE_SIZE)   # 32
L2_SETS = L2_SIZE  // (WAYS * LINE_SIZE)   # 256

HIT_CYCLES  = 1
MISS_CYCLES = 10

NOP = "NOP"


# ── PLRU set (2-way) ──────────────────────────────────────────────────────────
class PLRUSet:
    """
    1-bit pseudo-LRU for a 2-way set.

    state = index of the LRU way (evicted on the next miss).
    Accessing/installing way W sets state ← 1 − W.
    """

    def __init__(self):
        self.tags:  list[Optional[int]] = [None, None]
        self.state: int = 0

    def lookup(self, tag: int) -> bool:
        for w, t in enumerate(self.tags):
            if t == tag:
                self.state = 1 - w
                return True
        return False

    def install(self, tag: int) -> Optional[int]:
        for w in range(WAYS):                   # use a cold slot first
            if self.tags[w] is None:
                self.tags[w] = tag
                self.state = 1 - w
                return None
        victim = self.state                     # evict the LRU way
        evicted = self.tags[victim]
        self.tags[victim] = tag
        self.state = 1 - victim
        return evicted

    def invalidate(self, tag: int) -> bool:
        for w in range(WAYS):
            if self.tags[w] == tag:
                self.tags[w] = None
                return True
        return False


# ── set-associative cache ─────────────────────────────────────────────────────
class Cache:
    def __init__(self, num_sets: int, name: str = ""):
        self.name     = name
        self.num_sets = num_sets
        self.sets     = [PLRUSet() for _ in range(num_sets)]

    def _idx_tag(self, addr: int) -> tuple[int, int]:
        line = addr // LINE_SIZE
        return line % self.num_sets, line // self.num_sets

    def access(self, addr: int) -> bool:
        idx, tag = self._idx_tag(addr)
        return self.sets[idx].lookup(tag)

    def install(self, addr: int) -> Optional[int]:
        idx, tag = self._idx_tag(addr)
        ev_tag = self.sets[idx].install(tag)
        if ev_tag is None:
            return None
        ev_line = ev_tag * self.num_sets + idx
        return ev_line * LINE_SIZE

    def reset(self) -> None:
        self.sets = [PLRUSet() for _ in range(self.num_sets)]


# ── address helpers ───────────────────────────────────────────────────────────
def _ldr(reg: int, addr: int) -> str:
    return f"LDR R{reg}, [0x{addr:04X}]"


def parse_ldr(instr: str) -> Optional[int]:
    """Return the load address of a 'LDR Rx, [0xADDR]' string, or None for NOP."""
    if instr.strip().upper() == NOP:
        return None
    try:
        return int(instr.split("[")[1].rstrip("]").strip(), 16)
    except (IndexError, ValueError):
        return None


# ── load primitive ────────────────────────────────────────────────────────────
def _load(addr: int, l1: Cache, l2: Cache) -> int:
    """Perform one load through the (l1, l2) hierarchy; return cycle cost."""
    if l1.access(addr):
        return HIT_CYCLES
    if l2.access(addr):
        l1.install(addr)
        return HIT_CYCLES
    # Full miss: fill both caches (inclusive policy)
    l2.install(addr)
    l1.install(addr)
    return MISS_CYCLES


# ── execution engines ─────────────────────────────────────────────────────────
def run_solo(trace: list[str]) -> int:
    """Execute *trace* alone on a fresh (cold) cache hierarchy."""
    l1, l2 = Cache(L1_SETS), Cache(L2_SETS)
    total = 0
    for instr in trace:
        addr = parse_ldr(instr)
        total += HIT_CYCLES if addr is None else _load(addr, l1, l2)
    return total


def run_concurrent(trace1: list[str], trace2: list[str]) -> tuple[int, int]:
    """
    Execute two traces concurrently on a shared L2 cache (cycle-accurate).

    Each core issues its next instruction as soon as its own clock allows —
    i.e. after the previous instruction's latency has elapsed.  The two cores
    share a single L2 Cache object, so their accesses interleave in real-cycle
    order rather than by instruction index.

    The interleaving is resolved with a min-heap keyed on (next_free_cycle,
    core_id).  Ties (both cores free at the same cycle) are broken by core_id
    so that Core 0 always issues first when clocks are equal — this keeps the
    simulation deterministic.

    Cycle accounting
    ----------------
    Each core's total cycle count is the cycle at which its last instruction
    *completes*, i.e. next_free_cycle after processing the final instruction.
    This matches run_solo: a trace of N all-miss LDRs costs N × MISS_CYCLES.
    """
    import heapq

    l2   = Cache(L2_SETS, "L2-shared")
    l1_0 = Cache(L1_SETS, "L1-C0")
    l1_1 = Cache(L1_SETS, "L1-C1")

    traces = [trace1, trace2]
    l1s    = [l1_0, l1_1]
    finish = [0, 0]          # cycle at which each core finishes its last instr

    # heap entries: (next_free_cycle, core_id, instruction_index)
    heap: list[tuple[int, int, int]] = [(0, 0, 0), (0, 1, 0)]

    while heap:
        cycle, core, idx = heapq.heappop(heap)

        if idx >= len(traces[core]):
            continue                      # this core is done

        addr = parse_ldr(traces[core][idx])
        cost = HIT_CYCLES if addr is None else _load(addr, l1s[core], l2)

        next_cycle = cycle + cost
        finish[core] = next_cycle         # update completion time
        heapq.heappush(heap, (next_cycle, core, idx + 1))

    return finish[0], finish[1]


# ── random trace generator ────────────────────────────────────────────────────
# Address space: 4× the L2 capacity, rounded down to a line boundary.
# This gives a realistic mix of L2 hits (re-accesses) and misses (cold lines),
# and enough aliasing across L2 sets to produce natural cache interference.
_ADDR_SPACE = (4 * L2_SIZE // LINE_SIZE) * LINE_SIZE   # 65 536 bytes
_N_LINES    = _ADDR_SPACE // LINE_SIZE                  # 2 048 distinct lines


def _l2_sets_of(trace: list[str]) -> list[int]:
    """Return the list of L2 set indices accessed by *trace* (duplicates kept)."""
    sets = []
    for instr in trace:
        addr = parse_ldr(instr)
        if addr is not None:
            sets.append((addr // LINE_SIZE) % L2_SETS)
    return sets


def _random_trace(
    trace_len:     int,
    nop_prob:      float,
    conflict_sets: list[int] | None = None,
    bias_prob:     float = 0.0,
    reg_start:     int   = 0,
) -> list[str]:
    """
    Generate a single random trace of exactly *trace_len* instructions.

    Each instruction slot is independently:
      - NOP              with probability *nop_prob*
      - LDR Rx, [addr]   otherwise, where addr is sampled as follows:

        If *conflict_sets* is provided and non-empty, with probability
        *bias_prob* the address is drawn from a line that maps to one of the
        conflict sets (biased sampling); otherwise it is drawn uniformly from
        all lines in _ADDR_SPACE (unbiased sampling).

        Within a conflict set, the tag is chosen uniformly from the available
        tags so that all lines in that set are equally likely.

    Parameters
    ----------
    conflict_sets : L2 set indices to bias towards (typically the sets accessed
                    by the other trace in the pair).
    bias_prob     : probability of drawing from conflict_sets on each LDR slot.
                    0.0 = fully uniform (no bias); 1.0 = always conflict sets.
    """
    instrs   = []
    reg      = reg_start
    n_tags   = _N_LINES // L2_SETS      # number of tags per L2 set

    for _ in range(trace_len):
        if random.random() < nop_prob:
            instrs.append(NOP)
        else:
            if conflict_sets and random.random() < bias_prob:
                s    = random.choice(conflict_sets)
                tag  = random.randrange(n_tags)
                addr = (tag * L2_SETS + s) * LINE_SIZE
            else:
                addr = random.randrange(_N_LINES) * LINE_SIZE
            instrs.append(_ldr(reg % 8, addr))
            reg += 1

    return instrs


def generate_trace_pair(
    trace_len: int   = 16,
    nop_prob:  float = 0.25,
    bias_prob: float = 0.5,
) -> tuple[list[str], list[str], int]:
    """
    Generate a pair of random traces and return their interference level.

    Trace 1 is generated uniformly at random.  Trace 2 is generated with a
    bias towards the L2 sets that Trace 1 accesses: each LDR in Trace 2 is
    drawn from one of those sets with probability *bias_prob*, and from the
    full address space otherwise.

    This shifts the IFL distribution away from the degenerate all-zeros case
    without hard-coding any specific conflict pattern — IFL remains a pure
    emergent output of the simulator.

    Parameters
    ----------
    trace_len : int
        Exact number of instructions per trace.  Must be >= 1.
    nop_prob  : float
        Probability that any given instruction slot is a NOP (default 0.25).
    bias_prob : float
        Probability that each LDR in Trace 2 is drawn from a set already
        accessed by Trace 1 (default 0.5).
        0.0 → fully uniform, same as the unbiased generator.
        1.0 → Trace 2 always targets Trace 1's sets (maximum conflict pressure).

    Returns
    -------
    trace1 : list[str]   — exactly trace_len instructions, uniformly sampled
    trace2 : list[str]   — exactly trace_len instructions, biased towards
                           the L2 sets of trace1
    ifl    : int
        Total cycle degradation: |cyc_T1_concurrent - cyc_T1_solo|
                                + |cyc_T2_concurrent - cyc_T2_solo|
    """
    if not 0.0 <= nop_prob  <= 1.0:
        raise ValueError(f"nop_prob must be in [0, 1], got {nop_prob}")
    if not 0.0 <= bias_prob <= 1.0:
        raise ValueError(f"bias_prob must be in [0, 1], got {bias_prob}")
    trace_len = max(1, trace_len)

    # Generate T1 uniformly
    trace1 = _random_trace(trace_len, nop_prob)

    # Extract L2 sets T1 accesses, then generate T2 biased towards them
    conflict_sets = _l2_sets_of(trace1)
    trace2 = _random_trace(trace_len, nop_prob,
                           conflict_sets=conflict_sets, bias_prob=bias_prob)


    return trace1, trace2


# ── demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import collections

    print("=" * 72)
    print("Cache hierarchy")
    print(f"  L1 (private) : {L1_SIZE // 1024}KB, {WAYS}-way, "
          f"{LINE_SIZE}B lines, {L1_SETS} sets, PLRU")
    print(f"  L2 (shared)  : {L2_SIZE // 1024}KB, {WAYS}-way, "
          f"{LINE_SIZE}B lines, {L2_SETS} sets, PLRU")
    print(f"  Hit: {HIT_CYCLES} cy  |  Miss: {MISS_CYCLES} cy")
    print(f"  Address space: {_ADDR_SPACE} bytes  ({_N_LINES} distinct lines)")
    print()

    # Single example pair
    t1, t2, ifl = generate_trace_pair(trace_len=16)
    s0 = run_solo(t1);  s1 = run_solo(t2)
    c0, c1 = run_concurrent(t1, t2)
    print("Example pair (trace_len=16, bias_prob=0.5):")
    print(f"  trace1     = {t1}")
    print(f"  trace2     = {t2}")
    print(f"  Solo       : T1={s0:3d} cy   T2={s1:3d} cy")
    print(f"  Concurrent : T1={c0:3d} cy   T2={c1:3d} cy")
    print(f"  IFL        = {ifl} cy  (ΔT1={c0-s0:+d}  ΔT2={c1-s1:+d})")
    print()

    # Compare IFL distributions across bias_prob values
    N = 1000
    print(f"IFL distribution over {N} pairs (trace_len=16):")
    print(f"  {'IFL':>5}  ", end="")
    bias_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    for b in bias_levels:
        print(f"  bias={b:.2f}", end="")
    print()

    # Collect all counts first
    all_counts = {}
    for b in bias_levels:
        c: dict[int, int] = collections.Counter()
        for _ in range(N):
            _, _, v = generate_trace_pair(trace_len=16, bias_prob=b)
            c[v] += 1
        all_counts[b] = c

    all_ifls = sorted({k for c in all_counts.values() for k in c})
    for ifl_val in all_ifls:
        print(f"  {ifl_val:>5}  ", end="")
        for b in bias_levels:
            pct = all_counts[b].get(ifl_val, 0) / N * 100
            print(f"  {pct:>9.1f}%", end="")
        print()

