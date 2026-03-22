"""
Cache trace pair generator for a two-core system.

Hardware:
  - Private L1 per core : 2 KB, 2-way, 32-byte lines, PLRU
  - Shared  L2           : 16 KB, 2-way, 32-byte lines, PLRU (inclusive)

Public API
----------
    trace1, trace2, ifl = generate_trace_pair(trace_len=10,
                                               interference_level="random")

    ifl = |cyc_T1_concurrent - cyc_T1_solo|
        + |cyc_T2_concurrent - cyc_T2_solo|

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

Interference mechanism (_build_conflict_block)
----------------------------------------------
With a 2-way L2, line A is evicted by Core 1 through the following
cycle-accurate sequence (MISS=10 cy, HIT=1 cy):

  cycle  0: C0 loads A (miss)      → L2[S]=[A,_], PLRU evict_next=1
             C1 loads B (same S, miss) → L2[S]=[A,B], PLRU evict_next=0 (=A)
  cycle 10: C0 loads E1 (miss, same L1 set as A)
             C1 reloads B (hit)    → PLRU still evict_next=0 (A) ✓
  cycle 11: C1 loads C (same S, miss) → L2 evicts A ✓
  cycle 20: C0 loads E2 (miss, same L1 set as A) → L1 evicts A ✓
  cycle 30: C0 reloads A → L1 miss + L2 miss → 10 cy  (concurrent)
                         vs L1 miss + L2 hit  →  1 cy  (solo)

Interference contribution per block = 10 − 1 = 9 cycles on T1.
C1 naturally executes its reload-B and load-C during C0's stall on E1/E2,
so no NOP padding is required (unlike the old round-robin model).
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
def addr_for(set_idx: int, tag: int, num_sets: int = L2_SETS) -> int:
    """Return a line-aligned byte address that maps to *set_idx* with *tag*."""
    return (tag * num_sets + set_idx) * LINE_SIZE


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
    simulation deterministic and consistent with the conflict-block design.

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


# ── conflict block ────────────────────────────────────────────────────────────
def _build_conflict_block(
    l2_set: int,
) -> tuple[list[Optional[int]], list[Optional[int]]]:
    """
    Return a pair of raw address sequences (None = NOP) that produce 9 cycles
    of interference on trace1 under cycle-accurate simulation.

    The sequences must be converted to LDR strings *in order* — do not
    randomise their positions, as interference depends on the exact cycle
    interleaving produced by the latencies below.

    Cycle-accurate timeline (MISS=10 cy, HIT=1 cy)
    ------------------------------------------------
    C0 clock | C1 clock | event
    ---------+----------+--------------------------------------------------
       0     |    0     | C0 issues load A (miss, 10 cy)
       0     |    0     | C1 issues load B (miss, 10 cy)  → L2[S]=[A,B], PLRU→evict A next
      10     |   10     | C0 issues load E1 (miss, 10 cy)  [same L1 set as A]
      10     |   10     | C1 issues reload B (hit, 1 cy)   → PLRU still points at A ✓
      11     |          | C1 issues load C (miss, 10 cy)   [same L2 set S]
                           → L2 PLRU evicts A from L2  ✓
      20     |          | C0 issues load E2 (miss, 10 cy)  [same L1 set as A]
                           → C0-L1 PLRU evicts A from L1  ✓
      21     |          | C1 done
      30     |          | C0 issues reload A
                           → C0-L1 miss (A evicted) + L2 miss (A evicted) → 10 cy
                           In solo: C0-L1 miss + L2 hit → 1 cy
                           Interference on C0 = 10 − 1 = 9 cy  ✓

    The NOP padding required by the old round-robin model is no longer needed:
    C1 naturally executes 'reload B' and 'load C' during C0's stall on E1/E2,
    so the eviction happens at exactly the right moment without any padding.

    Addresses
    ---------
    A, B, C  map to the same L2 set (l2_set), different tags → will conflict.
    E1, E2   map to different L2 sets but the same L1 set as A → evict A
             from L1 without disturbing L2[l2_set].
    """
    A  = addr_for(l2_set, 0)
    B  = addr_for(l2_set, 1)
    C  = addr_for(l2_set, 2)
    E1 = addr_for((l2_set +     L1_SETS) % L2_SETS, 0)   # same L1 set as A
    E2 = addr_for((l2_set + 2 * L1_SETS) % L2_SETS, 0)   # same L1 set as A

    # 4 instructions each — no NOP padding needed
    t1: list[Optional[int]] = [A,  E1, E2, A   ]
    t2: list[Optional[int]] = [B,  B,  C,  None]
    return t1, t2


def _addrs_to_trace(addrs: list[Optional[int]], reg_start: int = 0) -> list[str]:
    """Convert a raw address sequence (None = NOP, int = LDR) to instruction strings."""
    instrs = []
    reg = reg_start
    for a in addrs:
        if a is None:
            instrs.append(NOP)
        else:
            instrs.append(_ldr(reg % 8, a))
            reg += 1
    return instrs


# ── main generator ────────────────────────────────────────────────────────────
def generate_trace_pair(
    trace_len: int = 10,
    interference_level: str = "random",
) -> tuple[list[str], list[str], int]:
    """
    Generate a pair of instruction traces and return their interference level.

    Parameters
    ----------
    trace_len : int
        Approximate total instruction count per trace (including NOPs). Min 5.
    interference_level : str
        ``'low'``    – disjoint address spaces; ifl = 0.
        ``'medium'`` – one conflict block; ifl = 9 cy.
        ``'high'``   – 2–4 conflict blocks; ifl = n_blocks × 9 cy.
        ``'random'`` – one of the above chosen randomly.

    Returns
    -------
    trace1 : list[str]
    trace2 : list[str]
    ifl    : int
        Total cycle degradation: Σ |concurrent − solo| over both cores.
    """
    if interference_level == "random":
        interference_level = random.choice(["low", "medium", "high"])

    trace_len = max(5, trace_len)

    # ── build raw address lists ──────────────────────────────────────────────
    if interference_level == "low":
        # Disjoint L2 set pools → no structural conflict, ifl = 0.
        half  = L2_SETS // 2
        n_mem = trace_len * 3 // 4
        pool0 = random.sample(range(0, half),       k=min(n_mem, half))
        pool1 = random.sample(range(half, L2_SETS), k=min(n_mem, half))
        t1_raw: list[Optional[int]] = \
            [addr_for(pool0[i % len(pool0)], 0) for i in range(n_mem)] + \
            [None] * (trace_len - n_mem)
        t2_raw: list[Optional[int]] = \
            [addr_for(pool1[i % len(pool1)], 0) for i in range(n_mem)] + \
            [None] * (trace_len - n_mem)
        # Shuffle the NOP positions uniformly
        random.shuffle(t1_raw)
        random.shuffle(t2_raw)

    elif interference_level == "medium":
        # One conflict block + a few independent accesses appended after it.
        s = random.randint(0, L2_SETS - 1)
        c0_block, c1_block = _build_conflict_block(s)

        used_sets = {s,
                     (s +     L1_SETS) % L2_SETS,
                     (s + 2 * L1_SETS) % L2_SETS}
        priv0 = random.sample([x for x in range(L2_SETS // 2)
                                if x not in used_sets], k=2)
        priv1 = random.sample([x for x in range(L2_SETS // 2, L2_SETS)
                                if x not in used_sets], k=2)

        # Conflict block comes first (order must be preserved), extras appended
        t1_raw = list(c0_block) + [addr_for(p, 0) for p in priv0]
        t2_raw = list(c1_block) + [addr_for(p, 0) for p in priv1]

    else:  # high
        # 2–4 independent conflict blocks concatenated.
        n_blocks = random.randint(2, 4)
        all_sets = list(range(L2_SETS))
        chosen   = random.sample(all_sets, k=n_blocks)
        t1_raw, t2_raw = [], []
        for s in chosen:
            c0, c1 = _build_conflict_block(s)
            t1_raw += list(c0)
            t2_raw += list(c1)

    # ── convert to instruction strings ───────────────────────────────────────
    trace1 = _addrs_to_trace(t1_raw, reg_start=0)
    trace2 = _addrs_to_trace(t2_raw, reg_start=0)

    return trace1, trace2


# ── demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 72)
    print("Cache hierarchy")
    print(f"  L1 (private) : {L1_SIZE // 1024}KB, {WAYS}-way, "
          f"{LINE_SIZE}B lines, {L1_SETS} sets, PLRU")
    print(f"  L2 (shared)  : {L2_SIZE // 1024}KB, {WAYS}-way, "
          f"{LINE_SIZE}B lines, {L2_SETS} sets, PLRU")
    print(f"  Hit: {HIT_CYCLES} cy  |  Miss: {MISS_CYCLES} cy")
    print()

    for level in ("low", "medium", "high"):
        t1, t2, ifl = generate_trace_pair(trace_len=10, interference_level=level)
        s0 = run_solo(t1)
        s1 = run_solo(t2)
        c0, c1 = run_concurrent(t1, t2)
        print(f"[{level.upper():6s}]")
        print(f"  trace1      = {t1}")
        print(f"  trace2      = {t2}")
        print(f"  Solo        : T1={s0:3d} cy   T2={s1:3d} cy")
        print(f"  Concurrent  : T1={c0:3d} cy   T2={c1:3d} cy")
        print(f"  IFL         = {ifl} cy   (ΔT1={c0 - s0:+d}  ΔT2={c1 - s1:+d})")
        print()

    print("-" * 72)
    print("generate_trace_pair() — direct API call:")
    t1, t2, ifl = generate_trace_pair()
    print(f"  trace1 = {t1}")
    print(f"  trace2 = {t2}")
    print(f"  ifl    = {ifl}")
