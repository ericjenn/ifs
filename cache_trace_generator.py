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
Round-robin interleaving: step i executes trace1[i] on Core 0 then trace2[i]
on Core 1 over a *shared* L2 Cache object.  Core 1's loads can therefore evict
lines that Core 0 just placed in L2, turning a cheap L2 hit into an expensive
memory fetch.

PLRU convention (2-way)
-----------------------
    state = index of the LRU way (the next victim).
    On access/install of way W:  state ← 1 − W   (the other way is now LRU).
    On miss: victim = state.

Interference mechanism (_build_conflict_block)
----------------------------------------------
With a 2-way L2 a line A is evicted by Core 1 through this exact sequence
(round-robin steps, one per pair):

  step 0: C0 loads A          → L2[S] = [A, _],  PLRU evict_next = 1
          C1 loads B (same S) → L2[S] = [A, B],  PLRU evict_next = 0 (= A's way)
  step 1: C0 loads E1 (same L1 set as A, different L2 set)
                               → C0-L1 now holds [A, E1]; PLRU will evict A from L1 next
          C1 re-loads B       → B becomes MRU; L2 PLRU still evict_next = 0 (A) ✓
  step 2: C0 loads E2 (same L1 set, yet another L2 set)
                               → C0-L1 PLRU evicts A (A was LRU)  ← A gone from C0-L1
          C1 loads C (same S) → L2 PLRU evicts A (way 0)          ← A gone from L2
  step 3: C0 NOP  |  C1 NOP
  step 4: C0 re-loads A       → C0-L1 miss + L2 miss → 10 cy  (concurrent)
                                  vs C0-L1 miss + L2 hit → 1 cy  (solo)
          C1 NOP

Interference contribution per block = 10 − 1 = 9 cycles on T1.
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
    Execute two traces concurrently on a shared L2 cache (round-robin).

    At each step i:  Core 0 executes trace1[i], then Core 1 executes trace2[i].
    The two cores share the same L2 Cache object; Core 1's fills can therefore
    evict lines placed by Core 0, creating cache interference.
    """
    l2   = Cache(L2_SETS, "L2-shared")
    l1_0 = Cache(L1_SETS, "L1-C0")
    l1_1 = Cache(L1_SETS, "L1-C1")
    cy0 = cy1 = 0
    for i in range(max(len(trace1), len(trace2))):
        if i < len(trace1):
            a = parse_ldr(trace1[i])
            cy0 += HIT_CYCLES if a is None else _load(a, l1_0, l2)
        if i < len(trace2):
            a = parse_ldr(trace2[i])
            cy1 += HIT_CYCLES if a is None else _load(a, l1_1, l2)
    return cy0, cy1


# ── conflict block ────────────────────────────────────────────────────────────
def _build_conflict_block(
    l2_set: int,
) -> tuple[list[Optional[int]], list[Optional[int]]]:
    """
    Return a pair of raw address sequences (None = NOP) of length 5 that
    produce 9 cycles of interference on trace1 in round-robin execution.

    The sequences must be converted to LDR strings *in order* — do not
    randomise their positions, as the interference depends on exact step alignment.

    Round-robin step layout:
      step 0: C0 loads A (miss)    | C1 loads B (miss)   → L2[S]=[A,B], evict_next=A
      step 1: C0 loads E1 (miss)   | C1 reloads B (hit)  → L2 PLRU still points at A
      step 2: C0 loads E2 (miss,   | C1 loads C (miss)   → C0-L1 evicts A; L2 evicts A
              A evicted from L1)
      step 3: C0 NOP               | C1 NOP
      step 4: C0 reloads A → full  | C1 NOP
              miss (10 cy); in solo
              this would be L2 hit (1 cy)
    """
    A  = addr_for(l2_set, 0)
    B  = addr_for(l2_set, 1)
    C  = addr_for(l2_set, 2)
    E1 = addr_for((l2_set +     L1_SETS) % L2_SETS, 0)   # same L1 set as A
    E2 = addr_for((l2_set + 2 * L1_SETS) % L2_SETS, 0)   # same L1 set as A

    t1: list[Optional[int]] = [A,    E1,   E2,   None, A   ]
    t2: list[Optional[int]] = [B,    B,    C,    None, None]
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

    # ── compute interference ──────────────────────────────────────────────────
    solo0        = run_solo(trace1)
    solo1        = run_solo(trace2)
    conc0, conc1 = run_concurrent(trace1, trace2)
    ifl = abs(conc0 - solo0) + abs(conc1 - solo1)

    return trace1, trace2, ifl


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
