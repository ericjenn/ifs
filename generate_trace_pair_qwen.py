import re
import random

def generate_trace_pair():
    """
    Generates a random pair of traces that may demonstrate cache interference 
    on a shared L2 cache with private L1 caches, and calculates the Interference Level (IFL).
    
    Hardware Config:
    - L1: 2KB, 2-way, 32B line, PLRU (Private per core)
    - L2: 16KB, 2-way, 32B line, PLRU (Shared)
    - Hit Latency: 1 cycle
    - Miss Latency: 10 cycles
    
    IFL Definition: (Total Cycles Concurrent) - (Total Cycles Isolated)
    """

    # --- Cache Simulator Implementation ---
    
    class CacheSet:
        def __init__(self, ways):
            self.ways = [None] * ways
            self.plru_bit = 0
            self.ways_count = ways

        def access(self, tag):
            for i in range(self.ways_count):
                if self.ways[i] == tag:
                    self.plru_bit = 1 - i
                    return True
            evict_way = self.plru_bit
            self.ways[evict_way] = tag
            self.plru_bit = 1 - evict_way
            return False

    class Cache:
        def __init__(self, size_kb, associativity, line_size):
            self.line_size = line_size
            self.num_sets = (size_kb * 1024) // (line_size * associativity)
            self.associativity = associativity
            self.sets = [CacheSet(associativity) for _ in range(self.num_sets)]
            
        def get_set_index(self, addr):
            return (addr // self.line_size) % self.num_sets
        
        def get_tag(self, addr):
            return addr // self.line_size // self.num_sets

        def access(self, addr):
            s_idx = self.get_set_index(addr)
            tag = self.get_tag(addr)
            return self.sets[s_idx].access(tag)

    def parse_instruction(instr):
        if instr.strip() == "NOP":
            return None
        match = re.search(r'\[(0x[0-9A-Fa-f]+)\]', instr)
        if match:
            return int(match.group(1), 16)
        return None

    def simulate_trace(trace, l1_cache, l2_cache=None):
        cycles = 0
        for instr in trace:
            addr = parse_instruction(instr)
            if addr is None:
                cycles += 1
                continue
            l1_hit = l1_cache.access(addr)
            if l1_hit:
                cycles += 1
            else:
                if l2_cache:
                    l2_hit = l2_cache.access(addr)
                    if l2_hit:
                        cycles += 1
                    else:
                        cycles += 10
                else:
                    cycles += 10
        return cycles

    # --- Random Trace Generation ---

    # Cache parameters
    L1_LINE_SIZE = 32
    L1_NUM_SETS = 32
    L2_NUM_SETS = 256

    def generate_random_address(preferred_l2_set=None, preferred_l1_set=None):
        if preferred_l2_set is not None:
            line_index = preferred_l2_set * L2_NUM_SETS + random.randint(0, 10) * L2_NUM_SETS
        elif preferred_l1_set is not None:
            line_index = preferred_l1_set + random.randint(0, 7) * L1_NUM_SETS
        else:
            line_index = random.randint(0, 20000)
        return line_index * L1_LINE_SIZE

    def generate_random_trace(length, conflict_probability=0.5):
        trace = []
        base_l2_set = random.randint(0, L2_NUM_SETS - 1)
        base_l1_set = random.randint(0, L1_NUM_SETS - 1)
        registers = ['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7']
        
        for i in range(length):
            if random.random() < 0.1:
                trace.append("NOP")
            else:
                reg = random.choice(registers)
                if random.random() < conflict_probability:
                    addr = generate_random_address(preferred_l2_set=base_l2_set)
                else:
                    addr = generate_random_address()
                trace.append(f"LDR {reg}, [0x{addr:04X}]")
        return trace

    trace1_length = random.randint(4, 8)
    trace2_length = random.randint(3, 7)
    trace1 = generate_random_trace(trace1_length, conflict_probability=0.6)
    trace2 = generate_random_trace(trace2_length, conflict_probability=0.6)

    # --- Calculation of IFL ---

    def get_isolated_time(trace):
        l1 = Cache(2, 2, 32)
        l2 = Cache(16, 2, 32)
        return simulate_trace(trace, l1, l2)

    t1_alone = get_isolated_time(trace1)
    t2_alone = get_isolated_time(trace2)
    total_isolated = t1_alone + t2_alone

    def get_concurrent_time(trace1, trace2):
        l1_core1 = Cache(2, 2, 32)
        l1_core2 = Cache(2, 2, 32)
        l2_shared = Cache(16, 2, 32)
        
        cycles_c1 = 0
        cycles_c2 = 0
        max_len = max(len(trace1), len(trace2))
        
        for i in range(max_len):
            if i < len(trace1):
                instr = trace1[i]
                addr = parse_instruction(instr)
                if addr is None:
                    cycles_c1 += 1
                else:
                    if l1_core1.access(addr):
                        cycles_c1 += 1
                    else:
                        if l2_shared.access(addr):
                            cycles_c1 += 1
                        else:
                            cycles_c1 += 10
            if i < len(trace2):
                instr = trace2[i]
                addr = parse_instruction(instr)
                if addr is None:
                    cycles_c2 += 1
                else:
                    if l1_core2.access(addr):
                        cycles_c2 += 1
                    else:
                        if l2_shared.access(addr):
                            cycles_c2 += 1
                        else:
                            cycles_c2 += 10
        return cycles_c1 + cycles_c2

    total_concurrent = get_concurrent_time(trace1, trace2)
    ifl = total_concurrent - total_isolated

    return trace1, trace2, ifl

# Example Usage
if __name__ == "__main__":
    while True:
        t1, t2, interference = generate_trace_pair()
        if interference:
            print(f"Trace 1: {t1}")
            print(f"Trace 2: {t2}")
            print(f"Interference Level (Cycles): {interference}")