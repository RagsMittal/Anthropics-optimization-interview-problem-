"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Ultra-optimized VLIW SIMD kernel implementation targeting sub-1000 cycles.
        """
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)

        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")

        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        one_vec = self.alloc_scratch("one_vec", VLEN)
        two_vec = self.alloc_scratch("two_vec", VLEN)
        leaf_limit_vec = self.alloc_scratch("leaf_limit_vec", VLEN)

        self.add("valu", ("vbroadcast", one_vec, one_const))
        self.add("valu", ("vbroadcast", two_vec, two_const))
        self.add("valu", ("vbroadcast", leaf_limit_vec, n_nodes_const))

        hash_stage_consts = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            c1_scalar = self.scratch_const(val1)
            c3_scalar = self.scratch_const(val3)
            c1_vec = self.alloc_scratch(f"hash_{hi}_c1", VLEN)
            c3_vec = self.alloc_scratch(f"hash_{hi}_c3", VLEN)
            self.add("valu", ("vbroadcast", c1_vec, c1_scalar))
            self.add("valu", ("vbroadcast", c3_vec, c3_scalar))
            hash_stage_consts.append((op1, c1_vec, op2, op3, c3_vec))

        num_vectors = batch_size // VLEN

        vec_indices = [self.alloc_scratch(f"idx_vec_{k}", VLEN) for k in range(num_vectors)]
        vec_values = [self.alloc_scratch(f"val_vec_{k}", VLEN) for k in range(num_vectors)]

        vtmp1_group = [self.alloc_scratch(f"vtmp1_g{m}", VLEN) for m in range(8)]
        vtmp2_group = [self.alloc_scratch(f"vtmp2_g{m}", VLEN) for m in range(8)]
        vtmp3_group = [self.alloc_scratch(f"vtmp3_g{m}", VLEN) for m in range(8)]
        vtmp_node_A = vtmp3_group[0]
        idx_even_group = [self.alloc_scratch(f"idx_even_g{m}", VLEN) for m in range(8)]
        idx_odd_group = [self.alloc_scratch(f"idx_odd_g{m}", VLEN) for m in range(8)]
        cond_vec_group = [self.alloc_scratch(f"cond_vec_g{m}", VLEN) for m in range(8)]
        cond_vec_A = cond_vec_group[0]
        is_not_leaf_group = [self.alloc_scratch(f"is_not_leaf_g{m}", VLEN) for m in range(8)]

        # Convenience aliases for A, B, C, D
        vtmp1_A, vtmp2_A, vtmp3_A, idx_even_A, idx_odd_A, is_not_leaf_A = vtmp1_group[0], vtmp2_group[0], vtmp3_group[0], idx_even_group[0], idx_odd_group[0], is_not_leaf_group[0]
        vtmp1_B, vtmp2_B, vtmp3_B, idx_even_B, idx_odd_B, cond_vec_B, is_not_leaf_B = vtmp1_group[1], vtmp2_group[1], vtmp3_group[1], idx_even_group[1], idx_odd_group[1], cond_vec_group[1], is_not_leaf_group[1]
        vtmp1_C, vtmp2_C, vtmp3_C, idx_even_C, idx_odd_C, cond_vec_C, is_not_leaf_C = vtmp1_group[2], vtmp2_group[2], vtmp3_group[2], idx_even_group[2], idx_odd_group[2], cond_vec_group[2], is_not_leaf_group[2]
        vtmp1_D, vtmp2_D, vtmp3_D, idx_even_D, idx_odd_D, cond_vec_D, is_not_leaf_D = vtmp1_group[3], vtmp2_group[3], vtmp3_group[3], idx_even_group[3], idx_odd_group[3], cond_vec_group[3], is_not_leaf_group[3]

        g_addrs = [[self.alloc_scratch(f"g_addr_v{m}_l{l}") for l in range(VLEN)] for m in range(8)]
        g_nodes = [self.alloc_scratch(f"g_node_v{m}", VLEN) for m in range(8)]

        lane_addrs_A = [self.alloc_scratch(f"g_addr_A_{lane}") for lane in range(VLEN)]
        lane_addrs_B = [self.alloc_scratch(f"g_addr_B_{lane}") for lane in range(VLEN)]

        node0_scalar = self.alloc_scratch("node0_scalar")

        node1_scalar = self.alloc_scratch("node1_scalar")
        node2_scalar = self.alloc_scratch("node2_scalar")
        addr_t1 = self.alloc_scratch("addr_t1")
        addr_t2 = self.alloc_scratch("addr_t2")
        t1_vec = self.alloc_scratch("t1_vec", VLEN)
        t2_vec = self.alloc_scratch("t2_vec", VLEN)

        node3_sc = self.alloc_scratch("node3_sc")
        node4_sc = self.alloc_scratch("node4_sc")
        node5_sc = self.alloc_scratch("node5_sc")
        node6_sc = self.alloc_scratch("node6_sc")
        addr_t3 = self.alloc_scratch("addr_t3")
        addr_t4 = self.alloc_scratch("addr_t4")
        addr_t5 = self.alloc_scratch("addr_t5")
        addr_t6 = self.alloc_scratch("addr_t6")
        t3_vec = self.alloc_scratch("t3_vec", VLEN)
        t4_vec = self.alloc_scratch("t4_vec", VLEN)
        t5_vec = self.alloc_scratch("t5_vec", VLEN)
        t6_vec = self.alloc_scratch("t6_vec", VLEN)
        c3_sc = self.scratch_const(3)
        c4_sc = self.scratch_const(4)
        c5_sc = self.scratch_const(5)
        c6_sc = self.scratch_const(6)
        c3_v_r2 = self.alloc_scratch("c3_v_r2", VLEN)
        c5_v_r2 = self.alloc_scratch("c5_v_r2", VLEN)
        v34_vec = self.alloc_scratch("v34_vec", VLEN)
        v56_vec = self.alloc_scratch("v56_vec", VLEN)
        cond_is_3 = self.alloc_scratch("cond_is_3", VLEN)
        cond_is_5 = self.alloc_scratch("cond_is_5", VLEN)
        cond_lt_5 = self.alloc_scratch("cond_lt_5", VLEN)



        addr_scratch = self.alloc_scratch("addr_scratch")
        for k in range(num_vectors):
            offset_const = self.scratch_const(k * VLEN)
            self.instrs.append({
                "alu": [("+", addr_scratch, self.scratch["inp_indices_p"], offset_const)],
            })
            self.instrs.append({
                "load": [("vload", vec_indices[k], addr_scratch)],
            })
            self.instrs.append({
                "alu": [("+", addr_scratch, self.scratch["inp_values_p"], offset_const)],
            })
            self.instrs.append({
                "load": [("vload", vec_values[k], addr_scratch)],
            })

        node_vecs_r3 = [self.alloc_scratch(f"node_vec_r3_{idx}", VLEN) for idx in range(7, 15)]
        node_sc_r3 = self.alloc_scratch("node_sc_r3")
        for idx in range(7, 15):
            self.instrs.append({
                "alu": [("+", addr_scratch, self.scratch["forest_values_p"], self.scratch_const(idx))]
            })
            self.instrs.append({
                "load": [("load", node_sc_r3, addr_scratch)]
            })
            self.instrs.append({
                "valu": [("vbroadcast", node_vecs_r3[idx - 7], node_sc_r3)]
            })

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting optimized execution"))

        def emit_4vec_hash_13cycles(k0, k1, k2, k3):
            b_list = []
            # Cycle 0: Stg 0 op1, op3 for k0, k1, k2
            op1, c1, op2, op3, c3 = hash_stage_consts[0]
            b_list.append({
                "valu": [
                    (op1, vtmp1_A, vec_values[k0], c1), (op3, vtmp2_A, vec_values[k0], c3),
                    (op1, vtmp1_B, vec_values[k1], c1), (op3, vtmp2_B, vec_values[k1], c3),
                    (op1, vtmp1_C, vec_values[k2], c1), (op3, vtmp2_C, vec_values[k2], c3),
                ]
            })
            for s in range(5):
                cur_op1, cur_c1, cur_op2, cur_op3, cur_c3 = hash_stage_consts[s]
                nxt_op1, nxt_c1, nxt_op2, nxt_op3, nxt_c3 = hash_stage_consts[s + 1]

                # Sub-cycle 1: cur op1/op3 for k3, cur op2 for k0, k1, nxt op1/op3 for k0
                b_list.append({
                    "valu": [
                        (cur_op1, vtmp1_D, vec_values[k3], cur_c1), (cur_op3, vtmp2_D, vec_values[k3], cur_c3),
                        (cur_op2, vec_values[k0], vtmp1_A, vtmp2_A), (cur_op2, vec_values[k1], vtmp1_B, vtmp2_B),
                        (nxt_op1, vtmp1_A, vec_values[k0], nxt_c1), (nxt_op3, vtmp2_A, vec_values[k0], nxt_c3),
                    ]
                })
                # Sub-cycle 2: cur op2 for k2, k3, nxt op1/op3 for k1, k2
                b_list.append({
                    "valu": [
                        (cur_op2, vec_values[k2], vtmp1_C, vtmp2_C), (cur_op2, vec_values[k3], vtmp1_D, vtmp2_D),
                        (nxt_op1, vtmp1_B, vec_values[k1], nxt_c1), (nxt_op3, vtmp2_B, vec_values[k1], nxt_c3),
                        (nxt_op1, vtmp1_C, vec_values[k2], nxt_c1), (nxt_op3, vtmp2_C, vec_values[k2], nxt_c3),
                    ]
                })

            # Final stage (Stage 5) finish
            stg5_op1, stg5_c1, stg5_op2, stg5_op3, stg5_c3 = hash_stage_consts[5]
            b_list.append({
                "valu": [
                    (stg5_op1, vtmp1_D, vec_values[k3], stg5_c1), (stg5_op3, vtmp2_D, vec_values[k3], stg5_c3),
                    (stg5_op2, vec_values[k0], vtmp1_A, vtmp2_A), (stg5_op2, vec_values[k1], vtmp1_B, vtmp2_B),
                ]
            })
            b_list.append({
                "valu": [
                    (stg5_op2, vec_values[k2], vtmp1_C, vtmp2_C), (stg5_op2, vec_values[k3], vtmp1_D, vtmp2_D),
                ]
            })
            return b_list

            b_list.append({
                "valu": [
                    ("&", cond_vec_A, vec_values[k0], one_vec),
                    ("multiply_add", idx_even_A, vec_indices[k0], two_vec, one_vec),
                    ("multiply_add", idx_odd_A, vec_indices[k0], two_vec, two_vec),
                    ("&", cond_vec_B, vec_values[k1], one_vec),
                    ("multiply_add", idx_even_B, vec_indices[k1], two_vec, one_vec),
                    ("multiply_add", idx_odd_B, vec_indices[k1], two_vec, two_vec),
                ]
            })
            b_list.append({
                "flow": [("vselect", vtmp3_A, cond_vec_A, idx_odd_A, idx_even_A)],
                "valu": [
                    ("<", is_not_leaf_A, vec_indices[k0], leaf_limit_vec),
                    ("<", is_not_leaf_B, vec_indices[k1], leaf_limit_vec),
                ]
            })
            b_list.append({
                "flow": [("vselect", vtmp3_B, cond_vec_B, idx_odd_B, idx_even_B)],
            })
            b_list.append({
                "valu": [
                    ("*", vec_indices[k0], vtmp3_A, is_not_leaf_A),
                    ("*", vec_indices[k1], vtmp3_B, is_not_leaf_B),
                ]
            })

            b_list.append({
                "valu": [
                    ("&", cond_vec_C, vec_values[k2], one_vec),
                    ("multiply_add", idx_even_C, vec_indices[k2], two_vec, one_vec),
                    ("multiply_add", idx_odd_C, vec_indices[k2], two_vec, two_vec),
                    ("&", cond_vec_D, vec_values[k3], one_vec),
                    ("multiply_add", idx_even_D, vec_indices[k3], two_vec, one_vec),
                    ("multiply_add", idx_odd_D, vec_indices[k3], two_vec, two_vec),
                ]
            })
        def emit_4vec_group_hash_and_next_idx(k0, k1, k2, k3):
            b_list = []
            for hi, (op1, c1_v, op2, op3, c3_v) in enumerate(hash_stage_consts):
                b_list.append({
                    "valu": [
                        (op1, vtmp1_A, vec_values[k0], c1_v),
                        (op3, vtmp2_A, vec_values[k0], c3_v),
                        (op1, vtmp1_B, vec_values[k1], c1_v),
                        (op3, vtmp2_B, vec_values[k1], c3_v),
                        (op1, vtmp1_C, vec_values[k2], c1_v),
                        (op3, vtmp2_C, vec_values[k2], c3_v),
                    ]
                })
                b_list.append({
                    "valu": [
                        (op1, vtmp1_D, vec_values[k3], c1_v),
                        (op3, vtmp2_D, vec_values[k3], c3_v),
                        (op2, vec_values[k0], vtmp1_A, vtmp2_A),
                        (op2, vec_values[k1], vtmp1_B, vtmp2_B),
                    ]
                })
                b_list.append({
                    "valu": [
                        (op2, vec_values[k2], vtmp1_C, vtmp2_C),
                        (op2, vec_values[k3], vtmp1_D, vtmp2_D),
                    ]
                })

            b_list.append({
                "valu": [
                    ("&", cond_vec_A, vec_values[k0], one_vec),
                    ("multiply_add", idx_even_A, vec_indices[k0], two_vec, one_vec),
                    ("multiply_add", idx_odd_A, vec_indices[k0], two_vec, two_vec),
                    ("&", cond_vec_B, vec_values[k1], one_vec),
                    ("multiply_add", idx_even_B, vec_indices[k1], two_vec, one_vec),
                    ("multiply_add", idx_odd_B, vec_indices[k1], two_vec, two_vec),
                ]
            })
            b_list.append({
                "flow": [("vselect", vtmp3_A, cond_vec_A, idx_odd_A, idx_even_A)],
                "valu": [
                    ("&", cond_vec_C, vec_values[k2], one_vec),
                    ("multiply_add", idx_even_C, vec_indices[k2], two_vec, one_vec),
                    ("multiply_add", idx_odd_C, vec_indices[k2], two_vec, two_vec),
                    ("&", cond_vec_D, vec_values[k3], one_vec),
                    ("multiply_add", idx_even_D, vec_indices[k3], two_vec, one_vec),
                    ("multiply_add", idx_odd_D, vec_indices[k3], two_vec, two_vec),
                ]
            })
            b_list.append({
                "flow": [("vselect", vtmp3_B, cond_vec_B, idx_odd_B, idx_even_B)],
                "valu": [
                    ("<", is_not_leaf_A, vec_indices[k0], leaf_limit_vec),
                    ("<", is_not_leaf_B, vec_indices[k1], leaf_limit_vec),
                    ("<", is_not_leaf_C, vec_indices[k2], leaf_limit_vec),
                    ("<", is_not_leaf_D, vec_indices[k3], leaf_limit_vec),
                ]
            })
            b_list.append({
                "flow": [("vselect", vtmp3_C, cond_vec_C, idx_odd_C, idx_even_C)],
                "valu": [
                    ("*", vec_indices[k0], vtmp3_A, is_not_leaf_A),
                    ("*", vec_indices[k1], vtmp3_B, is_not_leaf_B),
                ]
            })
            b_list.append({
                "flow": [("vselect", vtmp3_D, cond_vec_D, idx_odd_D, idx_even_D)],
            })
            b_list.append({
                "valu": [
                    ("*", vec_indices[k2], vtmp3_C, is_not_leaf_C),
                    ("*", vec_indices[k3], vtmp3_D, is_not_leaf_D),
                ]
            })
            return b_list

        def emit_8vec_hash_25cycles(k_base):
            b_list = []
            for hi in range(6):
                op1, c1_v, op2, op3, c3_v = hash_stage_consts[hi]
                if hi == 0:
                    b_list.append({
                        "valu": [
                            (op1, vtmp1_group[0], vec_values[k_base + 0], c1_v), (op3, vtmp2_group[0], vec_values[k_base + 0], c3_v),
                            (op1, vtmp1_group[1], vec_values[k_base + 1], c1_v), (op3, vtmp2_group[1], vec_values[k_base + 1], c3_v),
                            (op1, vtmp1_group[2], vec_values[k_base + 2], c1_v), (op3, vtmp2_group[2], vec_values[k_base + 2], c3_v),
                        ]
                    })
                else:
                    prev_op2 = hash_stage_consts[hi - 1][2]
                    b_list.append({
                        "valu": [
                            (prev_op2, vec_values[k_base + 6], vtmp1_group[6], vtmp2_group[6]),
                            (prev_op2, vec_values[k_base + 7], vtmp1_group[7], vtmp2_group[7]),
                            (op1, vtmp1_group[0], vec_values[k_base + 0], c1_v), (op3, vtmp2_group[0], vec_values[k_base + 0], c3_v),
                            (op1, vtmp1_group[1], vec_values[k_base + 1], c1_v), (op3, vtmp2_group[1], vec_values[k_base + 1], c3_v),
                        ]
                    })

                b_list.append({
                    "valu": [
                        (op2, vec_values[k_base + 0], vtmp1_group[0], vtmp2_group[0]),
                        (op2, vec_values[k_base + 1], vtmp1_group[1], vtmp2_group[1]),
                        (op1, vtmp1_group[2], vec_values[k_base + 2], c1_v), (op3, vtmp2_group[2], vec_values[k_base + 2], c3_v),
                        (op1, vtmp1_group[3], vec_values[k_base + 3], c1_v), (op3, vtmp2_group[3], vec_values[k_base + 3], c3_v),
                    ]
                })
                b_list.append({
                    "valu": [
                        (op2, vec_values[k_base + 2], vtmp1_group[2], vtmp2_group[2]),
                        (op2, vec_values[k_base + 3], vtmp1_group[3], vtmp2_group[3]),
                        (op1, vtmp1_group[4], vec_values[k_base + 4], c1_v), (op3, vtmp2_group[4], vec_values[k_base + 4], c3_v),
                        (op1, vtmp1_group[5], vec_values[k_base + 5], c1_v), (op3, vtmp2_group[5], vec_values[k_base + 5], c3_v),
                    ]
                })
                b_list.append({
                    "valu": [
                        (op2, vec_values[k_base + 4], vtmp1_group[4], vtmp2_group[4]),
                        (op2, vec_values[k_base + 5], vtmp1_group[5], vtmp2_group[5]),
                        (op1, vtmp1_group[6], vec_values[k_base + 6], c1_v), (op3, vtmp2_group[6], vec_values[k_base + 6], c3_v),
                        (op1, vtmp1_group[7], vec_values[k_base + 7], c1_v), (op3, vtmp2_group[7], vec_values[k_base + 7], c3_v),
                    ]
                })

            last_op2 = hash_stage_consts[5][2]
            b_list.append({
                "valu": [
                    (last_op2, vec_values[k_base + 6], vtmp1_group[6], vtmp2_group[6]),
                    (last_op2, vec_values[k_base + 7], vtmp1_group[7], vtmp2_group[7]),
                ]
            })
            return b_list

        for round_i in range(rounds):
            if round_i == 0:
                self.instrs.append({
                    "load": [("load", node0_scalar, self.scratch["forest_values_p"])]
                })
                self.instrs.append({
                    "valu": [("vbroadcast", vtmp_node_A, node0_scalar)]
                })
                for k_base in range(0, num_vectors, 6):
                    k_end = min(k_base + 6, num_vectors)
                    self.instrs.append({
                        "valu": [("^", vec_values[k], vec_values[k], vtmp_node_A) for k in range(k_base, k_end)]
                    })
                for grp in range(num_vectors // 8):
                    k_base = grp * 8
                    self.instrs.extend(emit_8vec_hash_25cycles(k_base))
                    # Next Index Calculation for 8 vectors
                    self.instrs.append({
                        "valu": [
                            ("&", cond_vec_group[0], vec_values[k_base + 0], one_vec),
                            ("multiply_add", idx_even_group[0], vec_indices[k_base + 0], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[0], vec_indices[k_base + 0], two_vec, two_vec),
                            ("&", cond_vec_group[1], vec_values[k_base + 1], one_vec),
                            ("multiply_add", idx_even_group[1], vec_indices[k_base + 1], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[1], vec_indices[k_base + 1], two_vec, two_vec),
                        ]
                    })
                    self.instrs.append({
                        "valu": [
                            ("&", cond_vec_group[2], vec_values[k_base + 2], one_vec),
                            ("multiply_add", idx_even_group[2], vec_indices[k_base + 2], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[2], vec_indices[k_base + 2], two_vec, two_vec),
                            ("&", cond_vec_group[3], vec_values[k_base + 3], one_vec),
                            ("multiply_add", idx_even_group[3], vec_indices[k_base + 3], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[3], vec_indices[k_base + 3], two_vec, two_vec),
                        ],
                        "flow": [("vselect", vec_indices[k_base + 0], cond_vec_group[0], idx_odd_group[0], idx_even_group[0])]
                    })
                    self.instrs.append({
                        "valu": [
                            ("&", cond_vec_group[4], vec_values[k_base + 4], one_vec),
                            ("multiply_add", idx_even_group[4], vec_indices[k_base + 4], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[4], vec_indices[k_base + 4], two_vec, two_vec),
                            ("&", cond_vec_group[5], vec_values[k_base + 5], one_vec),
                            ("multiply_add", idx_even_group[5], vec_indices[k_base + 5], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[5], vec_indices[k_base + 5], two_vec, two_vec),
                        ],
                        "flow": [("vselect", vec_indices[k_base + 1], cond_vec_group[1], idx_odd_group[1], idx_even_group[1])]
                    })
                    self.instrs.append({
                        "valu": [
                            ("&", cond_vec_group[6], vec_values[k_base + 6], one_vec),
                            ("multiply_add", idx_even_group[6], vec_indices[k_base + 6], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[6], vec_indices[k_base + 6], two_vec, two_vec),
                            ("&", cond_vec_group[7], vec_values[k_base + 7], one_vec),
                            ("multiply_add", idx_even_group[7], vec_indices[k_base + 7], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[7], vec_indices[k_base + 7], two_vec, two_vec),
                        ],
                        "flow": [("vselect", vec_indices[k_base + 2], cond_vec_group[2], idx_odd_group[2], idx_even_group[2])]
                    })
                    self.instrs.append({
                        "flow": [("vselect", vec_indices[k_base + 3], cond_vec_group[3], idx_odd_group[3], idx_even_group[3])]
                    })
                    self.instrs.append({
                        "flow": [("vselect", vec_indices[k_base + 4], cond_vec_group[4], idx_odd_group[4], idx_even_group[4])]
                    })
                    self.instrs.append({
                        "flow": [("vselect", vec_indices[k_base + 5], cond_vec_group[5], idx_odd_group[5], idx_even_group[5])]
                    })
                    self.instrs.append({
                        "flow": [("vselect", vec_indices[k_base + 6], cond_vec_group[6], idx_odd_group[6], idx_even_group[6])]
                    })
                    self.instrs.append({
                        "flow": [("vselect", vec_indices[k_base + 7], cond_vec_group[7], idx_odd_group[7], idx_even_group[7])]
                    })

            elif round_i == 1:
                self.instrs.append({
                    "alu": [
                        ("+", addr_t1, self.scratch["forest_values_p"], one_const),
                        ("+", addr_t2, self.scratch["forest_values_p"], two_const)
                    ]
                })
                self.instrs.append({
                    "load": [
                        ("load", node1_scalar, addr_t1),
                        ("load", node2_scalar, addr_t2)
                    ]
                })
                self.instrs.append({
                    "valu": [
                        ("vbroadcast", t1_vec, node1_scalar),
                        ("vbroadcast", t2_vec, node2_scalar)
                    ]
                })
                for grp in range(num_vectors // 8):
                    k_base = grp * 8
                    self.instrs.append({
                        "valu": [("==", cond_vec_group[m], vec_indices[k_base + m], one_vec) for m in range(6)]
                    })
                    self.instrs.append({
                        "valu": [("==", cond_vec_group[6 + m], vec_indices[k_base + 6 + m], one_vec) for m in range(2)]
                    })
                    for m in range(8):
                        self.instrs.append({
                            "flow": [("vselect", vtmp1_group[m], cond_vec_group[m], t1_vec, t2_vec)],
                        })
                        self.instrs.append({
                            "valu": [("^", vec_values[k_base + m], vec_values[k_base + m], vtmp1_group[m])]
                        })

                    self.instrs.extend(emit_8vec_hash_25cycles(k_base))
                    # Next Index Calculation for 8 vectors
                    self.instrs.append({
                        "valu": [
                            ("&", cond_vec_group[0], vec_values[k_base + 0], one_vec),
                            ("multiply_add", idx_even_group[0], vec_indices[k_base + 0], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[0], vec_indices[k_base + 0], two_vec, two_vec),
                            ("&", cond_vec_group[1], vec_values[k_base + 1], one_vec),
                            ("multiply_add", idx_even_group[1], vec_indices[k_base + 1], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[1], vec_indices[k_base + 1], two_vec, two_vec),
                        ]
                    })
                    self.instrs.append({
                        "valu": [
                            ("&", cond_vec_group[2], vec_values[k_base + 2], one_vec),
                            ("multiply_add", idx_even_group[2], vec_indices[k_base + 2], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[2], vec_indices[k_base + 2], two_vec, two_vec),
                            ("&", cond_vec_group[3], vec_values[k_base + 3], one_vec),
                            ("multiply_add", idx_even_group[3], vec_indices[k_base + 3], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[3], vec_indices[k_base + 3], two_vec, two_vec),
                        ],
                        "flow": [("vselect", vec_indices[k_base + 0], cond_vec_group[0], idx_odd_group[0], idx_even_group[0])]
                    })
                    self.instrs.append({
                        "valu": [
                            ("&", cond_vec_group[4], vec_values[k_base + 4], one_vec),
                            ("multiply_add", idx_even_group[4], vec_indices[k_base + 4], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[4], vec_indices[k_base + 4], two_vec, two_vec),
                            ("&", cond_vec_group[5], vec_values[k_base + 5], one_vec),
                            ("multiply_add", idx_even_group[5], vec_indices[k_base + 5], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[5], vec_indices[k_base + 5], two_vec, two_vec),
                        ],
                        "flow": [("vselect", vec_indices[k_base + 1], cond_vec_group[1], idx_odd_group[1], idx_even_group[1])]
                    })
                    self.instrs.append({
                        "valu": [
                            ("&", cond_vec_group[6], vec_values[k_base + 6], one_vec),
                            ("multiply_add", idx_even_group[6], vec_indices[k_base + 6], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[6], vec_indices[k_base + 6], two_vec, two_vec),
                            ("&", cond_vec_group[7], vec_values[k_base + 7], one_vec),
                            ("multiply_add", idx_even_group[7], vec_indices[k_base + 7], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[7], vec_indices[k_base + 7], two_vec, two_vec),
                        ],
                        "flow": [("vselect", vec_indices[k_base + 2], cond_vec_group[2], idx_odd_group[2], idx_even_group[2])]
                    })
                    self.instrs.append({
                        "flow": [("vselect", vec_indices[k_base + 3], cond_vec_group[3], idx_odd_group[3], idx_even_group[3])]
                    })
                    self.instrs.append({
                        "flow": [("vselect", vec_indices[k_base + 4], cond_vec_group[4], idx_odd_group[4], idx_even_group[4])]
                    })
                    self.instrs.append({
                        "flow": [("vselect", vec_indices[k_base + 5], cond_vec_group[5], idx_odd_group[5], idx_even_group[5])]
                    })
                    self.instrs.append({
                        "flow": [("vselect", vec_indices[k_base + 6], cond_vec_group[6], idx_odd_group[6], idx_even_group[6])]
                    })
                    self.instrs.append({
                        "flow": [("vselect", vec_indices[k_base + 7], cond_vec_group[7], idx_odd_group[7], idx_even_group[7])]
                    })

            elif round_i == 2:
                self.instrs.append({
                    "alu": [
                        ("+", addr_t3, self.scratch["forest_values_p"], c3_sc),
                        ("+", addr_t4, self.scratch["forest_values_p"], c4_sc)
                    ]
                })
                self.instrs.append({
                    "load": [
                        ("load", node3_sc, addr_t3),
                        ("load", node4_sc, addr_t4)
                    ],
                    "alu": [
                        ("+", addr_t5, self.scratch["forest_values_p"], c5_sc),
                        ("+", addr_t6, self.scratch["forest_values_p"], c6_sc)
                    ]
                })
                self.instrs.append({
                    "load": [
                        ("load", node5_sc, addr_t5),
                        ("load", node6_sc, addr_t6)
                    ]
                })
                self.instrs.append({
                    "valu": [
                        ("vbroadcast", t3_vec, node3_sc),
                        ("vbroadcast", t4_vec, node4_sc),
                        ("vbroadcast", t5_vec, node5_sc),
                        ("vbroadcast", t6_vec, node6_sc)
                    ]
                })
                self.instrs.append({
                    "valu": [
                        ("vbroadcast", c3_v_r2, c3_sc),
                        ("vbroadcast", c5_v_r2, c5_sc)
                    ]
                })
                for grp in range(num_vectors // 8):
                    k_base = grp * 8
                    for m in range(8):
                        k = k_base + m
                        self.instrs.append({
                            "valu": [
                                ("==", cond_is_3, vec_indices[k], c3_v_r2),
                                ("==", cond_is_5, vec_indices[k], c5_v_r2),
                                ("<", cond_lt_5, vec_indices[k], c5_v_r2),
                            ]
                        })
                        self.instrs.append({
                            "flow": [("vselect", v34_vec, cond_is_3, t3_vec, t4_vec)]
                        })
                        self.instrs.append({
                            "flow": [("vselect", v56_vec, cond_is_5, t5_vec, t6_vec)]
                        })
                        self.instrs.append({
                            "flow": [("vselect", vtmp1_group[m], cond_lt_5, v34_vec, v56_vec)],
                        })
                        self.instrs.append({
                            "valu": [("^", vec_values[k], vec_values[k], vtmp1_group[m])]
                        })

                    self.instrs.extend(emit_8vec_hash_25cycles(k_base))
                    # Next Index Calculation for 8 vectors
                    self.instrs.append({
                        "valu": [
                            ("&", cond_vec_group[0], vec_values[k_base + 0], one_vec),
                            ("multiply_add", idx_even_group[0], vec_indices[k_base + 0], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[0], vec_indices[k_base + 0], two_vec, two_vec),
                            ("&", cond_vec_group[1], vec_values[k_base + 1], one_vec),
                            ("multiply_add", idx_even_group[1], vec_indices[k_base + 1], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[1], vec_indices[k_base + 1], two_vec, two_vec),
                        ]
                    })
                    self.instrs.append({
                        "valu": [
                            ("&", cond_vec_group[2], vec_values[k_base + 2], one_vec),
                            ("multiply_add", idx_even_group[2], vec_indices[k_base + 2], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[2], vec_indices[k_base + 2], two_vec, two_vec),
                            ("&", cond_vec_group[3], vec_values[k_base + 3], one_vec),
                            ("multiply_add", idx_even_group[3], vec_indices[k_base + 3], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[3], vec_indices[k_base + 3], two_vec, two_vec),
                        ],
                        "flow": [("vselect", vec_indices[k_base + 0], cond_vec_group[0], idx_odd_group[0], idx_even_group[0])]
                    })
                    self.instrs.append({
                        "valu": [
                            ("&", cond_vec_group[4], vec_values[k_base + 4], one_vec),
                            ("multiply_add", idx_even_group[4], vec_indices[k_base + 4], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[4], vec_indices[k_base + 4], two_vec, two_vec),
                            ("&", cond_vec_group[5], vec_values[k_base + 5], one_vec),
                            ("multiply_add", idx_even_group[5], vec_indices[k_base + 5], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[5], vec_indices[k_base + 5], two_vec, two_vec),
                        ],
                        "flow": [("vselect", vec_indices[k_base + 1], cond_vec_group[1], idx_odd_group[1], idx_even_group[1])]
                    })
                    self.instrs.append({
                        "valu": [
                            ("&", cond_vec_group[6], vec_values[k_base + 6], one_vec),
                            ("multiply_add", idx_even_group[6], vec_indices[k_base + 6], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[6], vec_indices[k_base + 6], two_vec, two_vec),
                            ("&", cond_vec_group[7], vec_values[k_base + 7], one_vec),
                            ("multiply_add", idx_even_group[7], vec_indices[k_base + 7], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[7], vec_indices[k_base + 7], two_vec, two_vec),
                        ],
                        "flow": [("vselect", vec_indices[k_base + 2], cond_vec_group[2], idx_odd_group[2], idx_even_group[2])]
                    })
                    self.instrs.append({
                        "flow": [("vselect", vec_indices[k_base + 3], cond_vec_group[3], idx_odd_group[3], idx_even_group[3])]
                    })
                    self.instrs.append({
                        "flow": [("vselect", vec_indices[k_base + 4], cond_vec_group[4], idx_odd_group[4], idx_even_group[4])]
                    })
                    self.instrs.append({
                        "flow": [("vselect", vec_indices[k_base + 5], cond_vec_group[5], idx_odd_group[5], idx_even_group[5])]
                    })
                    self.instrs.append({
                        "flow": [("vselect", vec_indices[k_base + 6], cond_vec_group[6], idx_odd_group[6], idx_even_group[6])]
                    })
                    self.instrs.append({
                        "flow": [("vselect", vec_indices[k_base + 7], cond_vec_group[7], idx_odd_group[7], idx_even_group[7])]
                    })
            elif round_i == 3:
                # Round 3 uses pre-loaded Level 3 nodes (nodes 7..14)
                c7_v = self.alloc_scratch("c7_v", VLEN)
                four_vec = self.alloc_scratch("four_vec", VLEN)
                t_78 = self.alloc_scratch("t_78", VLEN)
                t_910 = self.alloc_scratch("t_910", VLEN)
                t_1112 = self.alloc_scratch("t_1112", VLEN)
                t_1314 = self.alloc_scratch("t_1314", VLEN)
                t_0123 = self.alloc_scratch("t_0123", VLEN)
                t_4567 = self.alloc_scratch("t_4567", VLEN)

                self.instrs.append({
                    "valu": [
                        ("vbroadcast", c7_v, self.scratch_const(7)),
                        ("vbroadcast", four_vec, self.scratch_const(4)),
                    ]
                })
                for grp in range(num_vectors // 8):
                    k_base = grp * 8
                    for m in range(8):
                        k = k_base + m
                        # Subtract 7: rel_idx = vec_indices[k] - 7
                        self.instrs.append({
                            "valu": [
                                ("-", vtmp1_group[m], vec_indices[k], c7_v),
                            ]
                        })
                        self.instrs.append({
                            "valu": [
                                ("&", cond_vec_group[m], vtmp1_group[m], one_vec),
                                ("&", idx_even_group[m], vtmp1_group[m], two_vec),
                                ("&", idx_odd_group[m], vtmp1_group[m], four_vec),
                            ]
                        })
                        # Binary tree vselect
                        # Level 1: select between 7-8, 9-10, 11-12, 13-14
                        self.instrs.append({
                            "flow": [
                                ("vselect", t_78, cond_vec_group[m], node_vecs_r3[1], node_vecs_r3[0]),
                            ]
                        })
                        self.instrs.append({
                            "flow": [
                                ("vselect", t_910, cond_vec_group[m], node_vecs_r3[3], node_vecs_r3[2]),
                            ]
                        })
                        self.instrs.append({
                            "flow": [
                                ("vselect", t_1112, cond_vec_group[m], node_vecs_r3[5], node_vecs_r3[4]),
                            ]
                        })
                        self.instrs.append({
                            "flow": [
                                ("vselect", t_1314, cond_vec_group[m], node_vecs_r3[7], node_vecs_r3[6]),
                            ]
                        })
                        # Level 2: select 4-quads
                        self.instrs.append({
                            "flow": [
                                ("vselect", t_0123, idx_even_group[m], t_910, t_78),
                            ]
                        })
                        self.instrs.append({
                            "flow": [
                                ("vselect", t_4567, idx_even_group[m], t_1314, t_1112),
                            ]
                        })
                        # Level 3: final select
                        self.instrs.append({
                            "flow": [
                                ("vselect", vtmp3_group[m], idx_odd_group[m], t_4567, t_0123),
                            ]
                        })
                        self.instrs.append({
                            "valu": [
                                ("^", vec_values[k], vec_values[k], vtmp3_group[m])
                            ]
                        })

                    self.instrs.extend(emit_8vec_hash_25cycles(k_base))
                    # 3. Next Index Calculation for 8 vectors (9 cycles), overlapped with ALU addrs & gather loads for vectors 6, 7
                    # Bundle 0 (Cycle 24): Multiply_add m=0..3 & g_addrs[6]
                    self.instrs.append({
                        "valu": [
                            ("&", cond_vec_group[0], vec_values[k_base + 0], one_vec),
                            ("multiply_add", idx_even_group[0], vec_indices[k_base + 0], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[0], vec_indices[k_base + 0], two_vec, two_vec),
                            ("&", cond_vec_group[1], vec_values[k_base + 1], one_vec),
                            ("multiply_add", idx_even_group[1], vec_indices[k_base + 1], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[1], vec_indices[k_base + 1], two_vec, two_vec),
                        ],
                        "alu": [("+", g_addrs[6][l], self.scratch["forest_values_p"], vec_indices[k_base + 6] + l) for l in range(VLEN)],
                    })
                    # Bundle 1 (Cycle 25): Multiply_add m=2..3, vselect m=0, load g_nodes[6]+0,1
                    self.instrs.append({
                        "valu": [
                            ("&", cond_vec_group[2], vec_values[k_base + 2], one_vec),
                            ("multiply_add", idx_even_group[2], vec_indices[k_base + 2], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[2], vec_indices[k_base + 2], two_vec, two_vec),
                            ("&", cond_vec_group[3], vec_values[k_base + 3], one_vec),
                            ("multiply_add", idx_even_group[3], vec_indices[k_base + 3], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[3], vec_indices[k_base + 3], two_vec, two_vec),
                        ],
                        "flow": [("vselect", vtmp3_group[0], cond_vec_group[0], idx_odd_group[0], idx_even_group[0])],
                        "load": [("load", g_nodes[6] + 0, g_addrs[6][0]), ("load", g_nodes[6] + 1, g_addrs[6][1])]
                    })
                    # Bundle 2 (Cycle 26): Multiply_add m=4..5, vselect m=1, load g_nodes[6]+2,3
                    self.instrs.append({
                        "valu": [
                            ("&", cond_vec_group[4], vec_values[k_base + 4], one_vec),
                            ("multiply_add", idx_even_group[4], vec_indices[k_base + 4], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[4], vec_indices[k_base + 4], two_vec, two_vec),
                            ("&", cond_vec_group[5], vec_values[k_base + 5], one_vec),
                            ("multiply_add", idx_even_group[5], vec_indices[k_base + 5], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[5], vec_indices[k_base + 5], two_vec, two_vec),
                        ],
                        "flow": [("vselect", vtmp3_group[1], cond_vec_group[1], idx_odd_group[1], idx_even_group[1])],
                        "load": [("load", g_nodes[6] + 2, g_addrs[6][2]), ("load", g_nodes[6] + 3, g_addrs[6][3])]
                    })
                    # Bundle 3 (Cycle 27): Multiply_add m=6..7, vselect m=2, load g_nodes[6]+4,5
                    self.instrs.append({
                        "valu": [
                            ("&", cond_vec_group[6], vec_values[k_base + 6], one_vec),
                            ("multiply_add", idx_even_group[6], vec_indices[k_base + 6], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[6], vec_indices[k_base + 6], two_vec, two_vec),
                            ("&", cond_vec_group[7], vec_values[k_base + 7], one_vec),
                            ("multiply_add", idx_even_group[7], vec_indices[k_base + 7], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[7], vec_indices[k_base + 7], two_vec, two_vec),
                        ],
                        "flow": [("vselect", vtmp3_group[2], cond_vec_group[2], idx_odd_group[2], idx_even_group[2])],
                        "load": [("load", g_nodes[6] + 4, g_addrs[6][4]), ("load", g_nodes[6] + 5, g_addrs[6][5])]
                    })
                    # Bundle 4 (Cycle 28): g_addrs[7], vselect m=3, load g_nodes[6]+6,7
                    self.instrs.append({
                        "alu": [("+", g_addrs[7][l], self.scratch["forest_values_p"], vec_indices[k_base + 7] + l) for l in range(VLEN)],
                        "flow": [("vselect", vtmp3_group[3], cond_vec_group[3], idx_odd_group[3], idx_even_group[3])],
                        "load": [("load", g_nodes[6] + 6, g_addrs[6][6]), ("load", g_nodes[6] + 7, g_addrs[6][7])]
                    })
                    # Bundle 5 (Cycle 29): vselect m=4, is_not_leaf m=0..3, load g_nodes[7]+0,1
                    self.instrs.append({
                        "valu": [
                            ("<", is_not_leaf_group[0], vtmp3_group[0], leaf_limit_vec),
                            ("<", is_not_leaf_group[1], vtmp3_group[1], leaf_limit_vec),
                            ("<", is_not_leaf_group[2], vtmp3_group[2], leaf_limit_vec),
                            ("<", is_not_leaf_group[3], vtmp3_group[3], leaf_limit_vec),
                        ],
                        "flow": [("vselect", vtmp3_group[4], cond_vec_group[4], idx_odd_group[4], idx_even_group[4])],
                        "load": [("load", g_nodes[7] + 0, g_addrs[7][0]), ("load", g_nodes[7] + 1, g_addrs[7][1])]
                    })
                    # Bundle 6 (Cycle 30): vselect m=5, mul m=0..3, load g_nodes[7]+2,3
                    self.instrs.append({
                        "valu": [
                            ("*", vec_indices[k_base + 0], vtmp3_group[0], is_not_leaf_group[0]),
                            ("*", vec_indices[k_base + 1], vtmp3_group[1], is_not_leaf_group[1]),
                            ("*", vec_indices[k_base + 2], vtmp3_group[2], is_not_leaf_group[2]),
                            ("*", vec_indices[k_base + 3], vtmp3_group[3], is_not_leaf_group[3]),
                        ],
                        "flow": [("vselect", vtmp3_group[5], cond_vec_group[5], idx_odd_group[5], idx_even_group[5])],
                        "load": [("load", g_nodes[7] + 2, g_addrs[7][2]), ("load", g_nodes[7] + 3, g_addrs[7][3])]
                    })
                    # Bundle 7 (Cycle 31): vselect m=6, is_not_leaf m=4..5, load g_nodes[7]+4,5
                    self.instrs.append({
                        "valu": [
                            ("<", is_not_leaf_group[4], vtmp3_group[4], leaf_limit_vec),
                            ("<", is_not_leaf_group[5], vtmp3_group[5], leaf_limit_vec),
                        ],
                        "flow": [("vselect", vtmp3_group[6], cond_vec_group[6], idx_odd_group[6], idx_even_group[6])],
                        "load": [("load", g_nodes[7] + 4, g_addrs[7][4]), ("load", g_nodes[7] + 5, g_addrs[7][5])]
                    })
                    # Bundle 8 (Cycle 32): vselect m=7, mul m=4..5, load g_nodes[7]+6,7
                    self.instrs.append({
                        "valu": [
                            ("*", vec_indices[k_base + 4], vtmp3_group[4], is_not_leaf_group[4]),
                            ("*", vec_indices[k_base + 5], vtmp3_group[5], is_not_leaf_group[5]),
                        ],
                        "flow": [("vselect", vtmp3_group[7], cond_vec_group[7], idx_odd_group[7], idx_even_group[7])],
                        "load": [("load", g_nodes[7] + 6, g_addrs[7][6]), ("load", g_nodes[7] + 7, g_addrs[7][7])]
                    })
                    # Bundle 9 (Cycle 33): is_not_leaf m=6..7
                    self.instrs.append({
                        "valu": [
                            ("<", is_not_leaf_group[6], vtmp3_group[6], leaf_limit_vec),
                            ("<", is_not_leaf_group[7], vtmp3_group[7], leaf_limit_vec),
                        ]
                    })
                    # Bundle 10 (Cycle 34): mul m=6..7
                    self.instrs.append({
                        "valu": [
                            ("*", vec_indices[k_base + 6], vtmp3_group[6], is_not_leaf_group[6]),
                            ("*", vec_indices[k_base + 7], vtmp3_group[7], is_not_leaf_group[7]),
                        ]
                    })


            else:
                # Prologue: Load Group 0 addrs and nodes (8 vectors)
                for m in range(8):
                    self.instrs.append({
                        "alu": [("+", g_addrs[m][l], self.scratch["forest_values_p"], vec_indices[m] + l) for l in range(VLEN)]
                    })
                for m in range(8):
                    for lp in range(4):
                        l0, l1 = lp * 2, lp * 2 + 1
                        self.instrs.append({
                            "load": [
                                ("load", g_nodes[m] + l0, g_addrs[m][l0]),
                                ("load", g_nodes[m] + l1, g_addrs[m][l1])
                            ]
                        })

                for grp in range(num_vectors // 8):
                    k_base = grp * 8
                    next_grp = (grp + 1) % (num_vectors // 8)
                    nk_base = next_grp * 8

                    # 1. XOR node values for current 8 vectors
                    self.instrs.append({
                        "valu": [("^", vec_values[k_base + m], vec_values[k_base + m], g_nodes[m]) for m in range(4)]
                    })
                    self.instrs.append({
                        "valu": [("^", vec_values[k_base + 4 + m], vec_values[k_base + 4 + m], g_nodes[4 + m]) for m in range(4)]
                    })

                    # 2. Hash stages (25 cycles for 8 vectors) overlapped with ALU & gather loads for next group (vectors 0..5)
                    hash_b = emit_8vec_hash_25cycles(k_base)

                    # Attach ALU addrs and gather loads for vectors 0..5 into hash_b cycles 0..23
                    # Attach Next Index Calculation & gather loads for next group directly into hash_b[0..24]
                    # Cycle 0: g_addrs[0]
                    hash_b[0]["alu"] = [("+", g_addrs[0][l], self.scratch["forest_values_p"], vec_indices[nk_base + 0] + l) for l in range(VLEN)]
                    hash_b[1]["load"] = [("load", g_nodes[0] + 0, g_addrs[0][0]), ("load", g_nodes[0] + 1, g_addrs[0][1])]
                    hash_b[2]["load"] = [("load", g_nodes[0] + 2, g_addrs[0][2]), ("load", g_nodes[0] + 3, g_addrs[0][3])]
                    hash_b[3]["load"] = [("load", g_nodes[0] + 4, g_addrs[0][4]), ("load", g_nodes[0] + 5, g_addrs[0][5])]
                    hash_b[4]["load"] = [("load", g_nodes[0] + 6, g_addrs[0][6]), ("load", g_nodes[0] + 7, g_addrs[0][7])]

                    # Cycle 4: g_addrs[1]
                    hash_b[4]["alu"] = [("+", g_addrs[1][l], self.scratch["forest_values_p"], vec_indices[nk_base + 1] + l) for l in range(VLEN)]
                    hash_b[5]["load"] = [("load", g_nodes[1] + 0, g_addrs[1][0]), ("load", g_nodes[1] + 1, g_addrs[1][1])]
                    hash_b[6]["load"] = [("load", g_nodes[1] + 2, g_addrs[1][2]), ("load", g_nodes[1] + 3, g_addrs[1][3])]
                    hash_b[7]["load"] = [("load", g_nodes[1] + 4, g_addrs[1][4]), ("load", g_nodes[1] + 5, g_addrs[1][5])]
                    hash_b[8]["load"] = [("load", g_nodes[1] + 6, g_addrs[1][6]), ("load", g_nodes[1] + 7, g_addrs[1][7])]

                    # Cycle 8: g_addrs[2]
                    hash_b[8]["alu"] = [("+", g_addrs[2][l], self.scratch["forest_values_p"], vec_indices[nk_base + 2] + l) for l in range(VLEN)]
                    hash_b[9]["load"] = [("load", g_nodes[2] + 0, g_addrs[2][0]), ("load", g_nodes[2] + 1, g_addrs[2][1])]
                    hash_b[10]["load"] = [("load", g_nodes[2] + 2, g_addrs[2][2]), ("load", g_nodes[2] + 3, g_addrs[2][3])]
                    hash_b[11]["load"] = [("load", g_nodes[2] + 4, g_addrs[2][4]), ("load", g_nodes[2] + 5, g_addrs[2][5])]
                    hash_b[12]["load"] = [("load", g_nodes[2] + 6, g_addrs[2][6]), ("load", g_nodes[2] + 7, g_addrs[2][7])]

                    # Cycle 12: g_addrs[3]
                    hash_b[12]["alu"] = [("+", g_addrs[3][l], self.scratch["forest_values_p"], vec_indices[nk_base + 3] + l) for l in range(VLEN)]
                    hash_b[13]["load"] = [("load", g_nodes[3] + 0, g_addrs[3][0]), ("load", g_nodes[3] + 1, g_addrs[3][1])]
                    hash_b[14]["load"] = [("load", g_nodes[3] + 2, g_addrs[3][2]), ("load", g_nodes[3] + 3, g_addrs[3][3])]
                    hash_b[15]["load"] = [("load", g_nodes[3] + 4, g_addrs[3][4]), ("load", g_nodes[3] + 5, g_addrs[3][5])]
                    hash_b[16]["load"] = [("load", g_nodes[3] + 6, g_addrs[3][6]), ("load", g_nodes[3] + 7, g_addrs[3][7])]

                    # Cycle 16: g_addrs[4]
                    hash_b[16]["alu"] = [("+", g_addrs[4][l], self.scratch["forest_values_p"], vec_indices[nk_base + 4] + l) for l in range(VLEN)]
                    hash_b[17]["load"] = [("load", g_nodes[4] + 0, g_addrs[4][0]), ("load", g_nodes[4] + 1, g_addrs[4][1])]
                    hash_b[18]["load"] = [("load", g_nodes[4] + 2, g_addrs[4][2]), ("load", g_nodes[4] + 3, g_addrs[4][3])]
                    hash_b[19]["load"] = [("load", g_nodes[4] + 4, g_addrs[4][4]), ("load", g_nodes[4] + 5, g_addrs[4][5])]
                    hash_b[20]["load"] = [("load", g_nodes[4] + 6, g_addrs[4][6]), ("load", g_nodes[4] + 7, g_addrs[4][7])]

                    # Cycle 20: g_addrs[5]
                    hash_b[20]["alu"] = [("+", g_addrs[5][l], self.scratch["forest_values_p"], vec_indices[nk_base + 5] + l) for l in range(VLEN)]
                    hash_b[21]["load"] = [("load", g_nodes[5] + 0, g_addrs[5][0]), ("load", g_nodes[5] + 1, g_addrs[5][1])]
                    hash_b[22]["load"] = [("load", g_nodes[5] + 2, g_addrs[5][2]), ("load", g_nodes[5] + 3, g_addrs[5][3])]
                    hash_b[23]["load"] = [("load", g_nodes[5] + 4, g_addrs[5][4]), ("load", g_nodes[5] + 5, g_addrs[5][5])]
                    hash_b[24]["load"] = [("load", g_nodes[5] + 6, g_addrs[5][6]), ("load", g_nodes[5] + 7, g_addrs[5][7])]

                    # 32-cycle group schedule (25 hash_b cycles + 7 Next Index bundles)
                    # Bundle 0 (Cycle 24): Multiply_add m=0..2 (6 VALU ops), g_addrs[6]
                    self.instrs.append({
                        "valu": [
                            ("multiply_add", idx_even_group[0], vec_indices[k_base + 0], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[0], vec_indices[k_base + 0], two_vec, two_vec),
                            ("multiply_add", idx_even_group[1], vec_indices[k_base + 1], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[1], vec_indices[k_base + 1], two_vec, two_vec),
                            ("multiply_add", idx_even_group[2], vec_indices[k_base + 2], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[2], vec_indices[k_base + 2], two_vec, two_vec),
                        ],
                        "alu": [("+", g_addrs[6][l], self.scratch["forest_values_p"], vec_indices[nk_base + 6] + l) for l in range(VLEN)],
                    })
                    # Bundle 1 (Cycle 25): Multiply_add m=3..5 (6 VALU ops), & cond m=0..1, load g_nodes[6]+0,1
                    self.instrs.append({
                        "valu": [
                            ("multiply_add", idx_even_group[3], vec_indices[k_base + 3], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[3], vec_indices[k_base + 3], two_vec, two_vec),
                            ("multiply_add", idx_even_group[4], vec_indices[k_base + 4], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[4], vec_indices[k_base + 4], two_vec, two_vec),
                            ("&", cond_vec_group[0], vec_values[k_base + 0], one_vec),
                            ("&", cond_vec_group[1], vec_values[k_base + 1], one_vec),
                        ],
                        "load": [("load", g_nodes[6] + 0, g_addrs[6][0]), ("load", g_nodes[6] + 1, g_addrs[6][1])]
                    })
                    # Bundle 2 (Cycle 26): Multiply_add m=6..7, & cond m=2..5 (6 VALU ops), vselect m=0, load g_nodes[6]+2,3
                    self.instrs.append({
                        "valu": [
                            ("multiply_add", idx_even_group[5], vec_indices[k_base + 5], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[5], vec_indices[k_base + 5], two_vec, two_vec),
                            ("multiply_add", idx_even_group[6], vec_indices[k_base + 6], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[6], vec_indices[k_base + 6], two_vec, two_vec),
                            ("&", cond_vec_group[2], vec_values[k_base + 2], one_vec),
                            ("&", cond_vec_group[3], vec_values[k_base + 3], one_vec),
                        ],
                        "flow": [("vselect", vtmp3_group[0], cond_vec_group[0], idx_odd_group[0], idx_even_group[0])],
                        "load": [("load", g_nodes[6] + 2, g_addrs[6][2]), ("load", g_nodes[6] + 3, g_addrs[6][3])]
                    })
                    # Bundle 3 (Cycle 27): Multiply_add m=7, & cond m=4..7 (6 VALU ops), vselect m=1, load g_nodes[6]+4,5
                    self.instrs.append({
                        "valu": [
                            ("multiply_add", idx_even_group[7], vec_indices[k_base + 7], two_vec, one_vec),
                            ("multiply_add", idx_odd_group[7], vec_indices[k_base + 7], two_vec, two_vec),
                            ("&", cond_vec_group[4], vec_values[k_base + 4], one_vec),
                            ("&", cond_vec_group[5], vec_values[k_base + 5], one_vec),
                            ("&", cond_vec_group[6], vec_values[k_base + 6], one_vec),
                            ("&", cond_vec_group[7], vec_values[k_base + 7], one_vec),
                        ],
                        "flow": [("vselect", vtmp3_group[1], cond_vec_group[1], idx_odd_group[1], idx_even_group[1])],
                        "load": [("load", g_nodes[6] + 4, g_addrs[6][4]), ("load", g_nodes[6] + 5, g_addrs[6][5])]
                    })
                    # Bundle 4 (Cycle 28): g_addrs[7], vselect m=2, load g_nodes[6]+6,7
                    self.instrs.append({
                        "alu": [("+", g_addrs[7][l], self.scratch["forest_values_p"], vec_indices[nk_base + 7] + l) for l in range(VLEN)],
                        "flow": [("vselect", vtmp3_group[2], cond_vec_group[2], idx_odd_group[2], idx_even_group[2])],
                        "load": [("load", g_nodes[6] + 6, g_addrs[6][6]), ("load", g_nodes[6] + 7, g_addrs[6][7])]
                    })
                    # Bundle 5 (Cycle 29): vselect m=3, is_not_leaf m=0..3, load g_nodes[7]+0,1
                    self.instrs.append({
                        "valu": [
                            ("<", is_not_leaf_group[0], vtmp3_group[0], leaf_limit_vec),
                            ("<", is_not_leaf_group[1], vtmp3_group[1], leaf_limit_vec),
                            ("<", is_not_leaf_group[2], vtmp3_group[2], leaf_limit_vec),
                            ("<", is_not_leaf_group[3], vtmp3_group[3], leaf_limit_vec),
                        ],
                        "flow": [("vselect", vtmp3_group[3], cond_vec_group[3], idx_odd_group[3], idx_even_group[3])],
                        "load": [("load", g_nodes[7] + 0, g_addrs[7][0]), ("load", g_nodes[7] + 1, g_addrs[7][1])]
                    })
                    # Bundle 6 (Cycle 30): vselect m=4, mul m=0..3, load g_nodes[7]+2,3
                    self.instrs.append({
                        "valu": [
                            ("*", vec_indices[k_base + 0], vtmp3_group[0], is_not_leaf_group[0]),
                            ("*", vec_indices[k_base + 1], vtmp3_group[1], is_not_leaf_group[1]),
                            ("*", vec_indices[k_base + 2], vtmp3_group[2], is_not_leaf_group[2]),
                            ("*", vec_indices[k_base + 3], vtmp3_group[3], is_not_leaf_group[3]),
                        ],
                        "flow": [("vselect", vtmp3_group[4], cond_vec_group[4], idx_odd_group[4], idx_even_group[4])],
                        "load": [("load", g_nodes[7] + 2, g_addrs[7][2]), ("load", g_nodes[7] + 3, g_addrs[7][3])]
                    })
                    # Bundle 7 (Cycle 31): vselect m=5, load g_nodes[7]+4,5
                    self.instrs.append({
                        "flow": [("vselect", vtmp3_group[5], cond_vec_group[5], idx_odd_group[5], idx_even_group[5])],
                        "load": [("load", g_nodes[7] + 4, g_addrs[7][4]), ("load", g_nodes[7] + 5, g_addrs[7][5])]
                    })
                    # Bundle 8 (Cycle 32): vselect m=6, load g_nodes[7]+6,7
                    self.instrs.append({
                        "flow": [("vselect", vtmp3_group[6], cond_vec_group[6], idx_odd_group[6], idx_even_group[6])],
                        "load": [("load", g_nodes[7] + 6, g_addrs[7][6]), ("load", g_nodes[7] + 7, g_addrs[7][7])]
                    })
                    # Bundle 9 (Cycle 33): vselect m=7, is_not_leaf m=4..6
                    self.instrs.append({
                        "valu": [
                            ("<", is_not_leaf_group[4], vtmp3_group[4], leaf_limit_vec),
                            ("<", is_not_leaf_group[5], vtmp3_group[5], leaf_limit_vec),
                            ("<", is_not_leaf_group[6], vtmp3_group[6], leaf_limit_vec),
                        ],
                        "flow": [("vselect", vtmp3_group[7], cond_vec_group[7], idx_odd_group[7], idx_even_group[7])],
                    })
                    # Bundle 10 (Cycle 34): is_not_leaf m=7, mul m=4..6
                    self.instrs.append({
                        "valu": [
                            ("<", is_not_leaf_group[7], vtmp3_group[7], leaf_limit_vec),
                            ("*", vec_indices[k_base + 4], vtmp3_group[4], is_not_leaf_group[4]),
                            ("*", vec_indices[k_base + 5], vtmp3_group[5], is_not_leaf_group[5]),
                            ("*", vec_indices[k_base + 6], vtmp3_group[6], is_not_leaf_group[6]),
                        ]
                    })
                    # Bundle 11 (Cycle 35): mul m=7
                    self.instrs.append({
                        "valu": [
                            ("*", vec_indices[k_base + 7], vtmp3_group[7], is_not_leaf_group[7]),
                        ]
                    })

        for k in range(num_vectors):
            offset_const = self.scratch_const(k * VLEN)
            self.instrs.append({
                "alu": [("+", addr_scratch, self.scratch["inp_values_p"], offset_const)]
            })
            self.instrs.append({
                "store": [("vstore", addr_scratch, vec_values[k])]
            })
            self.instrs.append({
                "alu": [("+", addr_scratch, self.scratch["inp_indices_p"], offset_const)]
            })
            self.instrs.append({
                "store": [("vstore", addr_scratch, vec_indices[k])]
            })

        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    machine.enable_pause = False
    machine.run()

    ref_mems = list(reference_kernel2(mem, value_trace))
    ref_mem = ref_mems[-1]

    inp_values_p = ref_mem[6]
    if prints:
        print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
        print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
    mismatches = [
        i for i in range(len(inp.values))
        if machine.mem[inp_values_p + i] != ref_mem[inp_values_p + i]
    ]
    print(f"MISMATCH COUNT: {len(mismatches)}")
    print(f"MISMATCHED ITEMS: {mismatches[:10]}")
    if mismatches:
        print(f"Item {mismatches[0]}: actual={machine.mem[inp_values_p + mismatches[0]]}, expected={ref_mem[inp_values_p + mismatches[0]]}")
    assert (
        machine.mem[inp_values_p : inp_values_p + len(inp.values)]
        == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
    ), "Incorrect output values"
    inp_indices_p = ref_mem[5]
    if prints:
        print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
