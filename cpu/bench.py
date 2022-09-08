from subprocess import Popen, PIPE, STDOUT
import os
import argparse
import copy
import re
import math


CurveParameters = {
    69:  [32, 20, 23],
    72:  [34, 22, 5],
    76:  [36, 22, 32],
    82:  [38, 24, 41],
    87:  [40, 26, 41],
    89:  [42, 26, 37],
    95:  [44, 28, 73],
    99:  [46, 30, 29],
    100: [48, 30, 13],
    107: [50, 32, 41],
    111: [52, 32, 197],
    117: [54, 33, 409],
    118: [56, 36, 19],
    124: [58, 37, 155],
    126: [60, 36, 307],
    129: [62, 37, 281],
    131: [64, 38, 85],
    140: [70, 39, 233],
    152: [76, 45, 19],
    164: [82, 47, 95],
    176: [88, 49, 1151],
}


class Range(argparse.Action):
    def __init__(self, *args, **kwargs):
        super(Range, self).__init__(*args, **kwargs)

    def parse_single_number(self, value: str):
        try:
            matches = re.findall("[0-9]+", value)
            if len(matches) != 1:
                return False, 0
            return True, int(matches[0])

        except:
            return False, 0

    def parse_double_number(self, value: str):
        try:
            matches = re.findall("[0-9]*,[0-9]*", value)
            if len(matches) != 1:
                return False, [0, 0]

            splits = matches[0].split(",")
            return True, [int(a) for a in splits]

        except:
            return False, [0, 0]

    def __call__(self, parser, namespace, value: str, option_string=None):
        ret, parsed_value = self.parse_single_number(value)
        if not ret:
            ret, parsed_value = self.parse_double_number(value)
            if not ret:
                msg = "Invalid format: " + value
                raise argparse.ArgumentError(self, msg)
            else:
                self.min = parsed_value[0]
                self.max = parsed_value[1]
        else:
            self.min = parsed_value
            self.max = parsed_value

        if self.min > self.max:
            msg = "invalid paramters: " + str(self.min) + " > " + str(self.max)
            raise argparse.ArgumentError(self, msg)

        setattr(namespace, self.dest, [self.min, self.max])


class MemoryStringsRange(argparse.Action):
    def __init__(self, *args, **kwargs):
        super(MemoryStringsRange, self).__init__(*args, **kwargs)

    def parse_single_number(self, value: str):
        try:
            ret = int(value)
            return True, ret
        except:
            return False, 0

    def parse_double_number(self, value: str):
        try:
            matches = re.findall("[0-9]*,[0-9]*", value)
            if len(matches) != 1:
                return False, [0, 0]

            splits = matches[0].split(",")
            return True, [int(a) for a in splits]

        except:
            return False, [0, 0]

    def parse_memory_specification(self, value: str):
        valid_specifications = ["kb", "KB", "mb", "MB", "gb", "GB", "tb", "TB"]
        value = value.strip()
        spec = value[-2:]
        print(spec)
        if spec not in valid_specifications:
            raise argparse.ArgumentError(self, spec + " is not a valid "
                    "specifier, must be in " + str(valid_specifications))

        mult = 1
        mmmm = 1000
        if spec in ["kb", "KB"]:
            mult = mmmm
        if spec in ["mb", "MB"]:
            mult = mmmm**2
        if spec in ["gb", "GB"]:
            mult = mmmm**3
        if spec in ["tb", "TB"]:
            mult = mmmm**4

        mem_value = value[:-2]
        try:
            mem_value = float(mem_value)
            return True, mem_value*mult
        except:
            return False, 0.0

    def __call__(self, parser, namespace, value: str, option_string=None):
        # first try to parse an integer
        ret, parsed_value = self.parse_single_number(value)
        if ret:
            setattr(namespace, self.dest, [parsed_value, parsed_value])
            return

        # next try to parse `x,y`
        ret, parsed_value = self.parse_double_number(value)
        if ret:
            setattr(namespace, self.dest, parsed_value)
            return

        # finally parse `128GB`
        ret, parsed_value = self.parse_memory_specification(value)
        if ret:
            self.val = parsed_value
            setattr(namespace, self.dest, self.val)
            return

        msg = "Invalid format: " + value
        raise argparse.ArgumentError(self, msg)


def mitm_mem(bytes: float, delta: int, e2: int, e3: int):
    """
    bytes: memory available
    delta: precomputation depth
    """
    memory_bytes = bytes
    memory_coefficient = e2 - e2 // 2
    memory_j_invariant = 2 * (e2 + e3 * math.log2(3))
    memory_leaf = (memory_coefficient + memory_j_invariant) // 8
    memory_bytes -= 2 * 4 * ((memory_j_invariant + 7) // 8) * (2 ** delta)
    W = memory_bytes / memory_leaf
    return int(math.log2(W))


def vowgcs_mem(bytes: float, delta: int, e2: int, e3: int):
    """
    bytes: memory available
    delta: precomputation depth
    """
    memory_bytes = bytes
    memory_j_invariant = 2 * (e2 + e3 * math.log2(3))
    e = (e2 - 2) // 2
    cell_bytes = (math.ceil(e + 10 + math.log2(20)) + 7) // 8
    memory_bytes -= 2 * 4 * ((memory_j_invariant + 7) // 8) * (2 ** delta)
    W = memory_bytes / cell_bytes
    return int(math.log2(W))


def vowgcs_max_mem(e2: int, e3: int):
    e = (e2 - 3) // 2
    if e2 % 2 == 0:
        logN = e + math.log(3, 2)
    else:
        logN = e + 1
    w_max = e2 - 13
    while( math.floor(0.5*logN - math.log(2.25, 2) - 0.5*w_max) + 4 + w_max > e ):
        w_max -= 1
    return int(w_max)


def parse_run_output(data: str, args):
    """
    return 0.123, 10,
        time,
        iters,
        colls avg per func,
        diff_colls avg per func,
        funcs_evals log2
        clocks in log
    """
    # print("parse_run_output", data)
    regex_int = r"(\d+)"
    regex_float = r"(\d+\.\d+)"

    time = -1.0
    iters = 0
    colls = -1.0
    diff_colls = -1.0
    evals = -1.0
    clock = -1.0

    for line in data:
        if line.__contains__("#(prf):"):
            match = re.findall(regex_int, line)
            assert(len(match))
            iters = int(match[0])

        if line.__contains__("#(collisions):") and args.algorithm != "mitm":
            match = re.findall(regex_float, line)
            assert(len(match))
            colls = float(match[0])

        if line.__contains__("#(different collisions):") and args.algorithm != "mitm":
            match = re.findall(regex_float, line)
            assert(len(match))
            diff_colls = float(match[0])

        if line.__contains__("#(function evaluations):") and args.algorithm != "mitm":
            match = re.findall(regex_float, line)
            assert(len(match))
            evals = float(match[0])

        if line.__contains__("clock cycles:"):
            match = re.findall(regex_float, line)
            assert(len(match))
            clock = float(match[0])

        if line.startswith("Time:"):
            match = re.findall(regex_int, line)
            assert(len(match))
            time = int(match[0])/(10**6)

    data2 = (time, iters, colls, diff_colls, evals, clock)
    return data2


def rebuild(path: str, args):
    """
    runs the `make` command in the current directory
    :return: nothing
    """
    # befor we can build the target, we need to check if we setup the
    # precomputation
    if args.d != 0:
        # ok we need to run the precomputation script
        cmd = ["python3", "src/setup.py", "--e2", str(args.e2), "--e3",
               str(args.e3), "--f", str(args.ff), "--pc", str(args.d)]
        # print(cmd)
        p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
        p.wait()
        if p.returncode != 0:
            print("ERROR precomputation Build", p.returncode,
                  p.stdout.read().decode("utf-8"))

    # make sure that we are able to build everything
    algorithm_target = args.algorithm if args.algorithm != "mitmdfs" else "mitm"
    algorithm = "instance_generator" if args.random else algorithm_target
    model = args.model
    arith = args.arith
    target = "tests" + str(args.p)

    # next recreate the config
    cmd = ["make", "ALGORITHM="+algorithm, target, "MODEL="+model,
           "ARITH="+arith, "-B"]
    # print("rebuild", cmd)
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True,
              cwd=path)
    p.wait()
    if p.returncode != 0:
        print("ERROR Build", p.returncode, p.stdout.read().decode("utf-8"))

    data = p.stdout.read()
    return p.returncode, data


def run(args):
    """
    runs `./main` in `build`
    :return: returncode, stdout
    """

    side = "-s" + args.side
    cores = "-c" + str(args.c) if args.algorithm == "vowgcs" else "-c" + str(int(math.log2(args.c)))
    memory = "-w" + str(args.m)
    functions = "-f " + str(args.f) if args.f != -1 else ""
    path = "SIPFD" + str(args.p)
    target = "vow-gcs" if args.algorithm == "vowgcs" else "mitm-basic"
    target = target if args.algorithm != "mitmdfs" else "mitm-dfs"

    if args.m != 0 and args.algorithm == "mitm":
        target = "mitm-memory-limited"
    elif args.m == 0 and args.algorithm == "mitm":
        memory = ''

    if args.m != 0 and args.algorithm == "mitmdfs":
        target = "mitm-dfs-memory-limited"
    elif args.m == 0 and args.algorithm == "mitmdfs":
        memory = ''

    target += "_x64_"

    cmd = ["./" + target, side, memory, cores, functions]

    # check if a precomputation depth is given
    if args.d != 0 and args.algorithm == "vowgcs":
        cmd.append("-p")

    # check of we need to add a beta flag
    if args.beta != 0 and args.algorithm == "vowgcs":
        cmd.append("-b " + str(args.beta))
    # print("run", cmd)
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=STDOUT,
              preexec_fn=os.setsid, cwd=path)
    p.wait()

    if p.returncode != 0:
        print("ERROR RUN", p.returncode, p.stdout.read().decode("utf-8"))

    data = p.stdout.readlines()

    # the first replace, removes b' from each string
    # the second removes all the unnecessary \\b
    # and the last lstrip removes leading whitespaces.
    data = [str(a).replace("b'", "")
            .replace("\\n'", "")
            .lstrip() for a in data]

    print(data)
    return p.returncode, parse_run_output(data, args)


def run_all_benches(path: str, args):
    """
    """
    bench_data = {}
    splits = args.algorithm.split(",")

    for algo in splits:
        for d_value in range(args.d[0], args.d[1] + 1):
            m_lower, m_upper = 0, 0

            # check if a cell count was given or a mem bound
            if type(args.m) is not list:
                if algo == "mitm":
                    w = mitm_mem(args.m, d_value, args.e2, args.e3)
                else:
                    w = vowgcs_mem(args.m, d_value, args.e2, args.e3)
                m_lower = w
                m_upper = w
            else:
                m_lower = args.m[0]
                m_upper = args.m[1]

            if algo == "vowgcs":
                m_max = vowgcs_max_mem(args.e2, args.e3)
                m_lower = min(m_lower, m_max)
                m_upper = min(m_upper, m_max)

            if algo == "vowgcs" and m_lower == 0:
                print("vowgcs and m=0, does not make any sense, skipping")
                continue

            for m_value in range(m_lower, m_upper + 1):
                for c_value in range(args.c[0], args.c[1] + 1):
                    for beta_value in range(args.beta[0], args.beta[1] + 1):
                        data_point = (algo, m_value, c_value, d_value,
                                      beta_value)
                        args2 = copy.copy(args)
                        args2.m = m_value
                        args2.c = c_value
                        args2.d = d_value
                        args2.beta = beta_value
                        args2.algorithm = algo

                        print("Building", args2)
                        returncode, _ = rebuild(path, args2)
                        if returncode != 0:
                            continue

                        print("Running", args2)

                        time_points, iter_points = [], []
                        for _ in range(args.i):
                            returncode, data = run(args2)
                            if returncode != 0:
                                continue

                            time_points.append(data[0])
                            iter_points.append(data[1])

                        bench_data[data_point] = [time_points, iter_points]

    return bench_data


def show_benchinformation(args, bench_data: dict):
    """
    """
    for data_point in bench_data:
        algo, m_value, c_value, d_value, beta_value = data_point
        headers = []
        timing_points, iter_points = bench_data[data_point]
        if args.plot:
            import matplotlib.pyplot as plt
            # del(headers[0])
            plt.plot(timing_points, headers)
            plt.show()

        if args.table:
            from tabulate import tabulate
            print(tabulate(timing_points, headers=headers))

        if args.plot or args.table:
            return

        print("Algo:", algo, "Mem:", m_value, "Cores:", c_value, "Precomp:",
              d_value, "Beta:", beta_value)
        print("timing=", timing_points)
        print("iters =", iter_points)
        print("\n")

    return


def main():
    parser = argparse.ArgumentParser(description='SIKE')
    parser.add_argument('-p', required=True, type=int,
                        help='primesize', default=69)
    parser.add_argument('-m', required=False, type=str, action=MemoryStringsRange,
                        help="log mem, can be 8,10 to iterate over all"
                        "values from 8 to 10 including the bounds",
                        default=[8, 8])
    parser.add_argument('-c', required=False, type=str, action=Range,
                        help='log cpu, can be 8,10 (see memory)',
                        default=[8, 8])
    parser.add_argument('-d', required=False, type=str, action=Range,
                        help='depth of precomputation, can be 8,10, (see mem)',
                        default=[0, 0])

    parser.add_argument('-f', required=False, type=int,
                        help='number of function evaluations, only for vOW',
                        default=-1)
    parser.add_argument('-i', required=False, type=int,
                        help='number of iterations per configuration',
                        default=1)
    parser.add_argument('-r', '--random', required=False, action='store_true',
                        help='generate a new random instance.')
    parser.add_argument('-a', '--algorithm', required=False, type=str,
                        help='vowgcs/mitm/mitmdfs', default="vowgcs")
    parser.add_argument('-b', '--beta', required=False, type=str, action=Range,
                        help='beta value', default=[0, 0])

    parser.add_argument('--arith', required=False, type=str,
                        help='asm/fiat.', default="asm")
    parser.add_argument('--model', required=False, type=str,
                        help='mont/shortw', default="mont")
    parser.add_argument('--side', required=False, type=str,
                        help='Alice/Bob', default="Alice")

    parser.add_argument('--e2', required=False, type=int, default=0,
                        help='e2 value.')
    parser.add_argument('--e3', required=False, type=int, default=0,
                        help='e3 value.')
    parser.add_argument('--ff', required=False, type=int, default=0,
                        help='f value.')

    parser.add_argument('--plot', required=False, action='store_true',
                        help="Plot the result with matplotlib, instead of"
                        "printing it. (NOT FINISHED)")
    parser.add_argument('--table', required=False, action='store_true',
                        help='Prints the data in table. (NOT FINISHED)')
    args = parser.parse_args()

    if args.arith not in ["asm", "fiat"]:
        print("wrong arith")
        return

    if args.algorithm not in ["vowgcs", "mitm", "mitm,vowgcs", "vowgcs,mitm",
            "mitmdfs", "mitm,mitmdfs", "mitmdfs,mitm", "vowgcs,mitmdfs",
            "mitmdfs,vowgcs", "mitm,mitmdfs,vowgcs", "mitm,vowgcs,mitmds",
            "vowgcs,mitm,mitmdfs", "vowgcs,mitmdfs,mitm", "mitmdfs,mitm,vowgcs",
            "mitmdfs,vowgcs,mitm"]:
        print("wrong algorithm")
        return

    if args.model not in ["mont", "shortw"]:
        print("wrong model")
        return

    if args.p in CurveParameters.keys():
        args.e2 = CurveParameters[args.p][0]
        args.e3 = CurveParameters[args.p][1]
        args.ff = CurveParameters[args.p][2]
    else:
        if args.e2 == 0 or args.e3 == 0 or args.ff == 0:
            print("ERROR could not find the prime in my internal list of"
                  "parameters. Please specify e2, e3, and f")
            return 1

    path = "./"
    bench_data = run_all_benches(path, args)
    show_benchinformation(args, bench_data)


if __name__ == "__main__":
    main()
