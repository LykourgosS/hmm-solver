import math
from collections import deque


def reload_viterbi():
    global states, obs, start_p, trans_p, emit_p, log_probs
    obs = []
    states = []
    start_p = {}
    trans_p = {}
    emit_p = {}
    log_probs = True
    no_Start_n_End = 0


def my_log(prob):
    if prob == 0:
        prob = float("-inf")
    else:
        prob = math.log(prob, 2)
    return prob


# use of global variable log_probs to keep the selection


def probs_in_log():
    global states, start_p, trans_p, emit_p, log_probs
    ans = input("\nDo you want to use logarithmic ratings for calculating viterbi score? [y/n] ").lower()
    if ans == "y":
        start_p = {key: my_log(val_) for key, val_ in start_p.items()}
        trans_p = {key: {key_: my_log(val_) for key_, val_ in val.items()}
                   for key, val in trans_p.items()}
        emit_p = {key: {key_: my_log(val_) for key_, val_ in val.items()}
                  for key, val in emit_p.items()}
        log_probs = True
    else:
        log_probs = False


def get_observations():
    global obs
    obs = input("\nGive the sequence of observations, separating them with a comma (,): ").replace(" ", "").split(",")


def hmm_114():
    global states, start_p, trans_p, emit_p
    states = ["a", "b"]
    start_p = {"a": 0.5, "b": 0.5}
    trans_p = {"a": {"a": 0.9, "b": 0.1},
               "b": {"a": 0.1, "b": 0.9}}
    emit_p = {"a": {"A": 0.4, "G": 0.4, "T": 0.1, "C": 0.1},
              "b": {"T": 0.3, "C": 0.3, "A": 0.2, "G": 0.2}}


def hmm_116():
    global states, start_p, trans_p, emit_p
    states = ["Start", "D1", "D2", "End"]
    start_p = {"Start": 1, "D1": 0, "D2": 0, "End": 0}
    trans_p = {"Start": {"Start": 0, "D1": 0.5, "D2": 0.5, "End": 0},
               "D1": {"Start": 0, "D1": 0.5, "D2": 0.25, "End": 0.25},
               "D2": {"Start": 0, "D1": 0.25, "D2": 0.5, "End": 0.25},
               "End": {"Start": 0, "D1": 0, "D2": 0, "End": 0}}
    emit_p = {"Start": {"1": 0, "2": 0, "3": 0},
              "D1": {"1": 0.5, "2": 0.25, "3": 0.25},
              "D2": {"1": 0.25, "2": 0.5, "3": 0.25},
              "End": {"1": 0, "2": 0, "3": 0}}


def custom_hmm():
    global states, start_p, trans_p, emit_p
    print("\nCustom HMM configuration:\n(WARNING: Don't forget to use quotemarks)")
    states = eval(input('''\nGive the states, using list (up to 8 char per state)...\nExample: ["Start", "D1", "D2", "End"]\n\
(WARNING: If you have start and end state, use the keywords 'Start' and 'End' !)\n\n'''))
    start_p = eval(input('''\nGive the matrix of initial probabilities, using dictionary...\nExample: \
{"Start": 1, "D1": 0, "D2": 0, "End": 0}\n\n'''))
    trans_p = eval(input('''\nGive the matrix of transition probabilities, using dictionary...\nExample: \
{"Start": {"Start": 0, "D1": 0.5, "D2": 0.5, "End": 0},\n\
          "D1": {"Start": 0, "D1": 0.5, "D2": 0.25, "End": 0.25},\n\
          "D2": {"Start": 0, "D1": 0.25, "D2": 0.5, "End": 0.25},\n\
          "End": {"Start": 0, "D1": 0, "D2": 0, "End": 0}}\n\n'''))
    emit_p = eval(input(
        '''\nGive the matrix of emission probabilities, using dictionary (up to 8 char per symbol)...\nExample: \
{"Start": {"1": 0, "2": 0, "3": 0},\n\
          "D1": {"1": 0.5, "2": 0.25, "3": 0.25},\n\
          "D2": {"1": 0.25, "2": 0.5, "3": 0.25},\n\
          "End": {"1": 0, "2": 0, "3": 0}}\n\n'''))


def print_mtrx():
    print(
        "\n--HMM states are %s and the sequence of the observations are %s\n\n--The initial probabilities are: \n%s\n\n"
        "--The transition probabilities are: \n%s\n\n--The emission probabilities are: \n%s"
        % (states, obs, start_p, trans_p, emit_p))


# Used, when having 'Start' and 'End' states, to find the end of observations' sequence --> Exiting hmm
# (last column of the Trellis Diagram)
def add_special_ending_symbol():
    global emit_p, obs
    for key, value in emit_p.items():
        prob = 0
        if key == "End":
            prob = 1
        value["(end)"] = prob
    obs.append("(end)")


def calc_Vscore(a, b):
    if log_probs:
        return a + b
    else:
        return a * b


def best_anc(V_prev_t, st, emit_prob):
    ancestors = []
    # calculating local Viterbi score for all nodes
    for prev_st in states:
        ancestors.append(calc_Vscore(V_prev_t[prev_st]["prob"], trans_p[prev_st][st]))
    # finding the maximum(s) probability(-ies) and the (best) ancestor nodes as well
    max_tr_prob = max(ancestors)
    best_ancestors = [states[i] for i, x in enumerate(ancestors) if x == max_tr_prob]
    max_prob = calc_Vscore(max_tr_prob, emit_prob)
    if max_prob == float("-inf"):
        best_ancestors = []
    return {"prob": max_prob, "prev": best_ancestors}


# Viterbi initialization (when t = 0, for both Case1 and Case2)
# (Case1): starting from the 1st observation, when there is Start state!
# (Case2): starting from the 2nd observation, when there isn't Start state!
def initialize(V):
    global no_Start_n_End
    if "Start" and "End" in states:
        add_special_ending_symbol()
        probs_in_log()
        init_node = {st: {"prob": start_p[st], "prev": [None]} for st in states}
        no_Start_n_End = 0
    else:
        probs_in_log()
        init_node = {st: {"prob": calc_Vscore(
            start_p[st], emit_p[st][obs[0]]), "prev": [None]} for st in states}
        no_Start_n_End = 1
    V.append(init_node)


def dict2table(V):
    # Print a table of steps from dictionary
    if no_Start_n_End == 0:
        obs.insert(0, "(start)")
    for state in V[0]:
        yield "|" + state.center(8) + "||" + "".join(
            (str(float("{0:.2f}".format(v[state]["prob"]))).center(8) + "|") for v in V)
    yield "|" + "St/Sym".center(8) + "||" + "".join((str(i).center(8) + "|") for i in obs)


def trellis_diagram(V):
    td = "".center(len(V) * 9 + 11, "-")
    for line in dict2table(V):
        td += "\n%s" % line
    td += "\n" + "".center(len(V) * 9 + 11, "-")
    return td


def paths_to_str(paths):
    paths_to_print = []
    for i in range(len(paths)):
        paths_to_print.append(paths[i][0])
        for j in range(1, len(paths[i])):
            paths_to_print[-1] += (", %s " % paths[i][j])
    return paths_to_print


def find_best_paths(max_p, V):
    # optimal paths' list
    paths = deque()
    best_paths = []
    # Get most probable state and its backtrack (data = {"prob": ..., "prev": ...})
    # for key,val in dict.items() (ex. key = "D1", val = {"prob": ..., "prev": ...})
    for st, data in V[-1].items():
        if data["prob"] == max_p:
            paths.append([st])
    # Follow the backtrack till the first observation
    while paths:
        tmp_path = paths.popleft()
        t = len(V) - len(tmp_path)
        for prev_st in V[t][tmp_path[0]]["prev"]:
            if prev_st is None:
                best_paths.append(tmp_path)
            else:
                paths.append([prev_st] + tmp_path)
    return paths_to_str(best_paths)


def run_viterbi():
    V = []
    # Viterbi initialization (when t = 0, for both Case1 and Case2)
    initialize(V)
    # build Viterbi table, for t >= 1 (used to make the Trellis Diagram)
    for t in range(no_Start_n_End, len(obs)):
        V.append({st: best_anc(V[-1], st, emit_p[st][obs[t]]) for st in states})
    print("\nTrellis Diagram has been completed...")
    print(trellis_diagram(V))
    # The highest probability of the last column, i.e. of the last symbol of observations
    # (example of value: {"prob": ..., "prev":...})
    max_prob = max(value["prob"] for value in V[-1].values())
    best_paths = find_best_paths(max_prob, V)
    # print best path(s) along with their probability (if selected using logarithmic ratings)
    print("\nThe sequence of states are:\n" + "\n".join(
        " - " + str(path) for path in best_paths) + "\nwith highest probability of %s." % max_prob)


flag = "y"
while flag != "n":
    reload_viterbi()
    while True:
        selection = input(
            "\n a. Run HMM of exercise 11.4.\n "
            "b. Run HMM of exercise 11.6.\n "
            "c. Run an HMM with custom configurations.\n\nSelect an option from above, typing a, b or c: ").lower()
        if selection in ["a", "b", "c"]:
            if selection == "a":
                hmm_114()
            elif selection == "b":
                hmm_116()
            else:
                custom_hmm()
            get_observations()
            print_mtrx()
            run_viterbi()
            break
        else:
            print("Invalid input...")
    flag = input("\nDo you want to restart the algorithm? [y/n] ").lower()
